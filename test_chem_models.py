"""
CPU unit test for the chemistry late-fusion head in CPMLP / CPTransformer (--chem_fusion). Run from the repo root:
    python test_chem_models.py
Uses tiny models with random data; takes a few seconds.
"""
import argparse
import torch
from models import CPMLP, CPTransformer

B, L, LEN, OUT = 4, 10, 20, 28   # batch, cycles, resampled length, output_num (28 = geo_bins; 1 = regression)


def cfg(fusion='none', embed=16, out=OUT):
    return argparse.Namespace(d_model=16, d_ff=32, charge_discharge_length=LEN, early_cycle_threshold=L, dropout=0.1,
                              e_layers=2, d_layers=2, output_num=out, n_heads=4, factor=1, activation='relu',
                              chem_fusion=fusion, chem_embed_dim=embed, chem_num=4)


x = torch.randn(B, L, 3, LEN)
mask = torch.ones(B, L)
ids_a = torch.tensor([0, 1, 2, 3])
ids_b = torch.tensor([3, 2, 1, 0])

for name, mod, head in [('CPMLP', CPMLP, 'head_output'), ('CPTransformer', CPTransformer, 'projection')]:
    # ---- baseline: unchanged behaviour ----
    torch.manual_seed(0)
    base = mod.Model(cfg('none')).eval()
    assert isinstance(getattr(base, head), torch.nn.Linear), 'baseline head must stay a plain Linear'
    assert not any('embedding' in n for n, _ in base.named_parameters())
    y0 = base(x, mask)
    assert y0.shape == (B, OUT)
    assert torch.equal(y0, base(x, mask, chemistry_ids=ids_a)), 'baseline must ignore chemistry_ids'
    assert base(x, mask, return_embedding=True)[0].shape == (B, OUT)
    # regression output size too
    assert mod.Model(cfg('none', out=1)).eval()(x, mask).shape == (B, 1)

    # ---- late_mlp with embedding ----
    m = mod.Model(cfg('late_mlp', embed=16)).eval()
    ya, yb = m(x, mask, chemistry_ids=ids_a), m(x, mask, chemistry_ids=ids_b)
    assert ya.shape == (B, OUT)
    assert torch.equal(ya, m(x, mask, chemistry_ids=ids_a)), 'deterministic in eval mode'
    assert not torch.allclose(ya, yb), 'different chemistry ids must change the output'
    same = m(x, mask, chemistry_ids=torch.zeros(B, dtype=torch.long))
    assert torch.allclose(same[0], m(x[:1], mask[:1], chemistry_ids=torch.zeros(1, dtype=torch.long))[0], atol=1e-5), 'rows are independent'
    try:
        m(x, mask); raise SystemExit(f'{name}: missing chemistry_ids should raise')
    except ValueError:
        pass
    m.train()
    m(x, mask, chemistry_ids=ids_a).sum().backward()
    g = getattr(m, head).embedding.weight.grad
    assert g is not None and g.abs().sum() > 0, 'embedding must receive gradient'
    assert mod.Model(cfg('late_mlp', embed=16, out=1)).eval()(x, mask, chemistry_ids=ids_a).shape == (B, 1)

    # ---- late_mlp, embed 0: capacity control, no chemistry input ----
    c = mod.Model(cfg('late_mlp', embed=0)).eval()
    assert getattr(c, head).embedding is None
    assert torch.equal(c(x, mask), c(x, mask, chemistry_ids=ids_a)), 'embed_dim=0 must ignore chemistry'

    n_base = sum(p.numel() for p in base.parameters()); n_m = sum(p.numel() for p in m.parameters()); n_c = sum(p.numel() for p in c.parameters())
    print(f'{name}: OK | params baseline={n_base:,}  late_mlp(E=16)={n_m:,}  late_mlp(E=0 control)={n_c:,}')
print('ALL OK')
