"""
Checks for models/CPMamba.py. Run on SCC from the repo root, inside the mamba venv:

    module load gcc/12.2.0 cuda/12.2 && source venv_mamba/bin/activate
    python test_cpmamba.py            # CPU, pure-PyTorch scan (selective_scan_ref); works on a login node
    python test_cpmamba.py --gpu      # also: CUDA kernel vs ref scan, and a long (L=4000) sequence fwd+bwd on GPU

What it checks, for --mamba_layer vanilla and mamba_init:
  1. shapes: full-timescale style [1, L, 300, 3] with an all-ones mask, and pooled val style [B, 100, 300, 3]
     right-padded; regression (output_num=1) and geo_bins-sized heads
  2. padding invariance: a prefix of length n padded to 100 inside a mixed batch gives the same prediction as the
     same prefix fed alone unpadded (this is what makes the CLS readout correct on the pooled val/test loaders)
  3. causality: changing cycles AFTER the prefix (in the padded region) doesn't change the output
  4. gradient flow into every parameter incl. cls_token and (mamba_init) initial_state
  5. chem_fusion none / late_mlp / early_concat build, and chemistry id changes the output when E > 0
"""
import argparse
import sys
from types import SimpleNamespace

import torch

from models import CPMamba

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
cli = p.parse_args()

LEN = 300


def make_args(**kw):
    a = dict(d_model=32, d_ff=64, charge_discharge_length=LEN, dropout=0.0, e_layers=2, output_num=1,
             mamba_n_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_layer='vanilla',
             mamba_scan='ref', chem_fusion='none', chem_embed_dim=16, chem_num=4)
    a.update(kw)
    return SimpleNamespace(**a)


def padded_batch(lengths, L=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), L, LEN, 3, generator=g)
    m = torch.zeros(len(lengths), L)
    for b, n in enumerate(lengths):
        m[b, :n] = 1
        x[b, n:] = 0            # the pooled collate zeroes padded cycles too
    return x, m


fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


def run_suite(layer, device, scan):
    print(f'=== mamba_layer={layer} scan={scan} device={device} ===')
    torch.manual_seed(0)
    model = CPMamba.Model(make_args(mamba_layer=layer, mamba_scan=scan)).float().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params}')

    # 1. shapes
    x = torch.randn(1, 137, LEN, 3, device=device)
    y = model(x, torch.ones(1, 137, device=device))
    check('full-timescale shape', tuple(y.shape) == (1, 1), str(tuple(y.shape)))
    lengths = [1, 5, 37, 100]
    xb, mb = padded_batch(lengths)
    xb, mb = xb.to(device), mb.to(device)
    yb = model(xb, mb)
    check('padded batch shape', tuple(yb.shape) == (4, 1), str(tuple(yb.shape)))
    check('finite outputs', bool(torch.isfinite(yb).all()))

    # 2. padding invariance
    with torch.no_grad():
        max_err = 0.0
        for b, n in enumerate(lengths):
            alone = model(xb[b:b + 1, :n], torch.ones(1, n, device=device))
            max_err = max(max_err, (alone - yb[b:b + 1]).abs().max().item())
    check('padded == unpadded prefix', max_err < 1e-4, f'max abs diff {max_err:.2e}')

    # 3. causality: garbage in the padded region must not matter
    with torch.no_grad():
        xg = xb.clone()
        for b, n in enumerate(lengths):
            xg[b, n:] = torch.randn_like(xg[b, n:]) * 10
        diff = (model(xg, mb) - yb).abs().max().item()
    check('padded region ignored', diff < 1e-4, f'max abs diff {diff:.2e}')

    # 4. gradients
    model.train()
    model.zero_grad()
    model(xb, mb).pow(2).sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    check('every parameter gets gradient', not no_grad, str(no_grad[:5]))
    if layer == 'mamba_init':
        g = [p.grad.abs().sum().item() for n, p in model.named_parameters() if n.endswith('initial_state')]
        check('initial_state params present with grad', len(g) == 2 and all(v > 0 for v in g), str(g))

    # 5. chemistry fusion variants + geo_bins-sized head
    for fusion, emb in [('late_mlp', 16), ('late_mlp', 0), ('early_concat', 16), ('early_concat', 0)]:
        m = CPMamba.Model(make_args(mamba_layer=layer, mamba_scan=scan, chem_fusion=fusion, chem_embed_dim=emb,
                                    output_num=57)).float().to(device).eval()
        with torch.no_grad():
            c0 = torch.zeros(4, dtype=torch.long, device=device)
            y0 = m(xb, mb, chemistry_ids=c0)
            y1 = m(xb, mb, chemistry_ids=c0 + 2)
        changed = (y0 - y1).abs().max().item() > 1e-6
        check(f'{fusion} E={emb} shape', tuple(y0.shape) == (4, 57))
        check(f'{fusion} E={emb} chemistry {"changes" if emb else "does not change"} output', changed == (emb > 0))


run_suite('vanilla', 'cpu', 'ref')
run_suite('mamba_init', 'cpu', 'ref')

if cli.gpu:
    assert torch.cuda.is_available(), '--gpu but no CUDA device'
    dev = 'cuda'
    run_suite('vanilla', dev, 'cuda')
    run_suite('mamba_init', dev, 'cuda')

    # CUDA kernel vs pure-PyTorch scan, same weights
    for layer in ('vanilla', 'mamba_init'):
        torch.manual_seed(0)
        m_cuda = CPMamba.Model(make_args(mamba_layer=layer, mamba_scan='cuda')).float().to(dev).eval()
        xb, mb = padded_batch([3, 50, 100])
        xb, mb = xb.to(dev), mb.to(dev)
        with torch.no_grad():
            y_cuda = m_cuda(xb, mb)
            CPMamba.get_mamba_mixer_cls(layer, 'ref')          # swap the scan to the pure-PyTorch reference
            y_ref = m_cuda(xb, mb)
            CPMamba.get_mamba_mixer_cls(layer, 'cuda')         # and back to the kernel
        d = (y_cuda - y_ref).abs().max().item()
        check(f'{layer}: cuda kernel vs ref scan', d < 1e-3, f'max abs diff {d:.2e}')

    # long sequence, like the longest full-timescale prefixes
    for layer in ('vanilla', 'mamba_init'):
        torch.cuda.reset_peak_memory_stats()
        m = CPMamba.Model(make_args(mamba_layer=layer, mamba_scan='cuda', d_model=64, mamba_n_layers=4)).float().to(dev).train()
        x = torch.randn(1, 4000, LEN, 3, device=dev)
        import time
        t = time.time()
        y = m(x, torch.ones(1, 4000, device=dev))
        y.sum().backward()
        torch.cuda.synchronize()
        check(f'{layer}: L=4000 fwd+bwd', bool(torch.isfinite(y).all()),
              f'{time.time() - t:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
