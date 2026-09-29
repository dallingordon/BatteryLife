"""
Checks for models/LongMamba.py. Run on SCC from the repo root, inside the mamba venv:

    module load gcc/12.2.0 cuda/12.2 && source venv_mamba/bin/activate
    python test_longmamba.py            # CPU, pure-PyTorch scan, small cycles (20 points x 10 cycles); login node OK
    python test_longmamba.py --gpu      # also: CUDA kernel vs ref scan, and REAL size (100 cycles x 300 points,
                                        # ~30k steps) fwd+bwd timing / peak memory at a few batch sizes

What it checks, for --mamba_layer vanilla / mamba_init x --long_boundary none / shared / index:
  1. shapes on the loaders' layout [B, L, 3, P], right-padded batch; regression and geo_bins-sized heads
  2. sequence length seen by the mixers = n_cycles * S + 1 (S = P, or P+1 with a boundary token)
  3. padding invariance: a prefix padded inside a mixed batch == the same prefix alone
  4. causality: garbage in the unseen cycles (after the CLS position) doesn't change the output
  5. gradient flow into every parameter incl. cls_token, boundary_token / cycle_embed, (mamba_init) initial_state
  6. boundary modes: 'shared' uses the same token at every boundary, 'index' a different one per cycle
"""
import argparse
import sys
import time
from types import SimpleNamespace

import torch

from models import LongMamba

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
cli = p.parse_args()

fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


def make_args(**kw):
    a = dict(d_model=32, charge_discharge_length=20, early_cycle_threshold=10, output_num=1,
             mamba_n_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_layer='vanilla',
             mamba_scan='ref', chem_fusion='none', long_boundary='none')
    a.update(kw)
    return SimpleNamespace(**a)


def padded_batch(lengths, L, P, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), L, 3, P, generator=g)
    m = torch.zeros(len(lengths), L)
    for b, n in enumerate(lengths):
        m[b, :n] = 1
        x[b, n:] = 0            # the pooled collate zeroes unseen cycles too
    return x, m


def run_suite(layer, boundary, device, scan):
    print(f'=== mamba_layer={layer} long_boundary={boundary} scan={scan} device={device} ===')
    torch.manual_seed(0)
    args = make_args(mamba_layer=layer, mamba_scan=scan, long_boundary=boundary)
    L, P = args.early_cycle_threshold, args.charge_discharge_length
    model = LongMamba.Model(args).float().to(device).eval()
    print(f'  params: {sum(q.numel() for q in model.parameters())}')

    # 1. shapes
    lengths = [1, 3, 7, L]
    xb, mb = padded_batch(lengths, L, P)
    xb, mb = xb.to(device), mb.to(device)
    seen = []
    hook = model.mamba_layers[0].mixer.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].shape[1]))
    yb = model(xb, mb)
    check('padded batch shape', tuple(yb.shape) == (4, 1), str(tuple(yb.shape)))
    check('finite outputs', bool(torch.isfinite(yb).all()))

    # 2. sequence length = max(n) * S + 1
    S = P + (0 if boundary == 'none' else 1)
    check('sequence length', seen[-1] == L * S + 1, f'{seen[-1]} vs expected {L * S + 1}')
    model(xb[1:2, :3], torch.ones(1, 3, device=device))
    check('sequence length (3 cycles alone)', seen[-1] == 3 * S + 1, f'{seen[-1]} vs {3 * S + 1}')
    hook.remove()

    # 3. padding invariance
    with torch.no_grad():
        err = max((model(xb[b:b + 1, :n], torch.ones(1, n, device=device)) - yb[b:b + 1]).abs().max().item()
                  for b, n in enumerate(lengths))
    check('padded == unpadded prefix', err < 1e-4, f'max abs diff {err:.2e}')

    # 4. causality
    with torch.no_grad():
        xg = xb.clone()
        for b, n in enumerate(lengths):
            xg[b, n:] = torch.randn_like(xg[b, n:]) * 10
        diff = (model(xg, mb) - yb).abs().max().item()
    check('unseen cycles ignored', diff < 1e-4, f'max abs diff {diff:.2e}')

    # 5. gradients
    model.train()
    model.zero_grad()
    model(xb, mb).pow(2).sum().backward()
    no_grad = [n for n, q in model.named_parameters() if q.requires_grad and (q.grad is None or q.grad.abs().sum() == 0)]
    check('every parameter gets gradient', not no_grad, str(no_grad[:5]))

    # 6. boundary tokens
    if boundary == 'shared':
        check('shared token param present', hasattr(model, 'boundary_token') and not hasattr(model, 'cycle_embed'))
    if boundary == 'index':
        check('cycle_embed has one row per cycle', tuple(model.cycle_embed.weight.shape) == (L, args.d_model))
        # the embedding rows actually used get gradient; row k only for samples with > k cycles
        g = model.cycle_embed.weight.grad.abs().sum(1)
        check('all 10 cycle rows used by the full-length sample', bool((g > 0).all()), str(g.tolist()))

    # geo_bins-sized head
    m2 = LongMamba.Model(make_args(mamba_layer=layer, mamba_scan=scan, long_boundary=boundary, output_num=28)).to(device).eval()
    with torch.no_grad():
        check('geo_bins head shape', tuple(m2(xb, mb).shape) == (4, 28))


for layer in ('vanilla', 'mamba_init'):
    for boundary in LongMamba.BOUNDARY_MODES:
        run_suite(layer, boundary, 'cpu', 'ref')

# the three modes must actually differ (same seed / weights otherwise)
with torch.no_grad():
    xb, mb = padded_batch([4, 10], 10, 20)
    outs = {}
    for boundary in LongMamba.BOUNDARY_MODES:
        torch.manual_seed(0)
        outs[boundary] = LongMamba.Model(make_args(long_boundary=boundary)).eval()(xb, mb)
    check('shared != none', (outs['shared'] - outs['none']).abs().max().item() > 1e-6)
    check('index != shared', (outs['index'] - outs['shared']).abs().max().item() > 1e-6)

if cli.gpu:
    assert torch.cuda.is_available(), '--gpu but no CUDA device'
    dev = 'cuda'
    for layer in ('vanilla', 'mamba_init'):
        for boundary in LongMamba.BOUNDARY_MODES:
            run_suite(layer, boundary, dev, 'cuda')

    # CUDA kernel vs pure-PyTorch scan, same weights (small size)
    for layer in ('vanilla', 'mamba_init'):
        torch.manual_seed(0)
        m = LongMamba.Model(make_args(mamba_layer=layer, mamba_scan='cuda', long_boundary='index')).float().to(dev).eval()
        xb, mb = padded_batch([2, 6, 10], 10, 20)
        xb, mb = xb.to(dev), mb.to(dev)
        with torch.no_grad():
            y_cuda = m(xb, mb)
            LongMamba.get_mamba_mixer_cls(layer, 'ref')
            y_ref = m(xb, mb)
            LongMamba.get_mamba_mixer_cls(layer, 'cuda')
        d = (y_cuda - y_ref).abs().max().item()
        check(f'{layer}: cuda kernel vs ref scan', d < 1e-3, f'max abs diff {d:.2e}')

    # REAL size: 100 cycles x 300 points (~30k steps), default model (d_model 64, 4 layers, d_state 16).
    # Prints fwd+bwd time and peak memory per batch size -> pick BATCH / ACCUM in LongMamba_pooled_qsub.sh.
    for layer in ('vanilla', 'mamba_init'):
        for B in (1, 4, 8, 16, 32):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = x = y = None
            try:
                m = LongMamba.Model(make_args(mamba_layer=layer, mamba_scan='cuda', d_model=64, mamba_n_layers=4,
                                              charge_discharge_length=300, early_cycle_threshold=100,
                                              long_boundary='index', output_num=28)).float().to(dev).train()
                x = torch.randn(B, 100, 3, 300, device=dev)
                torch.cuda.synchronize(); t = time.time()
                y = m(x, torch.ones(B, 100, device=dev))
                y.sum().backward()
                torch.cuda.synchronize()
                print(f'  {layer} B={B:2d} 100 cycles: fwd+bwd {time.time() - t:.2f}s, '
                      f'peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
            except torch.cuda.OutOfMemoryError:
                print(f'  {layer} B={B:2d} 100 cycles: OUT OF MEMORY')
                break
            finally:
                del m, x, y

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
