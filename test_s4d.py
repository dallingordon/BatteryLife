"""
Checks for models/S4D.py and LongMamba --mamba_layer s4d. Needs torch only (no mamba_ssm kernels), so the CPU part
runs on the SCC login node:

    source venv_mamba/bin/activate
    python test_s4d.py            # CPU checks
    python test_s4d.py --gpu      # also: real size (100 cycles x 300 points, ~30k steps) fwd+bwd time / peak memory

What it checks:
  1. S4D FFT convolution == explicit complex recurrence (the kernel math, ZOH discretisation, D skip, gating)
  2. dt init lies in [dt_min, dt_max], log-uniform spread; SSM tensors are tagged _ssm_no_wd (and nothing else is)
  3. LongMamba s4d, boundary none / shared / readout (+ r_first): shapes, padded == unpadded prefix, unseen cycles
     ignored (causality), every parameter gets gradient, readout R outputs padded == unpadded
  4. long memory: with dt_min=1e-5 the kernel of the slowest channel is still non-negligible after 30k steps,
     with the old default dt_min=1e-3 it is ~0
"""
import argparse
import math
import sys
import time
from types import SimpleNamespace

import torch

from models import LongMamba
from models.S4D import S4DMixer, S4DKernel

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
cli = p.parse_args()
fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


def make_args(**kw):
    a = dict(d_model=16, charge_discharge_length=20, early_cycle_threshold=10, output_num=1,
             mamba_n_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_layer='s4d',
             mamba_scan='ref', chem_fusion='none', long_boundary='none', long_chem_cls=0, long_chem_boundary=0,
             chem_num=4, s4_dt_min=1e-3, s4_dt_max=1e-1)   # small sequences in tests: faster timescales
    a.update(kw)
    return SimpleNamespace(**a)


def padded_batch(lengths, L, P, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), L, 3, P, generator=g)
    m = torch.zeros(len(lengths), L)
    for b, n in enumerate(lengths):
        m[b, :n] = 1
        x[b, n:] = 0
    return x, m


torch.manual_seed(0)
print('=== S4D mixer ===')
mix = S4DMixer(8, d_state=8, d_conv=4, expand=2, dt_min=1e-3, dt_max=0.5).double()
x = torch.randn(3, 60, 8, dtype=torch.float64)
with torch.no_grad():
    y_fft = mix(x)
    y_rec = mix.step_reference(x)
d = (y_fft - y_rec).abs().max().item()
check('FFT conv == recurrence', d < 1e-4, f'max abs diff {d:.2e} (|y| max {y_rec.abs().max().item():.2f})')
with torch.no_grad():
    x2 = x.clone(); x2[:, 30:] = torch.randn_like(x2[:, 30:]) * 10
    dc = (mix(x2)[:, :30] - y_fft[:, :30]).abs().max().item()
check('mixer is causal', dc < 1e-6, f'{dc:.1e}')

k = S4DKernel(4096, d_state=16, dt_min=1e-5, dt_max=1e-1)
dt = torch.exp(k.log_dt)
check('dt init within [1e-5, 1e-1]', bool((dt >= 1e-5 * 0.999).all() and (dt <= 1e-1 * 1.001).all()),
      f'min {dt.min().item():.2e} max {dt.max().item():.2e}')
frac = (dt < 1e-3).float().mean().item()
check('dt log-uniform (half the channels below 1e-3)', abs(frac - 0.5) < 0.05, f'{frac:.3f}')
m = LongMamba.Model(make_args())
tagged = [n for n, q in m.named_parameters() if getattr(q, '_ssm_no_wd', False)]
check('only S4D SSM tensors tagged _ssm_no_wd', len(tagged) == 4 * 2 and all('.kernel.' in n for n in tagged), str(tagged))

# long memory at the real scale: slowest channel's kernel after 30k steps
for dt_min, want_long in ((1e-5, True), (1e-3, False)):
    torch.manual_seed(0)
    kk = S4DKernel(64, d_state=16, dt_min=dt_min, dt_max=1e-1)
    with torch.no_grad():
        K = kk(30001)
        slow = int(torch.argmin(kk.log_dt))
        tail = K[slow, 29000:].abs().max().item() / K[slow].abs().max().item()
    check(f'dt_min={dt_min}: slowest channel kernel at ~30k steps is {"still there" if want_long else "~0"}',
          (tail > 1e-2) == want_long, f'|K[30k]| / max|K| = {tail:.2e}')

for boundary, rf in (('none', 0), ('shared', 0), ('readout', 0), ('readout', 1)):
    tag = f'LongMamba s4d boundary={boundary} r_first={rf}'
    print(f'=== {tag} ===')
    torch.manual_seed(0)
    model = LongMamba.Model(make_args(long_boundary=boundary, long_r_first=rf, long_r_dim=8)).float().eval()
    check('mixers are S4DMixer', all(isinstance(b.mixer, S4DMixer) for b in model.mamba_layers))
    lengths = [1, 3, 7, 10]
    xb, mb = padded_batch(lengths, 10, 20)
    with torch.no_grad():
        yb = model(xb, mb)
        check('shape', tuple(yb.shape) == (4, 1))
        err = max((model(xb[b:b + 1, :n], torch.ones(1, n)) - yb[b:b + 1]).abs().max().item()
                  for b, n in enumerate(lengths))
        check('padded == unpadded prefix', err < 1e-4, f'{err:.2e}')
        xg = xb.clone()
        for b, n in enumerate(lengths):
            xg[b, n:] = torch.randn_like(xg[b, n:]) * 10
        diff = (model(xg, mb) - yb).abs().max().item()
        check('unseen cycles ignored', diff < 1e-4, f'{diff:.2e}')
        if boundary == 'readout':
            _, rp, rm = model(xb, mb, return_r=True)
            errR = max((model(xb[b:b + 1, :n], torch.ones(1, n), return_r=True)[1][0] - rp[b, :n - 1]).abs().max().item()
                       for b, n in enumerate(lengths) if n > 1)
            check('R outputs padded == unpadded', errR < 1e-4, f'{errR:.2e}')
    model.train(); model.zero_grad()
    if boundary == 'readout':
        f, rp, rm = model(xb, mb, return_r=True)
        (f.pow(2).sum() + (rp.pow(2).sum(-1) * rm).sum()).backward()
    else:
        model(xb, mb).pow(2).sum().backward()
    no_grad = [n for n, q in model.named_parameters() if q.grad is None or q.grad.abs().sum() == 0]
    check('every parameter gets gradient', not no_grad, str(no_grad[:5]))
    m28 = LongMamba.Model(make_args(long_boundary=boundary, long_r_first=rf, output_num=28)).eval()
    with torch.no_grad():
        check('geo_bins head shape', tuple(m28(xb, mb).shape) == (4, 28))

if cli.gpu:
    assert torch.cuda.is_available()
    dev = 'cuda'
    for boundary, rf in (('none', 0), ('readout', 1)):
        for B in (1, 4, 8, 16):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = x = y = None
            try:
                m = LongMamba.Model(make_args(d_model=64, mamba_n_layers=4, charge_discharge_length=300,
                                              early_cycle_threshold=100, long_boundary=boundary, long_r_first=rf,
                                              output_num=28, s4_dt_min=1e-5, s4_dt_max=1e-1)).float().to(dev).train()
                x = torch.randn(B, 100, 3, 300, device=dev)
                y = m(x, torch.ones(B, 100, device=dev)); y.sum().backward(); m.zero_grad()   # warm-up
                torch.cuda.synchronize(); t = time.time()
                y = m(x, torch.ones(B, 100, device=dev)); y.sum().backward()
                torch.cuda.synchronize()
                print(f'  s4d boundary={boundary} r_first={rf} B={B:2d} 100 cycles: fwd+bwd {time.time() - t:.2f}s, '
                      f'peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
            except torch.cuda.OutOfMemoryError:
                print(f'  s4d B={B} OUT OF MEMORY'); break
            finally:
                del m, x, y

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
