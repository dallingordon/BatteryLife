"""
Checks for models/S4D.py and LongMamba --mamba_layer s4d. Needs torch only (no mamba_ssm kernels), so the CPU part
runs on the SCC login node:

    source venv_mamba/bin/activate
    python test_s4d.py            # CPU checks
    python test_s4d.py --gpu      # also: real size (100 cycles x 300 points, ~30k steps) fwd+bwd time / peak memory,
                                  #       scan backend (CUDA selective scan) == fft, and their speed side by side
    python test_s4d.py --gpu --long   # also: scan backend at full-history lengths (1000 / 2000 / 3842 cycles, B=1,
                                      #       readout + r_first, grad ckpt): fwd+bwd time and peak memory

What it checks:
  1. S4D FFT convolution == explicit complex recurrence (the kernel math, ZOH discretisation, D skip, gating)
  2. dt init lies in [dt_min, dt_max], log-uniform spread; SSM tensors are tagged _ssm_no_wd (and nothing else is)
  3. LongMamba s4d, boundary none / shared / readout (+ r_first): shapes, padded == unpadded prefix, unseen cycles
     ignored (causality), every parameter gets gradient, readout R outputs padded == unpadded
  4. long memory: with dt_min=1e-5 the kernel of the slowest channel is still non-negligible after 30k steps,
     with the old default dt_min=1e-3 it is ~0
  5. --s4_backend scan (needs mamba_ssm importable; skipped otherwise): outputs and gradients == fft backend, on CPU
     via selective_scan_ref; with --gpu also via the CUDA kernel (complex A/B/C)
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
p.add_argument('--long', action='store_true', help='with --gpu: full-history lengths with the scan backend')
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


def scan_vs_fft(dev, scan_impl, B=2, T=300, d=16, tol=1e-3):
    """Same S4DMixer weights, backend fft vs scan: outputs and every parameter gradient."""
    torch.manual_seed(0)
    mix = S4DMixer(d, d_state=16, d_conv=4, expand=2, dt_min=1e-4, dt_max=1e-1, scan_impl=scan_impl).to(dev)
    x = torch.randn(B, T, d, device=dev)
    outs, grads = {}, {}
    for be in ('fft', 'scan'):
        mix.backend = be
        mix.zero_grad()
        y = mix(x)
        (y * torch.linspace(-1, 1, y.numel(), device=dev).view_as(y)).sum().backward()
        outs[be] = y.detach()
        grads[be] = {n: q.grad.detach().clone() for n, q in mix.named_parameters()}
    scale = outs['fft'].abs().max().item()
    dy = (outs['scan'] - outs['fft']).abs().max().item() / scale
    check(f'scan ({scan_impl}, {dev}) == fft: output', dy < tol, f'max rel diff {dy:.2e}')
    worst = max(((grads['scan'][n] - grads['fft'][n]).abs().max() / grads['fft'][n].abs().max().clamp_min(1e-12)).item()
                for n in grads['fft'])
    check(f'scan ({scan_impl}, {dev}) == fft: every parameter gradient', worst < 10 * tol, f'max rel diff {worst:.2e}')


print('=== --s4_backend scan ===')
try:
    import mamba_ssm.ops.selective_scan_interface  # noqa: F401
    HAVE_MAMBA = True
except Exception as e:   # login node without the kernels' libs, etc.
    HAVE_MAMBA = False
    print(f'  [skip] mamba_ssm not importable here ({e!r}); scan backend not tested')
if HAVE_MAMBA:
    scan_vs_fft('cpu', 'ref', T=200)
    torch.manual_seed(0)
    mf = LongMamba.Model(make_args(long_boundary='readout', long_r_first=1, long_r_dim=8)).eval()
    ms = LongMamba.Model(make_args(long_boundary='readout', long_r_first=1, long_r_dim=8, s4_backend='scan')).eval()
    ms.load_state_dict(mf.state_dict())
    check('LongMamba s4_backend scan builds scan mixers', all(b.mixer.backend == 'scan' for b in ms.mamba_layers))
    xb, mb = padded_batch([2, 5], 5, 20)
    with torch.no_grad():
        a, ra, _ = mf(xb, mb, return_r=True)
        b, rb, _ = ms(xb, mb, return_r=True)
    d1 = max((a - b).abs().max().item(), (ra - rb).abs().max().item())
    check('LongMamba readout: scan == fft (F and R outputs)', d1 < 1e-3, f'{d1:.2e}')

if cli.gpu:
    assert torch.cuda.is_available()
    dev = 'cuda'
    if HAVE_MAMBA:
        scan_vs_fft(dev, 'cuda', T=300)
        scan_vs_fft(dev, 'cuda', B=1, T=30000, d=64, tol=2e-3)
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
    if HAVE_MAMBA:   # same real-size run, scan backend (CUDA kernel)
        for B in (1, 8):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = LongMamba.Model(make_args(d_model=64, mamba_n_layers=4, charge_discharge_length=300, early_cycle_threshold=100,
                                          long_boundary='readout', long_r_first=1, output_num=28, s4_dt_min=1e-5,
                                          s4_dt_max=1e-1, s4_backend='scan', mamba_scan='cuda')).float().to(dev).train()
            x = torch.randn(B, 100, 3, 300, device=dev)
            y = m(x, torch.ones(B, 100, device=dev)); y.sum().backward(); m.zero_grad()
            torch.cuda.synchronize(); t = time.time()
            y = m(x, torch.ones(B, 100, device=dev)); y.sum().backward()
            torch.cuda.synchronize()
            print(f'  s4d SCAN boundary=readout r_first=1 B={B:2d} 100 cycles: fwd+bwd {time.time() - t:.2f}s, '
                  f'peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
            del m, x, y
    if cli.long and HAVE_MAMBA:   # full-history lengths: one cell, readout + r_first, grad ckpt, F + R loss
        for n_cyc in (1000, 2000, 3842):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = x = None
            try:
                m = LongMamba.Model(make_args(d_model=64, mamba_n_layers=4, charge_discharge_length=300,
                                              early_cycle_threshold=100, long_boundary='readout', long_r_first=1,
                                              output_num=28, s4_dt_min=1e-5, s4_dt_max=1e-1, s4_backend='scan',
                                              mamba_scan='cuda', long_grad_ckpt=1)).float().to(dev).train()
                x = torch.randn(1, n_cyc, 3, 300, device=dev)
                torch.cuda.synchronize(); t = time.time()
                f, rp, rm = m(x, torch.ones(1, n_cyc, device=dev), return_r=True)
                (f.sum() + (rp.sum(-1) * rm).sum() / rm.sum()).backward()
                torch.cuda.synchronize()
                print(f'  s4d SCAN long: {n_cyc} cycles ({n_cyc * 301} steps), B=1, grad ckpt: fwd+bwd '
                      f'{time.time() - t:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
            except torch.cuda.OutOfMemoryError:
                print(f'  s4d SCAN long: {n_cyc} cycles OUT OF MEMORY'); break
            finally:
                del m, x

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
