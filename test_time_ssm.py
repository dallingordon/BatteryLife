"""
Checks for the recorded-time-step code (10/13): models/time_ssm.py, LongMamba --long_time_dt / --long_dt_input,
data_provider/data_loader_dt.py. CPU part needs the venv (mamba_ssm's selective_scan_ref), runs on a login node:

    source venv_mamba/bin/activate
    python test_time_ssm.py                     # CPU checks
    python test_time_ssm.py --gpu               # + CUDA selective scan == ref, real-size fwd+bwd time / memory
    python test_time_ssm.py --data --dt_c 20    # + Dataset_pooled_dt vs Dataset_pooled on the Li-ion val cells
                                                #   (reads the pkls; same samples/curves/labels, tau shape/edges/stats)

What it checks:
  1. times_to_tau: interior steps / c, edge positions (0, 1 of each half) = 1, negative -> 1, zero stays 0,
     missing cycles -> 1, clip
  2. TimeS4DMixer / TimeMambaMixer: selective scan (ref) == explicit per-step recurrence, with random tau incl. 0 and
     large steps; tau=None == tau 1; only the rate*tau product matters (c-invariance: tau*k with log_dt - log k);
     causal in hidden AND tau; gradients reach log_dt and every parameter; weight-decay tags
  3. LongMamba (s4d scan / vanilla x const / real x readout+r_first / none x dt_input): shapes, padded == unpadded
     prefix, unseen cycles (incl. their tau) ignored, const == real with tau 1, every parameter gets gradient,
     dt_input -> in_proj 4 inputs, flags off -> old model (no time mixers)
"""
import argparse
import copy
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.data_loader_dt import times_to_tau

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
p.add_argument('--data', action='store_true')
p.add_argument('--dt_c', type=float, default=20.0)
p.add_argument('--dt_clip', type=float, default=0.0)
p.add_argument('--root_path', default='./dataset')
cli = p.parse_args()
fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------------------------------------------------------
print('=== times_to_tau ===')
H = 5
t = np.zeros((3, 2 * H))
t[0] = [0, 10, 20, 35, 45, 645, 650, 655, 670, 680]     # charge then discharge; discharge step 1 spans a "rest"
t[1] = [0, 10, 20, 20, 15, 30, 40, 50, 60, 70]           # zero step (20 -> 20) and negative step (20 -> 15)
tau = times_to_tau(t, c=5.0)
check('cycle 0', np.allclose(tau[0], [1, 1, 2, 3, 2, 1, 1, 1, 3, 2]), str(tau[0]))
check('zero step stays 0, negative -> 1', np.allclose(tau[1], [1, 1, 2, 0, 1, 1, 1, 2, 2, 2]), str(tau[1]))
check('missing cycle -> 1', np.allclose(tau[2], 1))
check('clip', np.allclose(times_to_tau(t, c=5.0, clip=2.5)[0], [1, 1, 2, 2.5, 2, 1, 1, 1, 2.5, 2]))
check('float32', tau.dtype == np.float32)

try:
    import mamba_ssm.ops.selective_scan_interface  # noqa: F401
except Exception as e:
    print(f'\nmamba_ssm not importable ({e!r}): run inside venv_mamba (module load gcc/12.2.0) for the model checks')
    raise SystemExit(1)

from models import LongMamba
from models.time_ssm import TimeS4DMixer, TimeMambaMixer

torch.manual_seed(0)


def rel(a, b):
    """max |a - b| / max |b|  (selective_scan_ref computes in float32, so compare relative to ~1e-5)"""
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


def rand_tau(B, T, g=None):
    tau = torch.exp(torch.randn(B, T, generator=g) * 0.7)
    tau[:, ::17] = 0.0                                  # repeated timestamps
    tau[:, 5::23] = 30.0                                # rest-sized steps
    return tau.double()


# ---------------------------------------------------------------------------------------------------------------
for name, cls in (('TimeS4DMixer', TimeS4DMixer), ('TimeMambaMixer', TimeMambaMixer)):
    print(f'=== {name} ===')
    m = cls(8, d_state=8, d_conv=4, expand=2, dt_min=1e-3, dt_max=0.3, scan_impl='ref').double()
    x = torch.randn(3, 70, 8, dtype=torch.float64)
    tau = rand_tau(3, 70)
    with torch.no_grad():
        y = m(x, tau)
        y_ref = m.step_reference(x, tau)
        d = rel(y, y_ref)
        check('scan == per-step recurrence', d < 1e-4, f'max rel diff {d:.1e} (|y| max {y_ref.abs().max().item():.2f})')
        d1 = (m(x) - m(x, torch.ones(3, 70, dtype=torch.float64))).abs().max().item()
        check('tau=None == tau 1', d1 == 0, f'{d1:.1e}')
        d2 = (m(x, tau) - m(x, tau * 1.3)).abs().max().item()
        check('tau actually matters', d2 > 1e-6, f'{d2:.1e}')
        m2 = copy.deepcopy(m)
        lp = m2.kernel.log_dt if name == 'TimeS4DMixer' else m2.log_dt
        lp.add_(-math.log(4.0))
        dk = rel(m2(x, tau * 4.0), m(x, tau))
        check('only rate*tau matters (c-invariance)', dk < 1e-5, f'max rel diff {dk:.1e}')
        x2, tau2 = x.clone(), tau.clone()
        x2[:, 40:] = torch.randn_like(x2[:, 40:]) * 5
        tau2[:, 40:] = 7.0
        dc = (m(x2, tau2)[:, :40] - y[:, :40]).abs().max().item()
        check('causal in hidden and tau', dc < 1e-12, f'{dc:.1e}')
    m.zero_grad()
    m(x, tau).pow(2).mean().backward()
    lp = m.kernel.log_dt if name == 'TimeS4DMixer' else m.log_dt
    check('log_dt gets gradient', lp.grad is not None and lp.grad.abs().sum().item() > 0)
    missing = [n for n, q in m.named_parameters() if q.grad is None or q.grad.abs().sum().item() == 0]
    check('every parameter gets gradient', not missing, str(missing))
    tagged = sorted(n for n, q in m.named_parameters() if getattr(q, '_ssm_no_wd', False))
    want = ['kernel.A_imag', 'kernel.C', 'kernel.log_A_real', 'kernel.log_dt'] if name == 'TimeS4DMixer' else ['log_dt']
    check('no-weight-decay tags', tagged == want, str(tagged))

# ---------------------------------------------------------------------------------------------------------------


def make_args(**kw):
    a = dict(d_model=16, charge_discharge_length=20, early_cycle_threshold=10, output_num=3,
             mamba_n_layers=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_layer='s4d', s4_backend='scan',
             mamba_scan='ref', chem_fusion='none', long_boundary='none', long_chem_cls=0, long_chem_boundary=0,
             chem_num=4, s4_dt_min=1e-3, s4_dt_max=1e-1, long_time_dt='real', long_dt_input=0, long_r_first=0)
    a.update(kw)
    return SimpleNamespace(**a)


def batch(lengths, L, P, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), L, 3, P, generator=g)
    tau = torch.exp(torch.randn(len(lengths), L, P, generator=g) * 0.5)
    m = torch.zeros(len(lengths), L)
    for b, n in enumerate(lengths):
        m[b, :n] = 1
        x[b, n:] = 0
    return x, m, tau


for layer in ('s4d', 'vanilla'):
    for time_dt in ('const', 'real'):
        for boundary, r_first, dt_input in (('readout', 1, 0), ('none', 0, 1)):
            print(f'=== LongMamba {layer} time_dt={time_dt} boundary={boundary} r_first={r_first} dt_input={dt_input} ===')
            torch.manual_seed(1)
            args = make_args(mamba_layer=layer, long_time_dt=time_dt, long_boundary=boundary, long_r_first=r_first,
                             long_dt_input=dt_input)
            model = LongMamba.Model(args).double().eval()
            kinds = {type(l.mixer).__name__ for l in model.mamba_layers}
            check('time mixers', kinds == {'TimeS4DMixer' if layer == 's4d' else 'TimeMambaMixer'}, str(kinds))
            check('in_proj inputs', model.in_proj.in_features == 3 + dt_input, str(model.in_proj.in_features))
            x, m, tau = batch([10, 4, 7], 10, 20)
            x, tau = x.double(), tau.double()
            with torch.no_grad():
                out = model(x, m, tau=tau)
                check('shape', tuple(out.shape) == (3, 3), str(tuple(out.shape)))
                alone = model(x[1:2, :4], m[1:2, :4], tau=tau[1:2, :4])
                d = rel(alone, out[1:2])
                check('padded == unpadded prefix', d < 1e-5, f'max rel diff {d:.1e}')
                x2, tau2 = x.clone(), tau.clone()
                x2[1, 4:] = torch.randn_like(x2[1, 4:]); tau2[1, 4:] = 9.0
                d = rel(model(x2, m, tau=tau2)[1], out[1])
                check('unseen cycles (and their tau) ignored', d < 1e-5, f'max rel diff {d:.1e}')
                if time_dt == 'real' and not dt_input:
                    ones = torch.ones_like(tau)
                    model.time_dt = 'const'
                    a_ = model(x, m)
                    model.time_dt = 'real'
                    d = rel(model(x, m, tau=ones), a_)
                    check('const == real with tau 1', d < 1e-6, f'max rel diff {d:.1e}')
                    d = rel(model(x, m, tau=tau * 2), out)
                    check('real: output depends on tau', d > 1e-4, f'max rel diff {d:.1e}')
            needs = time_dt == 'real' or dt_input
            try:
                model(x, m)
                check('missing tau raises' if needs else 'no tau needed', not needs)
            except ValueError:
                check('missing tau raises' if needs else 'no tau needed', needs)
            model.train()
            model.zero_grad()
            if boundary == 'readout':
                o, r, rm = model(x, m, tau=tau, return_r=True)
                (o.pow(2).mean() + (r.pow(2).mean(-1) * rm).mean()).backward()
            else:
                model(x, m, tau=tau).pow(2).mean().backward()
            missing = [n for n, q in model.named_parameters() if q.grad is None or q.grad.abs().sum().item() == 0]
            missing = [n for n in missing if not n.startswith('cycle_embed')]
            check('every parameter gets gradient', not missing, str(missing[:6]))

print('=== flags off -> old model ===')
for layer in ('s4d',):
    args = make_args(mamba_layer=layer, long_time_dt='off', long_dt_input=0, s4_backend='fft')
    model = LongMamba.Model(args)
    kinds = {type(l.mixer).__name__ for l in model.mamba_layers}
    check('old mixers, 3 inputs', kinds == {'S4DMixer'} and model.in_proj.in_features == 3, str(kinds))

# ---------------------------------------------------------------------------------------------------------------
if cli.gpu:
    assert torch.cuda.is_available(), '--gpu needs a GPU'
    dev = 'cuda'
    print(f'=== GPU: {torch.cuda.get_device_name()} ===')
    for name, cls in (('TimeS4DMixer', TimeS4DMixer), ('TimeMambaMixer', TimeMambaMixer)):
        torch.manual_seed(0)
        m = cls(64, d_state=16, expand=2, scan_impl='cuda').to(dev)
        x = torch.randn(2, 3000, 64, device=dev)
        tau = rand_tau(2, 3000).float().to(dev)
        y = m(x, tau)
        m.scan_impl = 'ref'
        y_ref = m(x, tau)
        rel = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
        check(f'{name}: cuda == ref', rel < 1e-3, f'max rel diff {rel:.1e}')
        m.scan_impl = 'cuda'
        g = torch.autograd.grad(m(x, tau).sum(), [p_ for p_ in m.parameters()])
        m.scan_impl = 'ref'
        g_ref = torch.autograd.grad(m(x, tau).sum(), [p_ for p_ in m.parameters()])
        worst = max(((a - b).abs().max() / (b.abs().max() + 1e-12)).item() for a, b in zip(g, g_ref))
        check(f'{name}: cuda grads == ref', worst < 5e-3, f'worst rel diff {worst:.1e}')
    for layer in ('s4d', 'vanilla'):
        args = make_args(d_model=64, charge_discharge_length=300, early_cycle_threshold=100, output_num=50,
                         mamba_n_layers=4, mamba_d_state=16, mamba_layer=layer, mamba_scan='cuda', s4_dt_min=1e-5,
                         long_boundary='readout', long_r_first=1, long_time_dt='real')
        model = LongMamba.Model(args).to(dev).train()
        for B in (1, 8):
            x = torch.randn(B, 100, 3, 300, device=dev)
            m = torch.ones(B, 100, device=dev)
            tau = torch.exp(torch.randn(B, 100, 300, device=dev) * 0.5)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
            o, r, rm = model(x, m, tau=tau, return_r=True)
            (o.sum() + r.sum()).backward()
            torch.cuda.synchronize()
            print(f'  {layer} time_dt=real readout r_first B={B} 100 cycles: fwd+bwd {time.time() - t0:.2f}s, '
                  f'peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
            model.zero_grad(set_to_none=True)

# ---------------------------------------------------------------------------------------------------------------
if cli.data:
    print(f'=== Dataset_pooled_dt vs Dataset_pooled (Li-ion val, c={cli.dt_c}) ===')
    from data_provider.data_loader_pooled import Dataset_pooled
    from data_provider.data_loader_dt import Dataset_pooled_dt
    da = SimpleNamespace(root_path=cli.root_path, seq_len=1, charge_discharge_length=300, dataset='POOLED',
                         early_cycle_threshold=100, weighted_loss=False, chem_loss_alpha=0.0, target_dataset=None,
                         dt_c=cli.dt_c, dt_clip=cli.dt_clip)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.array([[100.0], [3000.0]]))
    a = Dataset_pooled(da, flag='val', chemistries=['Li-ion'], label_scaler=sc, life_class_scaler=sc)
    b = Dataset_pooled_dt(da, flag='val', chemistries=['Li-ion'], label_scaler=sc, life_class_scaler=sc)
    check('same number of samples', len(a) == len(b), f'{len(a)} vs {len(b)}')
    same = all(np.array_equal(np.asarray(a.total_charge_discharge_curves[i]), np.asarray(b.total_charge_discharge_curves[i]))
               for i in range(len(a)))
    check('same curves', same)
    check('same labels', np.array_equal(np.asarray(a.total_labels), np.asarray(b.total_labels)))
    s = b[0]
    check('cycle_tau shape', tuple(s['cycle_tau'].shape) == (100, 300), str(tuple(s['cycle_tau'].shape)))
    allt = np.stack([b[i]['cycle_tau'].numpy() for i in range(0, len(b), 50)])
    check('edge positions are 1', bool((allt[..., [0, 1, 150, 151]] == 1).all()))
    q = np.quantile(allt, [0.01, 0.5, 0.99])
    print(f'  tau p1 {q[0]:.3g}  p50 {q[1]:.3g}  p99 {q[2]:.3g}  max {allt.max():.3g}')

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
