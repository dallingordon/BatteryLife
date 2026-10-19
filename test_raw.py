"""
Checks for the raw-record code (10/17): data_provider/data_loader_raw.py, LongMamba --long_raw 1.

    source venv_mamba/bin/activate
    python test_raw.py                          # CPU checks (needs mamba_ssm importable: selective_scan_ref)
    python test_raw.py --gpu                    # + real-size fwd+bwd time / peak memory at raw lengths
    python test_raw.py --data --dt_c 5 --dt_clip 100   # + Dataset_pooled_raw vs Dataset_pooled on the Li-ion val cells
    python test_raw.py --data --raw_tau_mode cycle_norm --dt_c 5 --dt_clip 100   # same, per-cycle-normalized tau

What it checks:
  1. raw_times_to_tau / raw_cell_arrays / make_budget_batches (numpy only)
  2. LongMamba --long_raw == the 300-point grid path when every cycle has exactly P records (same weights, same data
     laid out flat): s4d / vanilla x off / const / real x none / shared / index / readout(+-r_first) x dt_input,
     incl. the R-token outputs. This pins the flat layout (token / record / CLS positions, tau per position).
  3. Variable cycle lengths: padded batch == each sample alone; a cycle with no records; extra records after the
     prefix (cid >= n_valid) and padding ignored; real depends on tau, const does not; grad checkpointing same output;
     every parameter gets gradient.
  4. --data: same samples / labels / prefixes as Dataset_pooled, records packed in cycle order, prefix lengths match
     starts, tau stats, and the DataLoader (through accelerate.prepare, as in run_main) yields every sample once.
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.data_loader_raw import raw_times_to_tau, raw_times_to_tau_cycle_norm, make_budget_batches, my_collate_fn_pooled_raw

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
p.add_argument('--data', action='store_true')
p.add_argument('--dt_c', type=float, default=5.0)
p.add_argument('--dt_clip', type=float, default=0.0)
p.add_argument('--raw_tau_mode', default='real', choices=['real', 'cycle_norm'])
p.add_argument('--raw_tau_boundary', type=float, default=1.0)
p.add_argument('--root_path', default='./dataset')
cli = p.parse_args()
fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


def rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


# ---------------------------------------------------------------------------------------------------------------
print('=== numpy helpers ===')
tt = np.array([0, 10, 10, 5, np.nan, 100.])
check('tau: first 1, step/c, zero stays 0, negative/nan -> 1, clip',
      raw_times_to_tau(tt, 5.0, clip=3).tolist() == [1, 2, 0, 1, 1, 1])
check('tau: no c -> ones', (raw_times_to_tau(tt, 0) == 1).all())
# cycle_norm: cycle 0 = records 0..3, cycle 1 empty, cycle 2 = records 4..8 (one negative step, one long pause),
# cycle 3 = 1 record
tc = np.array([0, 1, 3, 6, 1000, 1002, 1004, 1003, 3000, 5000.])
st = np.array([0, 4, 4, 9, 10])
for bnd in (1.0, 0.0):
    t = raw_times_to_tau_cycle_norm(tc, st, scale=300, boundary=bnd).astype(np.float64)
    sums = [t[1:4].sum(), t[5:9].sum()]
    check(f'tau cycle_norm (boundary {bnd:g}): firsts = boundary, each cycle sums to 300, spacing kept',
          np.allclose(t[[0, 4, 9]], bnd) and np.allclose(sums, 300, rtol=1e-6) and np.allclose(t[1:4], [50, 100, 150]),
          f'{np.round(t, 3).tolist()}')
t = raw_times_to_tau_cycle_norm(tc, st, scale=300, clip_s=10).astype(np.float64)
check('tau cycle_norm: negative step -> cycle median, clip_s caps pauses before normalizing',
      np.allclose(t[5:9], 300 * np.array([2, 2, 2, 10]) / 16, rtol=1e-6), f'{np.round(t[5:9], 3).tolist()}')
t = raw_times_to_tau_cycle_norm(np.array([5, 5, 5.]), np.array([0, 3]), scale=300)
check('tau cycle_norm: zero-duration cycle -> equal steps', np.allclose(t[1:], 150))
rng = np.random.default_rng(0)
Ls = rng.integers(10, 1000, size=300)
Ls[3] = 5000
for shuf in (False, True):
    bs = make_budget_batches(Ls, 8, 2000, shuf, np.random.default_rng(1))
    ok = sorted(i for b in bs for i in b) == list(range(300)) and all(len(b) <= 8 for b in bs) and \
        all(len(b) * max(Ls[b]) <= 2000 or len(b) == 1 for b in bs)
    check(f'budget batches (shuffle={shuf}): every sample once, <= 8, <= budget', ok)

try:
    import mamba_ssm.ops.selective_scan_interface  # noqa: F401
except Exception as e:
    print(f'\nmamba_ssm not importable ({e!r}): run inside venv_mamba (module load gcc/12.2.0) for the model checks')
    raise SystemExit(1)

from models import LongMamba

L_, P_ = 6, 12


def make_args(**kw):
    a = dict(d_model=16, charge_discharge_length=P_, early_cycle_threshold=L_, output_num=3,
             mamba_n_layers=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_layer='s4d', s4_backend='scan',
             mamba_scan='ref', chem_fusion='none', long_boundary='none', long_chem_cls=0, long_chem_boundary=0,
             chem_num=4, s4_dt_min=1e-3, s4_dt_max=1e-1, long_time_dt='real', long_dt_input=0, long_r_first=0,
             long_raw=0)
    a.update(kw)
    return SimpleNamespace(**a)


def grid_batch(lengths, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = len(lengths)
    x = torch.randn(B, L_, C, P_, generator=g, dtype=torch.float64)
    tau = torch.exp(torch.randn(B, L_, P_, generator=g, dtype=torch.float64) * 0.5)
    m = torch.zeros(B, L_, dtype=torch.float64)
    for b, n in enumerate(lengths):
        m[b, :n] = 1
        x[b, n:] = 0
    return x, m, tau


def grid_to_flat(x, m, tau):
    """[B, L, C, P] grid -> flat records of the first n_valid cycles, padded: x [B, T, C], tau [B, T], cid [B, T]."""
    samples = []
    for b in range(x.shape[0]):
        n = int(m[b].sum())
        xs = x[b, :n].transpose(1, 2).reshape(n * P_, -1)
        samples.append(dict(raw_x=xs, raw_tau=tau[b, :n].reshape(-1),
                            raw_cid=torch.arange(n).repeat_interleave(P_), curve_attn_mask=m[b], labels=0.,
                            life_class=0, scaled_life_class=0, weight=1., seen_unseen_id=1, chemistry_id=b % 4))
    out = my_collate_fn_pooled_raw(samples)
    return out[0].double(), out[8].double(), out[9]


print('=== raw path == grid path when every cycle has P records ===')
configs = []
for layer in ('s4d', 'vanilla'):
    for time_dt in ('off', 'const', 'real'):
        if layer == 'vanilla' and time_dt == 'off':
            continue          # old vanilla mixer = the fused mamba_ssm block (CUDA); covered in --gpu
        for boundary, r_first in (('none', 0), ('shared', 0), ('index', 0), ('readout', 0), ('readout', 1)):
            configs.append((layer, time_dt, boundary, r_first, 0))
configs += [('s4d', 'real', 'readout', 1, 1), ('vanilla', 'real', 'none', 0, 1)]
for layer, time_dt, boundary, r_first, dt_input in configs:
    extra = dict(s4_backend='fft') if (layer == 's4d' and time_dt == 'off') else {}
    torch.manual_seed(1)
    ga = make_args(mamba_layer=layer, long_time_dt=time_dt, long_boundary=boundary, long_r_first=r_first,
                   long_dt_input=dt_input, **extra)
    grid = LongMamba.Model(ga).double().eval()
    ra = make_args(mamba_layer=layer, long_time_dt=time_dt, long_boundary=boundary, long_r_first=r_first,
                   long_dt_input=dt_input, long_raw=1, raw_n_channels=3, **extra)
    raw = LongMamba.Model(ra).double().eval()
    raw.load_state_dict(grid.state_dict())
    x, m, tau = grid_batch([6, 2, 4])
    xf, tf, cf = grid_to_flat(x, m, tau)
    name = f'{layer} {time_dt} {boundary}{" r_first" if r_first else ""}{" dt_input" if dt_input else ""}'
    with torch.no_grad():
        if boundary == 'readout':
            o1, r1, rm1 = grid(x, m, tau=tau, return_r=True)
            o2, r2, rm2 = raw(xf, m, tau=tf, cid=cf, return_r=True)
            d = max(rel(o2, o1), rel(r2 * rm2.unsqueeze(-1), r1 * rm1.unsqueeze(-1)))
            ok = d < 1e-10 and torch.equal(rm1, rm2)
        else:
            o1 = grid(x, m, tau=tau)
            o2 = raw(xf, m, tau=tf, cid=cf)
            d = rel(o2, o1)
            ok = d < 1e-10
    check(name, ok, f'max rel diff {d:.1e}')

# ---------------------------------------------------------------------------------------------------------------


def var_samples(cycle_lens_list, C=4, seed=0):
    """samples with variable records per cycle (0 = missing cycle); prefix = all listed cycles."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for b, lens in enumerate(cycle_lens_list):
        n = sum(lens)
        m = torch.zeros(L_)
        m[:len(lens)] = 1
        out.append(dict(raw_x=torch.randn(n, C, generator=g, dtype=torch.float64),
                        raw_tau=torch.exp(torch.randn(n, generator=g, dtype=torch.float64)),
                        raw_cid=torch.cat([torch.full((k,), i) for i, k in enumerate(lens)]).long(),
                        curve_attn_mask=m, labels=0., life_class=0, scaled_life_class=0, weight=1.,
                        seen_unseen_id=1, chemistry_id=b % 4))
    return out


def run(model, samples, **kw):
    x, msk, *_, ch, tau, cid = my_collate_fn_pooled_raw(samples)
    return model(x.double(), msk.double(), tau=tau.double(), cid=cid, chemistry_ids=ch, **kw)


print('=== variable cycle lengths ===')
lens = [[5, 9, 0, 3, 7], [2, 2], [11, 1, 4]]
for layer in ('s4d', 'vanilla'):
    for boundary, r_first, chem in (('readout', 1, 0), ('readout', 0, 1), ('index', 0, 0), ('none', 0, 0)):
        torch.manual_seed(2)
        args = make_args(mamba_layer=layer, long_time_dt='real', long_boundary=boundary, long_r_first=r_first,
                         long_raw=1, long_chem_cls=chem, long_chem_boundary=chem)
        model = LongMamba.Model(args).double().eval()
        name = f'{layer} {boundary}{" r_first" if r_first else ""}{" chem" if chem else ""}'
        S = var_samples(lens)
        with torch.no_grad():
            out = run(model, S)
            d = max(rel(run(model, [S[i]]), out[i:i + 1]) for i in range(3))
            check(f'{name}: padded == alone', d < 1e-10, f'{d:.1e}')
            S2 = var_samples(lens)
            S2[1]['raw_x'] = torch.cat([S2[1]['raw_x'], torch.randn(6, 4, dtype=torch.float64)])   # records of an unseen cycle
            S2[1]['raw_tau'] = torch.cat([S2[1]['raw_tau'], torch.full((6,), 50., dtype=torch.float64)])
            S2[1]['raw_cid'] = torch.cat([S2[1]['raw_cid'], torch.full((6,), 2)])
            d = rel(run(model, S2)[1], out[1])
            check(f'{name}: records past the prefix ignored', d < 1e-10, f'{d:.1e}')
            S3 = var_samples(lens)
            S3[0]['raw_tau'] = S3[0]['raw_tau'] * 3
            d = rel(run(model, S3)[0], out[0])
            check(f'{name}: real depends on tau', d > 1e-6, f'{d:.1e}')
            model.time_dt = 'const'
            d = rel(run(model, S3)[0], run(model, S)[0])
            check(f'{name}: const ignores tau', d < 1e-12, f'{d:.1e}')
            model.time_dt = 'real'
        model.train()
        model.zero_grad()
        if boundary == 'readout':
            o, r, rm = run(model, S, return_r=True)
            check(f'{name}: r shapes', tuple(r.shape) == (3, len(lens[0]) - 1, 3) and rm.sum().item() == 4 + 1 + 2,
                  f'{tuple(r.shape)} mask sum {rm.sum().item()}')
            (o.pow(2).mean() + (r.pow(2).mean(-1) * rm).mean()).backward()
        else:
            o = run(model, S)
            o.pow(2).mean().backward()
        missing = [n for n, q in model.named_parameters() if q.grad is None or q.grad.abs().sum().item() == 0]
        if chem:
            missing = [n for n in missing if not n.startswith('chem_')]   # 3 of 4 chemistries used
        check(f'{name}: every parameter gets gradient', not missing, str(missing[:6]))
        model.grad_ckpt = True
        o_ck = run(model, S)
        model.grad_ckpt = False
        d = rel(o_ck, run(model, S))
        check(f'{name}: grad checkpointing same output', d < 1e-12, f'{d:.1e}')

try:
    LongMamba.Model(make_args(long_raw=1))(torch.zeros(1, 5, 4), torch.ones(1, L_), tau=torch.ones(1, 5))
    check('missing cid raises', False)
except ValueError:
    check('missing cid raises', True)

# ---------------------------------------------------------------------------------------------------------------
if cli.gpu:
    assert torch.cuda.is_available(), '--gpu needs a GPU'
    dev = 'cuda'
    print(f'=== GPU: {torch.cuda.get_device_name()} ===')
    # old vanilla mixer (fused CUDA block) through the raw path == grid path
    args = make_args(mamba_layer='vanilla', long_time_dt='off', long_boundary='readout', long_r_first=1, mamba_scan='cuda')
    torch.manual_seed(1)
    grid = LongMamba.Model(args).to(dev).eval()
    raw = LongMamba.Model(make_args(**{**vars(args), 'long_raw': 1, 'raw_n_channels': 3})).to(dev).eval()
    raw.load_state_dict(grid.state_dict())
    x, m, tau = grid_batch([6, 2, 4])
    xf, tf, cf = grid_to_flat(x, m, tau)
    with torch.no_grad():
        d = rel(raw(xf.float().to(dev), m.float().to(dev), cid=cf.to(dev)), grid(x.float().to(dev), m.float().to(dev)))
    check('vanilla off readout: raw == grid (cuda)', d < 1e-4, f'{d:.1e}')
    for layer in ('s4d', 'vanilla'):
        for ckpt in (0, 1):
            a = make_args(d_model=64, early_cycle_threshold=100, output_num=50, mamba_n_layers=4, mamba_d_state=16,
                          mamba_layer=layer, mamba_scan='cuda', s4_dt_min=1e-5, long_boundary='readout',
                          long_r_first=1, long_time_dt='real', long_raw=1, long_grad_ckpt=ckpt)
            model = LongMamba.Model(a).to(dev).train()
            for B, per_cycle in ((8, 1100), (1, 6900)):    # MATR-like batch, one XJTU-like cell
                T = 100 * per_cycle
                x = torch.randn(B, T, 4, device=dev)
                msk = torch.ones(B, 100, device=dev)
                cid = torch.arange(100, device=dev).repeat_interleave(per_cycle).unsqueeze(0).expand(B, T).contiguous()
                tau = torch.exp(torch.randn(B, T, device=dev) * 0.5)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
                try:
                    o, r, rm = model(x, msk, tau=tau, cid=cid, return_r=True)
                    (o.sum() + r.sum()).backward()
                    torch.cuda.synchronize()
                    print(f'  {layer} raw real readout grad_ckpt={ckpt} B={B} x {T} steps ({B * T / 1e3:.0f}k): '
                          f'fwd+bwd {time.time() - t0:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
                except torch.cuda.OutOfMemoryError:
                    print(f'  {layer} raw real readout grad_ckpt={ckpt} B={B} x {T} steps: OOM')
                o = r = rm = None
                model.zero_grad(set_to_none=True)
                del x, cid, tau
                torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------------------------
if cli.data:
    print(f'=== Dataset_pooled_raw vs Dataset_pooled (Li-ion val, c={cli.dt_c}, clip={cli.dt_clip}) ===')
    from sklearn.preprocessing import StandardScaler
    from data_provider.data_loader_pooled import Dataset_pooled
    from data_provider.data_loader_raw import Dataset_pooled_raw, data_provider_pooled_raw
    da = SimpleNamespace(root_path=cli.root_path, seq_len=1, charge_discharge_length=300, dataset='POOLED',
                         early_cycle_threshold=100, weighted_loss=False, chem_loss_alpha=0.0, target_dataset=None,
                         dt_c=cli.dt_c, dt_clip=cli.dt_clip, raw_v_norm='cycle_max', raw_tau_mode=cli.raw_tau_mode,
                         raw_tau_boundary=cli.raw_tau_boundary, raw_tau_scale=300.0, batch_size=8, num_workers=0,
                         raw_max_batch_steps=600000)
    sc = StandardScaler().fit(np.array([[100.0], [3000.0]]))
    a = Dataset_pooled(da, flag='val', chemistries=['Li-ion'], label_scaler=sc, life_class_scaler=sc)
    b = Dataset_pooled_raw(da, flag='val', chemistries=['Li-ion'], label_scaler=sc, life_class_scaler=sc)
    check('same number of samples', len(a) == len(b), f'{len(a)} vs {len(b)}')
    check('same labels', np.array_equal(np.asarray(a.total_labels), np.asarray(b.total_labels)))
    check('same prefixes', all(np.array_equal(np.asarray(a.total_curve_attn_masks[i]), np.asarray(b.total_curve_attn_masks[i]))
                               for i in range(len(a))))
    ok_order, ok_len = True, True
    for c in b._cells:
        cid = c['cid'].astype(np.int64)
        ok_order &= bool(np.all(np.diff(cid) >= 0))
        ok_len &= bool(np.array_equal(np.bincount(cid, minlength=100)[:100], np.diff(c['starts'])))
    check('records in cycle order', ok_order)
    check('starts match cycle ids', ok_len)
    s = b[len(b) // 2]
    n = int(s['curve_attn_mask'].sum())
    check('sample = records of its prefix', int(s['raw_cid'].max()) <= n - 1 and len(s['raw_tau']) == b.lengths[len(b) // 2],
          f'{n} cycles, {len(s["raw_tau"])} records')
    check('4 channels, finite', s['raw_x'].shape[1] == 4 and bool(torch.isfinite(s['raw_x']).all()))
    from accelerate import Accelerator
    _, loader = data_provider_pooled_raw(da, 'val', sc, sc, chemistries=['Li-ion'])
    loader = Accelerator().prepare(loader)
    seen = 0
    t0 = time.time()
    for batch in loader:
        seen += batch[0].shape[0]
        if len(batch) != 10:
            break
    check('accelerate-prepared loader yields every sample once', seen == len(b), f'{seen} vs {len(b)} '
          f'({len(loader)} batches, {time.time() - t0:.0f}s)')

print('\nALL PASSED' if not fails else f'\n{len(fails)} FAILED: {fails}')
