"""
Checks for HierGen (models/HierGen.py, data_provider/data_loader_gen.py, train_hiergen.py).

    source venv_mamba/bin/activate
    python test_hiergen.py              # CPU, S4D mixers (pure PyTorch)
    python test_hiergen.py --gpu        # + vanilla Mamba stacks on CUDA (step == forward, model fwd/bwd, rollout)
    python test_hiergen.py --data       # + the gen cache: GenCells, collate, one real forward/backward + rollout

  1. SeqStack.step == SeqStack.forward (S4D; Mamba with --gpu)
  2. causality: cycle c's data never reaches the predictions for cycles <= c; reading point t never sees V/Q at >= t
  3. stepping the reading stack with the true previous values == the teacher-forced forward (so generation runs the
     model it was trained as); stepping the cycle stack == the parallel h
  4. rollout (latent + full) runs, cells are independent of their batch-mates, life is the first cycle <= 0.8
  5. every parameter gets a gradient (with the 11/08 unroll + reading-noise losses on)
  6. (11/08) generate_cycle with a given Q (fullcc), unrolled_h == free-running step mode for cycles <= cut + K,
     fullcc rollout Q = template x (predicted SOH / last observed SOH), free_run_metrics runs
  7. (11/10) generated_h: its generated grids and SOH-head means == rollout(mode='full', prefix=cut) for the K cycles
     after the cut; real cycles >= cut never reach hg up to cut + K; no grad through generation, grad into the encoder
  8. (11/11) read_selfgen: the sg_ reading loss == a pass whose inputs are generate_cycle's own curve (from the same
     teacher-forced state, the cycle's true I) and whose targets are the REAL curve; differs from teacher forcing;
     with read_selfgen 0.5 both losses are present and every parameter gets a gradient
"""
import argparse
from types import SimpleNamespace

import torch

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
p.add_argument('--data', action='store_true')
p.add_argument('--root_path', default='./dataset')
cli = p.parse_args()
fails = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)


def rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


from models.HierGen import HierGen, SeqStack, gaussian_nll


def cfg(layer='s4d', P=20, **kw):
    c = dict(n_points=P, d_model=32, d_ff=64, cycle_layer=layer, read_layer=layer, n_cycle_layers=2, n_read_layers=2,
             d_state=8, d_conv=4, expand=2, cyc_dt_min=1e-3, cyc_dt_max=1e-1, read_dt_min=1e-2, read_dt_max=1e-1,
             s4_backend='fft', mamba_scan='cuda', grad_ckpt=0)
    c.update(kw)
    return SimpleNamespace(**c)


def fake_cells(B, N, P, seed=0, dev='cpu'):
    """Plausible grids: V in volts, I in C, Q ramps per half; SOH declining."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, P // 2)
    curves = torch.zeros(B, N, 4, P)
    soh = torch.zeros(B, N)
    for b in range(B):
        rate = 0.0005 + 0.002 * torch.rand(1, generator=g).item()
        for c in range(N):
            cap = 1.0 - rate * c
            curves[b, c, 0] = torch.cat([3.0 + 0.6 * t, 3.6 - 1.0 * t]) + 0.01 * torch.randn(P, generator=g)
            curves[b, c, 1] = torch.cat([torch.full((P // 2,), 1.0), torch.full((P // 2,), -1.0)])
            curves[b, c, 2] = torch.cat([cap * t, cap * t])
            soh[b, c] = cap
    mask = torch.ones(B, N, dtype=torch.bool)
    return curves.to(dev), mask.to(dev), soh.to(dev), torch.zeros(B, dtype=torch.bool, device=dev)


def run_checks(layer, dev):
    torch.manual_seed(0)
    print(f'=== {layer} on {dev} ===')
    # 1. step == forward
    st = SeqStack(layer, 32, 2, d_state=8, dt_min=1e-2, dt_max=1e-1).to(dev).eval()
    x = torch.randn(3, 40, 32, device=dev)
    with torch.no_grad():
        y = st(x)
        s = st.init_state(3, dev)
        ys = []
        for t in range(40):
            o, s = st.step(x[:, t], s)
            ys.append(o)
        ys = torch.stack(ys, 1)
    check(f'{layer}: SeqStack.step == forward', rel(ys, y) < 1e-3, f'(rel {rel(ys, y):.1e})')

    P, N, B = 20, 12, 3
    m = HierGen(cfg(layer, P), torch.tensor([3.3, 0.0, 0.5]), torch.tensor([0.3, 1.0, 0.3])).to(dev)
    curves, mask, soh, dfirst = fake_cells(B, N, P, dev=dev)

    # 2a. causality across cycles: perturb cycle 5 -> SOH / zhat predictions for cycles <= 5 unchanged, 6 changes
    def cyc_preds(cv):
        z = m.encode(cv[:, :, :3])
        h = m.cycle_stack(torch.cat([m.start.expand(B, 1, m.D), z[:, :-1]], 1))
        return m.soh_out(h)[0], m.latent_head(h)
    m.eval()
    with torch.no_grad():
        mu0, zh0 = cyc_preds(curves)
        c2 = curves.clone()
        c2[:, 5, 0] += 0.5
        mu1, zh1 = cyc_preds(c2)
    check('causal over cycles: cycle 5 does not change predictions for cycles 0..5',
          torch.allclose(mu0[:, :6], mu1[:, :6], atol=1e-5) and torch.allclose(zh0[:, :6], zh1[:, :6], atol=1e-5))  # fft S4D leaks ~1e-7 (float32)
    check('... and does change cycle 6', not torch.allclose(mu0[:, 6], mu1[:, 6]))

    # 2b. causality inside a cycle: V and Q at point 7 do not reach outputs 0..7, do reach 8
    def read_out(g, hc, df):
        dmask = m.discharge_half_mask(df)
        f, qp = m.read_features(g[:, 0], g[:, 2], g[:, 1], dmask)
        x = m.read_in(f) + m.pos.weight.unsqueeze(0) + m.read_cond(hc).unsqueeze(1)
        return m.read_head(m.read_stack(x)), f
    g = curves[:, 3, :3]
    hc = torch.randn(B, 32, device=dev)
    with torch.no_grad():
        o0, f0 = read_out(g, hc, dfirst)
        g2 = g.clone()
        g2[:, 0, 7] += 0.5
        g2[:, 2, 7] += 0.1
        o1, _ = read_out(g2, hc, dfirst)
    check('causal inside a cycle: V/Q at point 7 do not change outputs 0..7', torch.allclose(o0[:, :8], o1[:, :8], atol=1e-5))
    check('... and do change output 8', not torch.allclose(o0[:, 8], o1[:, 8]))
    check('Q counter restarts at the half boundary (prev Q = 0 at points 0 and P/2)',
          bool((f0[:, 0, 1] == 0).all() and (f0[:, P // 2, 1] == 0).all()))

    # 3a. reading: stepping with true previous values == teacher-forced forward
    with torch.no_grad():
        dmask = m.discharge_half_mask(dfirst)
        stt = m.read_stack.init_state(B, dev)
        cond = m.read_cond(hc)
        outs = []
        for t in range(P):
            x = m.read_in(f0[:, t]) + m.pos.weight[t] + cond
            y, stt = m.read_stack.step(x, stt)
            outs.append(m.read_head(y))
        outs = torch.stack(outs, 1)
    check('reading: step with true inputs == teacher-forced forward', rel(outs, o0) < 1e-3, f'(rel {rel(outs, o0):.1e})')

    # 3b. generate_cycle feeds back its own predictions exactly like the features expect: regenerate the first point
    with torch.no_grad():
        gen = m.generate_cycle(hc, g[:, 1], dmask)
        fg, _ = m.read_features(gen[:, 0], gen[:, 2], gen[:, 1], dmask)
        og, _ = read_out(gen, hc, dfirst)
        vn = (gen[:, 0] - m.chan_mean[0, 0, 0]) / m.chan_std[0, 0, 0]
    check('generate_cycle: generated V equals the forward mean on its own generated history',
          rel(og[..., 0], vn) < 1e-3, f'(rel {rel(og[..., 0], vn):.1e})')

    # 3d. generate_cycle with a given Q (fullcc): Q is exactly the given one, V == forward mean on (gen V, given Q)
    with torch.no_grad():
        qg = g[:, 2] * 0.9
        genq = m.generate_cycle(hc, g[:, 1], dmask, q_given=qg)
        ogq, _ = read_out(torch.stack([genq[:, 0], g[:, 1], qg], 1), hc, dfirst)
        vnq = (genq[:, 0] - m.chan_mean[0, 0, 0]) / m.chan_std[0, 0, 0]
    check('generate_cycle(q_given): Q channel == given Q', torch.equal(genq[:, 2], qg))
    check('generate_cycle(q_given): generated V equals the forward mean on (own V, given Q)',
          rel(ogq[..., 0], vnq) < 1e-3, f'(rel {rel(ogq[..., 0], vnq):.1e})')

    # 3c. cycle stack stepping == parallel
    with torch.no_grad():
        z = m.encode(curves[:, :, :3])
        h = m.cycle_stack(torch.cat([m.start.expand(B, 1, m.D), z[:, :-1]], 1))
        s = m.cycle_stack.init_state(B, dev)
        hs, s = m.cycle_stack.step(m.start.expand(B, m.D), s)
        hh = [hs]
        for c in range(N - 1):
            hs, s = m.cycle_stack.step(z[:, c], s)
            hh.append(hs)
        hh = torch.stack(hh, 1)
    check('cycle stack: step == parallel h', rel(hh, h) < 1e-3, f'(rel {rel(hh, h):.1e})')

    # 3e. unrolled_h: exact free-running h (step mode, zhat fed back from the cut on) for cycles <= cut + K; causal
    K = 3
    cut = torch.tensor([3, 5, 8], device=dev)
    with torch.no_grad():
        z = m.encode(curves[:, :, :3])
        hu = m.unrolled_h(z, cut, K)
        s = m.cycle_stack.init_state(B, dev)
        hs, s = m.cycle_stack.step(m.start.expand(B, m.D), s)
        ref = [hs]
        for c in range(N - 1):
            x = torch.where((c < cut).unsqueeze(1), z[:, c], m.latent_head(hs))
            hs, s = m.cycle_stack.step(x, s)
            ref.append(hs)
        ref = torch.stack(ref, 1)
        z2 = z.clone()
        z2[:, 1] += torch.randn_like(z2[:, 1])          # random, not a constant: the pre-norm LayerNorms remove a shift
        hu2 = m.unrolled_h(z2, cut, K)
    exact = max(rel(hu[b, :int(cut[b]) + K + 1], ref[b, :int(cut[b]) + K + 1]) for b in range(B))
    check(f'unrolled_h (K={K}) == free-running step mode for cycles <= cut + K', exact < 1e-3, f'(rel {exact:.1e})')
    check('... and only approximate beyond (the test can tell)', rel(hu[0, 3 + K + 1:], ref[0, 3 + K + 1:]) > 1e-4)
    check('unrolled_h causal: z of cycle 1 does not change h for cycles 0..1', torch.allclose(hu[:, :2], hu2[:, :2], atol=1e-5))
    check('... and does change cycle 2', not torch.allclose(hu[:, 2], hu2[:, 2]))

    # 4. rollout
    span = torch.ones(B, device=dev)
    mask2 = mask.clone()
    mask2[1, 7:] = False                                                    # cell 1 has only 7 observed cycles
    for mode in ('latent', 'full', 'fullcc'):
        with torch.no_grad():
            r = m.rollout(curves, mask2, dfirst, span, prefix=10, max_cycles=40, mode=mode)
            r1 = m.rollout(curves[1:2], mask2[1:2], dfirst[1:2], span[1:2], prefix=10, max_cycles=40, mode=mode)
        ok_shape = r['soh'].shape == (B, 40) and r['life'].shape == (B,)
        same = torch.allclose(torch.nan_to_num(r['soh'][1], -9), torch.nan_to_num(r1['soh'][0], -9), atol=1e-5)
        check(f'rollout {mode}: shapes, generated from cycle n_obs on, cell independent of batch-mates',
              ok_shape and same and bool(torch.isnan(r['soh'][0, :10]).all()) and bool(torch.isfinite(r['soh'][1, 7]))
              and bool(torch.isnan(r['soh'][1, :7]).all()))
        L = r['life']
        ok_life = True
        for b in range(B):
            sb = r['soh'][b]
            below = ((sb <= 0.8) & torch.isfinite(sb)).nonzero()
            ok_life &= (torch.isnan(L[b]).item() if len(below) == 0 else L[b].item() == below[0].item() + 1)
        check(f'rollout {mode}: life = first generated cycle with SOH <= 0.8 (1-based)', bool(ok_life), f'{L.tolist()}')

    # 4b. fullcc: first generated cycle's Q = last observed Q x (its SOH / last observed SOH)
    with torch.no_grad():
        r = m.rollout(curves, mask2, dfirst, span, prefix=10, max_cycles=40, mode='fullcc', keep_grids=1)
    n_obs = mask2.sum(1).clamp(max=10)
    ok = True
    for b in range(B):
        qt = curves[b, n_obs[b] - 1, 2]
        s_last = (qt * m.discharge_half_mask(dfirst)[b]).amax() / span[b]
        qg = r['grids'][0][b, 2].to(dev)
        expect = qt * (r['soh'][b, n_obs[b]] / s_last).clamp(min=0)
        ok &= bool(torch.allclose(qg, expect, atol=1e-5))
    check('rollout fullcc: generated Q = template x predicted SOH / last observed SOH', ok)

    # 4c. free-running metrics
    with torch.no_grad():
        fr = m.free_run_metrics(curves, mask, dfirst, torch.tensor([[0, 4], [2, -1], [5, 9]], device=dev))
    check('free_run_metrics finite', set(fr) == {'fr_v_mae_V', 'fr_qd_mae', 'fr_cc_v_mae_V'} and
          all(torch.isfinite(v) for v in fr.values()), f'{ {k: round(float(v), 4) for k, v in fr.items()} }')

    # 4d. generated_h == full rollout from the same cut
    K = 3
    cut = torch.tensor([4, 2, 6], device=dev)
    with torch.no_grad():
        z = m.encode(curves[:, :, :3])
        h0 = m.cycle_stack(torch.cat([m.start.expand(B, 1, m.D), z[:, :-1]], 1))
        hg, G = m.generated_h(z, curves[:, :, :3], dfirst, cut, K, h0)
        maskc = torch.arange(N, device=dev).unsqueeze(0) < cut.unsqueeze(1)     # n_obs = cut per cell
        r = m.rollout(curves, maskc, dfirst, span, prefix=N, max_cycles=N, mode='full', threshold=-1.0, keep_grids=K)
        mu_g, _ = m.soh_out(hg)
    gerr = max(rel(G[:, k], r['grids'][k].to(dev)) for k in range(K))
    serr = max(abs(float(mu_g[b, cut[b] + k] - r['soh_head'][b, cut[b] + k])) for b in range(B) for k in range(K))
    check(f'generated_h grids == rollout full (prefix = cut, K={K})', gerr < 1e-3, f'(rel {gerr:.1e})')
    check('generated_h SOH-head means == rollout full soh_head on the K generated cycles', serr < 1e-3, f'({serr:.1e})')
    c2 = curves.clone()
    for b in range(B):
        c2[b, cut[b]:cut[b] + K, 0] += 0.5 * torch.randn_like(c2[b, cut[b]:cut[b] + K, 0])   # real cycles cut..cut+K-1
    with torch.no_grad():
        z2 = m.encode(c2[:, :, :3])
        hg2, _ = m.generated_h(z2, c2[:, :, :3], dfirst, cut, K, h0)
    leak = max(rel(hg2[b, :cut[b] + K + 1], hg[b, :cut[b] + K + 1]) for b in range(B))
    check('generated_h: real cycles cut..cut+K-1 do not reach hg up to cut + K', leak < 1e-5, f'(rel {leak:.1e})')
    c3 = curves.clone()
    c3[0, cut[0] - 1, 0] += 0.5 * torch.randn_like(c3[0, cut[0] - 1, 0])
    with torch.no_grad():
        z3 = m.encode(c3[:, :, :3])
        h03 = m.cycle_stack(torch.cat([m.start.expand(B, 1, m.D), z3[:, :-1]], 1))
        hg3, _ = m.generated_h(z3, c3[:, :, :3], dfirst, cut, K, h03)
    check('... and the last real cycle before the cut does', rel(hg3[0, cut[0]:cut[0] + K + 1], hg[0, cut[0]:cut[0] + K + 1]) > 1e-4)
    zg = m.encode(curves[:, :, :3])
    hgg, Gg = m.generated_h(zg, curves[:, :, :3], dfirst, cut, K, h0)
    check('generated_h: no grad through the generated grids, hg has grad', (not Gg.requires_grad) and hgg.requires_grad)

    # 5. gradients, with the unroll + reading-noise losses on
    torch.manual_seed(1)
    m = HierGen(cfg(layer, P, unroll_k=3, read_vnoise=0.05, read_qnoise=0.01, gen_k=2, gen_read=2),
                torch.tensor([3.3, 0.0, 0.5]),
                torch.tensor([0.3, 1.0, 0.3])).to(dev)
    m.train()
    ri = torch.tensor([[0, 3, 11], [2, -1, -1], [5, 6, 7]], device=dev)
    out = m(curves, mask, soh, dfirst, ri)
    check('unroll losses present', {'unroll_soh_nll', 'unroll_soh_mae', 'unroll_latent_mse'} <= set(out))
    check('gen losses present', {'gen_soh_nll', 'gen_soh_mae', 'gen_read_v_nll', 'gen_read_q_nll'} <= set(out))
    loss = out['soh_nll'] + out['latent_mse'] + out['read_v_nll'] + out['read_q_nll'] + out['unroll_soh_nll'] + \
        out['unroll_latent_mse'] + out['gen_soh_nll'] + out['gen_read_v_nll'] + out['gen_read_q_nll']
    loss.backward()
    nog = [n for n, q in m.named_parameters() if q.requires_grad and (q.grad is None or q.grad.abs().sum() == 0)]
    check('every parameter gets a gradient', not nog, f'{nog[:5]}')
    check('losses finite', all(torch.isfinite(v).all() for v in out.values()), f'{ {k: round(float(v), 3) for k, v in out.items()} }')

    # 8. read_selfgen
    torch.manual_seed(2)
    m = HierGen(cfg(layer, P, read_selfgen=1.0), torch.tensor([3.3, 0.0, 0.5]), torch.tensor([0.3, 1.0, 0.3])).to(dev)
    m.eval()
    ri = torch.tensor([[0, 3, 11], [2, -1, -1], [5, 6, 7]], device=dev)
    with torch.no_grad():
        out = m(curves, mask, soh, dfirst, ri)
        bi, rr = (ri >= 0).nonzero(as_tuple=True)
        ci = ri[bi, rr]
        z = m.encode(curves[:, :, :3])
        h = m.cycle_stack(torch.cat([m.start.expand(B, 1, m.D), z[:, :-1]], 1))
        g, hc, dm = curves[bi, ci, :3], h[bi, ci], m.discharge_half_mask(dfirst[bi])

        def nll_from(Vin, Qin):
            f, qp = m.read_features(Vin, Qin, g[:, 1], dm)
            x = m.read_in(f) + m.pos.weight.unsqueeze(0) + m.read_cond(hc).unsqueeze(1)
            o = m.read_head(m.read_stack(x))
            vt, qt = m.read_targets(g[:, 0], g[:, 2], qp)                  # targets: the real curve
            return gaussian_nll(o[..., 0], o[..., 1], vt).mean(), gaussian_nll(o[..., 2], o[..., 3], qt).mean()
        gen = m.generate_cycle(hc, g[:, 1], dm)
        ref_v, ref_q = nll_from(gen[:, 0], gen[:, 2])
        tf_v, tf_q = nll_from(g[:, 0], g[:, 2])
    check('read_selfgen 1: only the sg_ reading losses (+ input-drift metrics)', 'read_v_nll' not in out and
          {'sg_read_v_nll', 'sg_read_q_nll', 'sg_in_v_mae_V', 'sg_in_qd_mae'} <= set(out), f'{sorted(out)}')
    ev = abs(float(out['sg_read_v_nll'] - ref_v)) / (1 + abs(float(ref_v)))
    eq = abs(float(out['sg_read_q_nll'] - ref_q)) / (1 + abs(float(ref_q)))
    check('sg loss == inputs from generate_cycle (same state, true I), targets the real curve', max(ev, eq) < 1e-3,
          f'(rel {max(ev, eq):.1e})')
    check('... and differs from the teacher-forced loss (the test can tell)',
          abs(float(ref_v - tf_v)) + abs(float(ref_q - tf_q)) > 1e-4)
    check('sg_in_v_mae_V == |generated V - real V|', abs(float(out['sg_in_v_mae_V'] - (gen[:, 0] - g[:, 0]).abs().mean())) < 1e-5)
    torch.manual_seed(3)
    m = HierGen(cfg(layer, P, read_selfgen=0.5, read_vnoise=0.05, read_qnoise=0.01), torch.tensor([3.3, 0.0, 0.5]),
                torch.tensor([0.3, 1.0, 0.3])).to(dev)
    m.train()
    out = m(curves, mask, soh, dfirst, ri)
    check('read_selfgen 0.5: teacher-forced and sg reading losses both present', {'read_v_nll', 'sg_read_v_nll'} <= set(out))
    (out['soh_nll'] + out['latent_mse'] + out['read_v_nll'] + out['read_q_nll'] + out['sg_read_v_nll'] +
     out['sg_read_q_nll']).backward()
    nog = [n for n, q in m.named_parameters() if q.requires_grad and (q.grad is None or q.grad.abs().sum() == 0)]
    check('read_selfgen 0.5: every parameter gets a gradient', not nog, f'{nog[:5]}')


run_checks('s4d', 'cpu')
if cli.gpu:
    assert torch.cuda.is_available(), '--gpu needs CUDA'
    run_checks('vanilla', 'cuda')
    run_checks('s4d', 'cuda')

if cli.data:
    print('=== gen cache ===')
    from data_provider.data_loader_gen import GenCells, collate_cells
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr = GenCells(cli.root_path, 'train')
    va = GenCells(cli.root_path, 'val')
    mean, std = tr.channel_stats(max_cells=40)
    print(f'  channel mean {mean.tolist()} std {std.tolist()}')
    batch = collate_cells([tr[i] for i in range(4)])
    check('collate shapes', batch['curves'].shape[2:] == (4, 300) and batch['mask'].sum(1).tolist() ==
          [tr.cells[i]['n'] for i in range(4)])
    ok = all(abs(float(tr[i]['soh'][0]) - 1.0) < 0.3 for i in range(10))
    check('raw SOH of cycle 1 near 1 for the first train cells', ok, f'{[round(float(tr[i]["soh"][0]), 3) for i in range(10)]}')
    lab_ok = []
    for c in va.cells[:40]:
        s = c['soh']
        below = (s <= 0.8).nonzero()[0]
        if len(below):
            lab_ok.append(abs(int(below[0]) + 1 - c['eol']) <= 1)
    check('val cells that cross 0.8 in the data: first crossing == label (+-1)', all(lab_ok), f'{sum(lab_ok)}/{len(lab_ok)}')
    layer = 'vanilla' if dev == 'cuda' else 's4d'
    m = HierGen(cfg(layer, 300, d_model=64), mean, std).to(dev)
    b = {k: v.to(dev) for k, v in batch.items()}
    ri = torch.randint(0, 100, (4, 8), device=dev)
    out = m(b['curves'], b['mask'], b['soh'], b['discharge_first'], ri)
    sum(out[k] for k in ('soh_nll', 'latent_mse', 'read_v_nll', 'read_q_nll')).backward()
    check(f'real batch forward/backward ({layer})', all(torch.isfinite(v).all() for v in out.values()),
          f'{ {k: round(float(v), 3) for k, v in out.items()} }')
    with torch.no_grad():
        span = torch.tensor([tr.cells[i]['soc_span'] for i in range(4)], device=dev)
        for mode in ('latent', 'full', 'fullcc'):
            r = m.rollout(b['curves'], b['mask'], b['discharge_first'], span, prefix=100, max_cycles=110, mode=mode)
            check(f'real rollout {mode} (10 cycles past 100)', bool(torch.isfinite(r['soh'][:, 100:110]).any()))

print('\nALL PASSED' if not fails else f'\nFAILED: {fails}')
