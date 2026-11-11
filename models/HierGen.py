"""
HierGen (10/18): hierarchical generative model of battery cycling, SaShiMi-style two time scales.

Data: data_provider/data_loader_gen.py. Per cell, every cycle on the 300-point grid (150 charge + 150 discharge points,
by record index; order = order in the cycle), channels V (volts), I (C-rate), Q (/ nominal); per cycle the raw,
label-consistent SOH (max discharge capacity / nominal / SOC span; the label = first cycle with SOH <= 0.8).

    cycle c's grid [3, 300] --encoder (MLP)--> z_c                                         [B, N, D]
    CYCLE stack (Mamba / S4D, one step per cycle), causal, input [START, z_0, ..., z_{N-2}]  -> h_c = state BEFORE
        cycle c (it has seen cycles 0..c-1 only)
        heads on h_c:  SOH_c  (Gaussian: mean, log-variance)       cycle-level capacity
                       zhat_c (predicts z_c; target z_c with stop-grad)   latent rollout
    READING stack (Mamba / S4D, one step per grid point), one independent 300-step sequence per cycle, conditioned
        on h_c (added at every step = SaShiMi's upsampling of the coarse state). Input at point t:
            [V_{t-1}, Q_{t-1} (0 at the start of each half), I_t, I_{t-1}, is-discharge-half, is-first-point]
            + learned position embedding[t] + Linear(h_c)
        output: V_t and dQ_t = Q_t - Q_{t-1} (Gaussian each). I is given (the tester's protocol), V and Q are predicted,
        so a generated cycle's capacity = its predicted Q at the end of the discharge half.

Causality: nothing about cycle c (its z, its curve) reaches h_c, and inside a cycle point t only sees points < t
(plus the given I_t). test_hiergen.py checks both.

Rollout (eval; condition on cycles 1..P, then generate until SOH <= 0.8 or max_cycles):
    latent  h -> SOH head (the cycle's capacity) and zhat (fed back as the next cycle's z). Cheap.
    full    h -> the READING stack generates the cycle point by point (I from the last observed cycle, the protocol
            template), the generated grid goes through the encoder -> z, back into the CYCLE stack. The cycle's SOH is
            the generated Q at the end of discharge / SOC span (reported as `full`); the SOH head along the same path
            is reported as `full_head`.
    fullcc  (11/08) like full, but Q is NOT generated: the cycle's capacity is the SOH head's prediction, and Q is
            coulomb-counted = the last observed cycle's Q template scaled by (predicted SOH / last observed SOH), i.e.
            the same current over a proportionally shorter half. The reading stack generates V against that Q
            (q_prev given, not fed back), and [V_gen, I_tmpl, Q_cc] goes through the encoder -> z. Removes the 300-step
            dQ integration, which drifts badly when free-running (11/08 diagnosis: teacher-forced capacity error 8e-4,
            first free-running cycle off by ~0.3 SOH).

Training additions (11/08, all off by default):
    unroll_k K   K-pass parallel unroll of the cycle stack on its own zhat after a random cut per cell (pass k feeds the
                 zhat of pass k-1 for cycles >= cut; after K passes h is exactly the free-running h for cycles
                 <= cut + K, a K-deep approximation beyond). Passes 1..K-1 without grad. Losses unroll_soh_nll /
                 unroll_latent_mse on cycles > cut. Targets the latent rollout's collapse to a population-mean fade.
    gen_k K      (11/10) the FULL-mode analogue: from the teacher-forced state at a random cut per cell, generate K
                 cycles exactly as rollout(mode='full', prefix=cut) does (reading stack, protocol I = the last real
                 cycle before the cut; no grad), re-encode the generated grids WITH grad, run the cycle stack on
                 [real z < cut, generated z cut..cut+K-1] with grad, and train on the REAL cycles cut+1..cut+K:
                 gen_soh_nll (SOH head) + gen_read_v_nll / gen_read_q_nll (teacher-forced reading loss on the real
                 curves, conditioned on the state that came from the generated history). Teaches the cycle stack /
                 encoder / reading stack to continue from the model's own generated curves (11/08: encode(generated
                 grid) is out of distribution and full-mode SOH drifts to ~1.3). No backprop through generation.
    read_vnoise / read_qnoise   Gaussian noise on the reading stack's v_prev (normalized V) / q_prev (Q / nominal)
                 inputs; the dQ target is taken relative to the NOISY q_prev, so the model learns to correct drift.
    read_selfgen f  (11/11) self-generated reading inputs (scheduled sampling / DAgger on the 300-step reading stack):
                 a fraction f of the reading-loss cycles first GENERATE the cycle free-running (generate_cycle, no
                 grad: from the same teacher-forced cycle state, with the cycle's true I), then the teacher-forced-style
                 pass with grad takes v_prev / q_prev from that generated curve and is scored on the REAL V_t and
                 dQ_t = Q_real_t - q_prev_generated, i.e. "from where you actually are, get back to the data". The
                 inputs then carry the free-running error that full mode sees (11/10: fr_v_mae 0.036 V vs 0.005
                 teacher-forced), not the one-step error. Losses sg_read_v_nll / sg_read_q_nll; sg_in_v_mae_V /
                 sg_in_qd_mae = how far the generated inputs were from the real curve (a free-running metric).
                 No reading noise on these rows. selfgen_sample: generate by sampling instead of the means.
Both stacks run recurrently in rollout (S4D: explicit complex recurrence; Mamba: mamba_ssm's step()).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from models.CPMamba import ResidualMixerBlock

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0
N_READ_FEATURES = 6


# ---------------------------------------------------------------------------------------------------------- stacks
def _make_mixer(kind, d, i, d_state, d_conv, expand, dt_min, dt_max, s4_backend, scan_impl):
    if kind == 's4d':
        from models.S4D import S4DMixer
        return S4DMixer(d, d_state=d_state, d_conv=d_conv, expand=expand, dt_min=dt_min, dt_max=dt_max, layer_idx=i,
                        backend=s4_backend, scan_impl=scan_impl)
    if kind == 'vanilla':
        from models.CPMamba import get_mamba_mixer_cls
        return get_mamba_mixer_cls('vanilla', scan_impl)(d, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i)
    raise ValueError(f'mixer kind must be s4d or vanilla, got {kind}')


def _s4d_init_state(mixer, B, device):
    E, K = mixer.d_inner, mixer.conv1d.kernel_size[0]
    N2 = mixer.kernel.C.shape[1]
    return [torch.zeros(B, E, K - 1, device=device), torch.zeros(B, E, N2, dtype=torch.cfloat, device=device)]


def _s4d_step(mixer, h, state):
    """One step of S4DMixer: h [B, D] -> [B, D]; state = [conv buffer of raw u (last K-1), complex SSM state]."""
    buf, s = state
    u, z = mixer.in_proj(h).chunk(2, dim=-1)                                   # [B, E] each
    win = torch.cat([buf, u.unsqueeze(-1)], dim=-1)                            # [B, E, K]
    uc = F.silu((win * mixer.conv1d.weight[:, 0, :]).sum(-1) + mixer.conv1d.bias)
    dt, A, dB, C = mixer.kernel.discretize()
    dA = torch.exp(A * dt.unsqueeze(-1))
    s = dA * s + dB * uc.float().unsqueeze(-1)
    y = 2 * (C * s).sum(-1).real + mixer.D.float() * uc.float()
    y = y.to(uc.dtype) * F.silu(z)
    return mixer.out_proj(y), [win[..., 1:], s]


class SeqStack(nn.Module):
    """n pre-norm residual mixer blocks + final LayerNorm. forward: [B, T, D] -> [B, T, D] (causal).
    init_state / step: the same computation one position at a time (for rollout)."""

    def __init__(self, kind, d_model, n_layers, d_state=16, d_conv=4, expand=2, dt_min=1e-4, dt_max=1e-1,
                 s4_backend='fft', scan_impl='cuda', mixer_cls=None, grad_ckpt=False):
        super().__init__()
        self.kind = kind if mixer_cls is None else 'custom'
        make = mixer_cls or (lambda d, i: _make_mixer(kind, d, i, d_state, d_conv, expand, dt_min, dt_max, s4_backend,
                                                      scan_impl))
        self.layers = nn.ModuleList([ResidualMixerBlock(d_model, make(d_model, i)) for i in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.grad_ckpt = grad_ckpt
        if self.kind == 'vanilla':
            from functools import partial
            from mamba_ssm.models.mixer_seq_simple import _init_weights
            self.layers.apply(partial(_init_weights, n_layer=n_layers))
        elif self.kind == 's4d':
            from models.S4D import init_s4d_stack
            init_s4d_stack(self.layers, n_layers)

    def forward(self, x):
        h, res = x, None
        for blk in self.layers:
            if self.grad_ckpt and self.training:
                h, res = torch.utils.checkpoint.checkpoint(blk, h, res, use_reentrant=False)
            else:
                h, res = blk(h, res)
        return self.norm_f((h + res).to(self.norm_f.weight.dtype))

    def init_state(self, B, device):
        if self.kind == 's4d':
            return [_s4d_init_state(blk.mixer, B, device) for blk in self.layers]
        if self.kind == 'vanilla':
            return [list(blk.mixer.allocate_inference_cache(B, 1)) for blk in self.layers]
        raise NotImplementedError('step() needs s4d or vanilla mixers')

    def step(self, x, state, active=None):
        """x [B, D] -> y [B, D], new state. active [B] bool: rows where False keep their old state."""
        h, res, new_state = x, None, []
        for blk, st in zip(self.layers, state):
            res = h + res if res is not None else h
            hn = blk.norm(res.to(blk.norm.weight.dtype))
            res = res.float()
            if self.kind == 's4d':
                h, st_new = _s4d_step(blk.mixer, hn, st)
            else:
                cs, ss = st[0].clone(), st[1].clone()
                out, cs, ss = blk.mixer.step(hn.unsqueeze(1), cs, ss)
                h, st_new = out.squeeze(1), [cs, ss]
            if active is not None:
                st_new = [torch.where(active.view(-1, *([1] * (n.dim() - 1))), n, o) for n, o in zip(st_new, st)]
            new_state.append(st_new)
        return self.norm_f((h + res).to(self.norm_f.weight.dtype)), new_state


def gaussian_nll(mu, logvar, y):
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))


# ---------------------------------------------------------------------------------------------------------- model
class HierGen(nn.Module):
    def __init__(self, cfg, chan_mean, chan_std, mixer_cls=None):
        """cfg: argparse namespace (see train_hiergen.py). chan_mean / chan_std: [3] V, I, Q over the train grid."""
        super().__init__()
        D, P = cfg.d_model, cfg.n_points
        self.P, self.H, self.D = P, P // 2, D
        self.register_buffer('chan_mean', torch.as_tensor(chan_mean, dtype=torch.float32).view(1, 3, 1))
        self.register_buffer('chan_std', torch.as_tensor(chan_std, dtype=torch.float32).view(1, 3, 1))
        self.encoder = nn.Sequential(nn.Linear(3 * P, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, D), nn.LayerNorm(D))
        self.start = nn.Parameter(torch.randn(D) * 0.02)
        common = dict(d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, s4_backend=cfg.s4_backend,
                      scan_impl=cfg.mamba_scan, mixer_cls=mixer_cls)
        self.cycle_stack = SeqStack(cfg.cycle_layer, D, cfg.n_cycle_layers, dt_min=cfg.cyc_dt_min, dt_max=cfg.cyc_dt_max,
                                    **common)
        self.read_stack = SeqStack(cfg.read_layer, D, cfg.n_read_layers, dt_min=cfg.read_dt_min, dt_max=cfg.read_dt_max,
                                   grad_ckpt=bool(cfg.grad_ckpt), **common)
        self.soh_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))
        self.latent_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.read_in = nn.Linear(N_READ_FEATURES, D)
        self.read_cond = nn.Linear(D, D)
        self.pos = nn.Embedding(P, D)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.unroll_k = int(getattr(cfg, 'unroll_k', 0))
        self.read_vnoise = float(getattr(cfg, 'read_vnoise', 0.0))
        self.read_qnoise = float(getattr(cfg, 'read_qnoise', 0.0))
        self.gen_k = int(getattr(cfg, 'gen_k', 0))
        self.gen_read = int(getattr(cfg, 'gen_read', 4))
        self.read_selfgen = float(getattr(cfg, 'read_selfgen', 0.0))
        self.selfgen_sample = bool(getattr(cfg, 'selfgen_sample', 0))
        self.read_head = nn.Linear(D, 4)          # V mean, V logvar, dQ mean, dQ logvar

    # -------------------------------------------------------------------------------------------- pieces
    def encode(self, grid):
        """grid [..., 3, P] physical (V, I C-rate, Q/nominal) -> z [..., D]"""
        shp = grid.shape[:-2]
        g = (grid.reshape(-1, 3, self.P) - self.chan_mean) / self.chan_std
        return self.encoder(g.reshape(-1, 3 * self.P)).reshape(*shp, self.D)

    def soh_out(self, h):
        o = self.soh_head(h)
        return o[..., 0] + 1.0, o[..., 1]          # mean centred at SOH 1, log-variance

    def discharge_half_mask(self, discharge_first):
        """[B] bool -> [B, P] float, 1 on the discharge half."""
        t = torch.arange(self.P, device=discharge_first.device)
        second = (t >= self.H).float().unsqueeze(0)
        df = discharge_first.float().unsqueeze(1)
        return df * (1 - second) + (1 - df) * second

    def read_features(self, V, Q, I, dmask):
        """Teacher-forced reading inputs. V, Q, I [M, P] physical; dmask [M, P]. -> feats [M, P, 6], prev Q [M, P]."""
        M, P = V.shape
        vn = (V - self.chan_mean[0, 0, 0]) / self.chan_std[0, 0, 0]
        istd = self.chan_std[0, 1, 0]
        v_prev = torch.cat([torch.zeros_like(vn[:, :1]), vn[:, :-1]], 1)
        q_prev = torch.cat([torch.zeros_like(Q[:, :1]), Q[:, :-1]], 1)
        q_prev[:, self.H] = 0.0                                                # each half's Q counter starts at 0
        i_prev = torch.cat([torch.zeros_like(I[:, :1]), I[:, :-1]], 1)
        first = torch.zeros_like(V)
        first[:, 0] = 1.0
        f = torch.stack([v_prev, q_prev, I / istd, i_prev / istd, dmask, first], -1)
        return f, q_prev

    def read_targets(self, V, Q, q_prev):
        vn = (V - self.chan_mean[0, 0, 0]) / self.chan_std[0, 0, 0]
        return vn, (Q - q_prev) * self.H

    def unrolled_h(self, z, cut, K, h0=None):
        """Cycle-stack states with cycles >= cut[b] fed the model's own zhat instead of z, by K parallel passes
        (passes before the last without grad). z [B, N, D], cut [B] long. Exact free-running h for cycles <= cut + K.
        h0: the teacher-forced h (pass 0) if already computed."""
        B, N, D = z.shape
        keep = (torch.arange(N, device=z.device).unsqueeze(0) < cut.unsqueeze(1)).unsqueeze(-1)
        start = self.start.expand(B, 1, D)
        with torch.no_grad():
            h = h0.detach() if h0 is not None else self.cycle_stack(torch.cat([start, z[:, :-1]], 1))
            for _ in range(K - 1):
                x = torch.where(keep, z, self.latent_head(h))
                h = self.cycle_stack(torch.cat([start, x[:, :-1]], 1))
            zhat = self.latent_head(h)
        x = torch.where(keep, z, zhat)
        return self.cycle_stack(torch.cat([start, x[:, :-1]], 1))

    def generated_h(self, z, grid, discharge_first, cut, K, h0):
        """Cycle-stack states with cycles cut[b] .. cut[b]+K-1 replaced by the model's own GENERATED cycles.
        z [B, N, D] (with grad), grid [B, N, 3, P], cut [B] long in [1, n-1], h0 [B, N, D] teacher-forced h.
        Generation = rollout(mode='full', prefix=cut): cycle cut is generated from h0[:, cut] with the protocol I of
        cycle cut-1, encoded, fed to the cycle stack, the next cycle generated from the new state, ... (no grad).
        The generated grids are then re-encoded with grad and one cycle-stack pass with grad gives hg.
        -> hg [B, L, D] (L = max(cut) + K + 1 capped at N; hg[:, j] for cut < j <= cut + K is the state after the
           generated cycles cut..j-1), G [B, K, 3, P] the generated grids."""
        B, N, D = z.shape
        dev = z.device
        L = int(min(N, int(cut.max()) + K + 1))
        rows = torch.arange(B, device=dev)
        kk = torch.arange(K, device=dev)
        idx = cut.unsqueeze(1) + kk.unsqueeze(0)                                # [B, K] cycles replaced
        ok = idx < L
        idx_c = idx.clamp(max=L - 1)
        dmask = self.discharge_half_mask(discharge_first)
        I_tmpl = grid[rows, (cut - 1).clamp(min=0), 1]
        start = self.start.expand(B, 1, D)
        gens = []
        with torch.no_grad():
            x = z[:, :L].detach().clone()
            h = h0[:, :L].detach()
            for k in range(K):
                g = self.generate_cycle(h[rows, idx_c[:, k]], I_tmpl, dmask)
                gens.append(g)
                if k < K - 1:
                    zk = self.encode(g)
                    x[rows[ok[:, k]], idx_c[ok[:, k], k]] = zk[ok[:, k]]
                    h = self.cycle_stack(torch.cat([start, x[:, :-1]], 1))
        G = torch.stack(gens, 1)                                                # [B, K, 3, P]
        zg = self.encode(G)                                                     # [B, K, D], grad into the encoder
        x = z[:, :L].clone()
        bi = rows.unsqueeze(1).expand(B, K)
        x[bi[ok], idx_c[ok]] = zg[ok]
        hg = self.cycle_stack(torch.cat([start, x[:, :-1]], 1))
        return hg, G

    def _read_loss(self, g, hc, dmask, prefix='', selfgen=False):
        """Reading loss on real grids g [M, 3, P] conditioned on cycle states hc [M, D]. Teacher-forced (inputs = the
        real previous V / Q, + training noise), or selfgen: inputs = the model's own free-running generation of the
        cycle (no grad), targets still the real curve."""
        out = {}
        if selfgen:
            gen = self.generate_cycle(hc.detach(), g[:, 1], dmask, sample=self.selfgen_sample)    # no grad
            feats, q_prev = self.read_features(gen[:, 0], gen[:, 2], g[:, 1], dmask)
            out[prefix + 'in_v_mae_V'] = (gen[:, 0] - g[:, 0]).abs().mean()
            out[prefix + 'in_qd_mae'] = ((gen[:, 2] * dmask).amax(1) - (g[:, 2] * dmask).amax(1)).abs().mean()
        else:
            feats, q_prev = self.read_features(g[:, 0], g[:, 2], g[:, 1], dmask)
        if self.training and not selfgen and (self.read_vnoise > 0 or self.read_qnoise > 0):
            first_of_half = torch.zeros_like(q_prev, dtype=torch.bool)
            first_of_half[:, 0] = True
            first_of_half[:, self.H] = True
            vn_noise = self.read_vnoise * torch.randn_like(q_prev)
            vn_noise[:, 0] = 0.0                                                    # v_prev at t=0 stays 0
            q_noise = (self.read_qnoise * torch.randn_like(q_prev)).masked_fill(first_of_half, 0.0)
            q_prev = q_prev + q_noise                                               # dQ target vs the NOISY prev
            feats = torch.cat([feats[..., :1] + vn_noise.unsqueeze(-1), q_prev.unsqueeze(-1), feats[..., 2:]], -1)
        x = self.read_in(feats) + self.pos.weight.unsqueeze(0) + self.read_cond(hc).unsqueeze(1)
        o = self.read_head(self.read_stack(x))                              # [M, P, 4]
        vt, qt = self.read_targets(g[:, 0], g[:, 2], q_prev)
        out.update({prefix + 'read_v_nll': gaussian_nll(o[..., 0], o[..., 1], vt).mean(),
                    prefix + 'read_q_nll': gaussian_nll(o[..., 2], o[..., 3], qt).mean(),
                    prefix + 'read_v_mae_V': ((o[..., 0] - vt).abs().mean() * self.chan_std[0, 0, 0])})
        # capacity of the cycle from the teacher-forced dQ predictions vs the true grid Q at the end of discharge
        qd_pred = ((o[..., 2] / self.H + q_prev) * dmask).amax(1)
        qd_true = (g[:, 2] * dmask).amax(1)
        out[prefix + 'read_qd_mae'] = (qd_pred - qd_true).abs().mean()
        return out

    # -------------------------------------------------------------------------------------------- training
    def forward(self, curves, mask, soh, discharge_first, read_idx):
        """curves [B, N, 4, P], mask [B, N], soh [B, N], discharge_first [B], read_idx [B, R] (cycles for the reading
        loss; -1 = none). -> dict of per-element losses / metrics (means over valid elements)."""
        B, N = mask.shape
        grid = curves[:, :, :3]
        z = self.encode(grid)                                                   # [B, N, D]
        cyc_in = torch.cat([self.start.expand(B, 1, self.D), z[:, :-1]], 1)
        h = self.cycle_stack(cyc_in)                                            # h[:, c] = before cycle c
        m = mask.float()
        mu, lv = self.soh_out(h)
        soh_nll = (gaussian_nll(mu, lv, soh) * m).sum() / m.sum()
        soh_mae = ((mu - soh).abs() * m).sum() / m.sum()
        zhat = self.latent_head(h)
        lat = (((zhat - z.detach()) ** 2).mean(-1) * m).sum() / m.sum()

        out = dict(soh_nll=soh_nll, soh_mae=soh_mae, latent_mse=lat)
        if self.unroll_k > 0:
            n = mask.sum(1)
            cut = (torch.rand(B, device=z.device) * (n - 1).clamp(min=1).float()).long() + 1     # in [1, n-1]
            hu = self.unrolled_h(z, cut, self.unroll_k, h0=h)
            mu_u, lv_u = self.soh_out(hu)
            um = m * (torch.arange(N, device=z.device).unsqueeze(0) > cut.unsqueeze(1)).float()
            den = um.sum().clamp(min=1.0)
            out['unroll_soh_nll'] = (gaussian_nll(mu_u, lv_u, soh) * um).sum() / den
            out['unroll_soh_mae'] = ((mu_u - soh).abs() * um).sum() / den
            out['unroll_latent_mse'] = (((self.latent_head(hu) - z.detach()) ** 2).mean(-1) * um).sum() / den
        valid = read_idx >= 0
        if valid.any():
            bi, ri = valid.nonzero(as_tuple=True)
            ci = read_idx[bi, ri]
            g_r, h_r, dm_r = grid[bi, ci], h[bi, ci], self.discharge_half_mask(discharge_first[bi])
            M = g_r.shape[0]
            n_sg = int(round(self.read_selfgen * M)) if self.read_selfgen > 0 else 0
            sg = torch.zeros(M, dtype=torch.bool, device=z.device)
            if n_sg > 0:
                sg[torch.randperm(M, device=z.device)[:n_sg]] = True
            if (~sg).any():
                out.update(self._read_loss(g_r[~sg], h_r[~sg], dm_r[~sg]))
            if sg.any():
                out.update(self._read_loss(g_r[sg], h_r[sg], dm_r[sg], prefix='sg_', selfgen=True))
        if self.gen_k > 0:
            K = self.gen_k
            n = mask.sum(1)
            cut = (torch.rand(B, device=z.device) * (n - 1).clamp(min=1).float()).long() + 1     # in [1, n-1]
            hg, _ = self.generated_h(z, grid, discharge_first, cut, K, h0=h)
            L = hg.shape[1]
            j = torch.arange(L, device=z.device).unsqueeze(0)
            gm = m[:, :L] * ((j > cut.unsqueeze(1)) & (j <= (cut + K).unsqueeze(1))).float()
            den = gm.sum().clamp(min=1.0)
            mu_g, lv_g = self.soh_out(hg)
            out['gen_soh_nll'] = (gaussian_nll(mu_g, lv_g, soh[:, :L]) * gm).sum() / den
            out['gen_soh_mae'] = ((mu_g - soh[:, :L]).abs() * gm).sum() / den
            # reading loss on up to gen_read of the real cycles cut+1..cut+K per cell, from the generated-history state
            score = torch.rand(gm.shape, device=z.device).masked_fill(gm == 0, -1.0)
            top = score.topk(min(self.gen_read, L), dim=1)
            sel = top.values >= 0
            if sel.any():
                gb = torch.arange(B, device=z.device).unsqueeze(1).expand_as(top.indices)[sel]
                gc = top.indices[sel]
                out.update(self._read_loss(grid[gb, gc], hg[gb, gc], self.discharge_half_mask(discharge_first[gb]),
                                           prefix='gen_'))
        return out

    # -------------------------------------------------------------------------------------------- generation
    @torch.no_grad()
    def generate_cycle(self, h, I_tmpl, dmask, sample=False, q_given=None):
        """h [B, D] (state before the cycle), I_tmpl [B, P] C-rate, dmask [B, P] -> grid [B, 3, P] (V, I, Q).
        q_given [B, P]: use this Q (coulomb-counted) instead of feeding back the predicted dQ; only V is generated."""
        B = h.shape[0]
        dev = h.device
        st = self.read_stack.init_state(B, dev)
        cond = self.read_cond(h)
        vm, vs = self.chan_mean[0, 0, 0], self.chan_std[0, 0, 0]
        istd = self.chan_std[0, 1, 0]
        v_prev = torch.zeros(B, device=dev)
        q_prev = torch.zeros(B, device=dev)
        V = torch.zeros(B, self.P, device=dev)
        Q = torch.zeros(B, self.P, device=dev)
        for t in range(self.P):
            if t == self.H:
                q_prev = torch.zeros(B, device=dev)
            i_prev = I_tmpl[:, t - 1] if t > 0 else torch.zeros(B, device=dev)
            first = torch.full((B,), 1.0 if t == 0 else 0.0, device=dev)
            f = torch.stack([v_prev, q_prev, I_tmpl[:, t] / istd, i_prev / istd, dmask[:, t], first], -1)
            x = self.read_in(f) + self.pos.weight[t] + cond
            y, st = self.read_stack.step(x, st)
            o = self.read_head(y)
            v_n, dq = o[:, 0], o[:, 2]
            if sample:
                v_n = v_n + torch.randn_like(v_n) * torch.exp(0.5 * o[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX))
                dq = dq + torch.randn_like(dq) * torch.exp(0.5 * o[:, 3].clamp(LOGVAR_MIN, LOGVAR_MAX))
            q = q_prev + dq / self.H if q_given is None else q_given[:, t]
            V[:, t] = v_n * vs + vm
            Q[:, t] = q
            v_prev, q_prev = v_n, q
        return torch.stack([V, I_tmpl, Q], 1)

    @torch.no_grad()
    def rollout(self, curves, mask, discharge_first, soc_span, prefix=100, max_cycles=4000, mode='latent',
                threshold=0.8, sample=False, keep_grids=0):
        """Condition on each cell's first min(prefix, n) cycles, then generate cycles until SOH <= threshold.
        -> dict: life [B] (1-based cycle number of the first predicted SOH <= threshold; nan if never by max_cycles),
           soh [B, max_cycles] (observed cycles: nan; generated: predicted), soh_head [B, max_cycles] (full mode),
           grids: generated grids of the first `keep_grids` generated cycles (full mode)."""
        dev = curves.device
        B = mask.shape[0]
        n_obs = mask.sum(1).clamp(max=prefix)                                   # [B]
        st = self.cycle_stack.init_state(B, dev)
        h, st = self.cycle_stack.step(self.start.expand(B, self.D), st)
        grid = curves[:, :, :3]
        for c in range(int(n_obs.max())):
            act = c < n_obs
            z = self.encode(grid[:, c])
            h_new, st = self.cycle_stack.step(z, st, active=act)
            h = torch.where(act.unsqueeze(1), h_new, h)
        last = (n_obs - 1).clamp(min=0)
        I_tmpl = grid[torch.arange(B, device=dev), last, 1]                     # protocol template: last observed cycle
        dmask = self.discharge_half_mask(discharge_first)
        Q_tmpl = grid[torch.arange(B, device=dev), last, 2]                     # fullcc: Q template, scaled per cycle
        s_last = ((Q_tmpl * dmask).amax(1) / soc_span).clamp(min=1e-3)
        soh = torch.full((B, max_cycles), float('nan'), device=dev)
        soh_head = torch.full((B, max_cycles), float('nan'), device=dev)
        life = torch.full((B,), float('nan'), device=dev)
        cur = n_obs.clone()                                                     # 0-based index of the next cycle
        done = cur >= max_cycles
        grids = []
        while not bool(done.all()):
            mu, _ = self.soh_out(h)
            if mode == 'latent':
                z_next, s_next = self.latent_head(h), mu
            elif mode == 'full':
                g = self.generate_cycle(h, I_tmpl, dmask, sample=sample)
                if len(grids) < keep_grids:
                    grids.append(g.cpu())
                z_next = self.encode(g)
                s_next = (g[:, 2] * dmask).amax(1) / soc_span
            elif mode == 'fullcc':
                s_next = mu
                q_cc = Q_tmpl * (mu / s_last).clamp(min=0.0).unsqueeze(1)
                g = self.generate_cycle(h, I_tmpl, dmask, sample=sample, q_given=q_cc)
                if len(grids) < keep_grids:
                    grids.append(g.cpu())
                z_next = self.encode(g)
            else:
                raise ValueError(mode)
            idx = cur.clamp(max=max_cycles - 1)
            rows = torch.arange(B, device=dev)
            soh[rows[~done], idx[~done]] = s_next[~done]
            soh_head[rows[~done], idx[~done]] = mu[~done]
            crossed = (~done) & (s_next <= threshold)
            life = torch.where(crossed, (cur + 1).float(), life)
            done = done | crossed
            cur = cur + (~done).long()
            done = done | (cur >= max_cycles)
            h_new, st = self.cycle_stack.step(z_next, st, active=~done)
            h = torch.where((~done).unsqueeze(1), h_new, h)
        out = dict(life=life, soh=soh, n_obs=n_obs)
        if mode == 'full':
            out['soh_head'] = soh_head
            hl = torch.full((B,), float('nan'), device=dev)
            below = (soh_head <= threshold) & torch.isfinite(soh_head)
            any_b = below.any(1)
            hl[any_b] = below.float().argmax(1)[any_b].float() + 1
            out['life_head'] = hl
            out['grids'] = grids
        if mode == 'fullcc':
            out['grids'] = grids
        return out

    @torch.no_grad()
    def free_run_metrics(self, curves, mask, discharge_first, read_idx):
        """Generate the cycles in read_idx (-1 = none) from the TEACHER-FORCED cycle state, free-running the reading
        stack with the cycle's true I. gen: V and dQ fed back (full mode); cc: Q given = the cycle's true Q (fullcc with
        the true capacity). -> fr_v_mae_V, fr_qd_mae (capacity, Q / nominal) for gen; fr_cc_v_mae_V for cc."""
        B, N = mask.shape
        valid = read_idx >= 0
        if not valid.any():
            return {}
        grid = curves[:, :, :3]
        z = self.encode(grid)
        h = self.cycle_stack(torch.cat([self.start.expand(B, 1, self.D), z[:, :-1]], 1))
        bi, ri = valid.nonzero(as_tuple=True)
        ci = read_idx[bi, ri]
        g, hc = grid[bi, ci], h[bi, ci]
        dmask = self.discharge_half_mask(discharge_first[bi])
        gen = self.generate_cycle(hc, g[:, 1], dmask)
        cc = self.generate_cycle(hc, g[:, 1], dmask, q_given=g[:, 2])
        qd_true, qd_gen = (g[:, 2] * dmask).amax(1), (gen[:, 2] * dmask).amax(1)
        return dict(fr_v_mae_V=(gen[:, 0] - g[:, 0]).abs().mean(), fr_qd_mae=(qd_gen - qd_true).abs().mean(),
                    fr_cc_v_mae_V=(cc[:, 0] - g[:, 0]).abs().mean())
