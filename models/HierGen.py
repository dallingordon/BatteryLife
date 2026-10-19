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
        valid = read_idx >= 0
        if valid.any():
            bi, ri = valid.nonzero(as_tuple=True)
            ci = read_idx[bi, ri]
            g = grid[bi, ci]                                                    # [M, 3, P]
            hc = h[bi, ci]                                                      # [M, D]
            dmask = self.discharge_half_mask(discharge_first[bi])
            feats, q_prev = self.read_features(g[:, 0], g[:, 2], g[:, 1], dmask)
            x = self.read_in(feats) + self.pos.weight.unsqueeze(0) + self.read_cond(hc).unsqueeze(1)
            o = self.read_head(self.read_stack(x))                              # [M, P, 4]
            vt, qt = self.read_targets(g[:, 0], g[:, 2], q_prev)
            out['read_v_nll'] = gaussian_nll(o[..., 0], o[..., 1], vt).mean()
            out['read_q_nll'] = gaussian_nll(o[..., 2], o[..., 3], qt).mean()
            out['read_v_mae_V'] = ((o[..., 0] - vt).abs().mean() * self.chan_std[0, 0, 0])
            # capacity of the cycle from the teacher-forced dQ predictions vs the true grid Q at the end of discharge
            qd_pred = ((o[..., 2] / self.H + q_prev) * dmask).amax(1)
            qd_true = (g[:, 2] * dmask).amax(1)
            out['read_qd_mae'] = (qd_pred - qd_true).abs().mean()
        return out

    # -------------------------------------------------------------------------------------------- generation
    @torch.no_grad()
    def generate_cycle(self, h, I_tmpl, dmask, sample=False):
        """h [B, D] (state before the cycle), I_tmpl [B, P] C-rate, dmask [B, P] -> grid [B, 3, P] (V, I, Q)."""
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
            q = q_prev + dq / self.H
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
        return out
