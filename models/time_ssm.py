"""
Time-aware SSM mixers for LongMamba --long_time_dt const | real (10/13).

Both mixers use the recorded clock for the SSM step size instead of a fixed or input-dependent one:

    delta[b, e, t] = exp(log_dt[e]) * tau[b, t]

  tau [B, T]   normalised elapsed time of step t (real Δt / c, one global c in seconds; see data_loader_dt.py). It is
               the same for every channel and every layer. --long_time_dt const feeds tau = 1 everywhere (control arm:
               same code, constant step).
  log_dt [E]   learned per channel and per layer, no weight decay (tagged _ssm_no_wd). Init log-uniform in
               [dt_min, dt_max] = rate per typical step (tau ~ 1), the same range S4D uses.

The discretisation is the selective-scan kernel's: state_t = exp(delta_t * A) state_{t-1} + delta_t * B_t * u_t
(exact decay, first-order input term), run by mamba_ssm's selective_scan_fn (selective_scan_ref on CPU / tests).
With a different delta at every step the exact ZOH input term of models/S4D.py cannot be passed to the kernel,
so --long_time_dt const is NOT bit-identical to the old S4D / Mamba runs; it is the control for --long_time_dt real.

TimeS4DMixer    the S4DMixer block (models/S4D.py: same params, S4D-Lin init, complex A, B = 1, learned C, D skip,
                SiLU(z) gate), time-invariant A/B/C, step from the clock. With fixed A, B, C this is a continuous-time
                LTI system sampled at the recorded times.
TimeMambaMixer  the mamba_ssm Mamba block (in_proj -> causal depthwise conv -> SiLU -> SSM + D*u -> * SiLU(z) ->
                out_proj, A = -exp(A_log) real with S4D-Real init 1..N) with ONLY the dt path replaced: x_proj outputs
                B and C (still selective, input dependent) but no dt; there is no dt_proj; delta comes from the clock.
                So vs TimeS4DMixer the difference is selective B/C (and real vs complex A, Mamba's own param choices).

TimeResidualMixerBlock  ResidualMixerBlock (models/CPMamba.py) that forwards tau to its mixer.
init_time_stack         same init as the other stacks (zero Linear biases, out_proj / sqrt(n_layer)).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.S4D import S4DKernel, init_s4d_stack


def _scan_fn(x, impl):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    return selective_scan_fn if (x.is_cuda and impl == 'cuda') else selective_scan_ref


def _check_tau(tau, B, T, device):
    if tau is None:
        return torch.ones(B, T, device=device)
    if tuple(tau.shape) != (B, T):
        raise ValueError(f'tau must be [B, T] = {(B, T)}, got {tuple(tau.shape)}')
    return tau


def _log_uniform(n, dt_min, dt_max):
    if not 0 < dt_min <= dt_max:
        raise ValueError(f'needs 0 < dt_min <= dt_max, got {dt_min}, {dt_max}')
    return torch.rand(n) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)


class TimeS4DMixer(nn.Module):
    """[B, T, D], tau [B, T] -> [B, T, D], causal. Parameters as S4DMixer (kernel.log_dt is the per-channel rate)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-5, dt_max=1e-1, layer_idx=None, scan_impl='cuda'):
        super().__init__()
        if scan_impl not in ('cuda', 'ref'):
            raise ValueError(f'scan_impl must be cuda or ref, got {scan_impl}')
        self.scan_impl = scan_impl
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.kernel = S4DKernel(self.d_inner, d_state=d_state, dt_min=dt_min, dt_max=dt_max)   # tags _ssm_no_wd
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_params(self):
        k = self.kernel
        rate = torch.exp(k.log_dt.float())                                      # [E]
        A = -torch.exp(k.log_A_real.float()) + 1j * k.A_imag.float()            # [E, N2] complex
        C = torch.view_as_complex(k.C.float().contiguous())                     # [E, N2]
        return rate, A, C

    def forward(self, hidden_states, tau=None):
        B, T, _ = hidden_states.shape
        tau = _check_tau(tau, B, T, hidden_states.device)
        u, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)     # [B, E, T]
        u = F.silu(self.conv1d(u)[..., :T])
        rate, A, C = self._ssm_params()
        delta = (rate.view(1, -1, 1) * tau.float().unsqueeze(1)).to(u.dtype).contiguous()   # [B, E, T]
        Bm = torch.ones_like(A)                                                 # S4D: B = 1, input term delta * u
        y = _scan_fn(u, self.scan_impl)(u.contiguous(), delta, A.contiguous(), Bm, C.contiguous(), D=self.D.float(),
                                        z=z.contiguous(), delta_bias=None, delta_softplus=False)
        return self.out_proj(y.transpose(1, 2))

    def step_reference(self, hidden_states, tau=None):
        """Explicit complex recurrence with a per-step delta (slow; tests only)."""
        B, T, _ = hidden_states.shape
        tau = _check_tau(tau, B, T, hidden_states.device)
        u, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        u = F.silu(self.conv1d(u)[..., :T])
        k = self.kernel
        rate = torch.exp(k.log_dt)
        A = -torch.exp(k.log_A_real) + 1j * k.A_imag
        C = torch.view_as_complex(k.C.contiguous())
        state = torch.zeros(B, *A.shape, dtype=torch.complex128 if u.dtype == torch.float64 else torch.cfloat)
        ys = []
        for t in range(T):
            d = rate.view(1, -1, 1) * tau[:, t].view(-1, 1, 1)                 # [B, E, 1]
            state = torch.exp(d * A) * state + d * u[:, :, t].unsqueeze(-1)
            ys.append(2 * (C * state).sum(-1).real)
        y = (torch.stack(ys, -1) + self.D.unsqueeze(-1) * u) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))


class TimeMambaMixer(nn.Module):
    """Mamba block with delta = exp(log_dt) * tau (no input-dependent dt); B, C selective. [B, T, D], tau -> [B, T, D]."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-5, dt_max=1e-1, layer_idx=None, scan_impl='cuda'):
        super().__init__()
        if scan_impl not in ('cuda', 'ref'):
            raise ValueError(f'scan_impl must be cuda or ref, got {scan_impl}')
        self.scan_impl = scan_impl
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state, bias=False)          # -> B, C (no dt part)
        self.log_dt = nn.Parameter(_log_uniform(self.d_inner, dt_min, dt_max))
        self.log_dt._ssm_no_wd = True
        # Mamba's S4D-Real init. A_log / D keep the old Mamba runs' weight-decay treatment (not tagged).
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _proj(self, hidden_states, tau):
        B, T, _ = hidden_states.shape
        u, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)     # [B, E, T]
        u = F.silu(self.conv1d(u)[..., :T])
        Bm, Cm = self.x_proj(u.transpose(1, 2)).split(self.d_state, dim=-1)    # [B, T, N] each
        Bm, Cm = Bm.transpose(1, 2).contiguous(), Cm.transpose(1, 2).contiguous()   # [B, N, T]
        A = -torch.exp(self.A_log.float())                                      # [E, N]
        delta = (torch.exp(self.log_dt.float()).view(1, -1, 1) * tau.float().unsqueeze(1)).to(u.dtype).contiguous()
        return u, z, Bm, Cm, A, delta

    def forward(self, hidden_states, tau=None):
        B, T, _ = hidden_states.shape
        tau = _check_tau(tau, B, T, hidden_states.device)
        u, z, Bm, Cm, A, delta = self._proj(hidden_states, tau)
        y = _scan_fn(u, self.scan_impl)(u.contiguous(), delta, A, Bm, Cm, D=self.D.float(), z=z.contiguous(),
                                        delta_bias=None, delta_softplus=False)
        return self.out_proj(y.transpose(1, 2))

    def step_reference(self, hidden_states, tau=None):
        """Explicit real recurrence with per-step delta and selective B/C (slow; tests only)."""
        B, T, _ = hidden_states.shape
        tau = _check_tau(tau, B, T, hidden_states.device)
        u, z, Bm, Cm, A, delta = self._proj(hidden_states, tau)
        A = A.to(u.dtype)
        state = u.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            d = delta[:, :, t].unsqueeze(-1)                                    # [B, E, 1]
            state = torch.exp(d * A) * state + d * Bm[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
            ys.append((state * Cm[:, :, t].unsqueeze(1)).sum(-1))
        y = (torch.stack(ys, -1) + self.D.to(u.dtype).unsqueeze(-1) * u) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))


class TimeResidualMixerBlock(nn.Module):
    """Same computation as models.CPMamba.ResidualMixerBlock, with tau passed to the mixer."""

    def __init__(self, dim, mixer, residual_in_fp32=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mixer
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden_states, residual=None, tau=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return self.mixer(hidden_states, tau), residual


def time_mixer_cls(layer_kind):
    if layer_kind == 's4d':
        return TimeS4DMixer
    if layer_kind == 'vanilla':
        return TimeMambaMixer
    raise ValueError(f'--long_time_dt needs --mamba_layer s4d or vanilla, got {layer_kind}')


init_time_stack = init_s4d_stack   # zero Linear biases, kaiming out_proj / sqrt(n_layer): same as the other stacks
