"""
S4D mixer: the LTI (time-invariant) counterpart of the Mamba block, for LongMamba --mamba_layer s4d.

Same block as mamba_ssm's Mamba with ONE change: the selective (input-dependent dt, B, C) scan is replaced by a
diagonal S4D SSM whose dt, A, B, C are fixed parameters, so the layer applies the same causal convolution kernel at
every time step (linear time invariance). Everything else is kept so Mamba vs S4D isolates selectivity:

    x -> in_proj (D -> 2*E, no bias) -> split (u, z)
      u -> causal depthwise conv1d (width d_conv) -> SiLU -> S4D (per channel, E channels) + D*u
      y = S4D(u) * SiLU(z) -> out_proj (E -> D, no bias)            E = expand * d_model

S4D (Gu et al. 2022, "On the Parameterization and Initialization of Diagonal State Space Models"), S4D-Lin init,
zero-order-hold discretisation, computed as one FFT convolution (training and eval; no recurrence kernel needed):
    per channel h:  A_n = -exp(log_A_real) + i * A_imag   (init -0.5 + i*pi*n, n = 0..N/2-1)
                    dt  = exp(log_dt)                      (init log-uniform in [dt_min, dt_max], one per channel)
                    K[l] = 2 * Re( sum_n C_n (exp(dt*A_n) - 1)/A_n * exp(dt*A_n * l) ),   l = 0..T-1
                    y = causal_conv(u, K) + D*u
    d_state N counts real state dims: N/2 complex modes (conjugate pairs), same state size as Mamba's real N.

Timescales: mode n of channel h decays as exp(-0.5 * dt_h * l), i.e. memory ~ 2/dt_h steps, and oscillates with
period 2/(n * dt_h) steps. With 300 points per cycle, carrying information across 100 cycles (~30k steps) needs
dt ~ 7e-5; the default range [1e-5, 1e-1] spreads the channels log-uniformly from ~20 steps (within a cycle) to
~200k steps (~670 cycles). The usual S4D default [1e-3, 1e-1] only reaches ~2000 steps (~7 cycles).

Weight decay: log_dt, log_A_real, A_imag, C are tagged `_ssm_no_wd`; run_main puts them in a weight_decay=0 group.
Coupled L2 on log_dt would pull it toward 0 (dt -> 1, memory of ~2 steps), i.e. it would erase exactly the long
timescales the init sets up.

Memory: the kernel is materialised as [E, N/2, T] complex per layer (E=128, N/2=8, T=30k: ~250 MB), independent
of batch size. Fine for 100-cycle prefixes; full-history prefixes (up to 1.5M steps) would need a chunked kernel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    def __init__(self, n_channels, d_state=16, dt_min=1e-5, dt_max=1e-1):
        super().__init__()
        if d_state % 2:
            raise ValueError(f'S4D d_state must be even (N/2 complex conjugate modes), got {d_state}')
        if not 0 < dt_min <= dt_max:
            raise ValueError(f'S4D needs 0 < dt_min <= dt_max, got {dt_min}, {dt_max}')
        H, N2 = n_channels, d_state // 2
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N2, dtype=torch.cfloat)
        self.log_dt = nn.Parameter(log_dt)
        self.C = nn.Parameter(torch.view_as_real(C))                              # [H, N2, 2]
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N2).float().repeat(H, 1))
        for p in (self.log_dt, self.C, self.log_A_real, self.A_imag):
            p._ssm_no_wd = True

    def forward(self, T):
        """Kernel K [H, T] (float32)."""
        dt = torch.exp(self.log_dt.float())                                     # [H]
        C = torch.view_as_complex(self.C.float().contiguous())                  # [H, N2]
        A = -torch.exp(self.log_A_real.float()) + 1j * self.A_imag.float()      # [H, N2]
        dtA = A * dt.unsqueeze(-1)                                              # [H, N2]
        C = C * (torch.exp(dtA) - 1.) / A                                       # ZOH: fold B=1 and discretisation into C
        steps = torch.arange(T, device=dt.device, dtype=torch.float32)
        K = torch.exp(dtA.unsqueeze(-1) * steps)                                # [H, N2, T]
        return 2 * torch.einsum('hn,hnl->hl', C, K).real


def fft_causal_conv(u, K):
    """u [B, H, T], K [H, T] -> y [B, H, T], y[t] = sum_{s<=t} K[t-s] u[s]."""
    T = u.shape[-1]
    n = 2 * T
    y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(K, n=n), n=n)
    return y[..., :T]


class S4DMixer(nn.Module):
    """Drop-in for mamba_ssm Mamba(d_model, d_state, d_conv, expand, layer_idx): [B, T, D] -> [B, T, D], causal."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-5, dt_max=1e-1, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.kernel = S4DKernel(self.d_inner, d_state=d_state, dt_min=dt_min, dt_max=dt_max)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        u, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)     # [B, E, T] each
        u = F.silu(self.conv1d(u)[..., :T])                                     # causal depthwise conv
        y = fft_causal_conv(u.float(), self.kernel(T)) + self.D.float().unsqueeze(-1) * u.float()
        y = y.to(u.dtype) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))

    def step_reference(self, hidden_states):
        """Same output computed by the explicit complex recurrence (slow; tests only)."""
        B, T, _ = hidden_states.shape
        u, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        u = F.silu(self.conv1d(u)[..., :T]).float()
        k = self.kernel
        dt = torch.exp(k.log_dt)
        A = -torch.exp(k.log_A_real) + 1j * k.A_imag
        dA = torch.exp(A * dt.unsqueeze(-1))                                    # [E, N2]
        Bd = (dA - 1.) / A                                                      # ZOH input matrix (B = 1)
        C = torch.view_as_complex(k.C.contiguous())
        state = torch.zeros(B, *dA.shape, dtype=torch.cfloat)
        ys = []
        for t in range(T):
            state = dA * state + Bd * u[:, :, t].unsqueeze(-1)
            ys.append(2 * (C * state).sum(-1).real)
        y = torch.stack(ys, -1) + self.D.unsqueeze(-1) * u
        y = y * F.silu(z)
        return self.out_proj(y.transpose(1, 2))


def init_s4d_stack(layers, n_layer):
    """Same init the Mamba stacks get from mamba_ssm.models.mixer_seq_simple._init_weights (so the comparison keeps
    it), without importing mamba_ssm: zero Linear biases, and rescale every out_proj by 1/sqrt(n_layer) after
    kaiming-uniform (GPT-2 style prenorm-residual scaling)."""
    for name, p in layers.named_parameters():
        if name.endswith('out_proj.weight'):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_layer)
    for m in layers.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
