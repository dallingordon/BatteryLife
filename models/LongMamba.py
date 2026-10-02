"""
LongMamba: one Mamba stack over the raw per-point time series of every early cycle, laid end to end.

    pooled batch [B, L=100, 3, 300]  (per cycle: 3 channels -- voltage / max, current in C, capacity / nominal --
                                      on a shared 300-point axis: 150 resampled charge points then 150 discharge)
      -> keep the first n_valid cycles per sample (n_valid = curve_attn_mask.sum(1)); trim the batch to max(n_valid)
      -> per point: Linear(3 -> d_model)                                           [B, L, 300, D]
      -> optional cycle-boundary token in front of EVERY cycle (incl. the first):   [B, L, 301, D]
            --long_boundary none    no boundary token; cycles are simply concatenated        (300 steps / cycle)
            --long_boundary shared  one learned token, identical at every cycle boundary     (301 steps / cycle)
            --long_boundary index   learned per-cycle-index token, nn.Embedding(100, d_model):
                                    token k sits in front of cycle k                         (301 steps / cycle)
      -> flatten cycles into one sequence of n_valid * S steps (S = 300 or 301), ~30,000 steps for 100 cycles
      -> learned output CLS token right after the last real step (position n_valid * S)
      -> n x [residual add -> LayerNorm -> Mamba mixer]   (pre-norm residual, same blocks as CPMamba)
      -> final LayerNorm, hidden state at the CLS position -> ReLU -> Linear head (regression / geo_bins / dual)

Chemistry conditioning (pooled training; two independent switches, both need chemistry_ids):
  --long_chem_cls 1       the output CLS token is per chemistry: nn.Embedding(4, d_model) instead of one vector
  --long_chem_boundary 1  per-chemistry cycle-start embedding nn.Embedding(4, d_model):
                            with --long_boundary shared: REPLACES the single shared token (one token per chemistry)
                            with --long_boundary index:  ADDED to the cycle-index token: token_k = index[k] + chem[c]
                            (not allowed with --long_boundary none: there is no boundary token to condition)

Prefix length: --long_boundary none / shared take any number of cycles (full-timescale training, --full_timescale);
--long_boundary index has one token per cycle index 0..early_cycle_threshold-1 and refuses longer prefixes.
--long_grad_ckpt 1 recomputes each Mamba block in the backward pass instead of storing its activations (needed for
the longest full-history prefixes, ~5000 cycles = 1.5M steps); outputs and gradients are unchanged.

Only the final position carries the output (CLS) token. Every mixer is causal, so anything after the CLS position
(the zeroed, unseen cycles of a padded sample) cannot affect the prediction: a prefix padded inside a batch gives the
same output as the same prefix alone (test_longmamba.py checks this for all three boundary modes).

Compared with CPMamba, the sequence runs over POINTS (300 per cycle, ~30k for 100 cycles) instead of over per-cycle
MLP summaries (<= 100 steps); there is no per-cycle MLP encoder. --chem_fusion is not used here (see above).
Mixers: same mamba-init fork / --mamba_layer / --mamba_scan options as CPMamba.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.CPMamba import ResidualMixerBlock, get_mamba_mixer_cls

BOUNDARY_MODES = ('none', 'shared', 'index')


class Model(nn.Module):
    def __init__(self, configs, mixer_cls=None):
        """
        mixer_cls: optional override, a callable (d_model, layer_idx) -> nn.Module mapping [B, L, D] -> [B, L, D]
                   causally. Only for tests; normally built from --mamba_layer / --mamba_scan.
        """
        super().__init__()
        self.d_model = configs.d_model
        self.n_points = configs.charge_discharge_length          # 300 points per cycle
        self.n_channels = 3
        self.max_cycles = configs.early_cycle_threshold           # 100
        self.n_mamba = getattr(configs, 'mamba_n_layers', 4)
        self.boundary = getattr(configs, 'long_boundary', 'none')
        if self.boundary not in BOUNDARY_MODES:
            raise ValueError(f'--long_boundary must be one of {BOUNDARY_MODES}, got {self.boundary}')
        if getattr(configs, 'chem_fusion', 'none') != 'none':
            raise NotImplementedError('LongMamba uses --long_chem_cls / --long_chem_boundary, not --chem_fusion')
        self.grad_ckpt = bool(getattr(configs, 'long_grad_ckpt', 0))
        self.chem_cls = bool(getattr(configs, 'long_chem_cls', 0))
        self.chem_boundary = bool(getattr(configs, 'long_chem_boundary', 0))
        chem_num = getattr(configs, 'chem_num', 4)
        if self.chem_boundary and self.boundary == 'none':
            raise ValueError('--long_chem_boundary needs --long_boundary shared or index (there is no boundary token)')

        self.in_proj = nn.Linear(self.n_channels, self.d_model)
        if self.boundary == 'shared' and not self.chem_boundary:
            self.boundary_token = nn.Parameter(torch.randn(self.d_model) * 0.02)
        if self.boundary == 'index':
            self.cycle_embed = nn.Embedding(self.max_cycles, self.d_model)
            nn.init.normal_(self.cycle_embed.weight, std=0.02)
        if self.chem_boundary:
            self.chem_boundary_embed = nn.Embedding(chem_num, self.d_model)   # [chem] -> cycle-start token / offset
            nn.init.normal_(self.chem_boundary_embed.weight, std=0.02)
        if self.chem_cls:
            self.chem_cls_embed = nn.Embedding(chem_num, self.d_model)        # [chem] -> output token
            nn.init.normal_(self.chem_cls_embed.weight, std=0.02)
        else:
            self.cls_token = nn.Parameter(torch.randn(self.d_model) * 0.02)

        if mixer_cls is None:
            cls = get_mamba_mixer_cls(getattr(configs, 'mamba_layer', 'vanilla'), getattr(configs, 'mamba_scan', 'cuda'))
            mixer_cls = lambda d, i: cls(d, d_state=getattr(configs, 'mamba_d_state', 16),
                                         d_conv=getattr(configs, 'mamba_d_conv', 4),
                                         expand=getattr(configs, 'mamba_expand', 2), layer_idx=i)
            use_mamba_init_weights = True
        else:
            use_mamba_init_weights = False
        self.mamba_layers = nn.ModuleList([ResidualMixerBlock(self.d_model, mixer_cls(self.d_model, i))
                                           for i in range(self.n_mamba)])
        self.norm_f = nn.LayerNorm(self.d_model)
        if use_mamba_init_weights:
            from functools import partial
            from mamba_ssm.models.mixer_seq_simple import _init_weights
            self.mamba_layers.apply(partial(_init_weights, n_layer=self.n_mamba))

        self.head_output = nn.Linear(self.d_model, configs.output_num)

    @property
    def steps_per_cycle(self):
        return self.n_points + (0 if self.boundary == 'none' else 1)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, chemistry_ids=None):
        '''
        cycle_curve_data: [B, L, 3, n_points]  (the loaders' layout: channels, then points)
        curve_attn_mask:  [B, L], right-padded (1 for the first n_valid cycles, 0 after)
        '''
        B = cycle_curve_data.shape[0]
        if (self.chem_cls or self.chem_boundary) and chemistry_ids is None:
            raise ValueError('LongMamba chemistry conditioning needs chemistry_ids (pooled loader / --pooled)')
        assert cycle_curve_data.shape[2] == self.n_channels and cycle_curve_data.shape[3] == self.n_points, \
            f'LongMamba expects [B, L, {self.n_channels}, {self.n_points}], got {tuple(cycle_curve_data.shape)}'
        n_valid = curve_attn_mask.sum(dim=1).round().long()                     # [B]
        if bool((n_valid < 1).any()):
            raise ValueError('LongMamba: every sample needs at least one real cycle')
        L = int(n_valid.max().item())
        if self.boundary == 'index' and L > self.max_cycles:
            raise ValueError(f'LongMamba --long_boundary index has {self.max_cycles} cycle tokens, got a {L}-cycle prefix '
                             f'(for full-history training use --long_boundary none or shared)')

        x = cycle_curve_data[:, :L].transpose(2, 3)                              # [B, L, P, 3]: time = points
        x = self.in_proj(x)                                                      # [B, L, P, D]
        if self.boundary != 'none':
            if self.boundary == 'shared':
                if self.chem_boundary:                                           # one start token per chemistry
                    tok = self.chem_boundary_embed(chemistry_ids.long()).to(x.dtype).view(B, 1, 1, -1).expand(B, L, 1, -1)
                else:
                    tok = self.boundary_token.to(x.dtype).view(1, 1, 1, -1).expand(B, L, 1, -1)
            else:
                idx = torch.arange(L, device=x.device)
                tok = self.cycle_embed(idx).to(x.dtype).view(1, L, 1, -1).expand(B, L, 1, -1)
                if self.chem_boundary:                                           # index[k] + chem[c]
                    tok = tok + self.chem_boundary_embed(chemistry_ids.long()).to(x.dtype).view(B, 1, 1, -1)
            x = torch.cat([tok, x], dim=2)                                       # [B, L, P+1, D], token BEFORE each cycle
        S = x.shape[2]
        x = x.reshape(B, L * S, self.d_model)                                    # one long sequence

        # output CLS token right after the last real step; positions after it belong to unseen (zeroed) cycles
        x = torch.cat([x, x.new_zeros(B, 1, self.d_model)], dim=1)              # [B, L*S + 1, D]
        cls_pos = n_valid * S                                                    # [B]
        pos = torch.arange(L * S + 1, device=x.device)
        is_cls = (pos.unsqueeze(0) == cls_pos.unsqueeze(1)).unsqueeze(-1)       # [B, L*S+1, 1]
        if self.chem_cls:
            cls = self.chem_cls_embed(chemistry_ids.long()).to(x.dtype).view(B, 1, -1)   # per-chemistry output token
        else:
            cls = self.cls_token.to(x.dtype).view(1, 1, -1)
        x = torch.where(is_cls, cls, x)

        hidden, residual = x, None
        for layer in self.mamba_layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                hidden, residual = checkpoint(layer, hidden, residual, use_reentrant=False)
            else:
                hidden, residual = layer(hidden, residual)
        residual = hidden + residual if residual is not None else hidden
        hidden = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        emb = hidden[torch.arange(B, device=x.device), cls_pos]                 # [B, D]

        preds = self.head_output(F.relu(emb))
        if return_embedding:
            return preds, emb
        return preds
