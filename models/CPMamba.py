"""
CPMamba: per-cycle curve encoder -> stack of Mamba blocks over the cycle axis -> CLS readout -> output head.

    [B, L, len, 3] curves
      -> flatten each cycle, (optional early chemistry concat), intra_embed + intra MLP blocks   (same as CPMLP)
      -> append a learned CLS token right after the last real cycle
      -> n x [residual add -> LayerNorm -> Mamba mixer]   (pre-norm residual, like mamba_ssm's Block)
      -> final LayerNorm, take the hidden state at the CLS position
      -> ReLU -> head (Linear, or ChemHead for --chem_fusion late_mlp)                            (same as CPMLP)

Works with both loaders, no special casing:
  * full-timescale train loader: [1, L, ...], mask all ones, L up to ~5000
  * pooled val/test loaders:     [B, 100, ...], mask right-padded (first i positions are real)
The CLS token is written at position n_valid = mask.sum(1). Every mixer is causal (causal depthwise conv, causal
scan, and MambaInit's initial-state term is a cumsum), so the zero padding AFTER the CLS position cannot affect the
output. The sequence is trimmed to max(n_valid)+1 first so padding costs nothing. test_cpmamba.py checks that a
padded sample gives the same prediction as the same prefix unpadded.

Mamba mixers come from the mamba-init fork (/projectnb/textconv/dgordon/mamba, importable as mamba_ssm):
  --mamba_layer vanilla     mamba_ssm.modules.mamba_simple.Mamba
  --mamba_layer mamba_init  mamba_ssm.modules.mamba_init.MambaInit  (learned initial SSM state [d_inner, d_state])
  --mamba_scan cuda         selective_scan_cuda kernel (needs gcc/12.2.0 loaded, GPU with compute capability >= 7.0)
  --mamba_scan ref          pure-PyTorch selective_scan_ref (CPU / debugging; slow Python loop over L)
mamba_ssm is imported lazily, only when this model is built, so nothing else in the repo depends on it.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ChemHead import ChemHead, ChemEarlyEmbed


class MLPBlock(nn.Module):
    # identical to CPMLP.MLPBlock (kept local so this file doesn't depend on CPMLP's internals)
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super().__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        return self.ln(self.dropout(out) + x)


def get_mamba_mixer_cls(layer='vanilla', scan='cuda'):
    """Return the mixer class from the mamba-init fork. scan='ref' swaps in the pure-PyTorch selective scan
    (patches the module-level name the mixers call, i.e. process-wide)."""
    try:
        import mamba_ssm.modules.mamba_simple as ms
        import mamba_ssm.modules.mamba_init as mi
    except ImportError as e:
        raise ImportError(
            'CPMamba needs the mamba-init fork (mamba_ssm). Use the venv from scc/setup_venv_mamba.sh and '
            '`module load gcc/12.2.0` (selective_scan_cuda needs its libstdc++). Original error: ' + repr(e))
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    if scan not in ('cuda', 'ref'):
        raise ValueError(f'unknown mamba_scan {scan}')
    ms.selective_scan_fn = mi.selective_scan_fn = selective_scan_ref if scan == 'ref' else selective_scan_fn
    if layer == 'vanilla':
        return ms.Mamba
    if layer == 'mamba_init':
        return mi.MambaInit
    raise ValueError(f'unknown mamba_layer {layer}')


class ResidualMixerBlock(nn.Module):
    """Pre-norm residual block, same computation as mamba_ssm.modules.block.Block(fused_add_norm=False,
    mlp_cls=nn.Identity): residual = x + residual; out = mixer(LayerNorm(residual)). Returns (out, residual)."""
    def __init__(self, dim, mixer, residual_in_fp32=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mixer
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden_states, residual=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return self.mixer(hidden_states), residual


class Model(nn.Module):
    def __init__(self, configs, mixer_cls=None):
        """
        mixer_cls: optional override, a callable (d_model, layer_idx) -> nn.Module mapping [B, L, D] -> [B, L, D]
                   causally. Only for tests; normally built from --mamba_layer / --mamba_scan.
        """
        super().__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.charge_discharge_length = configs.charge_discharge_length
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.n_mamba = getattr(configs, 'mamba_n_layers', 4)
        self.intra_flatten = nn.Flatten(start_dim=2)

        self.chem_fusion = getattr(configs, 'chem_fusion', 'none')
        chem_embed_dim = getattr(configs, 'chem_embed_dim', 16)
        chem_num = getattr(configs, 'chem_num', 4)

        # ---- intra-cycle encoder (as CPMLP) ----
        if self.chem_fusion == 'early_concat':
            self.chem_early_embed = ChemEarlyEmbed(chem_embed_dim, chem_num)
            intra_in_dim = self.charge_discharge_length * 3 + self.chem_early_embed.embed_dim
        else:
            self.chem_early_embed = None
            intra_in_dim = self.charge_discharge_length * 3
        self.intra_embed = nn.Linear(intra_in_dim, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate)
                                        for _ in range(self.e_layers)])

        # ---- inter-cycle Mamba stack ----
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
            # mamba_ssm's own from-scratch init (zero Linear biases except dt_proj, out_proj rescaled by 1/sqrt(n_layer))
            from mamba_ssm.models.mixer_seq_simple import _init_weights
            self.mamba_layers.apply(partial(_init_weights, n_layer=self.n_mamba))

        # ---- head (as CPMLP) ----
        if self.chem_fusion == 'late_mlp':
            self.head_output = ChemHead(self.d_model, self.d_ff, configs.output_num, chem_embed_dim, chem_num, self.drop_rate)
        else:
            self.head_output = nn.Linear(self.d_model, configs.output_num)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, chemistry_ids=None):
        '''
        cycle_curve_data: [B, L, fixed_len, num_var]
        curve_attn_mask:  [B, L], right-padded (1 for the first n_valid cycles, 0 after)
        '''
        B = cycle_curve_data.shape[0]
        n_valid = curve_attn_mask.sum(dim=1).round().long()                     # [B]
        if bool((n_valid < 1).any()):
            raise ValueError('CPMamba: every sample needs at least one real cycle')
        L = int(n_valid.max().item())
        x = self.intra_flatten(cycle_curve_data[:, :L])                          # [B, L, fixed_len*num_var]
        if self.chem_early_embed is not None and self.chem_early_embed.embedding is not None:
            chem_embed = self.chem_early_embed(chemistry_ids).unsqueeze(1).expand(-1, L, -1)
            x = torch.cat([x, chem_embed.to(x.dtype)], dim=-1)
        x = self.intra_embed(x)
        for blk in self.intra_MLP:
            x = blk(x)                                                           # [B, L, D]

        # CLS token at position n_valid (right after the last real cycle); positions > n_valid are padding
        x = torch.cat([x, x.new_zeros(B, 1, self.d_model)], dim=1)              # [B, L+1, D]
        pos = torch.arange(L + 1, device=x.device)
        is_cls = (pos.unsqueeze(0) == n_valid.unsqueeze(1)).unsqueeze(-1)        # [B, L+1, 1]
        x = torch.where(is_cls, self.cls_token.to(x.dtype).view(1, 1, -1), x)

        hidden, residual = x, None
        for layer in self.mamba_layers:
            hidden, residual = layer(hidden, residual)
        residual = hidden + residual if residual is not None else hidden
        hidden = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        emb = hidden[torch.arange(B, device=x.device), n_valid]                 # [B, D]

        feats = F.relu(emb)
        preds = self.head_output(feats, chemistry_ids) if self.chem_fusion == 'late_mlp' else self.head_output(feats)
        if return_embedding:
            return preds, emb
        return preds
