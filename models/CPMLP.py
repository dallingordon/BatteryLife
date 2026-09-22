import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.ChemHead import ChemHead, ChemEarlyEmbed

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.intra_flatten = nn.Flatten(start_dim=2)

        self.chem_fusion = getattr(configs, 'chem_fusion', 'none')
        chem_embed_dim = getattr(configs, 'chem_embed_dim', 16)
        chem_num = getattr(configs, 'chem_num', 4)

        if self.chem_fusion == 'early_concat':
            # early fusion: [flattened per-cycle-step curve ; chemistry embedding] -> intra_embed -> rest of the
            # model unchanged. Every downstream layer (intra MLP, inter MLP, output head) sees chemistry info,
            # unlike late_mlp which only touches the final output head. embed_dim=0 -> no embedding, no extra
            # input width (capacity control: identical architecture to chem_fusion='none').
            self.chem_early_embed = ChemEarlyEmbed(chem_embed_dim, chem_num)
            intra_in_dim = self.charge_discharge_length * 3 + self.chem_early_embed.embed_dim
        else:
            self.chem_early_embed = None
            intra_in_dim = self.charge_discharge_length * 3
        self.intra_embed = nn.Linear(intra_in_dim, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])

        self.inter_flatten = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model))
        self.inter_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.d_layers)])
        if self.chem_fusion == 'late_mlp':
            # pooled multi-chemistry: [features ; chemistry embedding] -> MLP -> output (chem_embed_dim=0: same head, no chemistry input)
            self.head_output = ChemHead(self.d_model, self.d_ff, configs.output_num, chem_embed_dim, chem_num, self.drop_rate)
        else:
            self.head_output = nn.Linear(self.d_model, configs.output_num)
        # self.flatten_head = nn.Sequential(nn.Linear(self.early_cycle_threshold*self.d_model, self.d_ff), nn.ReLU(),
        #                                  nn.Dropout(self.drop_rate), nn.Linear(self.d_ff, 1))



    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, chemistry_ids=None):
        '''
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, early_cycle, fixed_len * num_var]
        if self.chem_early_embed is not None and self.chem_early_embed.embedding is not None:
            chem_embed = self.chem_early_embed(chemistry_ids)  # [B, embed_dim]
            chem_embed = chem_embed.unsqueeze(1).expand(-1, cycle_curve_data.shape[1], -1)  # [B, early_cycle, embed_dim]
            cycle_curve_data = torch.cat([cycle_curve_data, chem_embed], dim=-1)  # [B, early_cycle, fixed_len*num_var + embed_dim]
        cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, early_cycle, d_model]

        cycle_curve_data = self.inter_flatten(cycle_curve_data) # [B, d_model]
        for i in range(self.d_layers):
            cycle_curve_data = self.inter_MLP[i](cycle_curve_data) # [B, d_model]

        feats = F.relu(cycle_curve_data)
        preds = self.head_output(feats, chemistry_ids) if self.chem_fusion == 'late_mlp' else self.head_output(feats)
        if return_embedding:
            return preds, cycle_curve_data
        else:
            return preds
