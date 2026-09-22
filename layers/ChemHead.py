import torch
import torch.nn as nn


class ChemHead(nn.Module):
    """
    Late-fusion output head for pooled multi-chemistry training:
        [pre-head features ; chemistry embedding] -> Linear -> ReLU -> Dropout -> Linear -> output
    A plain concat feeding a single linear layer could only add a per-chemistry offset to the outputs; the hidden
    layer lets the chemistry interact with the features.
    embed_dim == 0 gives the same head with NO chemistry input (capacity control for ablations).
    """
    def __init__(self, in_dim, hidden_dim, out_dim, embed_dim, num_chem, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_chem, embed_dim) if embed_dim > 0 else None
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feats, chemistry_ids=None):
        if self.embedding is not None:
            if chemistry_ids is None:
                raise ValueError('ChemHead with embed_dim > 0 needs chemistry_ids (use the pooled loader / --pooled).')
            feats = torch.cat([feats, self.embedding(chemistry_ids.long())], dim=-1)
        return self.mlp(feats)


class ChemEarlyEmbed(nn.Module):
    """
    Early-fusion chemistry embedding for pooled multi-chemistry training (--chem_fusion early_concat):
    looks up a per-sample chemistry embedding and hands it back for the CALLER to concatenate onto the raw
    per-cycle-step input, BEFORE the model's first embedding layer (intra_embed) -- so every downstream layer
    (intra MLP, inter MLP / Transformer encoder, output head) sees a chemistry-aware representation, unlike
    ChemHead's late fusion which only touches the final output head.
    embed_dim == 0 gives no embedding at all (embedding is None); the caller should skip concatenation entirely
    in that case (this is the capacity control: model architecture identical to --chem_fusion none, since there's
    no extra input width to control for).
    """
    def __init__(self, embed_dim, num_chem):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_chem, embed_dim) if embed_dim > 0 else None

    def forward(self, chemistry_ids):
        if self.embedding is None:
            raise ValueError('ChemEarlyEmbed has embed_dim=0 (no embedding); caller should not invoke forward() in that case.')
        if chemistry_ids is None:
            raise ValueError('ChemEarlyEmbed needs chemistry_ids (use the pooled loader / --pooled).')
        return self.embedding(chemistry_ids.long())
