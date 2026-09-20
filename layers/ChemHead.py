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
