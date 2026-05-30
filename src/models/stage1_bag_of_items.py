"""Stage 1 baseline: average-pool item embeddings, no sequence order.

Tests how well next-item prediction works using only the unordered set of
items the user has interacted with. The same pooled user representation is
broadcast to every position so the interface (B, L, D) matches Stages 2/3
and the unified train/eval loops don't need stage-specific branching.
"""

import torch
import torch.nn as nn


class BagOfItemsModel(nn.Module):
    def __init__(self, n_items, hidden_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.hidden_dim = hidden_dim

    def forward(self, input_seq):
        # input_seq: (B, L)
        item_emb = self.item_embedding(input_seq)               # (B, L, D)
        mask = (input_seq != 0).float().unsqueeze(-1)           # (B, L, 1)
        pooled = (item_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.unsqueeze(1).expand(-1, input_seq.size(1), -1)
