"""Stage 2: causal self-attention, NO positional embeddings.

Tests how much attention alone contributes. In theory, attention without
position info is permutation-invariant. In practice item identity acts as
a weak positional signal, so we'd expect this to beat Stage 1's pure
bag-of-items but trail Stage 3.

Causal mask is REQUIRED here (it's a correctness invariant for
autoregressive training, not an ablation knob — without it, position i
attends to position i+1 and the prediction target leaks into the input).
The only thing ablated between Stage 2 and Stage 3 is positional embeddings.
"""

import torch
import torch.nn as nn

from src.models.transformer_block import TransformerBlock

MAX_SEQ_LEN = 50


class AttentionNoPositionModel(nn.Module):
    def __init__(self, n_items, hidden_dim=64, n_blocks=2, n_heads=1, dropout=0.5):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = MAX_SEQ_LEN
        # Causal mask is a function of seq length only; precompute once and
        # move with the model. registered as buffer so .to(device) handles it.
        causal = torch.triu(
            torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, input_seq):
        # input_seq: (B, L)
        x = self.item_embedding(input_seq)              # (B, L, D)
        x = self.dropout(x)
        padding_mask = (input_seq == 0)                 # (B, L)
        for block in self.blocks:
            x = block(x, self.causal_mask, padding_mask)
        return self.layer_norm(x)                       # (B, L, D)
