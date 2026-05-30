"""Stage 3: full SASRec (Kang & McAuley 2018).

Stage 2 + learned positional embeddings added to item embeddings before
the transformer blocks. Architecturally, this is the canonical SASRec.

A quirk of left-padding + learned PE: most users have short sequences
(avg ~9 here), so only the *last* few positional embeddings get trained
heavily. pos_embedding[0..40] rarely receive gradient because they sit
under padding for most users. Standard SASRec convention accepts this;
the comparable-to-paper number is what matters.

Expect a small delta vs Stage 2 on Amazon Games (~1-2% per the paper's
own Table IV: removing PE drops NDCG@10 only 0.5360 -> 0.5301). A small
delta here is the expected finding, not a bug.
"""

import torch
import torch.nn as nn

from models.transformer_block import TransformerBlock

MAX_SEQ_LEN = 50


class SASRec(nn.Module):
    def __init__(self, n_items, hidden_dim=64, n_blocks=2, n_heads=1,
                 max_seq_len=MAX_SEQ_LEN, dropout=0.5):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        causal = torch.triu(
            torch.ones(max_seq_len, max_seq_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal, persistent=False)
        # Position ids 0..L-1 are static; precompute and ride along on .to(device).
        self.register_buffer("positions", torch.arange(max_seq_len), persistent=False)

    def forward(self, input_seq):
        # input_seq: (B, L)
        item_emb = self.item_embedding(input_seq)              # (B, L, D)
        pos_emb = self.pos_embedding(self.positions).unsqueeze(0)  # (1, L, D)
        x = self.dropout(item_emb + pos_emb)
        padding_mask = (input_seq == 0)
        for block in self.blocks:
            x = block(x, self.causal_mask, padding_mask)
        return self.layer_norm(x)
