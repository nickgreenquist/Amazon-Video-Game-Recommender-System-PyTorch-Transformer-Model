"""Pre-norm transformer block shared by Stages 2 and 3.

Pre-norm convention: LayerNorm BEFORE attn/ff (not after, as in vanilla
Transformer). Matches SASRec, and makes deep stacks more stable.

NaN trap (the reason there's a masked_fill on attn_out): with left-padding,
a query at a padding position only has padding keys inside its causal
window. attn_mask + key_padding_mask both block them → softmax(all -inf) →
NaN at that position. NaN would then propagate through the residual into
every later block and eventually corrupt the loss (NaN * loss_mask = NaN).
We zero attn_out at padding positions before the residual add so nothing
downstream ever sees a NaN.
"""

import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask, padding_mask):
        # x: (B, L, D)  causal_mask: (L, L) bool  padding_mask: (B, L) bool
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        # Kill NaNs from padding queries before they enter the residual stream.
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x
