"""Shared leave-one-out evaluation: Hit@10 and NDCG@10, both sampled (100
fresh per-user negatives) and full-corpus.

The 100 sampled negs are the SASRec paper's standard reporting protocol; the
full-corpus rank is the honest evaluation (Krichene & Rendle 2020). Both are
reported so the ablation table shows where the sampled metric is optimistic.

Negatives are resampled fresh at every call (matches the project-wide
"never precompute negatives" invariant). Run-to-run variance on NDCG@10 is
~±0.005; acceptable for ablation deltas.

User-seen items are masked out of the full-corpus scores BUT the target
itself is preserved (we're trying to find it). The target's rank is
1 + #items strictly out-ranking it — ties don't help the target.
"""

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, n_items, device, n_neg=100):
    """Returns dict with hit10_sampled, ndcg10_sampled, hit10_full, ndcg10_full."""
    model.eval()
    hs, ns, hf, nf = [], [], [], []
    # weight[1:] strips the padding row so column j maps to item id j+1.
    all_item_emb = model.item_embedding.weight[1:]

    for input_seq, target, seen_list in loader:
        input_seq = input_seq.to(device)
        target = target.to(device)

        seq_out = model(input_seq)          # (B, L, D)
        user_emb = seq_out[:, -1, :]        # (B, D) — most recent position

        # ---- sampled metrics ----
        # Per-user: 100 unique items not in the user's seen set.
        neg_items = _sample_unseen(seen_list, n_neg, n_items).to(device)  # (B, n_neg)
        candidates = torch.cat([target.unsqueeze(1), neg_items], dim=1)   # (B, 1+n_neg)
        cand_emb = model.item_embedding(candidates)                       # (B, 1+n_neg, D)
        scores = torch.einsum("bd,bcd->bc", user_emb, cand_emb)
        target_score = scores[:, 0:1]
        rank = (scores > target_score).sum(dim=1) + 1                      # (B,)
        hit = (rank <= 10).float()
        hs.append(hit)
        ns.append(hit / torch.log2(rank.float() + 1))

        # ---- full-corpus metrics ----
        full_scores = user_emb @ all_item_emb.T                            # (B, n_items)
        # Mask seen items (except the target itself) so they can't out-rank it.
        for b, seen_t in enumerate(seen_list):
            mask_ids = seen_t[seen_t != target[b].cpu()].to(device)
            full_scores[b, mask_ids - 1] = float("-inf")
        target_idx = (target - 1).unsqueeze(1)
        target_score_full = full_scores.gather(1, target_idx)
        rank_full = (full_scores > target_score_full).sum(dim=1) + 1
        hit_f = (rank_full <= 10).float()
        hf.append(hit_f)
        nf.append(hit_f / torch.log2(rank_full.float() + 1))

    return {
        "hit10_sampled":  torch.cat(hs).mean().item(),
        "ndcg10_sampled": torch.cat(ns).mean().item(),
        "hit10_full":     torch.cat(hf).mean().item(),
        "ndcg10_full":    torch.cat(nf).mean().item(),
    }


def _sample_unseen(seen_list, n, n_items):
    """(B, n) long tensor of unique unseen item ids in [1, n_items]."""
    B = len(seen_list)
    out = torch.empty(B, n, dtype=torch.long)
    for b in range(B):
        forbidden = set(seen_list[b].tolist())
        chosen = set()
        while len(chosen) < n:
            cand = int(np.random.randint(1, n_items + 1))
            if cand not in forbidden and cand not in chosen:
                chosen.add(cand)
        out[b] = torch.tensor(list(chosen), dtype=torch.long)
    return out
