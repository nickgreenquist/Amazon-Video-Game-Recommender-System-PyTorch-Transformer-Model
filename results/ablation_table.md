# Ablation: Building Up Sequential Recommendation on Amazon Video Games

Dataset: Amazon Video Games 5-core 2018 (McAuley UCSD), preprocessed with
dedup + iterative 5-core. 50,626 users · 16,882 items · 453,881 interactions
· avg seq length 8.97.

Training (identical across Stages 1-3): Adam(lr=1e-3, betas=(0.9, 0.98)),
batch_size=128, grad clip max_norm=1.0, BCE-with-logits with 1 fresh
sampled negative per non-pad position, max 200 epochs, val every 5 epochs
on sampled NDCG@10 with patience=4 val checks. hidden_dim=64.

Eval (identical across stages): leave-one-out. Sampled = rank target against
100 fresh per-user unseen negatives. Full = rank target against all 16,882
items, with the user's seen items (except the target) masked out.

| Stage | Model | Components | Params | Hit@10 (sampled) | NDCG@10 (sampled) | Hit@10 (full) | NDCG@10 (full) | Δ NDCG@10 (sampled) vs prev |
|-------|-------|------------|--------|------------------|-------------------|---------------|----------------|-----------------------------|
| 1 | BagOfItemsModel | item emb + avg pool | 1.08M | 0.3935 | 0.2230 | 0.0078 | 0.0037 | — |
| 2 | AttentionNoPositionModel | + causal self-attn | 1.13M | 0.7044 | 0.4756 | 0.0596 | 0.0301 | +0.2526 (+113%) |
| 3 | SASRec | + learned positional emb | — | — | — | — | — | — |
| 4 | BERT4Rec (optional) | bidirectional + masked LM | — | — | — | — | — | — |

## Per-stage notes

**Stage 1 (BagOfItemsModel).** Ran the full 200 epochs without ever
plateauing — val NDCG@10 improved monotonically (0.044 → 0.241) and was
still climbing at epoch 200. The plan caps training at 200, so this is the
"comparable to later stages" number; the *true* converged ceiling is
probably higher but isn't what we report. Sampled Hit@10 = 0.39 lands at
the top of the plan's expected 0.30-0.40 range. The 60x gap between
sampled NDCG@10 (0.223) and full NDCG@10 (0.0037) is the Krichene-Rendle
bias in action — against 100 random negatives this model looks decent,
against the full 17k corpus it can barely separate signal from noise. The
sanity gate (sampled NDCG@10 > 0.10) passes by a wide margin.

**Stage 2 (AttentionNoPositionModel).** Adding causal self-attention (still
no positional embeddings, 2 blocks × 1 head) more than doubles every
metric. Sampled NDCG@10 = 0.476 is **already within 11% of published
SASRec on Amazon Games (0.5360)** even without positional information.
Full-corpus NDCG@10 jumps 8× (0.0037 → 0.0301), so this isn't just an
artifact of the sampled protocol — the model genuinely ranks better
against the entire 17k corpus. Like Stage 1, never plateaued at 200
epochs (val NDCG@10 still climbing 0.5119 → 0.5161 → 0.5165 in the final
checks). The first-batch finite-loss assertion validated that the
left-pad + causal-mask NaN trap is properly handled by the `masked_fill`
inside TransformerBlock; without that fix, training dies on batch 0.

## Comparison to published baselines

| Source | Hit@10 (sampled) | NDCG@10 (sampled) |
|--------|------------------|-------------------|
| SASRec paper (Kang & McAuley 2018), Table III, Amazon Games | 0.7410 | 0.5360 |
| This implementation, Stage 3 | — | — |

Target: Stage 3 within 5-10% of published (NDCG@10 ≈ 0.48-0.54).
