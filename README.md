# Amazon Video Games Sequential Recommender — Building SASRec from First Principles

A PyTorch sequential recommender for the Amazon Video Games 5-core (2018) dataset,
built **progressively** so each architectural component can be measured in isolation.
The endpoint is [SASRec](https://arxiv.org/abs/1808.09781) (Kang & McAuley, 2018),
but the deliverable is the **ablation** — the deltas between stages, not just the
final number.

Each stage adds exactly one component on top of the previous one, with **identical**
data, training loop, loss, and evaluation. Only the model architecture changes, which
is what makes the comparison fair.

| Stage | Model | New component |
|-------|-------|---------------|
| 1 | Bag-of-items | item embeddings + average pooling (no sequence order) |
| 2 | Attention (no positions) | + causal self-attention |
| 3 | **SASRec** | + learned positional embeddings |
| 4 | BERT4Rec | *(optional, not built)* bidirectional + masked LM |

## Results

Dataset after dedup + iterative 5-core: **50,626 users · 16,882 items · 453,881
interactions** (avg sequence length 8.97). Leave-one-out split (train on
`[i₁..i_{T-2}]`, validate on `i_{T-1}`, test on `i_T`). All stages trained to 500
epochs on the same budget.

| Stage | Model | Params | Hit@10 (sampled) | NDCG@10 (sampled) | Hit@10 (full) | NDCG@10 (full) | Δ NDCG@10 (sampled) |
|-------|-------|--------|------------------|-------------------|---------------|----------------|----------------------|
| 1 | BagOfItems | 1.08M | 0.4829 | 0.2896 | 0.0164 | 0.0078 | — |
| 2 | Attention (no pos) | 1.13M | 0.7339 | 0.5106 | 0.0769 | 0.0400 | **+0.221 (+76%)** |
| 3 | SASRec | 1.13M | 0.7472 | **0.5188** | 0.0748 | 0.0387 | +0.008 (+1.6%) |

- **Sampled** = target ranked against 100 random unseen negatives (paper-comparable protocol).
- **Full** = target ranked against all 16,882 items with the user's history masked (honest protocol).

### Headline finding

**Causal self-attention is the entire architectural unlock on Amazon Games.**
Adding attention (Stage 1 → 2) lifts sampled NDCG@10 by **+76%** and full NDCG@10
by ~5×. Adding learned positional embeddings on top (Stage 2 → 3) contributes only
**+1.6%** — small, but consistent in sign and magnitude with the paper's own Table IV
ablation (removing PE drops their Games NDCG@10 by ~1.1%). If you only get to add one
thing on top of item embeddings, it's attention, not positional embeddings.

### Comparison to published SASRec

| Source | Hit@10 (sampled) | NDCG@10 (sampled) |
|--------|------------------|-------------------|
| SASRec paper (Kang & McAuley 2018), Table III, Amazon Games | 0.7410 | 0.5360 |
| **This implementation, Stage 3 (500 epochs)** | **0.7472** | **0.5188** |
| Relative gap | +0.8% | −3.2% |

Hit@10 matches/slightly beats published; NDCG@10 sits within the 5% band — inside the
project's 5–10% success target. The full write-up with per-stage notes is in
[`results/ablation_table.md`](results/ablation_table.md).

## Repository layout

```
src/
  preprocess.py                  raw reviews → dedup → iterative 5-core → interactions.parquet
  dataset.py                     leave-one-out splits, left-padding, fresh negative sampling
  models/
    stage1_bag_of_items.py       avg-pool baseline
    stage2_attention_no_pos.py   causal self-attention, no positional embeddings
    stage3_sasrec.py             full SASRec (causal attention + learned positions)
    transformer_block.py         shared pre-norm block (stages 2 & 3)
  train.py                       unified training loop (identical across stages 1-3)
  evaluate.py                    Hit@10 / NDCG@10, sampled + full
  main.py                        entry point: python src/main.py --stage <N>
results/
  ablation_table.md              full results write-up
  stage{1,2,3}_metrics.json      per-stage test metrics
saved_models/
  stage{1,2,3}_best.pth          best checkpoint (by val sampled NDCG@10)
data/
  raw/                           Video_Games_5.json.gz (download separately)
  processed/                     interactions.parquet + id→idx maps
```

## Getting started

```bash
pip install -r requirements.txt
```

### 1. Download the data

Download the Amazon Video Games **5-core** 2018 reviews from the
[McAuley UCSD dataset page](https://nijianmo.github.io/amazon/) and place it at
`data/raw/Video_Games_5.json.gz`:

```bash
# ~150 MB
curl -o data/raw/Video_Games_5.json.gz \
  https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
```

### 2. Preprocess

```bash
python src/preprocess.py
```

Writes `interactions.parquet`, `item_id_to_idx.json`, and `user_id_to_idx.json`
to `data/processed/`. Sanity-check the printed stats against 50,626 users /
16,882 items / 453,881 interactions.

### 3. Train a stage

```bash
python src/main.py --stage 1            # bag-of-items baseline
python src/main.py --stage 2            # + causal self-attention
python src/main.py --stage 3 --epochs 500   # full SASRec
```

Each run trains, evaluates on the held-out test set, saves the best checkpoint to
`saved_models/`, and writes metrics to `results/stage<N>_metrics.json`.

## Key design invariants

These are constant across all stages — see [`CLAUDE.md`](CLAUDE.md) for the full list.

- **Item idx 0 is padding.** Real items are 1-indexed; every embedding uses `padding_idx=0`.
- **Sequences are left-padded** to `MAX_SEQ_LEN=50` (drop oldest, keep most recent).
- **Causal mask is mandatory** in Stages 2 & 3 — it's required for autoregressive
  correctness, *not* a feature being ablated. The only thing ablated between Stage 2
  and 3 is positional embeddings.
- **Padding never contributes** — masked in attention (`key_padding_mask`) and in the loss.
- **Item embedding is tied** — the same table is used as input and for candidate scoring.
- **Negatives are resampled fresh every batch** (1 per position for training; 100 for sampled eval).
- **Timestamp ties break deterministically by `asin`.**

### Shared training & evaluation

- Loss: BCE-with-logits over the positive + one sampled negative per position.
- Adam, `lr=1e-3`, betas `(0.9, 0.98)`, grad clip `max_norm=1.0`.
- Defaults: `hidden_dim=64`, `n_blocks=2`, `n_heads=1`, `dropout=0.5`, `batch_size=128`, pre-norm blocks.
- 500 epochs, validate every 5 epochs on sampled NDCG@10, patience 10 val checks.

## Status & next steps

- [x] Stage 0 — preprocessing
- [x] Stage 0.5 — dataset / leave-one-out splits
- [x] Stage 1 — bag-of-items (verified NDCG@10 > 0.10 gate)
- [x] Stage 2 — attention, no positions
- [x] Stage 3 — SASRec (within 3.2% of published NDCG@10)
- [x] Ablation table + findings
- [ ] Stage 4 — BERT4Rec *(optional)*
- [ ] Streamlit demo app (sequential prediction + stage comparison)
- [ ] Deploy

## References

- Kang & McAuley, *Self-Attentive Sequential Recommendation* (SASRec), ICDM 2018 — https://arxiv.org/abs/1808.09781
- Sun et al., *BERT4Rec*, CIKM 2019 — https://arxiv.org/abs/1904.06690
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 (why both sampled and full metrics are reported)
- Amazon Review Data (2018), McAuley UCSD — https://nijianmo.github.io/amazon/
