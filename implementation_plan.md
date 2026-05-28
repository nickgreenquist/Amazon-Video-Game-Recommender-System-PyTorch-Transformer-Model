# Building SASRec from First Principles: An Ablation-Driven Sequential Recommender on Amazon Video Games

## Goal

Build up understanding of sequential recommendation **from first principles**, with SASRec as the endpoint. Each stage adds one architectural component and measures its contribution. The deliverable is a deployed Streamlit app demonstrating sequential next-item prediction, with offline evaluation metrics directly comparable to published SASRec/BERT4Rec papers, AND a clear ablation table showing what each component contributes.

This is a **parallel portfolio project**, not an integration with existing CG/Ranker repos.

## Why Progressive, Not Direct SASRec

Building SASRec wholesale and saying "I implemented SASRec" is a junior-engineer story. Building up from a bag-of-items baseline through attention, positional embeddings, and causal masking — measuring each step — is a senior-engineer story. The work is similar but the narrative and depth of understanding are meaningfully different.

The ablation also produces actual research findings about your specific dataset (e.g., "positional embeddings contributed X% lift on Amazon Video Games"), which is more interesting than reproducing a paper number.

---

## Non-Goals (Do Not Do These)

- Do NOT integrate with existing movie/book/game recommender repos
- Do NOT build a ranker on top of SASRec in this project
- Do NOT build a cross-encoder DIN-style architecture
- Do NOT add this to existing CG repos as another variant
- Do NOT try to beat published SASRec numbers — aim for comparable, not superior
- Do NOT skip ablation stages and jump straight to full SASRec

---

## Repo Structure

New standalone repo: `Amazon-Video-Game-Recommender-System-PyTorch-Transformer-Model`

```
Amazon-Video-Game-Recommender-System-PyTorch-Transformer-Model/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── Video_Games_5.json.gz       ← downloaded from McAuley 2018 dataset
│   └── processed/
│       ├── interactions.parquet
│       ├── item_id_to_idx.json
│       ├── user_id_to_idx.json
│       └── splits.pt
├── src/
│   ├── preprocess.py                   ← raw → 5-core → interactions.parquet
│   ├── dataset.py                      ← PyTorch Dataset, leave-one-out splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stage1_bag_of_items.py      ← baseline, no sequence
│   │   ├── stage2_attention_no_pos.py  ← causal attention, no position embeddings
│   │   ├── stage3_sasrec.py            ← full SASRec (causal + positions)
│   │   └── stage4_bert4rec.py          ← optional: masked LM bidirectional
│   ├── train.py                        ← unified training loop, takes model as arg
│   ├── evaluate.py                     ← Hit@10, NDCG@10 (sampled + full)
│   └── main.py                         ← entry point with --stage flag
├── streamlit/
│   ├── app.py                          ← deployed demo using best model
│   └── serving/
│       ├── model.pth                   ← best stage's checkpoint
│       ├── item_embeddings.pt
│       └── item_metadata.parquet
├── saved_models/
│   ├── stage1_bag_of_items_*.pth
│   ├── stage2_attention_*.pth
│   ├── stage3_sasrec_*.pth
│   └── stage4_bert4rec_*.pth
├── results/
│   └── ablation_table.md               ← metrics from each stage
└── notebooks/
    └── eda.ipynb                       ← optional EDA
```

---

## Stage 0: Data Preprocessing (Shared Across All Stages)

### Download
Amazon Video Games 5-core 2018 from McAuley UCSD page:
- Source: https://nijianmo.github.io/amazon/index.html
- File: `Video_Games_5.json.gz`
- URL pattern: `https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz`
- Approximate size: 500MB raw

Save to `data/raw/Video_Games_5.json.gz`.

### `src/preprocess.py`

```python
"""
Preprocess Amazon Video Games 5-core into sequential training data.
Output: data/processed/interactions.parquet
"""

def run(raw_path, out_dir):
    # 1. Load JSON (each line is one review record)
    #    Fields used: reviewerID, asin, unixReviewTime
    
    # 2. Apply 5-core filter iteratively until stable:
    #    Drop users with < 5 interactions
    #    Drop items with < 5 interactions
    #    Repeat (usually 2-3 iterations)
    
    # 3. Sort each user's interactions by unixReviewTime
    
    # 4. Handle timestamp collisions deterministically:
    #    When user has multiple reviews at same timestamp,
    #    sort secondarily by asin for reproducibility
    
    # 5. Build user_id_to_idx and item_id_to_idx mappings
    #    Item indices are 1-indexed (0 reserved for padding)
    
    # 6. Save:
    #    - interactions.parquet: (user_idx, item_idx, timestamp)
    #    - item_id_to_idx.json
    #    - user_id_to_idx.json
    
    # Print summary stats:
    #    n_users, n_items, n_interactions
    #    sequence length: avg, median, p95, max
```

### Expected stats (sanity check)
- ~50-80k users
- ~15-20k items
- ~500k-1M interactions
- Avg sequence length: 10-15 items
- Max sequence length: 100-500 items

---

## Stage 0.5: Dataset Construction (Shared Across All Stages)

### `src/dataset.py`

```python
"""
Leave-one-out evaluation protocol (standard SASRec convention):

For each user with sequence [i_1, i_2, ..., i_T] sorted by timestamp:
- Train:      [i_1, ..., i_{T-2}]
- Validation: predict i_{T-1} given [i_1, ..., i_{T-2}]
- Test:       predict i_T     given [i_1, ..., i_{T-1}]
"""

MAX_SEQ_LEN = 50

class SeqRecDataset(Dataset):
    def __init__(self, interactions_df, split='train'):
        # Build user → sequence dict
        # Apply leave-one-out split based on `split` argument
        # Left-pad each sequence to MAX_SEQ_LEN with 0
        # Store as tensors for fast batch loading
        ...
    
    def __getitem__(self, idx):
        return input_seq, pos_seq, neg_seq
        # input_seq: items [i_1, ..., i_{T-1}], LEFT-padded with 0
        # pos_seq:   items [i_2, ..., i_T],     LEFT-padded with 0
        # neg_seq:   random unseen items, one per position
```

### Padding convention (critical, do not change)
- Item idx 0 = padding token
- Real items have idx 1 to n_items
- Sequences are LEFT-padded (older items dropped, recent items kept)

### Negative sampling
- One random unseen item per position, resampled fresh each epoch
- Avoid sampling positives or padding

---

## Stage 1: Bag-of-Items Baseline (No Sequence Modeling)

**Hypothesis:** Establishes the no-sequence baseline. How well can we predict the next item using just an unordered set of past items?

### `src/models/stage1_bag_of_items.py`

```python
class BagOfItemsModel(nn.Module):
    """
    Simplest possible baseline:
    - Average pool item embeddings over the input sequence
    - Score candidates by dot product with the averaged representation
    - No attention, no position information, no sequence order
    
    This is essentially a degenerate two-tower model where the user tower
    is just average pooling. Tests how much we can predict from
    'set of items the user interacted with' alone.
    """
    def __init__(self, n_items, hidden_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
    
    def forward(self, input_seq):
        # input_seq: (batch, max_seq_len)
        
        # Embed all items in sequence
        item_emb = self.item_embedding(input_seq)  # (B, L, D)
        
        # Mask out padding positions
        mask = (input_seq != 0).float().unsqueeze(-1)  # (B, L, 1)
        
        # Average pool (mean over non-padding positions)
        pooled = (item_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # pooled: (B, D)
        
        # Expand to (B, L, D) so all positions get the same user representation
        # This keeps interface compatible with later stages
        return pooled.unsqueeze(1).expand(-1, input_seq.size(1), -1)
```

### Training & Evaluation
Use the unified `train.py` with `--stage 1`. Same loss as later stages (BCE with sampled negatives) for direct comparability.

### Expected result
Hit@10 (sampled) probably around 0.30-0.40. This is your floor — anything below this means something is broken. Anything above this is a problem since this model has no sequence information.

---

## Stage 2: Causal Self-Attention WITHOUT Positional Embeddings

**Hypothesis:** Does self-attention alone help, even without position information? Tests whether the attention mechanism's ability to dynamically weight past items adds value over uniform averaging.

### `src/models/stage2_attention_no_pos.py`

```python
class AttentionNoPositionModel(nn.Module):
    """
    Causal self-attention WITHOUT positional embeddings.
    
    Key question: does dynamic attention help even when the model
    can't tell positions apart? In theory, no — attention is
    permutation-invariant without positions. In practice, item identity
    serves as a weak positional signal. Worth measuring.
    """
    def __init__(self, n_items, hidden_dim=64, n_blocks=2, n_heads=1, dropout=0.5):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        # NOTE: NO pos_embedding here
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = MAX_SEQ_LEN
    
    def forward(self, input_seq):
        x = self.item_embedding(input_seq)  # (B, L, D)
        x = self.dropout(x)
        
        # Causal mask: position i can only attend to positions 1..i
        causal_mask = torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1
        ).bool().to(input_seq.device)
        padding_mask = (input_seq == 0)
        
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)
        
        return self.layer_norm(x)  # (B, L, D)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block. Shared between stages 2, 3."""
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
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out  # residual
        x = x + self.ff(self.ln2(x))  # residual
        return x
```

### Why causal mask is still used here

Even though we're testing "no position info," the causal mask is non-negotiable for autoregressive training. Without it, position i would attend to position i+1, leaking the answer. The causal mask is about training correctness, not about position information.

### Expected result
Hit@10 (sampled) somewhere between Stage 1 and Stage 3. The interesting question is HOW MUCH attention helps without positions. If it's barely better than Stage 1, position embeddings are doing most of the work. If it's close to Stage 3, attention itself is most of the value.

---

## Stage 3: Full SASRec (Causal Attention + Positional Embeddings)

**Hypothesis:** Adding positional embeddings lets the model distinguish positions in the sequence. This is the actual SASRec model.

> **Set expectations: the Stage 2→3 delta will be small on Games.** The paper's own ablation (Table IV, "Remove PE") shows NDCG@10 on Amazon Games moving only 0.5360 → 0.5301 — about a 1% relative drop. So PE is *not* a big unlock on this dataset; the honest finding is likely "positional embeddings barely move Games." That is still a legitimate, reportable result — don't treat a ~1% delta as a bug, and don't oversell PE in the write-up.

### `src/models/stage3_sasrec.py`

```python
class SASRec(nn.Module):
    """
    Full SASRec from Kang & McAuley 2018:
    - Item embeddings + learned positional embeddings
    - Causal self-attention (decoder-only transformer)
    - Pre-norm convention
    - Item embedding tied (same embedding used as input AND for scoring)
    """
    def __init__(self, n_items, hidden_dim=64, n_blocks=2, n_heads=1, 
                 max_seq_len=50, dropout=0.5):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)  # NEW vs Stage 2
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_seq):
        item_emb = self.item_embedding(input_seq)  # (B, L, D)
        
        positions = torch.arange(self.max_seq_len, device=input_seq.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # (1, L, D)
        x = item_emb + pos_emb  # NEW vs Stage 2
        x = self.dropout(x)
        
        causal_mask = torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1
        ).bool().to(input_seq.device)
        padding_mask = (input_seq == 0)
        
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)
        
        return self.layer_norm(x)  # (B, L, D)
```

### Expected result
Hit@10 (sampled) around 0.65-0.75 based on published SASRec results. This is the canonical benchmark.

### Hyperparameters (start with these defaults)

| Hyperparameter | Value | Notes |
|---|---|---|
| `hidden_dim` | 64 | Standard SASRec default |
| `n_blocks` | 2 | More rarely helps on this scale |
| `n_heads` | 1 | Paper uses 1 head |
| `max_seq_len` | 50 | Standard |
| `dropout` | 0.5 | Paper uses 0.5 for sparse datasets (Beauty/Games/Steam); 0.2 is the ML-1M value. Using 0.2 on Games will overfit and likely undershoot the published number. |

---

## Stage 4 (Optional): BERT4Rec — Bidirectional Masked LM

**Hypothesis:** Bidirectional attention with masked language modeling can leverage both past AND future context during training. Does this help on this dataset?

This stage is **optional** and adds significant implementation complexity (masked LM training objective is meaningfully different). Skip if iteration speed matters more than completeness.

### `src/models/stage4_bert4rec.py`

```python
class BERT4Rec(nn.Module):
    """
    Bidirectional transformer with masked LM training (BERT4Rec, Sun et al. 2019).
    
    Key differences from SASRec:
    - NO causal mask (bidirectional attention)
    - Training: randomly mask 15% of positions, predict masked items
    - Inference: append a [MASK] token at the end, predict it
    
    This requires a different training loop — handle separately.
    """
    # Implementation similar to SASRec but without causal_mask in forward
    # Requires masked LM training objective (different from Stages 1-3)
    ...
```

### Decision point
Decide before starting Stage 4 whether the additional complexity is worth it. Recent research (Petrov & Macdonald 2022) shows BERT4Rec advantages over SASRec are mostly from training-time differences, not architecture. Skipping Stage 4 and noting the finding in your write-up is also defensible.

---

## Unified Training Loop

### `src/train.py`

One training function that takes any model. Same hyperparameters, same loss, same evaluation — only the model architecture varies.

```python
def train(model, train_loader, val_loader, config):
    """
    Unified training loop for Stages 1-3.
    Stage 4 (BERT4Rec) needs its own loop due to masked LM objective.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
    )
    
    best_val_ndcg = 0
    
    for epoch in range(config['n_epochs']):
        model.train()
        for input_seq, pos_seq, neg_seq in train_loader:
            # SASRec §III-E: exactly ONE negative per timestep, resampled each
            # epoch (neg_seq, produced fresh by the dataset). Do NOT add the 100
            # eval negatives here — 100 is the evaluation protocol only. Mixing
            # them into training deviates from the paper and hurts comparability.
            seq_out = model(input_seq)  # (B, L, D)
            
            pos_emb = model.item_embedding(pos_seq)
            neg_emb = model.item_embedding(neg_seq)
            
            pos_logits = (seq_out * pos_emb).sum(dim=-1)
            neg_logits = (seq_out * neg_emb).sum(dim=-1)
            
            # Mask: only compute loss on real positions
            mask = (pos_seq != 0).float()
            
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction='none'
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction='none'
            )
            
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if epoch % config['val_every'] == 0:
            val_metrics = evaluate(model, val_loader, config)
            print(f"Epoch {epoch}: val_NDCG@10={val_metrics['ndcg10_sampled']:.4f}")
            
            if val_metrics['ndcg10_sampled'] > best_val_ndcg:
                best_val_ndcg = val_metrics['ndcg10_sampled']
                torch.save(model.state_dict(), config['checkpoint_path'])
    
    return best_val_ndcg
```

### Training hyperparameters (same across all stages)

| Hyperparameter | Value |
|---|---|
| `lr` | 1e-3 |
| `batch_size` | 128 |
| `n_epochs` | 200 with early stopping |
| training negatives | 1 per position | resampled each epoch (per paper) |
| `n_eval_neg` | 100 | **eval only** — sampled-metric protocol, not training |
| `val_every` | 5 epochs |

Keep these identical across stages for fair comparison.

---

## Evaluation (Shared Across All Stages)

### `src/evaluate.py`

Report BOTH sampled and full-ranking metrics. Sampled = comparable to published baselines. Full = honest evaluation.

```python
def evaluate(model, eval_loader, n_neg_samples=100):
    """
    Leave-one-out evaluation:
    - Sampled NDCG@10: rank target against 100 random unseen items
    - Full NDCG@10: rank target against entire corpus (minus user history)
    """
    model.eval()
    hit10_sampled, ndcg10_sampled = [], []
    hit10_full, ndcg10_full = [], []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_seq, target_item, user_history = batch
            
            seq_out = model(input_seq)
            user_emb = seq_out[:, -1, :]  # last position's representation
            
            # Sampled metrics
            neg_items = sample_unseen_items(user_history, n=n_neg_samples)
            candidates = torch.cat([target_item.unsqueeze(1), neg_items], dim=1)
            cand_embs = model.item_embedding(candidates)
            scores = torch.einsum('bd,bcd->bc', user_emb, cand_embs)
            
            rank = (scores > scores[:, 0:1]).sum(dim=1) + 1
            hit10_sampled.append((rank <= 10).float())
            ndcg10_sampled.append(
                (1.0 / torch.log2(rank.float() + 1)) * (rank <= 10).float()
            )
            
            # Full metrics
            all_item_embs = model.item_embedding.weight[1:]
            full_scores = user_emb @ all_item_embs.T
            full_scores = mask_seen_items(full_scores, user_history)
            target_score = full_scores.gather(1, target_item.unsqueeze(1))
            rank_full = (full_scores > target_score).sum(dim=1) + 1
            hit10_full.append((rank_full <= 10).float())
            ndcg10_full.append(
                (1.0 / torch.log2(rank_full.float() + 1)) * (rank_full <= 10).float()
            )
    
    return {
        'hit10_sampled': torch.cat(hit10_sampled).mean().item(),
        'ndcg10_sampled': torch.cat(ndcg10_sampled).mean().item(),
        'hit10_full': torch.cat(hit10_full).mean().item(),
        'ndcg10_full': torch.cat(ndcg10_full).mean().item(),
    }
```

### Why report both metrics

Sampled metrics with 100 negatives are the SASRec paper standard but have been criticized (Krichene & Rendle 2020) as biased — they don't reflect ranking quality against the full corpus. Reporting both demonstrates evaluation rigor and gives interviewers an opening to discuss tradeoffs.

---

## Ablation Results Table (Fill As You Go)

This table is the centerpiece of your portfolio narrative.

### `results/ablation_table.md`

```markdown
# Ablation: Building Up Sequential Recommendation on Amazon Video Games

| Stage | Model | Components | Hit@10 (sampled) | NDCG@10 (sampled) | NDCG@10 (full) | Δ vs prev |
|-------|-------|-----------|------------------|-------------------|----------------|-----------|
| 1 | Bag of Items | item embeddings + avg pool | ? | ? | ? | — |
| 2 | Attention (no pos) | + causal self-attention | ? | ? | ? | vs S1 |
| 3 | SASRec | + positional embeddings | ? | ? | ? | vs S2 |
| 4 | BERT4Rec (optional) | bidirectional + masked LM | ? | ? | ? | vs S3 |

## Findings

- Stage 1 → Stage 2: How much does self-attention help WITHOUT position info?
- Stage 2 → Stage 3: How much do positional embeddings contribute?
- Stage 3 → Stage 4 (optional): Does bidirectional attention help on this dataset?

## Comparison To Published Baselines

| Source | Hit@10 (sampled) | NDCG@10 (sampled) |
|--------|------------------|-------------------|
| SASRec paper (Kang & McAuley 2018), Table III, Amazon Games | 0.7410 | 0.5360 |
| BERT4Rec paper (Sun et al. 2019), Amazon Games | — | — |
| This implementation, Stage 3 | ? | ? |

Target: within 5-10% of published (NDCG@10 ≈ 0.48–0.54).

> ⚠️ **Verify before filling the BERT4Rec row.** The original BERT4Rec paper evaluated on Beauty, Steam, ML-1m, and ML-20m — **not Amazon Video Games**. Do not cite a Games number for BERT4Rec unless you find a source that actually reports one; otherwise leave it blank or drop the row.
```

---

## Streamlit Deployment

### `streamlit/app.py`

Use the **best-performing stage** (likely Stage 3 SASRec) for the deployed app. The other stages are for the ablation story, not for serving.

Three-tab demo:

**Tab 1 — Sequential Prediction**
- User inputs a sequence of games in order
- Top-10 predicted next games
- Optional: visualize attention weights showing which past items contributed most

**Tab 2 — Stage Comparison**
- Show same input through Stage 1, Stage 2, Stage 3 models
- Demonstrate how predictions change as architecture gets richer
- Strong visual story for the ablation work

**Tab 3 — About**
- Architecture diagrams for each stage
- Ablation results table
- Reference SASRec paper
- Link to GitHub README

### Serving artifacts (< 50 MB total)
- `model.pth` — best stage's checkpoint (~5-10 MB)
- `item_embeddings.pt` — precomputed item embeddings (~5 MB)
- `item_metadata.parquet` — game titles, ASIN, image URLs

---

## CLAUDE.md Contents

```markdown
# Amazon Video Games SASRec — CLAUDE.md

## Project Purpose
Progressive implementation of sequential recommendation, building from
bag-of-items baseline through SASRec (and optionally BERT4Rec).
Each stage adds one architectural component and is independently trained
and evaluated for ablation comparison.

## Stages
- Stage 1: Bag of items (no sequence)
- Stage 2: Causal self-attention WITHOUT positional embeddings
- Stage 3: SASRec (causal attention + positional embeddings)
- Stage 4 (optional): BERT4Rec (bidirectional + masked LM)

## Dataset
- Amazon Video Games 5-core 2018 (McAuley UCSD)
- ~50-80k users, ~17k items after 5-core
- Leave-one-out evaluation

## Architecture Defaults (Stages 1-3)
- hidden_dim=64, n_blocks=2, n_heads=1, max_seq_len=50, dropout=0.5
- Item embedding tied between input and output
- Pre-norm transformer convention (LN before attn/ff)
- Causal attention mask in Stages 2, 3 (REQUIRED for autoregressive training,
  not a feature being ablated)

## Training (Same Across Stages 1-3)
- BCE with logits + ONE sampled negative per position, resampled each epoch (100 negatives is eval-only)
- Adam optimizer, lr=1e-3, beta=(0.9, 0.98)
- Gradient clipping max_norm=1.0
- Early stop on val NDCG@10 plateau (20 epochs patience)

## Evaluation
- Report BOTH sampled (100 negatives) AND full-ranking metrics
- Primary metric: NDCG@10 (sampled) for paper comparability
- Secondary metric: NDCG@10 (full) for honest evaluation

## Timestamp Tie-Breaking
When multiple reviews share a timestamp, sort secondarily by asin.
Deterministic and reproducible. Documented in README.

## Implementation Order (Strict)
1. preprocess.py and verify dataset stats
2. dataset.py and verify batch iteration
3. Stage 1 model and train to completion
4. Stage 2 model and train to completion
5. Stage 3 model and train to completion
6. (Optional) Stage 4
7. Generate ablation table
8. Streamlit app
9. Deploy

Do not move to stage N+1 until stage N is verified working.

## Do NOT
- Do not integrate with existing CG/Ranker repos
- Do not skip stages — the ablation IS the portfolio value
- Do not use full softmax (too slow with sequence outputs)
- Do not deviate from SASRec defaults without measurement
- Do not try to beat published baselines — aim for comparable
- Do not add cross-encoder or DIN-style architecture
```

---

## Implementation Order (Strict)

1. **Stage 0**: `preprocess.py` working, summary stats verified
2. **Stage 0.5**: `dataset.py` working, can iterate over batches
3. **Stage 1**: Bag-of-items model, train to convergence, record metrics
4. **Verify Stage 1**: Sampled NDCG@10 > 0.10 — if not, debug before continuing
5. **Stage 2**: Attention without positions, train to convergence, record metrics
6. **Compare Stage 1 vs 2**: Quantify what attention alone contributes
7. **Stage 3**: Full SASRec, train to convergence, record metrics
8. **Compare Stage 2 vs 3**: Quantify what positional embeddings contribute
9. **Verify Stage 3**: Sampled NDCG@10 within 5-10% of published Games number (0.5360 → target ~0.48-0.54)
10. **Stage 4 (optional)**: BERT4Rec implementation
11. **Ablation write-up**: Complete ablation table with findings
12. **Streamlit app** with stage comparison tab
13. **Deploy** to Streamlit Community Cloud
14. **Document** in README with full results

Do not move to stage N+1 until stage N is verified working AND its metrics are recorded.

---

## Critical Warnings

1. **Causal attention is non-negotiable for Stages 2 and 3.** It's not a feature being ablated — it's required for autoregressive training. Removing it without switching to masked LM training would leak the answer. The "ablation" here is positional embeddings, not the causal mask.

2. **Padding must not contribute.** Use `padding_idx=0` in `nn.Embedding`. Use `key_padding_mask` in attention. Mask padding in loss computation.

   **NaN trap:** with left-padding + causal mask + `key_padding_mask`, a padding *query* position (e.g. position 0 when it is pad) has every key masked → softmax over all `-inf` → `NaN` out of `nn.MultiheadAttention`. Even though that position is masked in the loss, `NaN * 0 = NaN`, so `(loss * mask).sum()` becomes NaN and training dies on the first batch. Fix: zero the model output at padding positions before the loss (`seq_out = seq_out * (input_seq != 0).unsqueeze(-1)`), or guarantee every query attends to ≥1 valid key. Verify the loss is finite on batch 0.

3. **Item indices are 1-indexed.** Idx 0 is padding. All real items go from 1 to n_items.

4. **Resample negatives every batch.** Do not precompute.

5. **Tie-breaking for timestamps must be deterministic.** Sort secondarily by asin.

6. **Report both sampled and full metrics.** Modern papers report both. This is a strong portfolio signal.

7. **Verify Stage 1 NDCG@10 > 0.10 before continuing.** If your baseline is broken, all later stages are meaningless.

8. **Do not skip ablation stages.** The progressive build IS the portfolio value. If you skip straight to Stage 3, the project is just "I implemented SASRec" rather than "I built up sequential recommendation from first principles and measured each component's contribution."

9. **Keep hyperparameters identical across Stages 1-3.** Same lr, same batch size, same n_epochs, same 1-negative training scheme. Only the model architecture varies. This is what makes the comparison fair.

10. **All four metrics in the ablation table matter.** Hit@10 and NDCG@10 each, both sampled and full. Don't drop columns to make the table cleaner.

---

## Reference Implementations (For Sanity Checking Only)

- Original SASRec (TensorFlow): https://github.com/kang205/SASRec
- PyTorch port: https://github.com/pmixer/SASRec.pytorch
- RecBole library: https://github.com/RUCAIBox/RecBole

Use these to verify your Stage 3 metrics are in the right ballpark. Do NOT copy implementations directly — the value of this project is in building it yourself and understanding the ablation results.

---

## Success Criteria

Project is complete when:

1. All Stages 1-3 trained, evaluated, and recorded in ablation table
2. Stage 3 (SASRec) sampled NDCG@10 within 5-10% of published baselines
3. Ablation table shows clear progression with measurable deltas
4. Both sampled and full NDCG@10 reported for each stage
5. Streamlit app deployed with stage comparison
6. README documents architecture progression, hyperparameters, results
7. CLAUDE.md captures all decisions
8. Findings articulated: which component contributed most? Were any surprising?

Total estimated time: 2-3 weekends for Stages 1-3. Add 1 weekend for Stage 4.

---

## Out of Scope

- BERT4Rec is optional Stage 4
- Multi-task learning (clicks + purchases)
- Text embeddings from product descriptions
- Session-based variants (different dataset)
- Long sequence variants (max_seq_len > 50)
- Custom attention mechanisms beyond standard multi-head
- Integration with existing CG/Ranker repos
- Cross-encoder or DIN-style architectures
- Beating published baselines (aim for comparable)
