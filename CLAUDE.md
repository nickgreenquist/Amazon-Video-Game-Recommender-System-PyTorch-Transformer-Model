
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project Context

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status: Pre-implementation

The repo currently contains only `implementation_plan.md`, this file, and `LICENSE`. **No code exists yet.** `implementation_plan.md` is the authoritative spec — read it before writing any code. Build in the strict order it defines (preprocess → dataset → Stage 1 → 2 → 3 → optional 4 → ablation → Streamlit), and do not advance to stage N+1 until stage N is trained, verified, and its metrics recorded.

There is no `requirements.txt`, test suite, or build tooling yet. When you create the first ones, document the actual commands here (entry point is planned as `python src/main.py --stage <N>`).

## Goal

Progressive, ablation-driven implementation of sequential recommendation on the Amazon Video Games 5-core 2018 dataset (McAuley UCSD), culminating in SASRec. Each stage adds exactly one architectural component so its contribution can be measured. The deltas between stages — not just the final SASRec number — are the deliverable.

- Stage 1: Bag-of-items (avg-pool embeddings, no sequence order)
- Stage 2: Causal self-attention, **no** positional embeddings
- Stage 3: Full SASRec (causal attention + learned positional embeddings)
- Stage 4 (optional): BERT4Rec (bidirectional + masked LM — needs its own training loop)

## Invariants that span multiple files (do not break)

- **Item idx 0 is padding.** Real items are 1-indexed (1..n_items). Use `padding_idx=0` in every `nn.Embedding`.
- **Sequences are LEFT-padded** to `MAX_SEQ_LEN=50` (drop oldest items, keep most recent).
- **Causal mask is mandatory in Stages 2 and 3** — it is required for autoregressive correctness, NOT a feature being ablated. The only thing ablated between Stage 2 and 3 is positional embeddings.
- **Padding must never contribute**: mask it in attention (`key_padding_mask`) and in the loss.
- **Item embedding is tied**: the same embedding table is used for both input and candidate scoring (the training/eval loops call `model.item_embedding(...)` directly), so every stage's model must expose an `item_embedding` attribute and `forward` must return per-position output of shape `(B, L, D)`.
- **Negatives are resampled fresh every batch**, never precomputed.
- **Timestamp ties break deterministically by `asin`** during preprocessing.

## Shared training/eval (identical across Stages 1–3 for fair comparison)

- Loss: BCE-with-logits over positive + sampled negatives (`n_random_neg=100` per position).
- Adam, `lr=1e-3`, betas `(0.9, 0.98)`, grad clip `max_norm=1.0`.
- Defaults: `hidden_dim=64`, `n_blocks=2`, `n_heads=1`, `max_seq_len=50`, `dropout=0.2`, `batch_size=128`. Pre-norm transformer blocks.
- Leave-one-out split: train on `[i_1..i_{T-2}]`, validate on `i_{T-1}`, test on `i_T`.
- Report **both** sampled (100 negatives) and full-corpus metrics for Hit@10 and NDCG@10. Primary selection metric is sampled NDCG@10.

Keep these constant across stages; only the model architecture varies.

## Sanity checks

- Stage 1 sampled NDCG@10 must exceed 0.10, or the baseline is broken — debug before continuing.
- Stage 3 sampled metrics should land within ~5–10% of published SASRec (Hit@10 ≈ 0.737, NDCG@10 ≈ 0.523). Aim for comparable, not superior.

## Out of scope

No integration with other recommender repos, no ranker/cross-encoder/DIN architectures, no text/multi-task extensions, no skipping ablation stages.
