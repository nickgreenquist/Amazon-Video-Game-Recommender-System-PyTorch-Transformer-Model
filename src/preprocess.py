"""Turn the 5-core Video Games reviews into sequential training data.

Loads data/raw/Video_Games_5.json.gz, dedups (user, item) pairs (keeping the
earliest timestamp), then iteratively drops users and items with <5 unique
interactions until stable. McAuley's `_5` suffix only guarantees 5 raw rows
per user/item — once duplicates collapse, ~4.3k users and ~200 items fall
below the threshold, so we re-filter to restore the 5-core invariant every
later stage assumes. This matches how published SASRec/BERT4Rec papers
preprocess the Amazon datasets, keeping our metrics directly comparable.

Item indices are 1..n_items; idx 0 is reserved for padding.

Output:
    data/processed/interactions.parquet  — (user_idx, item_idx, timestamp)
    data/processed/item_id_to_idx.json
    data/processed/user_id_to_idx.json

Usage: python3 src/preprocess.py
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
REVIEWS = DATA / "raw" / "Video_Games_5.json.gz"
OUT = DATA / "processed"


def load_reviews():
    rows = []
    with gzip.open(REVIEWS, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], r["unixReviewTime"]))
    return pd.DataFrame(rows, columns=["reviewer_id", "asin", "timestamp"])


def iterative_5core(df):
    """Drop users and items with <5 interactions until the set is stable."""
    for _ in range(20):
        n0 = len(df)
        uc = df.groupby("reviewer_id").size()
        df = df[df["reviewer_id"].isin(uc[uc >= 5].index)]
        ic = df.groupby("asin").size()
        df = df[df["asin"].isin(ic[ic >= 5].index)]
        if len(df) == n0:
            return df
    raise RuntimeError("5-core filter did not converge in 20 iterations")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_reviews()
    n_raw = len(df)

    # Collapse repeat reviews of the same item to a single interaction,
    # keeping the earliest timestamp so chronology stays intact.
    df = (df.sort_values("timestamp", kind="mergesort")
            .drop_duplicates(["reviewer_id", "asin"], keep="first"))
    n_dedup = len(df)

    df = iterative_5core(df)
    n_filtered = len(df)

    # Deterministic index assignment: sort the unique IDs lexicographically.
    # Items are 1-indexed because idx 0 is reserved as the padding token.
    item_ids = sorted(df["asin"].unique())
    user_ids = sorted(df["reviewer_id"].unique())
    item_id_to_idx = {a: i + 1 for i, a in enumerate(item_ids)}
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}

    df["item_idx"] = df["asin"].map(item_id_to_idx).astype(np.int32)
    df["user_idx"] = df["reviewer_id"].map(user_id_to_idx).astype(np.int32)

    # Per-user chronological order, with asin as the deterministic tie-break.
    df = (df.sort_values(["user_idx", "timestamp", "asin"], kind="mergesort")
            .reset_index(drop=True))

    out_df = df[["user_idx", "item_idx", "timestamp"]]
    out_df.to_parquet(OUT / "interactions.parquet", index=False)
    (OUT / "item_id_to_idx.json").write_text(json.dumps(item_id_to_idx))
    (OUT / "user_id_to_idx.json").write_text(json.dumps(user_id_to_idx))

    seq_lens = df.groupby("user_idx").size().to_numpy()
    print(f"raw rows:       {n_raw:,}")
    print(f"after dedup:    {n_dedup:,}  ({n_raw - n_dedup:,} duplicates dropped)")
    print(f"after 5-core:   {n_filtered:,}  ({n_dedup - n_filtered:,} dropped)")
    print(f"n_users:        {len(user_ids):,}")
    print(f"n_items:        {len(item_ids):,}  (indices 1..{len(item_ids)}, 0 = pad)")
    print(f"n_interactions: {len(df):,}")
    print(f"seq length:     "
          f"avg={seq_lens.mean():.2f}  median={int(np.median(seq_lens))}  "
          f"p95={int(np.percentile(seq_lens, 95))}  max={int(seq_lens.max())}")


if __name__ == "__main__":
    main()
