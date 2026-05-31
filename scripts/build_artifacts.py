"""Build the two committed serving artifacts in serving/ (run locally, once).

This is a THIN transform of the already-rich data/item_metadata.parquet (output
of scripts/build_item_metadata.py) plus a copy of the trained checkpoint — it
does NOT re-implement metadata extraction, `kind` classification, or image
picking. That logic lives in build_item_metadata.py; keep it there.

Outputs (committed; the Streamlit app reads only these, never data/raw/):
    serving/stage3_best.pth        copied from saved_models/ as-is (~4.5 MB)
    serving/item_metadata.parquet  slim, idx-keyed, exactly 16,882 rows (~2-3 MB)

idx convention: item_id_to_idx.json values are 1-indexed (1..n_items; 0 is
padding). Those values ARE the item ids recommend()/most_similar() return, so
the parquet `idx` column == the raw item_id_to_idx value with NO off-by-one
shift — a lookup of a rec against `idx` is a direct join.

Usage: python -m scripts.build_artifacts   (or: python scripts/build_artifacts.py)
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RICH_META = ROOT / "data" / "item_metadata.parquet"
ASIN2IDX = ROOT / "data" / "processed" / "item_id_to_idx.json"
SRC_CKPT = ROOT / "saved_models" / "stage3_best.pth"

SERVING = ROOT / "serving"
OUT_META = SERVING / "item_metadata.parquet"
OUT_CKPT = SERVING / "stage3_best.pth"

# 6 cols (§5): the 5 display cols + interaction_count, which orders the demo
# pickers by popularity. interaction_count is carried straight from the rich
# source (raw _5-review count) — NOT re-derived from interactions.parquet.
COLS = ["idx", "title", "platform", "kind", "image", "interaction_count"]


def main():
    SERVING.mkdir(exist_ok=True)

    # 1. Checkpoint: straight copy.
    shutil.copyfile(SRC_CKPT, OUT_CKPT)

    # 2. Metadata: thin transform.
    asin2idx = json.loads(ASIN2IDX.read_text())
    n_items = len(asin2idx)

    df = pd.read_parquet(RICH_META)
    df["idx"] = df["asin"].map(asin2idx)
    df = df[df["idx"].notna()].copy()           # drop the ~524 non-corpus asins
    df["idx"] = df["idx"].astype(int)

    # 3. Placeholder rows for corpus items with no metadata record (17 asins
    #    genuinely absent from raw meta_Video_Games.json.gz — unrecoverable,
    #    not a build bug).
    have = set(df["idx"])
    missing = [i for i in asin2idx.values() if i not in have]
    placeholders = pd.DataFrame([
        {"idx": i, "title": "(no metadata)", "platform": "",
         "kind": "unknown", "image": "", "interaction_count": 0}
        for i in missing
    ])

    # 4. Slim to the 6 serving cols, append placeholders, sort by idx.
    slim = pd.concat([df[COLS], placeholders], ignore_index=True)
    slim["interaction_count"] = slim["interaction_count"].astype(int)
    slim = slim.sort_values("idx").reset_index(drop=True)

    # 5. Write idx-keyed parquet.
    slim.to_parquet(OUT_META, index=False)

    # Verification (§5).
    assert len(slim) == n_items, f"row count {len(slim)} != {n_items}"
    assert slim["idx"].is_unique, "idx column is not unique"
    assert set(slim["idx"]) == set(range(1, n_items + 1)), "idx does not cover 1..n_items"
    ckpt_mb = OUT_CKPT.stat().st_size / 1e6
    meta_mb = OUT_META.stat().st_size / 1e6
    print(f"serving/stage3_best.pth        {ckpt_mb:.1f} MB")
    print(f"serving/item_metadata.parquet  {meta_mb:.1f} MB  ({len(slim):,} rows, {len(missing)} placeholders)")
    print(f"serving/ total                 {ckpt_mb + meta_mb:.1f} MB")
    print("kind:", slim["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
