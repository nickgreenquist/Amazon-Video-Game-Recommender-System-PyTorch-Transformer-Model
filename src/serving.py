"""Serving helpers for the Streamlit demo (streamlit_app.py).

Only the genuinely-new serving glue lives here. The scoring math is IMPORTED,
not re-implemented, so the demo cannot drift from the offline path:
    recommend            <- src.canaries        (last-position scoring, history masked)
    most_similar, item_matrix <- src.similar_items  (cosine over the embedding table)

This module adds exactly three things (§7 of deployment_plan.md):
    load_model()      construct Stage 3 SASRec, load serving/stage3_best.pth, .eval()
    load_metadata()   read serving/item_metadata.parquet, idx-keyed, with display labels
    search_titles()   substring title search -> (idx, label) for the multiselect

The public load_* functions are Streamlit-cached; each delegates to an uncached
_impl so the parity gate (and any non-Streamlit caller) can run them directly.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.canaries import recommend  # noqa: F401 (re-exported for the app)
from src.dataset import MAX_SEQ_LEN
from src.main import get_device
from src.models.stage3_sasrec import SASRec
from src.similar_items import item_matrix, most_similar  # noqa: F401 (re-exported)

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "serving"
CKPT = SERVING / "stage3_best.pth"
META = SERVING / "item_metadata.parquet"

DEVICE = get_device()


def _build_labels(df):
    """Add a `label` column that maps 1:1 to idx. Base label is
    "{title} — {platform}" (or just the title when platform is empty); any base
    shared by more than one item gets a "(#idx)" suffix so every label is unique
    and resolves to exactly one item (§8.1)."""
    base = df.apply(
        lambda r: f"{r.title} — {r.platform}" if r.platform else str(r.title),
        axis=1,
    )
    collides = base.duplicated(keep=False)
    df = df.copy()
    df["label"] = [
        f"{b} (#{i})" if c else b
        for b, c, i in zip(base, collides, df["idx"])
    ]
    return df


def _load_metadata_impl():
    df = pd.read_parquet(META).sort_values("idx").reset_index(drop=True)
    df = _build_labels(df)
    return df.set_index("idx", drop=False)


@st.cache_data
def load_metadata():
    """idx-keyed DataFrame: columns idx, title, platform, kind, image, label."""
    return _load_metadata_impl()


def _load_model_impl():
    n_items = len(_load_metadata_impl())          # parquet has exactly n_items rows
    model = SASRec(n_items, hidden_dim=64, n_blocks=2, n_heads=1,
                   max_seq_len=MAX_SEQ_LEN, dropout=0.5).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_model():
    """Trained Stage 3 SASRec on DEVICE, in eval mode."""
    return _load_model_impl()


def search_titles(meta, query, k=50):
    """[(idx, label), ...] for items whose title contains `query` (case-insensitive),
    ordered by title. Mirrors similar_items.py's --search."""
    q = query.strip().lower()
    if not q:
        return []
    hits = meta[meta["title"].str.lower().str.contains(q, regex=False)]
    hits = hits.sort_values("title")
    return list(zip(hits["idx"].tolist(), hits["label"].tolist()))[:k]
