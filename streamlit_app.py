"""Amazon Video Games SASRec — interactive demo.

Run:      streamlit run streamlit_app.py
Requires: serving/stage3_best.pth, serving/item_metadata.parquet
          (build with: python scripts/build_artifacts.py)

The headline (Tab 1) is the transformer's one differentiating capability:
recommendations that depend on the ORDER of a purchase history — something a
two-tower model structurally cannot do. Everything else is supporting cast.
"""

import html
import random
import re
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from src.canaries import CANARIES, SHOWCASE
from src.serving import (DEVICE, item_matrix, load_metadata, load_model,
                         most_similar, recommend)

ROOT = Path(__file__).resolve().parent
N_RECS = 10
SIMILAR_CAVEAT = (
    "This reads the learned item-embedding geometry directly (cosine similarity, "
    "no sequence model, no forward pass). The embedding was trained only for "
    "next-item prediction, so \"similar\" means \"appears in similar co-purchase "
    "contexts\" — dominated by platform, then genre. It is not a content model."
)

# ── cover-art + card rendering ─────────────────────────────────────────────────
# Covers render as a background-image div (not <img>): a missing/empty URL degrades
# to a clean panel instead of a broken-image icon, and it sidesteps Streamlit's
# markdown margin:auto-on-images rule. The full-width grids use st.columns; the
# order-sensitivity comparison uses a CSS grid (cover → title → platform per card),
# mirroring the Steam app's ranker side-by-side so the two layouts read the same.

COVER_COLS = 5          # cards per row in the full-width grids

_STYLE = """
<style>
.main .block-container { overflow-x:hidden; max-width:100%; }
div[data-testid="stCaptionContainer"] p { word-break:break-word; white-space:normal; }
div[data-testid="stTabs"] > div:first-child { overflow-x:auto; white-space:nowrap; flex-wrap:nowrap; }
div[data-testid="stDataFrame"] { overflow-x:auto; max-width:100%; }
</style>
"""


def _cover_div(url):
    safe = html.escape(str(url)) if url else ""
    return (
        "<div style='width:100%;aspect-ratio:1/1;border-radius:6px;"
        "background-color:#2b2b3a;background-position:center;background-size:contain;"
        f"background-repeat:no-repeat;background-image:url(\"{safe}\")'></div>"
    )


def _title_div(title):
    """Title rendered the way the Steam app renders its result titles — wrapping,
    breaking on long tokens, with a small top margin under the cover."""
    safe = html.escape(str(title))
    return (f"<div style='font-size:.8rem;line-height:1.2;margin-top:4px;"
            f"white-space:normal;overflow-wrap:anywhere;word-break:break-word'>{safe}</div>")


def _card_html(idx, meta, extra=""):
    """Single result card as an HTML string: cover → title → platform (· extra).
    Used by the CSS-grid comparison; mirrors the Steam app's `_card`."""
    row = meta.loc[idx]
    bits = []
    if row.platform:
        bits.append(html.escape(str(row.platform)))
    if extra:
        bits.append(extra)
    meta_el = (f"<div style='font-size:.72rem;opacity:.6;margin-top:2px;"
               f"white-space:normal;overflow-wrap:anywhere'>{' · '.join(bits)}</div>"
               if bits else "")
    return f"<div style='min-width:0'>{_cover_div(row.image)}{_title_div(row.title)}{meta_el}</div>"


def render_cards(idxs, meta, scores=None, ncols=COVER_COLS):
    """Responsive grid of cover cards laid out with st.columns: one row of `ncols`
    at a time, the last row left-packed with the leftover columns blank. Each card
    is cover → title → platform · score caption (Steam-app result format)."""
    for row_start in range(0, len(idxs), ncols):
        chunk = idxs[row_start:row_start + ncols]
        cols = st.columns(ncols)
        for offset, (col, idx) in enumerate(zip(cols, chunk)):
            i = row_start + offset
            row = meta.loc[idx]
            with col:
                st.html(_cover_div(row.image))
                st.html(_title_div(row.title))
                bits = []
                if row.platform:
                    bits.append(str(row.platform))
                if scores is not None:
                    bits.append(f"cos {scores[i]:.3f}")
                if bits:
                    st.caption(" · ".join(bits))


# Row-per-rank CSS grid (rank | left card | right card), matching the Steam app's
# ranker comparison. INLINE styles + st.html (not st.columns): Streamlit columns
# stack on mobile, and `minmax(0,1fr)` lets each track shrink so long titles wrap
# instead of bleeding into the next column.
_CMP_ROW = ("display:grid;grid-template-columns:1.6rem minmax(0,1fr) minmax(0,1fr);"
            "column-gap:22px;align-items:start;margin-bottom:16px")
_CMP_RANK = "text-align:center;font-weight:700;opacity:.55;font-size:.8rem;padding-top:2px"


def _cmp_header(label, order_ids, meta):
    """Header cell: column label over the (numbered) input history for that column,
    so each result column sits under the order that produced it."""
    items = "".join(
        f"<div style='display:flex;gap:5px;margin-bottom:2px'>"
        f"<span style='opacity:.5;font-weight:700;flex:none'>{n}</span>"
        f"<span style='min-width:0;overflow-wrap:anywhere'>{html.escape(str(meta.loc[i].title))}</span>"
        f"</div>"
        for n, i in enumerate(order_ids, 1)
    )
    return (f"<div style='min-width:0'>"
            f"<div style='font-weight:600;font-size:.85rem;margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:.76rem;line-height:1.5;opacity:.85'>{items}</div></div>")


def render_comparison(left_ids, right_ids, meta, left_label, right_label,
                      left_order, right_order):
    """Two-column row-per-rank comparison of two recommendation lists, formatted like
    the Steam app's ranker side-by-side."""
    parts = [
        f"<div style='{_CMP_ROW}'><div style='{_CMP_RANK}'></div>"
        f"{_cmp_header(left_label, left_order, meta)}"
        f"{_cmp_header(right_label, right_order, meta)}</div>"
    ]
    for i in range(max(len(left_ids), len(right_ids))):
        left = _card_html(left_ids[i], meta) if i < len(left_ids) else "<div></div>"
        right = _card_html(right_ids[i], meta) if i < len(right_ids) else "<div></div>"
        parts.append(
            f"<div style='{_CMP_ROW}'><div style='{_CMP_RANK}'>{i + 1}</div>{left}{right}</div>"
        )
    st.html("".join(parts))


# ── scoring ────────────────────────────────────────────────────────────────────

def recommend_display(model, meta, history):
    """Top-N recommended item ids for a history — all video-game-category products
    (games, accessories, consoles, guides) are eligible, nothing is filtered out."""
    return recommend(model, history, len(meta), DEVICE, k=N_RECS)


def constrained_shuffle(ids, current=None):
    """A permutation that differs from `current` (the ordering on screen now —
    defaults to ids on the first shuffle) AND ends in a different last item than
    current. Order-sensitivity is last-item-dominated, so this guarantees every
    press visibly changes the most-recent item, not just the first (§8.1,
    Decision #9). Comparing against `current` (not always ids) lets a later press
    cycle back to the original ordering, which matters for 2-item histories."""
    if len(ids) < 2:
        return list(ids)
    current = ids if current is None else current
    for _ in range(100):
        p = ids[:]
        random.shuffle(p)
        if p[-1] != current[-1] and p != current:
            return p
    p = current[:]                   # fallback: swap to force a different last item
    p[0], p[-1] = p[-1], p[0]
    return p


# ── tabs ─────────────────────────────────────────────────────────────────────

def render_history_recs(model, meta, history, key):
    """Shuffle button + order-sensitivity comparison (or plain recs) for a history.
    Shared by the Recommend and Example-users tabs; `key` namespaces the button and
    session state so the two tabs don't collide."""
    shuf_key, base_key = f"{key}_shuf_ids", f"{key}_shuf_base"
    if st.button("🔀 Shuffle history", use_container_width=True, key=f"{key}_shuffle"):
        if len(history) >= 2:
            prev = st.session_state.get(shuf_key) if st.session_state.get(base_key) == history else None
            st.session_state[shuf_key] = constrained_shuffle(history, prev)
            st.session_state[base_key] = history[:]
        else:
            st.session_state.pop(shuf_key, None)

    shuf = st.session_state.get(shuf_key)
    valid_shuffle = shuf is not None and st.session_state.get(base_key) == history

    if valid_shuffle:
        render_comparison(
            recommend_display(model, meta, history),
            recommend_display(model, meta, shuf),
            meta, "Original order", "Shuffled order", history, shuf,
        )
        st.caption(
            "Same items, different order → different next-item prediction. A two-tower "
            "model pools the history orderlessly and would return identical lists; the "
            "transformer reads positional embeddings, so the most recent items steer the recs."
        )
    else:
        render_cards(recommend_display(model, meta, history), meta)


def tab_recommend(model, meta, label_to_idx, options):
    st.caption(
        "Pick items in the order you'd have bought them — the last one is the most "
        "recent. Then hit **Shuffle** to see the *same items in a different order* "
        "produce a different next-item prediction."
    )

    selection = st.multiselect(
        "Your purchase history (in order — last = most recent)",
        options=options,
        key="rec_sel",
    )
    history = [label_to_idx[l] for l in selection]

    if not history:
        st.info("Select at least one item to get recommendations.")
        return

    render_history_recs(model, meta, history, "rec")


def tab_personas(model, meta):
    st.caption("Hand-built histories with coherent taste — a quick way to see clean recs "
               "without assembling a history yourself.")
    name = st.selectbox("Example user", SHOWCASE, index=None,
                        placeholder="Choose an example user…", key="persona_sel")
    if name is None:
        st.info("Choose an example user to see their recommendations.")
        return
    render_history_recs(model, meta, CANARIES[name], "persona")


def tab_similar(model, meta, label_to_idx, options):
    st.caption(SIMILAR_CAVEAT)
    seed_label = st.selectbox(
        "Seed item",
        options=options,
        index=None,
        placeholder="Choose a seed item…",
        key="sim_sel",
    )
    if seed_label is None:
        st.info("Choose a seed item to see its nearest neighbors.")
        return
    seed_id = label_to_idx[seed_label]
    E = item_matrix(model)
    neighbors = most_similar(E, seed_id, N_RECS)
    ids = [i for i, _ in neighbors]
    scores = [s for _, s in neighbors]
    render_cards(ids, meta, scores=scores)


def tab_about():
    ablation = ROOT / "results" / "ablation_table.md"
    if not ablation.exists():
        st.warning("results/ablation_table.md not found.")
        return
    lines = ablation.read_text().splitlines()
    st.markdown(lines[0])           # the "# From Bag-of-Items …" title — the page's one H1
    st.markdown(
        "A from-scratch PyTorch implementation of **SASRec** "
        "([Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781)), built as a staged "
        "ablation on the Amazon Video Games 5-core 2018 dataset so each architectural "
        "component's contribution can be measured."
    )
    st.markdown(
        "Stage 3 (full SASRec) lands at **sampled NDCG@10 = 0.5188** vs the paper's "
        "published 0.5360 (within 3.2%)."
    )
    st.markdown(
        "Code & write-up: "
        "[GitHub](https://github.com/nickgreenquist/Amazon-Video-Game-Recommender-System-PyTorch-Transformer-Model)"
    )
    st.markdown("---")
    _render_writeup("\n".join(lines[1:]))


_WRAP_WIDTH = 243  # max chars per rendered prose <p>; keeps long paragraphs short


def _render_writeup(text):
    """Render the ablation markdown. Each prose paragraph is hard-wrapped to
    _WRAP_WIDTH chars and each chunk rendered as its own st.markdown, so long
    paragraphs don't render as one run-on <p> that overflows on mobile; the
    pipe-table renders as a scrollable st.dataframe."""
    prose, table = [], []

    def flush_prose():
        if not prose:
            return
        block = "\n".join(prose)
        prose.clear()
        for para in re.split(r'\n\s*\n', block):
            para = " ".join(para.split())
            if not para:
                continue
            if para.startswith("#"):
                st.markdown(para)
                continue
            for chunk in textwrap.wrap(para, width=_WRAP_WIDTH):
                st.markdown(chunk)

    def flush_table():
        if table:
            cells = [[c.strip().replace("**", "") for c in r.strip("|").split("|")]
                     for r in table]
            st.dataframe(pd.DataFrame(cells[2:], columns=cells[0]), hide_index=True)
            table.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            flush_prose()
            table.append(line.strip())
        else:
            flush_table()
            prose.append(line)
    flush_table()
    flush_prose()


# ── app ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Amazon Video Games Recommender (Transformer Model)", layout="wide")
st.markdown(_STYLE, unsafe_allow_html=True)
st.title("Amazon Video Games Recommender (Transformer Model)")
st.markdown(
    "<small>A transformer that recommends your next video game product — game, accessory, "
    "console, or guide — from your purchase history. It reads the <b>sequence</b> you bought "
    "items in, not just which ones.<br>Built from scratch in PyTorch on the "
    "<a href='https://nijianmo.github.io/amazon/' target='_blank'>UCSD Amazon 2018</a> "
    "dataset.</small>",
    unsafe_allow_html=True,
)

model = load_model()
meta = load_metadata()
idx_to_label = meta["label"].to_dict()
label_to_idx = {v: k for k, v in idx_to_label.items()}
# Picker options ordered by popularity (interaction count, desc) so the most
# recognizable items surface first instead of opaque ASIN order.
popular_labels = meta.sort_values("interaction_count", ascending=False)["label"].tolist()

rec_tab, persona_tab, similar_tab, about_tab = st.tabs(
    ["Recommend", "Examples", "Similar", "About"]
)
with rec_tab:
    tab_recommend(model, meta, label_to_idx, popular_labels)
with persona_tab:
    tab_personas(model, meta)
with similar_tab:
    tab_similar(model, meta, label_to_idx, popular_labels)
with about_tab:
    tab_about()
