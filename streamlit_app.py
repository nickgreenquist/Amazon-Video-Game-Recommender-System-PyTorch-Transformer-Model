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
from pathlib import Path

import streamlit as st

from src.canaries import CANARIES, SHOWCASE
from src.serving import (DEVICE, item_matrix, load_metadata, load_model,
                         most_similar, recommend)

ROOT = Path(__file__).resolve().parent
N_RECS = 10
OVERFETCH = 40          # score this many, then trim to N_RECS after games-only filter
SIMILAR_CAVEAT = (
    "This reads the learned item-embedding geometry directly (cosine similarity, "
    "no sequence model, no forward pass). The embedding was trained only for "
    "next-item prediction, so \"similar\" means \"appears in similar co-purchase "
    "contexts\" — dominated by platform, then genre. It is not a content model."
)

# ── cover-art + card rendering (Steam _cover_div pattern: a background-image div,
#    not <img> — a missing/empty URL degrades to a clean panel, never a broken
#    icon, and it sidesteps Streamlit's markdown margin:auto on images). ──────────

_STYLE = """
<style>
.cards { display:flex; flex-wrap:wrap; gap:14px; }
.card { width:132px; }
.card .cover {
    width:100%; aspect-ratio:1/1; border-radius:6px;
    background-color:#2b2b3a; background-position:center;
    background-size:contain; background-repeat:no-repeat;
}
.card .cap {
    font-size:.78rem; line-height:1.2; margin-top:5px;
    overflow-wrap:anywhere; word-break:break-word;
}
.card .cap .rank { opacity:.5; font-weight:700; margin-right:4px; }
.card .plat { font-size:.7rem; opacity:.6; margin-top:2px; }
.card .score { font-size:.7rem; opacity:.75; margin-top:2px; font-variant-numeric:tabular-nums; }
.order-list { font-size:.82rem; line-height:1.5; }
.order-list .n { opacity:.5; font-weight:700; margin-right:4px; }
div[data-testid="stTabs"] > div:first-child { overflow-x:auto; white-space:nowrap; flex-wrap:nowrap; }
</style>
"""


def _card_html(rank, idx, meta, score=None):
    row = meta.loc[idx]
    cover = (f"<div class='cover' style='background-image:url(\"{html.escape(str(row.image))}\")'></div>")
    rank_el = f"<span class='rank'>{rank}</span>" if rank is not None else ""
    title = html.escape(str(row.title))
    plat = f"<div class='plat'>{html.escape(str(row.platform))}</div>" if row.platform else ""
    score_el = f"<div class='score'>cos {score:.3f}</div>" if score is not None else ""
    return (f"<div class='card'>{cover}"
            f"<div class='cap'>{rank_el}{title}</div>{plat}{score_el}</div>")


def render_cards(idxs, meta, scores=None, start=1):
    """Responsive flex-wrap grid of cover cards. One st.html call (st.markdown
    would run the dense HTML through a sanitizer that mangles it)."""
    cards = [
        _card_html(start + n, idx, meta, None if scores is None else scores[n])
        for n, idx in enumerate(idxs)
    ]
    st.html(f"<div class='cards'>{''.join(cards)}</div>")


def order_html(ids, meta):
    items = "  ".join(
        f"<span class='n'>{n}</span>{html.escape(str(meta.loc[i].title))}"
        for n, i in enumerate(ids, 1)
    )
    return f"<div class='order-list'>{items}</div>"


# ── scoring ────────────────────────────────────────────────────────────────────

def recommend_display(model, meta, history, games_only):
    """Top-N item ids for a history. Over-fetch then trim so the games-only
    toggle never produces a short/ragged list (§6, Decision #10)."""
    recs = recommend(model, history, len(meta), DEVICE, k=OVERFETCH)
    if games_only:
        recs = [i for i in recs if meta.loc[i].kind == "game"]
    return recs[:N_RECS]


def constrained_shuffle(ids):
    """A permutation that never leaves the last item unchanged — order-sensitivity
    is last-item-dominated, so this guarantees a visible delta (§8.1, Decision #9)."""
    if len(ids) < 2:
        return list(ids)
    last = ids[-1]
    for _ in range(100):
        p = ids[:]
        random.shuffle(p)
        if p[-1] != last and p != ids:
            return p
    p = ids[:]                       # fallback: move the old last item off the end
    p[0], p[-1] = p[-1], p[0]
    return p


# ── tabs ─────────────────────────────────────────────────────────────────────

def tab_recommend(model, meta, label_to_idx, idx_to_label):
    st.subheader("Recommendations that depend on order")
    st.caption(
        "Pick games in the order you'd have bought them — the last one is the most "
        "recent. Then hit **Shuffle** to see the *same games in a different order* "
        "produce a different next-item prediction."
    )

    default = [idx_to_label[i] for i in CANARIES["JRPG fan"]]
    selection = st.multiselect(
        "Your purchase history (in order — last = most recent)",
        options=list(label_to_idx.keys()),
        default=default,
        key="rec_sel",
    )
    history = [label_to_idx[l] for l in selection]
    games_only = st.checkbox("Show games only (hide accessories / consoles)",
                             value=True, key="rec_go")

    if not history:
        st.info("Select at least one game to get recommendations.")
        return

    c1, c2 = st.columns([1, 4])
    if c1.button("🔀 Shuffle history", use_container_width=True):
        if len(history) >= 2:
            st.session_state["shuf_ids"] = constrained_shuffle(history)
            st.session_state["shuf_base"] = history[:]
        else:
            st.session_state.pop("shuf_ids", None)

    shuf = st.session_state.get("shuf_ids")
    valid_shuffle = shuf is not None and st.session_state.get("shuf_base") == history

    if valid_shuffle:
        left, right = st.columns(2)
        with left:
            st.markdown("**Original order**")
            st.html(order_html(history, meta))
            render_cards(recommend_display(model, meta, history, games_only), meta)
        with right:
            st.markdown("**Shuffled order**")
            st.html(order_html(shuf, meta))
            render_cards(recommend_display(model, meta, shuf, games_only), meta)
        st.caption(
            "Same games, different order → different next-item prediction. A two-tower "
            "model pools the history orderlessly and would return identical lists; the "
            "transformer reads positional embeddings, so the most recent items steer the recs."
        )
    else:
        render_cards(recommend_display(model, meta, history, games_only), meta)
        st.caption("Press **Shuffle history** to compare this list against a reordered history.")


def tab_personas(model, meta):
    st.subheader("Prebuilt personas")
    st.caption("Hand-built histories with coherent taste — a quick way to see clean recs "
               "without assembling a history yourself.")
    name = st.selectbox("Persona", SHOWCASE, key="persona_sel")
    history = CANARIES[name]
    games_only = st.checkbox("Show games only (hide accessories / consoles)",
                             value=True, key="persona_go")
    st.markdown("**History (chronological)**")
    st.html(order_html(history, meta))
    st.markdown("**Top recommendations**")
    render_cards(recommend_display(model, meta, history, games_only), meta)


def tab_similar(model, meta, label_to_idx, idx_to_label):
    st.subheader("Similar items")
    st.caption(SIMILAR_CAVEAT)
    default_seed = 4489  # Resident Evil 4 - PS2 (a clean mid-tail seed)
    seed_label = st.selectbox(
        "Seed game",
        options=list(label_to_idx.keys()),
        index=list(label_to_idx.keys()).index(idx_to_label[default_seed]),
        key="sim_sel",
    )
    seed_id = label_to_idx[seed_label]
    E = item_matrix(model)
    neighbors = most_similar(E, seed_id, N_RECS)
    ids = [i for i, _ in neighbors]
    scores = [s for _, s in neighbors]
    render_cards(ids, meta, scores=scores)


def tab_about():
    st.subheader("About this project")
    st.markdown(
        "A from-scratch PyTorch implementation of **SASRec** "
        "([Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781)), built as a staged "
        "ablation on the Amazon Video Games 5-core 2018 dataset so each architectural "
        "component's contribution can be measured. Stage 3 (full SASRec) lands at "
        "**sampled NDCG@10 = 0.5188** vs the paper's published 0.5360 (within 3.2%)."
    )
    st.markdown(
        "Code & write-up: "
        "[GitHub](https://github.com/nickgreenquist/Amazon-Video-Game-Recommender-System-PyTorch-Transformer-Model)"
    )
    st.markdown("---")
    ablation = ROOT / "results" / "ablation_table.md"
    if ablation.exists():
        st.markdown(ablation.read_text())
    else:
        st.warning("results/ablation_table.md not found.")


# ── app ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Amazon Video Games SASRec", layout="wide")
st.markdown(_STYLE, unsafe_allow_html=True)
st.title("Amazon Video Games — Sequential Recommender (SASRec)")
st.markdown(
    "<small>A transformer that recommends your next game from the <b>order</b> of your "
    "history. Built from scratch in PyTorch on the "
    "<a href='https://nijianmo.github.io/amazon/' target='_blank'>UCSD Amazon 2018</a> "
    "dataset.</small>",
    unsafe_allow_html=True,
)

model = load_model()
meta = load_metadata()
idx_to_label = meta["label"].to_dict()
label_to_idx = {v: k for k, v in idx_to_label.items()}

rec_tab, persona_tab, similar_tab, about_tab = st.tabs(
    ["Recommend", "Personas", "Similar", "About"]
)
with rec_tab:
    tab_recommend(model, meta, label_to_idx, idx_to_label)
with persona_tab:
    tab_personas(model, meta)
with similar_tab:
    tab_similar(model, meta, label_to_idx, idx_to_label)
with about_tab:
    tab_about()
