# deployment_plan.md

Authoritative spec for the **Streamlit demo / deployment stage**. The model build
(Stages 0–3) is specified in `implementation_plan.md` and is complete; this
document covers only serving the trained Stage 3 SASRec as an interactive demo.

Build in the order in §9 and do not advance to a tab until the previous one is
working and verified.

> **Status of this doc:** rewritten after a full design grill (2026-05-31). All
> 13 load-bearing decisions are resolved and recorded inline (see the
> **Decisions** callouts and §12). It is meant to be coded against directly.
>
> **Implemented 2026-05-31** (build order §9 steps 1–5 done and verified;
> step 6 deploy is the manual Streamlit Cloud action). Shipped: packageified
> `src/`, `scripts/build_artifacts.py` → `serving/` (5.3 MB total), `src/serving.py`,
> `streamlit_app.py` (4 tabs). Parity verified across all 11 personas; JRPG
> shuffle moves 8–10/10 recs. Divergences from this spec, all deliberate:
> - **`.gitignore` negation must be on its own line.** The §4.2 snippet showed
>   `!serving/*.pth  # comment` inline — git treats the `#` and everything after
>   as part of the pattern, so the negation silently fails. Comment goes above it.
> - **`search_titles` exists in `serving.py` but the UI doesn't call it.**
>   `st.multiselect` over all labels already type-to-filters (the exact Steam
>   pattern), so a custom search box was unnecessary. The helper stays per §7.
> - **Label disambiguation is collision-only:** append `(#idx)` to any base that
>   isn't unique (cleaner than the literal "platform empty OR collides" rule, and
>   still guarantees every label → exactly one idx). Verified unique catalog-wide.
> - **Slim parquet is ~0.7 MB** (not the 2–3 MB estimate); 6 cols, 16,882 rows.
> - **Covers use `background-size: contain` on a square panel** (Amazon product
>   photos have mixed aspect ratios; `contain` avoids cropping) rather than
>   Steam's `cover` banner aspect — same `_cover_div` background-image technique.

---

## 0. Before you start (read first)

**Entry point: begin at §9 step 1** (packageify `src/` + `.gitignore`). Everything
downstream imports `from src.X`, so *nothing else works until that lands.* Do not
start with the parquet or the UI.

**Local preconditions (these are gitignored, so a fresh clone won't have them):**
`scripts/build_artifacts.py` (§5) consumes two files that must already exist on
disk:
- `data/item_metadata.parquet` — produced by `python scripts/build_item_metadata.py`
  (needs the raw `data/raw/*.json.gz` download).
- `saved_models/stage3_best.pth` — produced by `python -m src.main --stage 3`
  (the trained Stage 3 checkpoint).

On the author's current machine both already exist — check first
(`ls data/item_metadata.parquet saved_models/stage3_best.pth`). Only if missing
(e.g. fresh clone) do you need to regenerate them via the commands above before
§5. They are inputs to the build, never committed themselves; only their slimmed
serving derivatives (`serving/`) are committed.

---

## 1. Goal

Deploy an interactive Streamlit app that makes the **transformer's one
differentiating capability visible**: recommendations that depend on the *order*
of a user's history. Everything else (retrieval, similar-items, personas) is
supporting cast.

The deliverable is a deployed URL plus the resume sentence it backs:

> Implemented SASRec from scratch with a staged ablation proving each
> architectural component's contribution, and deployed an interactive demo where
> reordering a purchase history changes the recommendations — something a
> two-tower model structurally cannot do.

## 2. Why this design (portfolio framing)

This repo's novelty, relative to the author's existing portfolio (two-tower
retrieval for movies/books/Steam; a Steam ranker — none using transformers), is
**the transformer and its sequence-awareness**. The demo must therefore lead with
order-sensitivity, not with generic "here are some recs" (which the existing
two-tower demos already show). The ablation table (Stage 1→2→3) is the secondary
proof of rigor — most portfolio projects have no ablation story; this one does.

This stage deliberately mirrors the author's **Steam two-tower repo**
(full path: `/Users/nickgreenquist/Documents/Game-Recommender-System-PyTorch-TwoTower-Model`;
key files to copy patterns from: `streamlit_app.py` — `st.multiselect` input @ ~L610,
`_cover_div` cover rendering @ ~L194 rendered via `st.html(...)` @ ~L280; root
`main.py` dispatcher; `.gitignore` `!serving/*.pth`; empty `src/__init__.py`)
for everything structural
(repo layout, `.gitignore`, artifact-commit pattern, `st.multiselect` input,
cover-art rendering) so deployment follows an already-proven path. Divergences
from Steam are only where the transformer/Amazon-data context demands them, and
are called out where they occur.

## 3. Non-goals (do not do these)

- **No new model training and no new features.** The demo serves the *existing*
  `stage3_best.pth` as-is. Rich item/user features, genre pooling, a ranking
  head, BERT4Rec — all out of scope (and the first three are already covered by
  other projects in the portfolio).
- **No re-scoring logic that diverges from `evaluate.py` / `canaries.py`.** The
  app reuses the exact serving math already written and verified (§7). A demo
  that scores differently from the offline metrics is a bug — guarded by the
  parity gate in §9 step 2.
- **No dependency on the multi-GB raw download at serve time.** Metadata is
  precomputed into a small parquet (§5). Streamlit Community Cloud cannot host
  `data/raw/`.
- **No games-only filtering anywhere.** Training, scoring, *and* display all use
  the full corpus — accessories/consoles/guides are recommended alongside games
  (§6, and the invariant in CLAUDE.md).
- **No second model served.** Only Stage 3 ships. The ablation is a *static
  table* (Tab 4); we do **not** serve Stage 1/2 checkpoints for live multi-stage
  scoring. No attention-weight visualization. (Decisions #4, #12.)

## 4. Repo layout (mirrors the Steam repo)

The Steam repo keeps both entry points at repo root and `src/` as a true package.
We adopt that exactly. **There is no `streamlit/` directory** — a top-level dir
named `streamlit/` would shadow the `streamlit` pip package on import.

```
streamlit_app.py              ← NEW. repo-ROOT entry; all UI/tabs. imports from src.*
src/
├── __init__.py               ← NEW. empty package marker (Steam's is 0 bytes)
├── serving.py                ← NEW. the 3 genuinely-new serving helpers (§7)
├── canaries.py               ← EXISTING. gains a SHOWCASE name-list (§8 Tab 2);
│                                imports become `from src.X` (§4.1)
├── similar_items.py          ← EXISTING. reused as-is; imports become `from src.X`
├── main.py, train.py, dataset.py, evaluate.py, preprocess.py, canary_experiments.py
└── models/                   ← EXISTING (already a package). imports → `from src.models.X`
scripts/
├── build_item_metadata.py    ← EXISTING. produces data/item_metadata.parquet (rich)
└── build_artifacts.py        ← NEW. thin transform → serving/item_metadata.parquet (§5)
serving/                      ← NEW. COMMITTED artifacts (Steam uses this dir name)
├── stage3_best.pth           ← copied from saved_models/ (~4.5 MB)
└── item_metadata.parquet     ← slim, idx-keyed (~2–3 MB)
```

`requirements.txt` gains `streamlit` (unpinned, matching the Steam repo's working
deploy). Everything else (torch, pandas, pyarrow, numpy) is already present. No
Python-version pin is needed — Streamlit Cloud's default works (Steam ships none).

### 4.1 Packageify `src/` (Decision #2)

Today `src/` has no `__init__.py` and its modules use **bare** imports (`from
dataset import …`), which only resolve when `src/` is the CWD. That works for
`python src/main.py` but **cannot** be imported as `from src.canaries import …`
from a root-level `streamlit_app.py`. We convert `src/` to a proper package — no
path-shim hacks.

**Steps:**
1. Add empty `src/__init__.py`.
2. Rewrite these **9 intra-`src` import statements** to absolute `from src.X`:
   - `src/main.py:15-19` — `dataset`, `models.stage1_bag_of_items`,
     `models.stage2_attention_no_pos`, `models.stage3_sasrec`, `train`
   - `src/train.py:23` — `from evaluate import evaluate`
   - `src/canaries.py:31-33` — `dataset`, `main`, `models.stage3_sasrec`
   - `src/similar_items.py:27` — `from canaries import (...)`
   - `src/canary_experiments.py:22` — `from canaries import (...)`
   - `src/models/stage2_attention_no_pos.py:17` and
     `src/models/stage3_sasrec.py:20` — `from models.transformer_block import …`
3. Run style changes from `python src/X.py` → `python -m src.X`. Update the
   command blocks in **CLAUDE.md** (§Commands) and **README.md**:
   - `python src/preprocess.py`        → `python -m src.preprocess`
   - `python src/main.py --stage N`    → `python -m src.main --stage N`
   - `python src/canaries.py …`        → `python -m src.canaries …`
   - `python src/canary_experiments.py`→ `python -m src.canary_experiments`
   - `python src/similar_items.py …`   → `python -m src.similar_items`

**Verify:** `python -m src.main --stage 3 --epochs 1` still trains a step;
`python -c "from src.canaries import recommend"` from repo root succeeds.
`ROOT = Path(__file__).resolve().parents[1]` in canaries.py still resolves to the
repo root under package import (parents[1] of `src/canaries.py`), so no path
constants change.

### 4.2 `.gitignore` (Decision #1)

The current `.gitignore` globally ignores `*.pth`, `*.pt`, `data/`,
`saved_models/`, and a stale `streamlit/serving/`. As-is it would **silently
swallow every committed serving artifact** — `git add serving/stage3_best.pth`
would be a no-op and Cloud would deploy with a missing checkpoint. Fix with a
negation scoped to `serving/` (the exact pattern the Steam repo uses and proves):

```gitignore
*.pth
!serving/*.pth          # the ONE checkpoint we deploy is intentionally tracked
```
- Delete the stale `streamlit/serving/` line (no such dir anymore).
- `serving/item_metadata.parquet` is matched by no ignore rule (the `data/` rule
  doesn't reach `serving/`), so it commits normally.
- `saved_models/` stays ignored — the *serving copy* lives in `serving/`, not
  `saved_models/`.

**Verify:** `git check-ignore serving/stage3_best.pth` prints nothing (i.e. it is
NOT ignored); `git add serving/ && git status` shows both files staged.

## 5. Serving artifacts build (`scripts/build_artifacts.py`, run locally) — Decisions #5, #6

A one-shot local script (not run at serve time) produces the two files in
`serving/`. It is a **thin transform of the already-rich
`data/item_metadata.parquet`** (output of `scripts/build_item_metadata.py`) — it
does **not** re-implement metadata extraction, `kind` classification, or image
picking. (DRY: that logic lives in one place.)

**`stage3_best.pth`** — copy `saved_models/stage3_best.pth` → `serving/` as-is.

**`item_metadata.parquet`** — transform:
1. Read `data/item_metadata.parquet` (cols include `asin, title, platform, kind,
   image, interaction_count, …`).
2. Join `data/processed/item_id_to_idx.json` → add an `idx` column; **filter to
   the 16,882 corpus asins** (drops the ~524 extra 5-core-review asins that
   preprocess dedup/re-filtered out of the trained corpus).
3. **Add 17 placeholder rows** for corpus items absent from the metadata table.
   These 17 asins are genuinely absent from raw `meta_Video_Games.json.gz`
   (0/17 present — confirmed), so they are unrecoverable, **not a build bug**.
   Placeholder = `title="(no metadata)", platform="", kind="unknown", image=""`.
4. **Slim to exactly 6 columns: `idx, title, platform, kind, image,
   interaction_count`.** Drop `description, also_buy, also_view, price,
   sales_rank, brand, category` — the UI renders none of them; they are the bulk
   of the 18.6 MB. **Keep `interaction_count`** (carried straight from the rich
   source, NOT re-derived): it orders the demo pickers by popularity so the most
   recognizable games surface first instead of opaque ASIN order (§8.1). Result
   ≈ 0.7 MB.
5. Write `serving/item_metadata.parquet`, **idx-keyed, exactly 16,882 rows.**

**idx convention (do not get this wrong):** `item_id_to_idx.json` values are
**1-indexed, 1..16,882** (0 is reserved for padding). These values ARE the item
ids that `recommend()` returns (`top.indices + 1`) and that the canary history
lists use directly. So the parquet `idx` column = the raw `item_id_to_idx` value,
**no off-by-one shift** — a lookup of `recommend()`'s output against `idx` is a
direct join. The 17 placeholder rows take whatever idx their asin maps to (all in
1..16,882).

**Image source (Decision #6):** the `image` column is `imageURLHighRes` →
`imageURL` (HighRes first) as picked by `build_item_metadata.py:pick_image`.
**Keep HighRes-first** — the old plan said "imageURL first," which was wrong and
gives lower-res art. URLs are Amazon CDN links
(`images-na.ssl-images-amazon.com/...`); **verified live 2026** (24/24 sampled
top/mid/tail items return HTTP 200), so they are hotlinked at serve time, never
downloaded/hosted. ~86% of items have art; the rest (2,420 empty + 17
placeholder) render the placeholder panel (§8.5).

**Verify (§9 step 1):** row count == 16,882; `idx` is unique and covers
1..n_items used by the corpus; spot-check 3 titles + their image URLs resolve;
`serving/` total < 50 MB.

## 6. Recommendations span all product types (no games-only filter) — Decisions #6, #10

The model trains and scores on the **full 5-core corpus** (consoles, accessories,
guides included — CLAUDE.md invariant; `kind` shows ~27% non-game: 4,114
accessory + 379 console + 232 unknown + 19 other). The demo displays those
recommendations **raw** — every video-game-category product type is eligible, so
accessories/consoles/guides surface alongside games. There is **no** games-only
toggle or filter at any layer.

**Implementation:** score the full corpus and take the first **10** for display.
The Tab 1 side-by-side lists are always exactly 10 rows (no filter that could
leave a short stub list).

## 7. Inference contract (must match the offline path exactly)

`src/serving.py` holds **only the 3 genuinely-new helpers**; the scoring math is
**imported from the existing modules**, not re-implemented:
- `recommend(model, history, n_items, device, k)` — import from `src.canaries`
- `most_similar(E, seed_id, k)` and `item_matrix(model)` — import from
  `src.similar_items`

`src/serving.py` adds:
1. **`load_model()`** — construct Stage 3 SASRec, load weights from
   `serving/stage3_best.pth` (NOT `saved_models/`), `.eval()`. Cached with
   `@st.cache_resource`. Construction mirrors `src/canaries.py:254`:
   `SASRec(n_items, hidden_dim=64, n_blocks=2, n_heads=1, max_seq_len=50,
   dropout=0.5)`.
2. **`load_metadata()`** — read `serving/item_metadata.parquet` into an
   idx-keyed structure (dict or DataFrame). **Replaces `canaries.load_metadata()`**
   (which reads the raw gz and is never called at serve time). Cached with
   `@st.cache_data`.
3. **`search_titles(query, k)`** — substring title match over the metadata,
   returning `(idx, label)` where `label = "{title} — {platform}"`. Backs the
   `st.multiselect` options (§8.1). (Mirrors `similar_items.py:85`'s `--search`.)

Invariants that must hold (from CLAUDE.md), all already satisfied by the imported
functions:
- Item idx 0 is padding; real items are 1-indexed. `padding_idx=0`.
- Sequences are **left-padded** to `MAX_SEQ_LEN=50`, dropping oldest items.
- Recommend = last-position output `model(seq)[:, -1, :]` scored against
  `model.item_embedding.weight[1:].T`, the user's own history masked to `-inf`,
  then top-k. (Exactly `recommend()`.)
- Similar-items = **cosine** similarity over `item_embedding.weight[1:]`, seed
  masked out. No forward pass, no positions. (Exactly `most_similar()`.)

The order-sensitivity demo (Tab 1) relies on the fact that `model(seq)` reads
positional embeddings, so permuting `history` genuinely changes `seq` and thus the
output — no special code needed beyond letting the user reorder the list.

**Empirically validated (2026-05-31):** reversing a showcase persona's history
changes **4–10 of the top-10** recs (never 0). Order-sensitivity is
last-item-dominated (e.g. PC-strategy reverses to 10/10 changed but moves only
1/10 when just the last two swap) — which is *why* the Tab 1 control is a
constrained shuffle that never leaves the last item unchanged (§8.1).

## 8. Tabs

### 8.1 Tab 1 — Recommend (the headline) — Decisions #7, #8, #9

- **Input = `st.multiselect`** over catalog titles (Steam's pattern,
  `streamlit_app.py:610`). Streamlit returns selections **in click order**, which
  IS the chronological history (last selected = most recent → matches the
  training convention). No search-box+add-button, **no drag, no up/down
  buttons.** Remove = deselect.
- **Picker options are ordered by `interaction_count`, descending** (most popular
  first) — so the type-to-filter list surfaces recognizable games instead of the
  opaque ASIN order the idx-keyed parquet would otherwise give. Same ordering for
  the Tab 3 seed `selectbox`.
- **Label disambiguation:** 824 corpus items share a title with another item
  ("Spider-Man" spans 7 platforms). Build multiselect labels as
  `"{title} — {platform}"`, appending a short idx/asin suffix when `platform` is
  empty or the pair still collides, so every label maps to exactly one idx.
- **Default seed history = the "JRPG fan" canary** (`src/canaries.py`). It is the
  most robustly order-sensitive history (9/10 changed on full reversal, 8/10 even
  on a minimal reorder), so the page demonstrates the point on first paint and
  any shuffle visibly moves the list. (Backups: "Sony handheld", "Western RPG".)
- Show top-10 recommendations (all product types — §6; no games-only filter).
- **Order-sensitivity panel (the whole point):** a **shuffle** button drives a
  **side-by-side** view — original recs (left) vs reordered recs (right), with
  each history's order shown above its column. The shuffle is a **constrained
  shuffle: reject any permutation that leaves the last item unchanged** (the user
  never knows it's constrained; this guarantees a visible delta even for
  last-item-dominated histories). A short caption names what happened ("same
  items, different order → different next-item prediction") so a non-expert gets
  the significance.

### 8.2 Tab 2 — Personas — Decision #12

- Dropdown of **SHOWCASE personas only**. Add a `SHOWCASE` name-list constant to
  `src/canaries.py` (the 8 dense/coherent personas) rather than a new
  `personas.py`; the app does `from src.canaries import CANARIES, SHOWCASE`. The
  DIAGNOSTIC three (horror/fighting/racing) stay out of the deployed demo per
  CLAUDE.md and `[[streamlit-exclude-sparse-genre-canaries]]`.
- Selecting one loads its history into the same view as Tab 1 and shows recs.
  Good for a visitor who doesn't want to build a history by hand.

### 8.3 Tab 3 — Similar to

- User picks a seed item (same search/multiselect input); show top-10 cosine
  neighbors with scores and cover art via `most_similar` / `item_matrix`
  (imported from `src.similar_items`).
- Caption that this reads the **embedding geometry directly** (no sequence
  model), and that "similar" means co-occurrence context — dominated by platform,
  then genre — not a content model. **Reuse the honest caveat already written in
  `similar_items.py`'s module docstring.**

### 8.4 Tab 4 — About / Ablation — Decision #12

- The ablation table (Stage 1 → 2 → 3 deltas) read directly from
  `results/ablation_table.md` (it is already rich enough — full hyperparameters,
  sampled+full metrics, deltas). No copy into `serving/`.
- One-line architecture summary per stage and the headline numbers (Stage 3
  sampled NDCG@10 = 0.5188 vs published 0.5360).
- Link to the GitHub README and the SASRec paper.

### 8.5 Cover-art rendering (all tabs) — Decision #6

Render covers with the **Steam `_cover_div` pattern** (`streamlit_app.py:194`): a
`background-image` div with a `background-color` fallback — **not** `st.image()`
or a markdown `<img>`. Two reasons (both from the Steam repo's hard-won notes):
1. A missing/empty URL (our 2,420 empties + 17 placeholders) degrades to a clean
   colored panel, **never a broken-image icon** — satisfies "render a
   placeholder, never crash."
2. Sidesteps Streamlit's markdown CSS forcing `margin:auto` on `<img>`.

Amazon covers have no derivable URL pattern (unlike Steam's id-derived headers),
so the stored `image` column is the only source — which is why §5 keeps it.

## 9. Build order (each step verifiable before the next)

1. **Packageify `src/` (§4.1) + `.gitignore` (§4.2).**
   **Verify:** `python -m src.main --stage 3 --epochs 1` trains; `python -c "from
   src.canaries import recommend"` works from root; `git check-ignore
   serving/stage3_best.pth` prints nothing.
2. **`scripts/build_artifacts.py` → `serving/item_metadata.parquet` + checkpoint
   copied.**
   **Verify:** row count == 16,882; spot-check 3 titles + image URLs resolve;
   `serving/` < 50 MB.
3. **`src/serving.py` load + `recommend()` parity (Decision #11).**
   **Verify:** for one SHOWCASE persona, the app's recs == `python -m src.canaries
   --only <name>` top-10. The demo applies no display filter, so this is a direct
   comparison. If the lists differ, the serving path diverged — fix before
   building UI.
4. **Tab 1** minimal (multiselect history → recs), then add the shuffle +
   side-by-side order-sensitivity panel.
   **Verify:** shuffling the JRPG seed visibly changes the rec list (expect
   ~8–9/10 rows change).
5. **Tab 2** personas, **Tab 3** similar-to, **Tab 4** about.
   **Verify (Tab 3):** app neighbors == `python -m src.similar_items --id <seed>`.
6. **Deploy.**

## 10. Deployment — Decision #1, #13

- **Streamlit Community Cloud**, pointed at this repo, **`streamlit_app.py`** (repo
  root) as the entry. Artifacts are committed under `serving/` (both < 100 MB per
  file, the GitHub hard limit), so no external storage / LFS / download-at-startup
  — deploy reads whatever is on `main`. Same pattern the author shipped for the
  book/game/movie recommenders.
- Cache the model with `@st.cache_resource` and the metadata parquet with
  `@st.cache_data` so they load once per container, not per interaction. CPU
  inference is fine at this model size; no GPU.
- `get_device()` already prefers MPS/CUDA then CPU; on Cloud it resolves to CPU.
- `requirements.txt` += `streamlit` (unpinned). No Python-version pin.

## 11. Out of scope (explicitly cut during the design grill)

- Live multi-stage (Stage 1/2/3) scoring in the UI — ablation is a static table
  (Decision #12). Stage 1/2 checkpoints are NOT shipped.
- Attention-weight visualization (Decision #12).
- Drag-and-drop / up-down manual reordering and the `streamlit-sortables`
  dependency — the shuffle button is the reorder UX (Decision #8).
- Any games-only filtering of training, scoring, or display (CLAUDE.md invariant).

## 12. Resolved decisions (quick index)

1. **Commit artifacts to `main`** via `!serving/*.pth` negation; deploy from main. (§4.2, §10)
2. **Packageify `src/`** (`from src.X`, `python -m src.main`); no path shim. (§4.1)
3. **Root `streamlit_app.py`**, no `streamlit/` dir; `src/serving.py` for new helpers only. (§4, §7)
4. **`serving/` = 2 files** (`stage3_best.pth` + `item_metadata.parquet`); single model. (§4, §11)
5. **Build fresh idx-keyed parquet** via thin transform; 17 placeholder rows; slim to 6 cols (incl. `interaction_count` for popularity-ordered pickers). (§5)
6. **All product types shown** (no games-only filter); **HighRes-first** images; CDN URLs verified live. (§5, §6, §8.5)
7. **Order-sensitivity empirically validated** (4–10/10); **JRPG default seed**. (§7, §8.1)
8. **`st.multiselect` ordered input**; `"{title} — {platform}"` labels; no drag/buttons. (§8.1)
9. **Constrained shuffle** (never same last item) → **side-by-side** comparison. (§8.1)
10. **All product types in recs** — top-10 raw, no games-only filter. (§6)
11. **Parity gate** compares the app's (unfiltered) serving math to `canaries.py`. (§9 step 3)
12. **Cut** live multi-stage + attention viz; ablation static; Tab 3 reuses honest caveat. (§8, §11)
13. **`requirements.txt` += `streamlit`** (unpinned); no Python pin. (§4, §10)
```
