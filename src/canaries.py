"""Synthetic 'canary' users: hand-built purchase histories run through the
trained Stage 3 SASRec, to eyeball whether recommendations are thematically
sensible before building the Streamlit demo.

Each canary is a chronological list of item indices (real in-corpus games,
titles in the trailing comment). Inference mirrors evaluate.py's full-corpus
branch: left-pad to MAX_SEQ_LEN, take the last position's representation,
score against the whole item table, mask the user's own history, top-k.

Recommendations are shown raw (no games-only filter) with a category tag so
consoles/accessories/guides are visible.

See canary_experiments.py for the history-tuning sweep behind the sparse-genre
personas (horror/fighting/racing).

Usage:
    python -m src.canaries                  # all canaries
    python -m src.canaries --list           # just the persona names
    python -m src.canaries --only vintage   # canaries whose name contains "vintage"
"""

import argparse
import gzip
import html
import json
import re
from pathlib import Path

import torch

from src.dataset import MAX_SEQ_LEN, _left_pad
from src.main import get_device
from src.models.stage3_sasrec import SASRec

ROOT = Path(__file__).resolve().parents[1]
META_GZ = ROOT / "data" / "raw" / "meta_Video_Games.json.gz"
CKPT = ROOT / "saved_models" / "stage3_best.pth"
TOP_K = 20

# Hand-built personas. Games are chosen from the higher-interaction part of the
# corpus (counts in trailing comments); the median item has only ~12
# interactions, so low-count picks give noisy recs. Histories are chronological
# because the model conditions strongly on the most recent item — most personas
# are platform/era-coherent, but a few (Western RPG, JRPG, Fighting) deliberately
# span hardware to test whether genre signal survives a platform mix.
#
# Two groups:
#   SHOWCASE   — dense, coherent genres that recommend cleanly; demo-ready.
#   DIAGNOSTIC — sparse genres (horror/fighting/racing) whose recs collapse to
#                platform popularity no matter the history (see
#                canary_experiments.py). Kept for analysis; do NOT feature them
#                in the deployed Streamlit demo.
CANARIES = {
    # ---- SHOWCASE ----
    "Western RPG fan": [
        7322,   # The Witcher: Enhanced Edition - PC        (108)
        7088,   # Alpha Protocol - PS3                      (58)
        7101,   # Two Worlds II - Xbox 360                  (63)
        9637,   # Kingdoms of Amalur: Reckoning - PC        (65)
        11939,  # Dragon's Dogma: Dark Arisen - Xbox 360    (75)
        12441,  # Fable Anniversary                         (83)
    ],
    "JRPG fan": [
        136,    # Final Fantasy VII                         (296)
        364,    # Final Fantasy VIII                        (262)
        6976,   # Shin Megami Tensei: Persona 3 FES - PS2   (83)
        7022,   # Tales of Vesperia - Xbox 360              (92)
        4978,   # Final Fantasy XIII - PS3                  (384)
        8647,   # NieR - PS3                                (59)
        11351,  # Dragon's Crown - PS Vita                  (67)
        16727,  # Persona 5 - PS4                           (101)
    ],
    "PC strategy fan": [
        415,    # Age of Empires II: Age of Kings - PC      (67)
        2439,   # Age of Mythology - PC                     (48)
        3674,   # Rome: Total War - PC                      (51)
        7239,   # Empire: Total War - PC                    (76)
        7444,   # Torchlight                                (75)
        9672,   # Sid Meier's Civilization V                (161)
    ],
    "Nintendo Wii U fan": [
        8660,   # New Super Mario Bros. Wii                 (467)
        12454,  # Super Mario 3D World - Wii U              (354)
        12456,  # Mario Kart 8 - Wii U                      (600)
        12467,  # Super Smash Bros. - Wii U                 (440)
        13892,  # Splatoon                                  (256)
        13978,  # The Legend of Zelda: Breath of the Wild - Wii U   (206)
    ],
    "Pokemon / 3DS handheld fan": [
        10359,  # Mario Kart 7                              (417)
        10353,  # Pokemon X                                 (346)
        10355,  # Pokemon Y                                 (297)
        12989,  # The Legend of Zelda: A Link Between Worlds 3D     (335)
        12468,  # Super Smash Bros. - 3DS                  (440)
        13722,  # Pokemon Alpha Sapphire - 3DS              (264)
    ],
    "Sony handheld fan (PSP / Vita)": [
        3965,   # Grand Theft Auto: Liberty City Stories - PSP     (89)
        7004,   # Crisis Core: Final Fantasy VII - PSP             (112)
        6066,   # God of War: Chains of Olympus - PSP             (93)
        6248,   # Final Fantasy Tactics: War of the Lions - PSP   (78)
        10202,  # Uncharted: Golden Abyss - PS Vita               (221)
        10223,  # Killzone Mercenary - PS Vita                     (151)
        11245,  # Assassin's Creed III: Liberation - PS Vita      (140)
    ],
    "Vintage gamer (PS2)": [
        1936,   # Grand Theft Auto III                      (293)
        1865,   # Devil May Cry                             (122)
        1823,   # Metal Gear Solid 2: Sons of Liberty       (182)
        2067,   # Final Fantasy X                           (393)
        2287,   # Kingdom Hearts                            (328)
        3735,   # God of War - PS2                          (220)
        4162,   # Shadow of the Colossus - PS2              (175)
    ],
    "Vintage PC gamer": [
        211,    # StarCraft Battle Chest - PC               (112)
        390,    # Diablo 2 - PC                             (96)
        3511,   # RollerCoaster Tycoon 2 - PC               (77)
        2144,   # WarCraft III: Reign of Chaos - PC         (85)
        3178,   # SimCity 4 Deluxe Edition - PC             (120)
        2388,   # Doom 3 - PC                               (174)
        2453,   # Half-Life 2 - PC                          (147)
    ],

    # ---- DIAGNOSTIC (sparse genres; keep OUT of the deployed demo) ----
    # None of these clear ~2/5 on-genre in the top-5, no matter the history
    # (canary_experiments.py swept 18 variants). The lists below are the best
    # found; deeper ranks drift to platform popularity. Good for the "where the
    # model struggles" story, not for the showcase.

    # All-Silent-Hill won on top-5 (Dead Space #1, Resident Evil 6 #4); niche.
    "Horror / zombie fan": [
        1807,   # Silent Hill 2 - PS2                       (110)
        2658,   # Silent Hill 3 - PS2                       (87)
        3415,   # Silent Hill 4: The Room - PS2             (55)
        6965,   # Silent Hill: Homecoming - PS3             (90)
    ],
    # Niche cross-era fighters; two denser anchors (SC IV, MvC3) mid-history
    # surface a couple of fighters, ending on the niche Soul Calibur V.
    "Fighting-game fan": [
        170,    # Tekken 3 - PlayStation                    (51)
        146,    # Soul Calibur - Dreamcast                  (52)
        2541,   # Marvel vs. Capcom 2                        (48)
        1332,   # Tekken Tag Tournament - PS2               (82)
        1937,   # Dead or Alive 3 - Xbox                    (83)
        2354,   # Mortal Kombat: Deadly Alliance - PS2      (62)
        4123,   # Soulcalibur 3 - PS2                       (66)
        6696,   # Soul Calibur IV - PS3                     (97)
        8119,   # Tekken 6 - Xbox 360                       (53)
        9231,   # Marvel vs. Capcom 3 - PS3                 (81)
        9318,   # Soul Calibur V - Xbox 360                 (52)
    ],
    # Xbox racing (Forza + Need for Speed); NFS lands at #1.
    "Racing fan": [
        9317,   # Forza Motorsport 4 - Xbox 360             (162)
        9387,   # Need for Speed: Hot Pursuit - Xbox 360    (115)
        10264,  # Forza Horizon - Xbox 360                  (122)
        12893,  # Forza Motorsport 5 - Xbox One             (134)
        12376,  # Need for Speed Rivals - Xbox One          (111)
        16790,  # Forza Horizon 3 - Xbox One                (104)
    ],
}

# The dense/coherent personas that recommend cleanly — the only ones featured in
# the deployed Streamlit demo (Tab 2). The DIAGNOSTIC three (horror/fighting/
# racing) are deliberately excluded: their recs collapse to platform popularity
# (see canary_experiments.py) and they are kept only for analysis.
SHOWCASE = [
    "Western RPG fan",
    "JRPG fan",
    "PC strategy fan",
    "Nintendo Wii U fan",
    "Pokemon / 3DS handheld fan",
    "Sony handheld fan (PSP / Vita)",
    "Vintage gamer (PS2)",
    "Vintage PC gamer",
]


_HTML_TAG = re.compile(r"<[^>]+>")


def _clean(s):
    """Strip residual HTML tags/entities the 2018 metadata dump leaves behind."""
    return html.unescape(_HTML_TAG.sub("", s or "")).strip()


def load_metadata():
    """idx -> (title, category_tag), restricted to the trained corpus."""
    asin2idx = json.loads((ROOT / "data/processed/item_id_to_idx.json").read_text())
    idx2meta = {}
    with gzip.open(META_GZ, "rt") as f:
        for line in f:
            r = json.loads(line)
            i = asin2idx.get(r.get("asin"))
            if i is None:
                continue
            title = _clean(r.get("title")) or "(no title)"
            cat = [c for c in (_clean(c) for c in (r.get("category") or [])) if c]
            tag = " > ".join(cat[1:3]) if len(cat) > 1 else _clean(r.get("main_cat"))
            idx2meta[i] = (title, tag)
    return idx2meta


def name(idx2meta, i):
    title, tag = idx2meta.get(i, ("(unknown)", ""))
    return f"{title}  [{tag}]" if tag else title


CAPS = (3, 45, 34)  # max width per column: #, Title, Category


def _fit(text, width):
    return text if len(text) <= width else text[: width - 1] + "…"


def _render_table(headers, rows, caps):
    w = [min(caps[c], max(len(r[c]) for r in (headers, *rows))) for c in range(len(headers))]
    bar = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    fmt = lambda r: "| " + " | ".join(_fit(r[c], w[c]).ljust(w[c]) for c in range(len(w))) + " |"
    print(bar)
    print(fmt(headers))
    print(bar)
    for r in rows:
        print(fmt(r))
    print(bar)


def print_rec_table(idx2meta, item_ids):
    """Bordered table: rank | title | category, with long cells truncated."""
    rows = [(str(rank), *idx2meta.get(i, ("(unknown)", "")))
            for rank, i in enumerate(item_ids, 1)]
    _render_table(("#", "Title", "Category"), rows, CAPS)


def recommend(model, history, n_items, device, k=TOP_K):
    """Top-k item indices for one history, with its own items masked out."""
    seq = torch.tensor([_left_pad(history, MAX_SEQ_LEN)], dtype=torch.long, device=device)
    with torch.no_grad():
        user_emb = model(seq)[:, -1, :]                       # (1, D)
        scores = (user_emb @ model.item_embedding.weight[1:].T).squeeze(0)  # (n_items,)
    seen = torch.tensor([h - 1 for h in history], dtype=torch.long, device=device)
    scores[seen] = float("-inf")                              # column j -> item id j+1
    top = torch.topk(scores, k).indices + 1                  # back to 1-indexed ids
    return top.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list persona names and exit")
    p.add_argument("--only", help="run only canaries whose name contains this (case-insensitive)")
    args = p.parse_args()

    if args.list:
        for persona in CANARIES:
            print(persona)
        return

    personas = {n: h for n, h in CANARIES.items()
                if not args.only or args.only.lower() in n.lower()}
    if not personas:
        raise SystemExit(f"no canary matches --only {args.only!r}. "
                         f"options: {', '.join(CANARIES)}")

    device = get_device()
    n_items = len(json.loads((ROOT / "data/processed/item_id_to_idx.json").read_text()))
    idx2meta = load_metadata()

    model = SASRec(n_items, hidden_dim=64, n_blocks=2, n_heads=1,
                   max_seq_len=MAX_SEQ_LEN, dropout=0.5).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f"device: {device}  |  Stage 3 SASRec  |  {n_items:,} items\n")

    for persona, history in personas.items():
        print("=" * 78)
        print(f"CANARY: {persona}")
        print("-" * 78)
        print("history (chronological):")
        for i in history:
            print(f"    {name(idx2meta, i)}")
        print(f"\ntop-{TOP_K} recommendations:")
        print_rec_table(idx2meta, recommend(model, history, n_items, device))
        print()


if __name__ == "__main__":
    main()
