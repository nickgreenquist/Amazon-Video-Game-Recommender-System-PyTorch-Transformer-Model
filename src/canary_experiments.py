"""Side experiment: find histories that make the *sparse* genres (fighting,
horror, racing) recommend on-genre instead of collapsing to platform
popularity.

Two hypotheses were swept: console/era (does the right platform help?) and
franchise loyalty (does an all-one-brand history pull that brand?). Each variant
is scored by how many of the top-5/10/20 recs are in-genre (keyword match on the
title). Conclusion: neither lever works — these genres are too sparse to cluster,
so the best top-5 hit rate is only ~1-2/5. The winners are promoted into
canaries.py; the rest stays here as evidence.

Usage:
    python -m src.canary_experiments                 # all personas
    python -m src.canary_experiments --only fighting
"""

import argparse
import json

import torch

from src.canaries import (CKPT, MAX_SEQ_LEN, ROOT, SASRec, get_device,
                          load_metadata, recommend)

TOP_K = 20

# {persona: {variant_label: [item_idx, ...]}}.
# Round 1 (console-era, pure-genre) showed >=4 platforms all collapse to
# platform popularity. Round 2 tests FRANCHISE-PURE histories: does brand
# loyalty pull the brand's other entries + closer genre neighbors?
EXPERIMENTS = {
    "fighting": {
        "PS3 mixed (baseline)": [6841, 6696, 9038, 9231, 8936, 10560],
        "All Street Fighter": [
            6841,   # Street Fighter IV - PS3       (116)
            9038,   # Super Street Fighter IV - PS3 (122)
            8936,   # Street Fighter X Tekken - PS3 (82)
            14633,  # Street Fighter V - PS4        (216)
        ],
        "All Tekken": [
            1332,   # Tekken Tag Tournament - PS2   (82)
            2263,   # Tekken 4 - PS2                (66)
            8119,   # Tekken 6 - X360               (53)
            16795,  # Tekken 7 - PS4                (63)
        ],
        "All Mortal Kombat": [
            2354,   # MK: Deadly Alliance - PS2     (62)
            7174,   # MK vs. DC Universe - PS3      (188)
            11151,  # MK: Komplete Edition - PS3    (304)
            13797,  # Mortal Kombat X - PS4         (260)
        ],
        "All Capcom (SF/MvC)": [
            2541,   # Marvel vs. Capcom 2           (48)
            9231,   # Marvel vs. Capcom 3 - PS3     (81)
            10560,  # Ultimate MvC3 - PS3           (86)
            9038,   # Super Street Fighter IV - PS3 (122)
        ],
    },
    "horror": {
        "PS3 mixed (baseline)": [6600, 9088, 6846, 10239, 6965, 12249],
        "All Resident Evil": [
            4489,   # Resident Evil 4 - PS2         (184)
            6846,   # Resident Evil 5 - PS3         (305)
            10239,  # Resident Evil 6 - PS3         (353)
            16804,  # Resident Evil 7                (255)
        ],
        "All Silent Hill": [
            1807,   # Silent Hill 2 - PS2           (110)
            2658,   # Silent Hill 3 - PS2           (87)
            3415,   # Silent Hill 4: The Room - PS2 (55)
            6965,   # Silent Hill: Homecoming - PS3 (90)
        ],
        "All Dead Space": [
            6600,   # Dead Space - PS3              (213)
            9088,   # Dead Space 2 - PS3            (160)
            10220,  # Dead Space 3 - X360           (174)
        ],
        "All zombie": [
            4853,   # Dead Rising - X360            (154)
            9980,   # Dead Island - X360            (153)
            8656,   # Left 4 Dead 2 - X360          (202)
            12371,  # Dying Light - PS4             (237)
        ],
    },
    "racing": {
        "PS3 mixed (baseline)": [8672, 12311, 9390, 10217, 5659, 9323],
        "All Forza": [
            8677,   # Forza Motorsport 3 - X360     (100)
            9317,   # Forza Motorsport 4 - X360     (162)
            10264,  # Forza Horizon - X360          (122)
            12893,  # Forza Motorsport 5            (134)
            16790,  # Forza Horizon 3 - Xbox One    (104)
        ],
        "All Gran Turismo": [
            1518,   # Gran Turismo 3 A-spec - PS2   (143)
            3082,   # Gran Turismo 4 - PS2          (117)
            8672,   # Gran Turismo 5 - PS3          (181)
            12311,  # Gran Turismo 6 - PS3          (189)
        ],
        "All Need for Speed": [
            9390,   # NFS: Hot Pursuit - PS3        (198)
            10217,  # NFS: Most Wanted - PS3        (178)
            9323,   # NFS: The Run - PS3            (105)
            12377,  # NFS: Rivals                   (248)
        ],
        "Burnout / arcade": [
            3646,   # Burnout 3: Takedown - PS2     (76)
            5659,   # Burnout Paradise - PS3        (119)
            8965,   # Midnight Club - PS3           (93)
            12377,  # NFS: Rivals                   (248)
        ],
    },
}

GENRE_KEYWORDS = {
    "fighting": ["tekken", "soul calibur", "soulcalibur", "street fighter",
                 "mortal kombat", "marvel vs", "capcom", "dead or alive",
                 "injustice", "guilty gear", "blazblue", "king of fighters",
                 "killer instinct", "naruto", "virtua fighter", "skullgirls",
                 "persona 4 arena", "samurai shodown", "dragon ball", "dissidia",
                 "smash bros"],
    "horror": ["resident evil", "silent hill", "dead space", "fatal frame",
               "siren", "outlast", "amnesia", "soma", "alien: isolation",
               "alien isolation", "evil within", "condemned", "f.e.a.r",
               "dead rising", "dead island", "left 4 dead", "dying light",
               "until dawn", "walking dead", "slender", "layers of fear",
               "state of decay", "zombi", "murdered", "clock tower",
               "dino crisis", "project zero", "daylight"],
    "racing": ["gran turismo", "forza", "need for speed", "burnout", "dirt",
               "project cars", "grid", "midnight club", "motogp", "driveclub",
               "nascar", "f1 ", "wrc", "split/second", "ridge racer", "mx vs",
               "test drive", "racing", "kart", "rallisport", "trackmania"],
}


def genre_hits(idx2meta, item_ids, keywords):
    """Indices in item_ids whose title matches any genre keyword."""
    hit = []
    for rank, i in enumerate(item_ids, 1):
        title = idx2meta.get(i, ("", ""))[0].lower()
        if any(k in title for k in keywords):
            hit.append(rank)
    return hit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="run only this persona (fighting/horror/racing)")
    args = p.parse_args()

    device = get_device()
    n_items = len(json.loads((ROOT / "data/processed/item_id_to_idx.json").read_text()))
    idx2meta = load_metadata()
    model = SASRec(n_items, hidden_dim=64, n_blocks=2, n_heads=1,
                   max_seq_len=MAX_SEQ_LEN, dropout=0.5).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f"device: {device}  |  Stage 3 SASRec  |  scoring genre hits in top-{TOP_K}\n")

    personas = {k: v for k, v in EXPERIMENTS.items()
                if not args.only or args.only.lower() in k}
    for persona, variants in personas.items():
        kws = GENRE_KEYWORDS[persona]
        print("#" * 80)
        print(f"# PERSONA: {persona.upper()}")
        print("#" * 80)
        summary = []
        for label, history in variants.items():
            recs = recommend(model, history, n_items, device, k=TOP_K)
            hits = genre_hits(idx2meta, recs, kws)
            top5 = sum(1 for r in hits if r <= 5)
            top10 = sum(1 for r in hits if r <= 10)
            summary.append((top5, top10, len(hits), label))
            print(f"\n=== {persona} / {label}  "
                  f"(ends on '{idx2meta.get(history[-1],('?',''))[0][:34]}') ===")
            print(f"  genre hits: {top5}/5 in top-5, {top10}/10 in top-10, "
                  f"{len(hits)}/{TOP_K} in top-20  (at ranks {hits})")
            for rank, i in enumerate(recs[:12], 1):
                title, tag = idx2meta.get(i, ("(unknown)", ""))
                mark = "  <-- genre" if rank in hits else ""
                print(f"    {rank:>2}. {title[:48]:48} [{tag[:22]}]{mark}")
        summary.sort(reverse=True)
        print(f"\n  >>> BEST for {persona} (by top-5): ", end="")
        print("  |  ".join(f"{lab}: {t5}/5 ({t10}/10, {n20}/20)"
                            for t5, t10, n20, lab in summary))
        print()


if __name__ == "__main__":
    main()
