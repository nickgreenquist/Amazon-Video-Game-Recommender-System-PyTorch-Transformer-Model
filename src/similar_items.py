"""Most-similar-item retrieval straight from the learned item embedding table.

This is DECOUPLED from the sequence model — it never calls model.forward(), so
no attention / positions / causal mask are involved. Each item's 64-dim
embedding is treated as coordinates and neighbors are ranked by COSINE
similarity, not the dot product the recommender uses: the dot product is
magnitude-sensitive (popular items have larger norms and crowd every neighbor
list), so cosine ("same direction") is the cleaner similarity readout.

The embedding was trained only for next-item prediction, so "similar" means
"appears in similar sequential / co-purchase contexts" — which the canary probes
showed is dominated by platform, then genre. It is NOT a content/genre model.

Usage:
    python3 src/similar_items.py                       # curated showcase seeds
    python3 src/similar_items.py --id 15040            # neighbors of one item id
    python3 src/similar_items.py --id 15040 --top 20
    python3 src/similar_items.py --search witcher      # find item ids by title
"""

import argparse
import json

import torch
import torch.nn.functional as F

from canaries import (CKPT, MAX_SEQ_LEN, ROOT, SASRec, _render_table,
                      get_device, load_metadata)

TOP_K = 10

# Curated seed items spanning platforms/genres/eras, for the no-arg showcase
# sweep. Drawn from the MID-TAIL (popularity rank ~280-880, 92-184 interactions;
# for reference rank 150 has ~259, rank 900 ~91). Mid-tail seeds give cleaner
# neighbor lists than blockbuster seeds: even with cosine normalization,
# top-popularity items sit in a dense, genre-blurred region of the embedding
# space (everything co-occurs with them), so their neighbors collapse toward
# "other popular same-platform titles." Mid-tail items occupy more specific
# niches, so their nearest neighbors stay on-genre/franchise.
SEEDS = [
    4489,   # Resident Evil 4 - PS2          (184)  survival horror
    1823,   # Metal Gear Solid 2 - PS2       (182)  stealth/action
    7708,   # Chrono Trigger - DS            (173)  JRPG (handheld)
    5659,   # Burnout Paradise - PS3         (119)  racing
    9672,   # Sid Meier's Civilization V - PC (161) strategy
    9033,   # Fallout: New Vegas - PC        (87)   western RPG
    7022,   # Tales of Vesperia - Xbox 360   (92)   JRPG
    390,    # Diablo 2 - PC                  (96)   action RPG
]


def item_matrix(model):
    """E[j] is the embedding of item id j+1 (row 0 is padding_idx, sliced off)."""
    return model.item_embedding.weight.detach()[1:]


def most_similar(E, seed_id, k):
    """[(item_id, cosine), ...] for the k nearest items to seed_id (seed dropped)."""
    v = E[seed_id - 1]                          # seed's learned vector
    sims = F.cosine_similarity(v.unsqueeze(0), E)
    sims[seed_id - 1] = float("-inf")           # never return the seed itself
    top = torch.topk(sims, k)
    return [(i + 1, s) for i, s in zip(top.indices.tolist(), top.values.tolist())]


def print_neighbors(idx2meta, seed_id, neighbors):
    title, tag = idx2meta.get(seed_id, ("(unknown)", ""))
    print("=" * 78)
    print(f"SEED [{seed_id}]: {title}" + (f"  [{tag}]" if tag else ""))
    print("-" * 78)
    rows = [(str(rank), f"{score:.3f}", *idx2meta.get(i, ("(unknown)", "")))
            for rank, (i, score) in enumerate(neighbors, 1)]
    _render_table(("#", "cos", "Title", "Category"), rows, (3, 5, 45, 30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, help="seed item id (use --search to find one)")
    p.add_argument("--search", help="list item ids whose title contains this string")
    p.add_argument("--top", type=int, default=TOP_K)
    args = p.parse_args()

    idx2meta = load_metadata()

    if args.search:
        q = args.search.lower()
        hits = sorted((i, t) for i, (t, _) in idx2meta.items() if q in t.lower())
        for i, t in hits[:40]:
            print(f"  [{i}] {t}")
        print(f"\n{len(hits)} match(es)" + (" (showing 40)" if len(hits) > 40 else ""))
        return

    device = get_device()
    n_items = len(json.loads((ROOT / "data/processed/item_id_to_idx.json").read_text()))
    model = SASRec(n_items, hidden_dim=64, n_blocks=2, n_heads=1,
                   max_seq_len=MAX_SEQ_LEN, dropout=0.5).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    E = item_matrix(model)

    seeds = [args.id] if args.id else SEEDS
    print(f"Stage 3 SASRec item embeddings  |  cosine similarity  |  top-{args.top}\n")
    for seed_id in seeds:
        print_neighbors(idx2meta, seed_id, most_similar(E, seed_id, args.top))
        print()


if __name__ == "__main__":
    main()
