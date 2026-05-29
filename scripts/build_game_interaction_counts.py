"""Build a CSV of all games (from the 5-core reviews) sorted by interaction count,
joined with their titles from the metadata file."""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
REVIEWS = DATA / "raw" / "Video_Games_5.json.gz"
META = DATA / "raw" / "meta_Video_Games.json.gz"
OUT = DATA / "games_by_interaction_count.csv"


def count_interactions():
    counts = Counter()
    with gzip.open(REVIEWS, "rt") as f:
        for line in f:
            counts[json.loads(line)["asin"]] += 1
    return counts


def load_titles(asins):
    titles = {}
    with gzip.open(META, "rt") as f:
        for line in f:
            d = json.loads(line)
            asin = d.get("asin")
            if asin in asins:
                titles[asin] = (d.get("title") or "").strip()
    return titles


def main():
    counts = count_interactions()
    titles = load_titles(set(counts))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asin", "interaction_count", "title"])
        for asin, n in counts.most_common():
            w.writerow([asin, n, titles.get(asin, "")])
    print(f"{len(counts)} games written to {OUT}")


if __name__ == "__main__":
    main()
