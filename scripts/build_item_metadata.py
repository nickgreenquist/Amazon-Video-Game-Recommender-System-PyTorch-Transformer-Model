"""Build the clean, classified item-metadata table for the 5-core Video Games corpus.

Counts interactions from the 5-core reviews, joins to the metadata file, cleans
messy fields (brand 'by\\n' prefixes, HTML-wrapped prices, list-valued rank,
HTML entities), classifies each item as game/console/accessory/other/unknown
via category-segment + title-keyword rules, and writes data/item_metadata.parquet
for downstream training and demo use.

Inspect with scripts/inspect_items.py.
"""

import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
REVIEWS = DATA / "raw" / "Video_Games_5.json.gz"
META = DATA / "raw" / "meta_Video_Games.json.gz"
OUT = DATA / "item_metadata.parquet"

# Platforms we can pull out of the category path, longest-match first.
PLATFORMS = [
    "PlayStation Vita", "PlayStation 4", "PlayStation 3", "PlayStation 2",
    "PlayStation", "Xbox One", "Xbox 360", "Xbox", "Nintendo Switch",
    "Nintendo 3DS", "Nintendo DS", "Wii U", "Wii", "GameCube",
    "Game Boy Advance", "Nintendo 64", "Sega Dreamcast", "PSP", "PC", "Mac",
]

PRICE_RE = re.compile(r"\$\s*([\d,]+\.\d{2})")
RANK_RE = re.compile(r"#([\d,]+)")

ACCESSORY_SEGS = {
    "Accessories", "Controllers", "Headsets", "Gaming Mice", "Gaming Keyboards",
    "Gamepads & Standard Controllers", "Cases & Storage", "Thumb Grips",
    "Accessory Kits", "Faceplates, Protectors & Skins", "Chargers", "Cables",
    "Adapters", "Memory", "Batteries", "Stands", "Mounts", "Skins",
    "Cooling Systems", "Interactive Gaming Figures",
}
CONSOLE_SEGS = {"Consoles"}
GAME_SEGS = {"Games", "Digital Games", "Digital Games & DLC"}
# NOTE: "Retro Gaming & Microconsoles" is the umbrella over retro Games/
# Consoles/Accessories — the leaf segment already classifies those, so don't
# tag the umbrella itself as console (it would mis-tag every retro game).

# Title-keyword fallback used when the category path is just a bare platform
# (e.g. "Video Games > PC") and gives us no kind signal. Lowercased substring match.
ACCESSORY_TITLE_KW = (
    "controller", "gamepad", "headset", "headphone", "charging", "charger",
    "charge kit", "adapter", "cable", "mount", "stand", "dock", "cooler",
    "cooling", "battery", "batteries", "faceplate", "thumb grip",
    "memory card", "mouse pad", "gaming mouse", "gaming keyboard", "wheel",
    "remote",
)
CONSOLE_TITLE_KW = ("console", "system")


def count_interactions():
    counts = Counter()
    with gzip.open(REVIEWS, "rt") as f:
        for line in f:
            counts[json.loads(line)["asin"]] += 1
    return counts


def load_best_metadata(asins):
    """Stream the metadata file, keep the most complete record per asin in our set."""
    best = {}
    with gzip.open(META, "rt") as f:
        for line in f:
            rec = json.loads(line)
            a = rec.get("asin")
            if a in asins and completeness(rec) > completeness(best.get(a, {})):
                best[a] = rec
    return best


def completeness(rec):
    return sum(1 for v in rec.values() if v not in (None, "", [], {}))


def clean_brand(raw):
    if not raw:
        return ""
    b = html.unescape(str(raw)).replace("\n", " ").strip()
    b = re.sub(r"^by\s+", "", b, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", b)


def clean_category(raw):
    if not isinstance(raw, list):
        return [], ""
    # Trailing segments are often HTML scrape junk ('</span>...'); drop them.
    segs = [html.unescape(s).strip() for s in raw
            if s and s.strip() and "<" not in s]
    # First segment is always the umbrella "Video Games" — drop it.
    if segs and segs[0].lower() == "video games":
        segs = segs[1:]
    platform = next((p for seg in segs for p in PLATFORMS if seg == p), "")
    return segs, platform


def classify_kind(segs, title="", platform=""):
    """Coarse item type from category path, with a title-keyword fallback for
    bare-platform paths where the segment list gives no signal."""
    s = set(segs)
    if s & CONSOLE_SEGS:
        return "console"
    if "Currency & Subscription Cards" in s:
        return "other"
    if s & GAME_SEGS:
        return "game"
    if s & ACCESSORY_SEGS:
        return "accessory"

    # Fallback: category is just a platform (or empty). Disambiguate from title.
    t = title.lower()
    if any(kw in t for kw in ACCESSORY_TITLE_KW):
        return "accessory"
    if platform and t.strip() == platform.lower():
        return "console"  # title is literally the platform name, e.g. "Wii"
    if any(kw in t for kw in CONSOLE_TITLE_KW):
        return "console"
    # Any segment is a known platform and title didn't match accessory/console
    # keywords -> almost certainly a game on that platform. Catches both
    # "Video Games > PC" (Diablo III) and "Retro Gaming > Xbox" (Halo-Xbox).
    if any(seg in PLATFORMS for seg in segs):
        return "game"
    return "unknown"


def clean_price(raw):
    m = PRICE_RE.search(str(raw or ""))
    return float(m.group(1).replace(",", "")) if m else None


def clean_rank(raw):
    """Overall 'Video Games' sales rank as an int, lower = more popular."""
    text = " ".join(raw) if isinstance(raw, list) else str(raw or "")
    m = RANK_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def clean_text(raw):
    if isinstance(raw, list):
        raw = " ".join(raw)
    return html.unescape(str(raw or "")).strip()


def pick_image(rec):
    for key in ("imageURLHighRes", "imageURL"):
        urls = rec.get(key)
        if urls:
            return urls[0]
    return ""


def main():
    counts = count_interactions()
    best = load_best_metadata(set(counts))

    rows = []
    for a, rec in best.items():
        cats, platform = clean_category(rec.get("category"))
        title = clean_text(rec.get("title"))
        rows.append({
            "asin": a,
            "interaction_count": counts[a],
            "title": title,
            "brand": clean_brand(rec.get("brand")),
            "platform": platform,
            "kind": classify_kind(cats, title, platform),
            "category": " > ".join(cats),
            "price": clean_price(rec.get("price")),
            "sales_rank": clean_rank(rec.get("rank")),
            "description": clean_text(rec.get("description")),
            "also_buy": rec.get("also_buy") or [],
            "also_view": rec.get("also_view") or [],
            "image": pick_image(rec),
        })

    df = pd.DataFrame(rows).sort_values("interaction_count", ascending=False)
    df.to_parquet(OUT, index=False)

    print(f"{len(df)} items written to {OUT}")
    print(f"  {len(counts) - len(best)} of {len(counts)} items have no metadata record")
    cov = {c: f"{100 * df[c].astype(bool).mean():.1f}%" for c in
           ["title", "brand", "platform", "category", "description", "image"]}
    cov["price"] = f"{100 * df['price'].notna().mean():.1f}%"
    cov["sales_rank"] = f"{100 * df['sales_rank'].notna().mean():.1f}%"
    print("  coverage:", cov)
    print("  kind:", df["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
