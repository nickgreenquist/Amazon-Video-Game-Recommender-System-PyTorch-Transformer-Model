"""Print a top-N view of data/item_metadata.parquet for quick eyeball inspection.

Usage: python3 scripts/inspect_items.py [N]   (default N=50)
Pipe through grep/less for search:
    python3 scripts/inspect_items.py 1000 | grep -i diablo
"""

import sys
from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).resolve().parents[1] / "data" / "item_metadata.parquet"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    df = (pd.read_parquet(PARQUET)
            .sort_values("interaction_count", ascending=False)
            .head(n))

    print(f"{'count':>6}  {'kind':<9}  {'platform':<16}  title")
    for _, r in df.iterrows():
        print(f"{r['interaction_count']:>6}  {r['kind']:<9}  "
              f"{r['platform'][:16]:<16}  {r['title']}")


if __name__ == "__main__":
    main()
