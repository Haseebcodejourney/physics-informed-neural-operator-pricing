#!/usr/bin/env python3
"""Download real option chain (Yahoo Finance) to data/raw/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.market_fetch import fetch_option_chain, save_chain_csv  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--out", default="data/raw/spy_options.csv")
    parser.add_argument("--max-expiries", type=int, default=12)
    args = parser.parse_args()

    df = fetch_option_chain(ticker=args.ticker, max_expiries=args.max_expiries)
    path = save_chain_csv(df, ROOT / args.out)
    print(f"Saved {len(df)} quotes -> {path}")
    print("Next: python scripts/train_market.py --csv", path)


if __name__ == "__main__":
    main()
