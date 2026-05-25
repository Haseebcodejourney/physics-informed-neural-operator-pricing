#!/usr/bin/env python3
"""
Download real option chains (Yahoo Finance) to data/raw/.

More data (recommended):
    python scripts/fetch_market_data.py --ticker SPY --max-expiries 0 --out data/raw/spy_options_full.csv

Multi-asset + puts:
    python scripts/fetch_market_data.py --tickers SPY,QQQ,IWM --max-expiries 0 --include-puts --out data/raw/multi_options.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.market_fetch import main  # noqa: E402

if __name__ == "__main__":
    main()
