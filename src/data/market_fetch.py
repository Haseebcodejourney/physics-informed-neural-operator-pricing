"""
Download real US equity/index option chains (Yahoo Finance) to CSV.

Requires: pip install yfinance pandas

Example:
    python -m src.data.market_fetch --ticker SPY --out data/raw/spy_options.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def fetch_option_chain(
    ticker: str = "SPY",
    max_expiries: int = 12,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> pd.DataFrame:
    """
    Fetch call options for nearest expiries.

    Returns DataFrame with columns:
        ticker, trade_date, expiry, dte, strike, mid, bid, ask, spot, volume, iv, rate, div_yield
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Install yfinance: pip install yfinance pandas") from e

    stock = yf.Ticker(ticker)
    spot = float(stock.history(period="1d")["Close"].iloc[-1])
    expiries = stock.options[:max_expiries]
    trade_date = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for exp in expiries:
        chain = stock.option_chain(exp)
        calls = chain.calls.copy()
        if calls.empty:
            continue
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        dte = max((exp_dt - datetime.now()).days, 1)

        for _, r in calls.iterrows():
            bid, ask = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
            last = float(r.get("lastPrice", 0) or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            if mid <= 0:
                continue
            iv = r.get("impliedVolatility", float("nan"))
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "expiry": exp,
                    "dte": dte,
                    "strike": float(r["strike"]),
                    "mid": float(mid),
                    "bid": bid,
                    "ask": ask,
                    "spot": spot,
                    "volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
                    "iv": float(iv) if pd.notna(iv) else "",
                    "rate": rate,
                    "div_yield": div_yield,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No option quotes returned for {ticker}")
    return df


def save_chain_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch real option chain to CSV")
    parser.add_argument("--ticker", default="SPY", help="SPY, ^SPX not always on yfinance; SPY is liquid proxy")
    parser.add_argument("--out", default="data/raw/spy_options.csv")
    parser.add_argument("--max-expiries", type=int, default=12)
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--div", type=float, default=0.013)
    args = parser.parse_args(argv)

    df = fetch_option_chain(
        ticker=args.ticker,
        max_expiries=args.max_expiries,
        rate=args.rate,
        div_yield=args.div,
    )
    path = save_chain_csv(df, Path(args.out))
    print(f"Saved {len(df)} quotes across {df['expiry'].nunique()} expiries -> {path}")
    print(f"Spot: {df['spot'].iloc[0]:.2f} | Strikes: {df['strike'].min():.0f}-{df['strike'].max():.0f}")


if __name__ == "__main__":
    main()
