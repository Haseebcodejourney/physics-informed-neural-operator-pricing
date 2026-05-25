"""
Download real US equity/index option chains (Yahoo Finance) to CSV.

Requires: pip install yfinance pandas

Examples:
    python scripts/fetch_market_data.py --ticker SPY --max-expiries 0 --out data/raw/spy_options.csv
    python scripts/fetch_market_data.py --tickers SPY,QQQ,IWM --out data/raw/multi_options.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _row_from_option(
    ticker: str,
    trade_date: str,
    expiry: str,
    dte: int,
    spot: float,
    r: pd.Series,
    option_type: str,
    rate: float,
    div_yield: float,
) -> Optional[dict]:
    bid = _safe_float(r.get("bid"))
    ask = _safe_float(r.get("ask"))
    last = _safe_float(r.get("lastPrice"))
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
    if mid <= 0.01:
        return None

    iv = r.get("impliedVolatility", float("nan"))
    oi = r.get("openInterest", float("nan"))
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "expiry": expiry,
        "dte": dte,
        "option_type": option_type,
        "strike": _safe_float(r["strike"]),
        "mid": mid,
        "bid": bid,
        "ask": ask,
        "spot": spot,
        "volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
        "open_interest": int(oi) if pd.notna(oi) else 0,
        "iv": _safe_float(iv, float("nan")) if pd.notna(iv) else "",
        "rate": rate,
        "div_yield": div_yield,
    }


def fetch_option_chain(
    ticker: str = "SPY",
    max_expiries: int = 0,
    rate: float = 0.04,
    div_yield: float = 0.013,
    include_puts: bool = False,
    min_mid: float = 0.01,
    sleep_sec: float = 0.15,
) -> pd.DataFrame:
    """
    Fetch option chain from Yahoo Finance.

    Args:
        ticker: e.g. SPY, QQQ, IWM
        max_expiries: 0 = all listed expiries; otherwise cap count
        include_puts: include put options (more data, wider strike coverage)
        min_mid: skip quotes below this mid price
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Install yfinance: pip install yfinance pandas") from e

    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}")
    spot = float(hist["Close"].iloc[-1])

    all_expiries = list(stock.options)
    if not all_expiries:
        raise RuntimeError(f"No option expiries listed for {ticker}")

    if max_expiries > 0:
        expiries = all_expiries[:max_expiries]
    else:
        expiries = all_expiries

    trade_date = datetime.now().strftime("%Y-%m-%d")
    rows: List[dict] = []
    failed = 0

    print(f"  {ticker}: spot={spot:.2f}, {len(expiries)}/{len(all_expiries)} expiries")

    for i, exp in enumerate(expiries):
        try:
            chain = stock.option_chain(exp)
        except Exception as e:
            failed += 1
            print(f"    skip expiry {exp}: {e}")
            time.sleep(sleep_sec)
            continue

        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte = max((exp_dt - datetime.now()).days, 1)
        except ValueError:
            dte = 30

        for opt_type, table in [("call", chain.calls)]:
            if table is None or table.empty:
                continue
            for _, r in table.iterrows():
                row = _row_from_option(
                    ticker, trade_date, exp, dte, spot, r, opt_type, rate, div_yield
                )
                if row and row["mid"] >= min_mid:
                    rows.append(row)

        if include_puts and chain.puts is not None and not chain.puts.empty:
            for _, r in chain.puts.iterrows():
                row = _row_from_option(
                    ticker, trade_date, exp, dte, spot, r, "put", rate, div_yield
                )
                if row and row["mid"] >= min_mid:
                    rows.append(row)

        if (i + 1) % 5 == 0:
            print(f"    {ticker}: {i+1}/{len(expiries)} expiries, {len(rows)} quotes so far")
        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No option quotes returned for {ticker} ({failed} expiries failed)")
    if failed:
        print(f"    {ticker}: {failed} expiries skipped")
    return df


def fetch_multiple_tickers(
    tickers: List[str],
    max_expiries: int = 0,
    include_puts: bool = False,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> pd.DataFrame:
    """Fetch and concatenate chains for several tickers."""
    frames = []
    for t in tickers:
        print(f"Fetching {t}...")
        try:
            frames.append(
                fetch_option_chain(
                    t,
                    max_expiries=max_expiries,
                    include_puts=include_puts,
                    rate=rate,
                    div_yield=div_yield,
                )
            )
        except Exception as e:
            print(f"  Failed {t}: {e}")
    if not frames:
        raise RuntimeError("No data fetched for any ticker")
    return pd.concat(frames, ignore_index=True)


def save_chain_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def print_summary(df: pd.DataFrame, path: Path) -> None:
    print(f"\nSaved {len(df)} quotes -> {path}")
    print(f"  Tickers: {df['ticker'].nunique()} - {', '.join(sorted(df['ticker'].unique()))}")
    print(f"  Expiries: {df['expiry'].nunique()}")
    if "option_type" in df.columns:
        print(f"  Types: {df['option_type'].value_counts().to_dict()}")
    for t in df["ticker"].unique():
        sub = df[df["ticker"] == t]
        print(
            f"  {t}: {len(sub)} quotes, spot~{sub['spot'].median():.2f}, "
            f"strikes {sub['strike'].min():.0f}-{sub['strike'].max():.0f}"
        )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch real option chains to CSV")
    parser.add_argument("--ticker", default="", help="Single ticker (ignored if --tickers set)")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated list, e.g. SPY,QQQ,IWM",
    )
    parser.add_argument("--out", default="data/raw/spy_options.csv")
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=0,
        help="0 = fetch ALL expiries (recommended for more data)",
    )
    parser.add_argument("--include-puts", action="store_true", help="Include put options")
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--div", type=float, default=0.013)
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = ["SPY"]

    if len(tickers) == 1:
        df = fetch_option_chain(
            tickers[0],
            max_expiries=args.max_expiries,
            include_puts=args.include_puts,
            rate=args.rate,
            div_yield=args.div,
        )
    else:
        df = fetch_multiple_tickers(
            tickers,
            max_expiries=args.max_expiries,
            include_puts=args.include_puts,
            rate=args.rate,
            div_yield=args.div,
        )

    path = save_chain_csv(df, Path(args.out))
    print_summary(df, path)


if __name__ == "__main__":
    main()
