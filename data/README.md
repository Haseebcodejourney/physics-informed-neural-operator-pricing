# Real market data

## Quick start (SPY options from Yahoo Finance)

```bash
cd cf_hpino
pip install yfinance pandas

# 1) Download live option chain (all expiries = more training surfaces)
python scripts/fetch_market_data.py --ticker SPY --max-expiries 0 --out data/raw/spy_options_full.csv

# Multi-asset (SPY + QQQ + IWM), calls only
python scripts/fetch_market_data.py --tickers SPY,QQQ,IWM --max-expiries 0 --out data/raw/multi_options.csv

# Even more rows: include puts (loader keeps calls only for BS training)
python scripts/fetch_market_data.py --ticker SPY --max-expiries 0 --include-puts --out data/raw/spy_all_types.csv

# 2) Train on real quotes (split by expiry)
python scripts/train_market.py --csv data/raw/spy_options.csv --device cuda --fetch

# 3) Test on held-out expiries
python scripts/test_market.py --checkpoint checkpoints/market/best.pt --csv data/raw/spy_options.csv
```

## What is “real” here?

| Component | Source |
|-----------|--------|
| **Quoted mids** | Live (or recent) bid/ask mid from Yahoo Finance |
| **Train/val/test split** | By **expiry date** — test maturities not seen during training |
| **market loss** | Direct MSE on listed strike prices at \(t \approx 0\) |
| **Full surface** | Anchored to quotes at \(t=0\); extended in \(t\) via BS with interpolated smile IV |

This is honest real-data supervision on **observed prices**, not only synthetic BS surfaces.

## CSV format (your own data)

Required columns (flexible names):

| Field | Aliases |
|-------|---------|
| strike | strike, K |
| price | mid, close, price |
| expiry | expiry, maturity, dte |
| spot | spot, S0 |
| rate | rate, r (optional, default 0.04) |
| div_yield | q (optional) |
| iv | iv, implied_vol (optional; inverted if missing) |

Example:

```csv
ticker,trade_date,expiry,dte,strike,mid,spot,rate,div_yield,iv
SPY,2025-05-25,2025-06-20,26,520,12.45,530.2,0.04,0.013,0.18
```

Place files under `data/raw/` and pass `--csv` to `train_market.py`.

## Notes

- **SPY vs SPX**: Yahoo provides SPY chains reliably; SPX index options may need a paid feed or manual CSV.
- Fetch **more expiries** if training fails: use `--max-expiries 0` (all listed dates) or add tickers with `--tickers SPY,QQQ,IWM`.
- For publication, report **test_quote_rmse** (dollars on mids) and **test_surface_rel_l2** from `checkpoints/market/logs/market_test.json`.
