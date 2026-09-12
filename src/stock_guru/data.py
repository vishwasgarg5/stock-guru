from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import time
import pandas as pd
import requests
import yfinance as yf

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"


def fetch_nifty500_symbols() -> pd.DataFrame:
    """Fetch the current NIFTY 500 constituent list from NSE Indices."""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    response = requests.get(NIFTY500_URL, headers=headers, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content))
    symbol_col = next(c for c in frame.columns if c.strip().lower() in {"symbol", "ticker"})
    frame = frame.rename(columns={symbol_col: "symbol"})
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["yf_symbol"] = frame["symbol"] + ".NS"
    return frame


def save_universe_snapshot(output: str = "data/universe_snapshots.csv", as_of: str | None = None) -> Path:
    """Persist the fetched membership with an observation date for future research."""
    universe = fetch_nifty500_symbols()
    snapshot_date = as_of or date.today().isoformat()
    universe.insert(0, "as_of", snapshot_date)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)
    return path


def download_prices(symbols: list[str], start: str, end: str | None = None, chunk_size: int = 50) -> pd.DataFrame:
    """Download daily NSE OHLCV data through yfinance and normalize its schema."""
    rows: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), chunk_size):
        chunk = symbols[offset : offset + chunk_size]
        tickers = [s if s.endswith(".NS") else f"{s}.NS" for s in chunk]
        raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False,
                           group_by="ticker", progress=False, threads=True)
        if raw.empty:
            continue
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            part = raw[ticker].reset_index()
            part.columns = [str(c).lower().replace(" ", "_") for c in part.columns]
            part["symbol"] = ticker.removesuffix(".NS")
            rows.append(part[["date", "symbol", "open", "high", "low", "close", "volume"]])
        time.sleep(0.25)
    if not rows:
        raise RuntimeError("No price data returned. Check symbols, dates, and network access.")
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None)
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])


def download_nifty500_prices(start: str, end: str | None = None, output: str = "data/prices.csv") -> Path:
    universe = fetch_nifty500_symbols()
    prices = download_prices(universe["symbol"].tolist(), start=start, end=end)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)
    universe_path = path.parent / "nifty500_universe.csv"
    universe.to_csv(universe_path, index=False)
    save_universe_snapshot(str(path.parent / "universe_snapshots.csv"))
    return path
