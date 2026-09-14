from __future__ import annotations

from pathlib import Path
import pandas as pd

# Point-in-time fundamentals are intentionally file-driven. A provider adapter can
# populate this schema without letting today's fundamentals leak into history.
REQUIRED = {"symbol", "available_date"}

NUMERIC_FIELDS = [
    "revenue_growth", "eps_growth", "roe", "roce", "debt_to_equity",
    "operating_margin", "net_margin", "pe", "pb", "peg",
    "free_cash_flow", "dividend_yield", "promoter_holding",
    "institutional_holding",
]


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["available_date"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing fundamental columns: {sorted(missing)}")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce").dt.normalize()
    if df["symbol"].eq("").any() or df["available_date"].isna().any():
        raise ValueError("Fundamentals contain blank symbols or invalid available_date values")
    if df.duplicated(["symbol", "available_date"]).any():
        raise ValueError("Fundamentals contain duplicate symbol/available_date rows")
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["symbol", "available_date"])


def asof_join(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach only information that was available on or before each market date."""
    required_prices = {"date", "symbol"}
    missing = required_prices - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    missing = REQUIRED - set(fundamentals.columns)
    if missing:
        raise ValueError(f"Missing fundamental columns: {sorted(missing)}")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    f = fundamentals.copy()
    f["available_date"] = pd.to_datetime(f["available_date"], errors="coerce").dt.normalize()
    f["symbol"] = f["symbol"].astype(str).str.strip()
    if f[["symbol", "available_date"]].isna().any().any():
        raise ValueError("Fundamentals contain invalid symbol/available_date values")
    if f.duplicated(["symbol", "available_date"]).any():
        raise ValueError("Fundamentals contain duplicate symbol/available_date rows")
    f = f.sort_values(["symbol", "available_date"])
    p = p.sort_values(["symbol", "date"])
    return pd.merge_asof(
        p, f, left_on="date", right_on="available_date", by="symbol",
        direction="backward", allow_exact_matches=True,
    )
