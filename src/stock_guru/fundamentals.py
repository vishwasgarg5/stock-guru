from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"symbol", "available_date"}
NUMERIC_FIELDS = [
    "revenue_growth", "eps_growth", "roe", "roce", "debt_to_equity",
    "operating_margin", "net_margin", "pe", "pb", "peg", "free_cash_flow",
    "dividend_yield", "promoter_holding", "institutional_holding",
]
DATE_FIELDS = ("available_date", "reported_date", "period_end")


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_FIELDS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _validate_pit_contract(df: pd.DataFrame) -> None:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing fundamental columns: {sorted(missing)}")
    if df["symbol"].eq("").any() or df["available_date"].isna().any():
        raise ValueError("Fundamentals contain blank symbols or invalid available_date values")
    if df.duplicated(["symbol", "available_date"]).any():
        raise ValueError("Fundamentals contain duplicate symbol/available_date rows")
    if "reported_date" in df.columns:
        invalid = df["reported_date"].notna() & (df["available_date"] < df["reported_date"])
        if invalid.any():
            raise ValueError("Fundamentals contain available_date earlier than reported_date")
    if "period_end" in df.columns and "reported_date" in df.columns:
        invalid = df["period_end"].notna() & df["reported_date"].notna() & (df["reported_date"] < df["period_end"])
        if invalid.any():
            raise ValueError("Fundamentals contain reported_date earlier than period_end")
    if "source" in df.columns:
        if df["source"].isna().any() or df["source"].astype(str).str.strip().eq("").any():
            raise ValueError("Fundamentals contain blank source values")
    if "source_id" in df.columns:
        if df["source_id"].isna().any() or df["source_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Fundamentals contain blank source_id values")


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.strip() if "symbol" in df.columns else ""
    df = _normalise_dates(df)
    _validate_pit_contract(df)
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("source", "source_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def asof_join(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach only information that was available on or before each market date.

    ``available_date`` is the only date used for the join.  ``period_end`` is the
    accounting period being described and ``reported_date`` is when the issuer
    reported it; neither is substituted for availability.  This prevents a
    period-end date from accidentally leaking future information into a backtest.
    """
    missing = {"date", "symbol"} - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.strip()
    if p["date"].isna().any() or p["symbol"].eq("").any():
        raise ValueError("Prices contain invalid date or blank symbol values")

    f = _normalise_dates(fundamentals)
    f["symbol"] = f["symbol"].astype(str).str.strip()
    _validate_pit_contract(f)
    f = f.sort_values(["symbol", "available_date"])
    p = p.sort_values(["symbol", "date"])
    return pd.merge_asof(
        p,
        f,
        left_on="date",
        right_on="available_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
