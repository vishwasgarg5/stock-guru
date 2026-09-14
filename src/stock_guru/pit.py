from __future__ import annotations

import pandas as pd

from .fundamentals_ingest import validate_pit_fundamentals


def join_pit_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Attach the latest publicly available fundamental observation to each price row.

    The join is point-in-time safe: a fundamental observation is eligible only when
    its ``available_date`` is on or before the price row's date.  No future value is
    forward-filled across a publication boundary.
    """
    if date_column not in prices.columns:
        raise ValueError(f"Missing price date column: {date_column}")
    if "symbol" not in prices.columns:
        raise ValueError("Missing price symbol column: symbol")

    p = prices.copy()
    p[date_column] = pd.to_datetime(p[date_column], errors="coerce").dt.normalize()
    if p[date_column].isna().any():
        raise ValueError("Prices contain invalid dates")
    p["symbol"] = p["symbol"].astype(str).str.strip()
    if p["symbol"].eq("").any():
        raise ValueError("Prices contain blank symbols")

    f = validate_pit_fundamentals(fundamentals)
    f = f.rename(columns={"available_date": "_pit_available_date"})
    f = f.sort_values(["symbol", "_pit_available_date"])
    p = p.sort_values(["symbol", date_column])

    merged = pd.merge_asof(
        p,
        f,
        left_on=date_column,
        right_on="_pit_available_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_index(kind="stable")
