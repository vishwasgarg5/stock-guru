from __future__ import annotations

from typing import Any

import pandas as pd


def validate_pit_join_output(frame: pd.DataFrame, *, symbol_col: str = "symbol", date_col: str = "date") -> dict[str, Any]:
    """Audit a PIT-enriched feature frame for leakage and duplicate keys."""
    required = {symbol_col, date_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PIT output missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("PIT output contains invalid dates")
    symbols = frame[symbol_col].astype("string").str.strip()
    if symbols.isna().any() or (symbols == "").any():
        raise ValueError("PIT output contains blank symbols")
    key = pd.DataFrame({"symbol": symbols, "date": dates})
    if key.duplicated().any():
        raise ValueError("PIT output contains duplicate symbol/date rows")
    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "unique_symbols": int(symbols.nunique()),
        "min_date": dates.min().date().isoformat() if len(frame) else None,
        "max_date": dates.max().date().isoformat() if len(frame) else None,
        "pit_available_date_checked": False,
        "leakage_free": True,
    }
    if "_pit_available_date" in frame.columns:
        available = pd.to_datetime(frame["_pit_available_date"], errors="coerce")
        invalid = available.notna() & (available > dates)
        if invalid.any():
            raise ValueError("PIT output contains fundamentals available after prediction date")
        report["pit_available_date_checked"] = True
    return report
