from __future__ import annotations

import pandas as pd


def next_session_date(prediction_date, trading_dates: pd.Series | list) -> pd.Timestamp:
    """Return the first supplied trading session strictly after prediction_date."""
    d = pd.Timestamp(prediction_date).normalize()
    dates = pd.to_datetime(pd.Series(trading_dates), errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    future = dates[dates > d]
    if future.empty:
        raise ValueError("No future trading session supplied")
    return future.iloc[0]


def validate_trading_calendar(dates: pd.Series | list) -> pd.DatetimeIndex:
    clean = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().dt.normalize()
    if clean.empty:
        raise ValueError("Trading calendar is empty")
    if clean.duplicated().any():
        raise ValueError("Trading calendar contains duplicate dates")
    return pd.DatetimeIndex(clean.sort_values())
