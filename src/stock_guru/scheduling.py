from __future__ import annotations

import pandas as pd


def validate_trading_calendar(dates: pd.Series | list) -> pd.DatetimeIndex:
    """Validate and normalize a supplied exchange-session calendar."""
    raw = pd.Series(dates)
    if raw.empty:
        raise ValueError("Trading calendar is empty")
    clean = pd.to_datetime(raw, errors="coerce")
    if clean.isna().any():
        raise ValueError("Trading calendar contains invalid dates")
    normalized = clean.dt.normalize()
    if normalized.duplicated().any():
        raise ValueError("Trading calendar contains duplicate dates")
    return pd.DatetimeIndex(normalized.sort_values())


def next_session_date(prediction_date, trading_dates: pd.Series | list) -> pd.Timestamp:
    """Return the first supplied trading session strictly after prediction_date."""
    d = pd.Timestamp(prediction_date).normalize()
    if pd.isna(d):
        raise ValueError("Prediction date is invalid")
    dates = validate_trading_calendar(trading_dates)
    future = dates[dates > d]
    if len(future) == 0:
        raise ValueError("No future trading session supplied")
    return future[0]


def previous_session_date(session_date, trading_dates: pd.Series | list) -> pd.Timestamp:
    """Return the last supplied trading session strictly before session_date."""
    d = pd.Timestamp(session_date).normalize()
    if pd.isna(d):
        raise ValueError("Session date is invalid")
    dates = validate_trading_calendar(trading_dates)
    previous = dates[dates < d]
    if len(previous) == 0:
        raise ValueError("No previous trading session supplied")
    return previous[-1]
