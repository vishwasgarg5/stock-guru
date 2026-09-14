from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scheduling import validate_trading_calendar


def load_trading_calendar(path: str | Path, date_column: str = "date") -> pd.DatetimeIndex:
    """Load a validated exchange-session calendar from CSV."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if date_column not in frame.columns:
        raise ValueError(f"Calendar missing date column: {date_column}")
    return validate_trading_calendar(frame[date_column].tolist())
