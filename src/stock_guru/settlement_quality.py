from __future__ import annotations

from typing import Any

import pandas as pd


def validate_settlement_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate settled prediction outcomes before feedback ingestion."""
    required = {"prediction_date", "symbol", "actual_close", "pred_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Settlement missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["prediction_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Settlement contains invalid prediction dates")
    if frame["symbol"].astype("string").str.strip().eq("").any():
        raise ValueError("Settlement contains blank symbols")
    if frame.duplicated(["prediction_date", "symbol"]).any():
        raise ValueError("Settlement contains duplicate prediction keys")
    for col in ("actual_close", "pred_close"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (~values.map(pd.notna)).any():
            raise ValueError(f"Settlement contains invalid {col}")
    return {"rows": int(len(frame)), "unique_symbols": int(frame["symbol"].nunique()), "validated": True}
