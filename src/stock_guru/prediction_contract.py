from __future__ import annotations

import pandas as pd


def validate_prediction_frame(frame: pd.DataFrame) -> dict:
    required = {"symbol", "pred_open", "pred_high", "pred_low", "pred_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction missing columns: {sorted(missing)}")
    if frame["symbol"].astype("string").str.strip().eq("").any():
        raise ValueError("Prediction contains blank symbols")
    if frame.duplicated("symbol").any():
        raise ValueError("Prediction contains duplicate symbols")
    for col in sorted(required - {"symbol"}):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (~values.map(pd.notna)).any():
            raise ValueError(f"Prediction contains invalid {col}")
    return {"rows": int(len(frame)), "unique_symbols": int(frame["symbol"].nunique()), "validated": True}
