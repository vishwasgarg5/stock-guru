from __future__ import annotations

from pathlib import Path
import pandas as pd

KEYS = ["prediction_date", "symbol", "model_version"]


def append_predictions(store: str | Path, predictions: pd.DataFrame) -> None:
    """Persist daily predictions idempotently for later outcome matching."""
    required = {"prediction_date", "symbol"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    incoming = predictions.copy()
    incoming["prediction_date"] = pd.to_datetime(incoming["prediction_date"], errors="coerce").dt.normalize()
    if incoming["prediction_date"].isna().any():
        raise ValueError("Predictions contain invalid prediction_date values")
    incoming["symbol"] = incoming["symbol"].astype(str).str.strip()
    if "model_version" not in incoming.columns:
        incoming["model_version"] = "unknown"

    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            existing["prediction_date"] = pd.to_datetime(existing["prediction_date"], errors="coerce").dt.normalize()
            if "model_version" not in existing.columns:
                existing["model_version"] = "unknown"
            incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    incoming = incoming.drop_duplicates(KEYS, keep="last")
    incoming = incoming.sort_values(KEYS[:2])
    incoming.to_csv(path, index=False)


def load_pending(store: str | Path, as_of: str | None = None) -> pd.DataFrame:
    """Return prediction rows whose next-session outcome has not been recorded."""
    path = Path(store)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce").dt.normalize()
    if as_of is not None:
        df = df[df["prediction_date"] <= pd.Timestamp(as_of).normalize()]
    return df.reset_index(drop=True)
