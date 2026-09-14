from __future__ import annotations

from pathlib import Path
import pandas as pd

KEYS = ["prediction_date", "symbol", "model_version"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prediction_date"] = pd.to_datetime(out["prediction_date"], errors="coerce").dt.normalize()
    if out["prediction_date"].isna().any():
        raise ValueError("Predictions contain invalid prediction_date values")
    out["symbol"] = out["symbol"].astype(str).str.strip()
    if out["symbol"].eq("").any():
        raise ValueError("Predictions contain blank symbols")
    if "model_version" not in out.columns:
        out["model_version"] = "unknown"
    out["model_version"] = out["model_version"].fillna("unknown").astype(str)
    return out


def append_predictions(store: str | Path, predictions: pd.DataFrame) -> None:
    """Persist daily predictions idempotently for later outcome matching."""
    required = {"prediction_date", "symbol"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    incoming = _normalise(predictions)
    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            existing = _normalise(existing)
            incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    incoming = incoming.drop_duplicates(KEYS, keep="last")
    incoming = incoming.sort_values(KEYS[:2]).reset_index(drop=True)
    incoming.to_csv(path, index=False)


def load_pending(store: str | Path, as_of: str | None = None,
                 settled_store: str | Path | None = None) -> pd.DataFrame:
    """Return prediction rows eligible for settlement and not already settled.

    ``as_of`` limits predictions to sessions at or before that date. When a
    feedback store is supplied, matching date/symbol/model keys are excluded so
    repeated daily runs remain idempotent.
    """
    path = Path(store)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = _normalise(df)
    if as_of is not None:
        df = df[df["prediction_date"] <= pd.Timestamp(as_of).normalize()]
    if settled_store is not None and Path(settled_store).exists():
        settled = pd.read_csv(settled_store)
        if not settled.empty and set(KEYS).issubset(set(settled.columns)):
            settled = _normalise(settled)
            keys = settled[KEYS].drop_duplicates()
            df = df.merge(keys.assign(_settled=True), on=KEYS, how="left")
            df = df[df["_settled"].isna()].drop(columns=["_settled"])
    return df.reset_index(drop=True)
