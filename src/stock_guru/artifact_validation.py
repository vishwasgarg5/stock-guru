from __future__ import annotations

from pathlib import Path
import json

REQUIRED_FILES = ("ranker.joblib", "ohlc.joblib", "features.csv", "model_metadata.json")


def validate_model_artifact(model_dir: str | Path, *, require_pit_context: bool = False) -> dict:
    """Fail closed when a model directory is incomplete or metadata is invalid."""
    root = Path(model_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Model artifact missing files: {missing}")
    try:
        metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("model_metadata.json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("model_metadata.json must contain an object")
    if not str(metadata.get("model_version", "")).strip():
        raise ValueError("model metadata requires model_version")
    features = metadata.get("features")
    if not isinstance(features, list) or not features or not all(str(x).strip() for x in features):
        raise ValueError("model metadata requires a non-empty features list")
    if require_pit_context:
        context = metadata.get("pit_context")
        if not isinstance(context, dict):
            raise ValueError("model metadata requires pit_context")
        if not context.get("fundamentals_supplied") or not context.get("universe_intervals_supplied"):
            raise ValueError("PIT production artifact requires fundamentals and universe intervals")
    return {"status": "valid", "model_version": metadata["model_version"], "features": features, "pit_context": metadata.get("pit_context", {})}
