from __future__ import annotations

from pathlib import Path
import hashlib
import json

REQUIRED_FILES = ("ranker.joblib", "ohlc.joblib", "features.csv", "model_metadata.json")
MANIFEST_KEYS = ("model_version", "trained_through", "ranker_sha256", "ohlc_sha256", "features_sha256", "metadata_sha256")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_manifest(manifest: dict) -> dict:
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Artifact manifest missing fields: {missing}")
    for key in MANIFEST_KEYS:
        if not str(manifest[key]).strip():
            raise ValueError(f"Artifact manifest field {key} must be nonblank")
    for key in MANIFEST_KEYS[2:]:
        if len(str(manifest[key]).strip()) != 64:
            raise ValueError(f"Artifact manifest field {key} must be a 64-character SHA-256")
    return dict(manifest)


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
    return {"status": "valid", "model_version": metadata["model_version"], "features": features, "pit_context": metadata.get("pit_context", {}), "file_sha256": {name: file_sha256(root / name) for name in REQUIRED_FILES}}
