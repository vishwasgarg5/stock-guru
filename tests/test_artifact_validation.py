import json
from pathlib import Path

import pytest

from stock_guru.artifact_validation import validate_model_artifact


def make_artifact(tmp_path: Path, pit=True):
    for name in ("ranker.joblib", "ohlc.joblib", "features.csv"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    context = {"fundamentals_supplied": pit, "universe_intervals_supplied": pit}
    (tmp_path / "model_metadata.json").write_text(json.dumps({"model_version": "abc123", "features": ["x"], "pit_context": context}), encoding="utf-8")


def test_artifact_validation_passes(tmp_path):
    make_artifact(tmp_path)
    report = validate_model_artifact(tmp_path, require_pit_context=True)
    assert report["status"] == "valid"
    assert report["model_version"] == "abc123"
    assert set(report["file_sha256"]) == {"ranker.joblib", "ohlc.joblib", "features.csv", "model_metadata.json"}
    assert all(len(value) == 64 for value in report["file_sha256"].values())


def test_artifact_validation_rejects_missing_file(tmp_path):
    make_artifact(tmp_path)
    (tmp_path / "ohlc.joblib").unlink()
    with pytest.raises(ValueError, match="missing files"):
        validate_model_artifact(tmp_path)


def test_artifact_validation_rejects_non_pit_artifact(tmp_path):
    make_artifact(tmp_path, pit=False)
    with pytest.raises(ValueError, match="PIT production"):
        validate_model_artifact(tmp_path, require_pit_context=True)
