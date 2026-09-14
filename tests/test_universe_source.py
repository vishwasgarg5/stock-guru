import json

import pandas as pd
import pytest

from stock_guru.universe_source import file_sha256, load_source_manifest, validate_historical_snapshots, validate_source_bundle, validate_source_manifest


def manifest():
    return {"dataset": "nifty500_membership", "source_name": "Example authoritative source", "source_url": "https://example.com/nifty500", "retrieved_at": "2026-09-14T00:00:00Z", "license_or_terms": "Use subject to source terms"}


def test_source_manifest_requires_provenance_fields():
    with pytest.raises(ValueError, match="Missing source manifest fields"):
        validate_source_manifest({"dataset": "nifty500_membership"})


def test_source_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    assert load_source_manifest(path)["dataset"] == "nifty500_membership"


def test_historical_snapshot_rejects_partial_set_when_source_requires_full_size():
    snapshots = pd.DataFrame({"as_of": pd.to_datetime(["2024-01-31", "2024-01-31"]), "symbol": ["A", "B"]})
    with pytest.raises(ValueError, match="not full sets"):
        validate_historical_snapshots(snapshots, expected_constituents=500, require_full_snapshot_size=True)


def test_historical_snapshot_can_be_validated_without_fabrication():
    snapshots = pd.DataFrame({"as_of": pd.to_datetime(["2024-01-31"] * 3), "symbol": ["A", "B", "C"]})
    out = validate_historical_snapshots(snapshots)
    assert len(out) == 3
    assert set(out["symbol"]) == {"A", "B", "C"}


def test_source_bundle_returns_audit_summary(tmp_path):
    snapshots = tmp_path / "snapshots.csv"
    snapshots.write_text("as_of,symbol\n2024-01-31,A\n2024-01-31,B\n2024-02-01,A\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    report = validate_source_bundle(snapshots, manifest_path)
    assert report["provenance_validated"] is True
    assert report["snapshot_dates"] == 2
    assert report["min_constituents"] == 1
    assert report["max_constituents"] == 2
    assert report["source_sha256"] == file_sha256(snapshots)


def test_source_bundle_requires_manifest(tmp_path):
    snapshots = tmp_path / "snapshots.csv"
    snapshots.write_text("as_of,symbol\n2024-01-31,A\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        validate_source_bundle(snapshots, tmp_path / "missing.json")


def test_source_bundle_rejects_mutated_source(tmp_path):
    snapshots = tmp_path / "snapshots.csv"
    snapshots.write_text("as_of,symbol\n2024-01-31,A\n", encoding="utf-8")
    m = manifest()
    m["source_sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_source_bundle(snapshots, manifest_path)
