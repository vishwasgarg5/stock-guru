import json

import pandas as pd
import pytest

from stock_guru.universe_source import (
    load_source_manifest,
    validate_historical_snapshots,
    validate_source_manifest,
)


def manifest():
    return {
        "dataset": "nifty500_membership",
        "source_name": "Example authoritative source",
        "source_url": "https://example.com/nifty500",
        "retrieved_at": "2026-09-14T00:00:00Z",
        "license_or_terms": "Use subject to source terms",
    }


def test_source_manifest_requires_provenance_fields():
    with pytest.raises(ValueError, match="Missing source manifest fields"):
        validate_source_manifest({"dataset": "nifty500_membership"})


def test_source_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    assert load_source_manifest(path)["dataset"] == "nifty500_membership"


def test_historical_snapshot_rejects_partial_nifty500_set():
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-31", "2024-01-31"]),
        "symbol": ["A", "B"],
    })
    with pytest.raises(ValueError, match="not full NIFTY 500"):
        validate_historical_snapshots(snapshots)


def test_historical_snapshot_can_be_validated_without_fabrication():
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-31"] * 3),
        "symbol": ["A", "B", "C"],
    })
    out = validate_historical_snapshots(
        snapshots,
        expected_constituents=3,
    )
    assert len(out) == 3
    assert set(out["symbol"]) == {"A", "B", "C"}
