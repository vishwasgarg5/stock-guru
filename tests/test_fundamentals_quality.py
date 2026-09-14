import pandas as pd
import pytest

from stock_guru.fundamentals_quality import audit_pit_fundamentals, validate_fundamentals_manifest


def manifest():
    return {"dataset":"pit_fundamentals","source_name":"test","source_url":"https://example.com","retrieved_at":"2026-09-14T00:00:00Z","license_or_terms":"test terms"}


def test_manifest_requires_all_provenance_fields():
    with pytest.raises(ValueError):
        validate_fundamentals_manifest({"dataset":"pit_fundamentals"})


def test_audit_reports_hash_and_coverage(tmp_path):
    path = tmp_path / "fundamentals.csv"
    pd.DataFrame([{"symbol":"ABC","reported_date":"2026-06-30","available_date":"2026-07-02","roe":10.0,"source":"s","source_id":"1"}]).to_csv(path, index=False)
    report = audit_pit_fundamentals(path, manifest())
    assert report["provenance_validated"] is True
    assert len(report["sha256"]) == 64
    assert report["unique_symbols"] == 1
