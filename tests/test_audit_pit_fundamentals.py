from pathlib import Path

import pandas as pd

from scripts.audit_pit_fundamentals import audit


def _frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": "AAA", "reported_date": "2025-03-31", "available_date": "2025-05-15",
        "available_timestamp": "2025-05-15T10:30:00Z", "source": "NSE filing",
        "source_id": "nse-aaa-2025q4-v1", "source_url": "https://example.invalid/filing",
        "source_sha256": "a" * 64, "version": "v1", "revenue": 100.0,
    }])


def test_missing_file_blocks(tmp_path: Path):
    report = audit(tmp_path / "missing.csv")
    assert report["status"] == "blocked"


def test_missing_timestamp_blocks(tmp_path: Path):
    path = tmp_path / "fundamentals.csv"
    _frame().drop(columns=["available_timestamp"]).to_csv(path, index=False)
    report = audit(path)
    assert report["status"] == "blocked"
    assert "available_timestamp" in report["reason"]


def test_valid_filing_is_accepted(tmp_path: Path):
    path = tmp_path / "fundamentals.csv"
    _frame().to_csv(path, index=False)
    report = audit(path)
    assert report["status"] == "validated"
    assert report["rows"] == 1
    assert report["symbols"] == 1


def test_duplicate_version_blocks(tmp_path: Path):
    path = tmp_path / "fundamentals.csv"
    pd.concat([_frame(), _frame()], ignore_index=True).to_csv(path, index=False)
    report = audit(path)
    assert report["status"] == "blocked"
    assert "Duplicate PIT observations" in report["reason"]
