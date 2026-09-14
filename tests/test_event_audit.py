from pathlib import Path

import pytest

from stock_guru.event_audit import audit_event_chain


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "nifty500_event_chain_manifest.csv"


def test_event_chain_audit_passes_current_manifest():
    report = audit_event_chain(MANIFEST)
    assert report["status"] == "PASS"
    assert report["source_count"] >= 16
    assert report["event_count"] >= 200
    assert report["duplicate_manifest_source_ids"] == []
    assert report["missing_files"] == []


def test_event_chain_audit_blocks_missing_evidence(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_id,effective_date,evidence_file,status\n"
        "missing,2024-01-01,data/does_not_exist.csv,verified\n",
        encoding="utf-8",
    )
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert report["missing_files"]
