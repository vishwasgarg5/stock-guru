from pathlib import Path

from stock_guru.event_audit import audit_event_chain


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "nifty500_event_chain_manifest.csv"


def test_event_chain_audit_passes_current_manifest():
    report = audit_event_chain(MANIFEST)
    assert report["status"] == "PASS"
    assert report["source_count"] >= 16
    assert report["event_count"] >= 200
    assert report["duplicate_manifest_source_ids"] == []
    assert report["date_mismatches"] == {}
    assert report["unknown_sources"] == []
    assert report["missing_files"] == []


def test_event_chain_audit_blocks_missing_evidence(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("source_id,effective_date,evidence_file,status\nmissing,2024-01-01,data/does_not_exist.csv,verified\n", encoding="utf-8")
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert report["missing_files"]


def test_event_chain_audit_blocks_invalid_manifest_status(tmp_path):
    evidence = tmp_path / "events.csv"
    evidence.write_text("effective_date,symbol,action,source,source_id\n2024-01-01,AAA,exclude,NSE,x\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"source_id,effective_date,evidence_file,status\nx,2024-01-01,{evidence},pending\n", encoding="utf-8")
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert "pending" in report["invalid_statuses"]


def test_event_chain_audit_blocks_effective_date_mismatch(tmp_path):
    evidence = tmp_path / "events.csv"
    evidence.write_text("effective_date,symbol,action,source,source_id\n2024-01-02,AAA,exclude,NSE,x\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"source_id,effective_date,evidence_file,status\nx,2024-01-01,{evidence},verified\n", encoding="utf-8")
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert report["date_mismatches"]["x"]["observed"] == ["2024-01-02"]


def test_event_chain_audit_blocks_duplicate_manifest_source_ids(tmp_path):
    evidence = tmp_path / "events.csv"
    evidence.write_text("effective_date,symbol,action,source,source_id\n2024-01-01,AAA,exclude,NSE,x\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"source_id,effective_date,evidence_file,status\nx,2024-01-01,{evidence},verified\nx,2024-01-01,{evidence},verified\n", encoding="utf-8")
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert report["duplicate_manifest_source_ids"] == ["x"]


def test_event_chain_audit_blocks_unknown_event_source(tmp_path):
    evidence = tmp_path / "events.csv"
    evidence.write_text("effective_date,symbol,action,source,source_id\n2024-01-01,AAA,exclude,NSE,other\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"source_id,effective_date,evidence_file,status\nx,2024-01-01,{evidence},verified\n", encoding="utf-8")
    report = audit_event_chain(manifest)
    assert report["status"] == "BLOCKED"
    assert report["unknown_sources"] == ["other"]
