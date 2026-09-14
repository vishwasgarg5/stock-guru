import json

import pytest

from stock_guru.universe_audit import (
    build_source_audit,
    compare_source_fingerprint,
    save_source_audit,
    sha256_file,
)


def _manifest():
    return {
        "dataset": "nifty500_membership",
        "source_name": "Example authoritative source",
        "source_url": "https://example.com/nifty500",
        "retrieved_at": "2026-09-14T00:00:00Z",
        "license_or_terms": "Use subject to source terms",
    }


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(b"as_of,symbol\n2024-01-31,ABC\n")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_source_audit_records_fingerprints(tmp_path):
    snapshot = tmp_path / "snapshots.csv"
    snapshot.write_text("as_of,symbol\n2024-01-31,ABC\n2024-02-01,XYZ\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    report = build_source_audit(snapshot, manifest, gap_threshold_days=0)
    assert report["audit_status"] == "PASS"
    assert report["snapshot_sha256"] == sha256_file(snapshot)
    assert report["manifest_sha256"] == sha256_file(manifest)
    assert compare_source_fingerprint(report, snapshot, manifest)


def test_source_audit_detects_mutation(tmp_path):
    snapshot = tmp_path / "snapshots.csv"
    snapshot.write_text("as_of,symbol\n2024-01-31,ABC\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    report = build_source_audit(snapshot, manifest)
    snapshot.write_text("as_of,symbol\n2024-01-31,DEF\n", encoding="utf-8")
    assert compare_source_fingerprint(report, snapshot, manifest) is False


def test_source_audit_save_round_trip(tmp_path):
    snapshot = tmp_path / "snapshots.csv"
    snapshot.write_text("as_of,symbol\n2024-01-31,ABC\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    report = build_source_audit(snapshot, manifest)
    output = save_source_audit(report, tmp_path / "audit.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["snapshot_sha256"] == report["snapshot_sha256"]


def test_source_audit_rejects_bad_source(tmp_path):
    snapshot = tmp_path / "snapshots.csv"
    snapshot.write_text("as_of,symbol\n2024-01-31,ABC\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing source manifest fields"):
        build_source_audit(snapshot, manifest)
