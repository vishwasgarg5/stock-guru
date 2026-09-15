import csv
from pathlib import Path

import pytest

from scripts.reconstruct_nifty500_history import load_anchor, load_verified_events, reconstruct


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reconstruction_requires_exact_500_row_anchor(tmp_path: Path):
    snapshot = _write(tmp_path / "snapshot.csv", "as_of,symbol\n2026-09-15,AAA\n")
    with pytest.raises(ValueError, match="Expected 500"):
        load_anchor(snapshot)


def test_reconstruction_rejects_unverified_event_source(tmp_path: Path):
    events = _write(
        tmp_path / "events.csv",
        "effective_date,symbol,action,source_id\n2026-01-01,AAA,exclude,bad-source\n",
    )
    sources = _write(tmp_path / "sources.csv", "source_id,status\nknown,verified\n")
    with pytest.raises(ValueError, match="unverified source IDs"):
        load_verified_events(events, sources)


def test_reconstruction_uses_snapshot_as_explicit_anchor(tmp_path: Path):
    rows = ["as_of,symbol"] + [f"2026-02-01,S{i:03d}" for i in range(500)]
    snapshot = _write(tmp_path / "snapshot.csv", "\n".join(rows) + "\n")
    events = _write(tmp_path / "events.csv", "effective_date,symbol,action,source_id\n")
    sources = _write(tmp_path / "sources.csv", "source_id,status\nknown,verified\n")
    intervals = reconstruct(snapshot, events, sources)
    assert len(intervals) == 500
    assert all(row["start_date"] == "2026-02-01" for row in intervals)
    # A snapshot-only interval is directly certified by the authoritative
    # anchor; EVENT_DERIVED is reserved for intervals established through
    # verified membership events.
    assert all(row["evidence_tier"] == "CERTIFIED" for row in intervals)
    assert all(not row["source_ids"] for row in intervals)


def test_reconstruction_does_not_fabricate_pre_event_history(tmp_path: Path):
    symbols = [f"S{i:03d}" for i in range(500)]
    snapshot = _write(tmp_path / "snapshot.csv", "as_of,symbol\n" + "\n".join(f"2026-02-01,{s}" for s in symbols) + "\n")
    events = _write(
        tmp_path / "events.csv",
        "effective_date,symbol,action,source_id\n2026-01-01,S000,include,known\n",
    )
    sources = _write(tmp_path / "sources.csv", "source_id,status\nknown,verified\n")
    intervals = reconstruct(snapshot, events, sources)
    assert any(row["evidence_tier"] == "BLOCKED" for row in intervals)
