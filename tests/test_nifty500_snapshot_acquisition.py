from pathlib import Path

import pytest

from scripts.acquire_nifty500_snapshot import save_snapshot


def test_snapshot_writer_records_distinct_source_and_file_hashes(monkeypatch, tmp_path: Path):
    class Response:
        content = b"Symbol,Company Name\nAAA,Alpha\nBBB,Beta\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.EXPECTED_COUNT", 2)

    output = tmp_path / "snapshot.csv"
    manifest = tmp_path / "manifest.csv"
    save_snapshot(output, manifest, as_of="2026-09-15")

    snapshot = output.read_text(encoding="utf-8")
    assert snapshot.splitlines()[0].startswith("as_of,")
    assert snapshot.splitlines()[1].startswith("2026-09-15,")

    text = manifest.read_text(encoding="utf-8")
    assert "as_of" in text
    assert "retrieved_at" in text
    assert "source_sha256" in text
    assert "file_sha256" in text
    source_hash, file_hash = text.strip().splitlines()[1].split(",")[-2:]
    assert source_hash
    assert file_hash
    assert source_hash != file_hash


def test_snapshot_acquisition_requires_explicit_as_of(monkeypatch, tmp_path: Path):
    class Response:
        content = b"Symbol\nAAA\nBBB\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.EXPECTED_COUNT", 2)

    with pytest.raises(TypeError, match="as_of"):
        save_snapshot(tmp_path / "snapshot.csv", tmp_path / "manifest.csv")


def test_snapshot_acquisition_blocks_wrong_cardinality(monkeypatch, tmp_path: Path):
    class Response:
        content = b"Symbol\nAAA\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr("scripts.acquire_nifty500_snapshot.EXPECTED_COUNT", 500)

    with pytest.raises(ValueError, match="Expected 500"):
        save_snapshot(tmp_path / "snapshot.csv", tmp_path / "manifest.csv", as_of="2026-09-15")


def test_snapshot_acquisition_rejects_invalid_as_of(monkeypatch, tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid isoformat string"):
        save_snapshot(tmp_path / "snapshot.csv", tmp_path / "manifest.csv", as_of="not-a-date")
