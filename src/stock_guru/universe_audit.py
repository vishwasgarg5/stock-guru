from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

import pandas as pd

from .universe_source import validate_source_bundle


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a source file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_audit(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_constituents: int | None = None,
    require_full_snapshot_size: bool = False,
    gap_threshold_days: int | None = None,
) -> dict[str, Any]:
    """Combine provenance, structural validation, and an immutable file fingerprint."""
    snapshot_path = Path(snapshot_path)
    manifest_path = Path(manifest_path)
    report = validate_source_bundle(
        snapshot_path,
        manifest_path,
        expected_constituents=expected_constituents,
        require_full_snapshot_size=require_full_snapshot_size,
        gap_threshold_days=gap_threshold_days,
    )
    report["snapshot_sha256"] = sha256_file(snapshot_path)
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["snapshot_file"] = str(snapshot_path)
    report["manifest_file"] = str(manifest_path)
    report["audit_status"] = "PASS"
    return report


def save_source_audit(report: dict[str, Any], output_path: str | Path) -> Path:
    """Persist an audit report as deterministic, machine-readable JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return output


def compare_source_fingerprint(report: dict[str, Any], snapshot_path: str | Path, manifest_path: str | Path) -> bool:
    """Return whether the current source bytes match fingerprints stored in an audit report."""
    return (
        report.get("snapshot_sha256") == sha256_file(snapshot_path)
        and report.get("manifest_sha256") == sha256_file(manifest_path)
    )
