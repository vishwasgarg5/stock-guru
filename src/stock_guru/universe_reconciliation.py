from __future__ import annotations

import pandas as pd

from .universe_source import validate_historical_snapshots


def reconcile_snapshots(
    supplied: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    require_all_reconstructed_dates: bool = False,
) -> dict[str, object]:
    """Compare two membership sources on common dates without filling missing history."""
    left = validate_historical_snapshots(supplied)
    right = validate_historical_snapshots(reconstructed)
    left_dates = set(left["as_of"])
    right_dates = set(right["as_of"])
    common = sorted(left_dates & right_dates)
    if require_all_reconstructed_dates and not right_dates.issubset(left_dates):
        missing = sorted(right_dates - left_dates)
        raise ValueError(f"Supplied snapshots missing reconstructed dates: {[str(d.date()) for d in missing]}")

    mismatches = []
    for date in common:
        a = set(left.loc[left["as_of"] == date, "symbol"])
        b = set(right.loc[right["as_of"] == date, "symbol"])
        if a != b:
            mismatches.append({
                "as_of": str(date.date()),
                "supplied_count": len(a),
                "reconstructed_count": len(b),
                "only_supplied": sorted(a - b),
                "only_reconstructed": sorted(b - a),
            })
    return {
        "status": "PASS" if not mismatches else "MISMATCH",
        "common_snapshot_dates": len(common),
        "supplied_snapshot_dates": len(left_dates),
        "reconstructed_snapshot_dates": len(right_dates),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "coverage_comparable": bool(common),
    }
