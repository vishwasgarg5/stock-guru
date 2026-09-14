from __future__ import annotations

import pandas as pd


def find_snapshot_gaps(snapshots: pd.DataFrame, *, expected_interval_days: int | None = None) -> dict[str, object]:
    """Report gaps between supplied snapshot dates without inventing missing history."""
    required = {"as_of", "symbol"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    dates = pd.to_datetime(snapshots["as_of"], errors="coerce").dt.normalize().dropna().drop_duplicates().sort_values()
    if dates.empty:
        return {"snapshot_dates": 0, "gaps": [], "gap_count": 0, "gap_detection_mode": "none"}
    deltas = dates.diff().dt.days
    gaps = []
    for current, delta in zip(dates.iloc[1:], deltas.iloc[1:]):
        if expected_interval_days is not None and delta > expected_interval_days:
            gaps.append({"from": str((current - pd.Timedelta(days=int(delta))).date()), "to": str(current.date()), "days": int(delta)})
    return {
        "snapshot_dates": int(len(dates)),
        "earliest_date": str(dates.iloc[0].date()),
        "latest_date": str(dates.iloc[-1].date()),
        "gaps": gaps,
        "gap_count": len(gaps),
        "gap_detection_mode": "expected_interval" if expected_interval_days is not None else "none",
    }
