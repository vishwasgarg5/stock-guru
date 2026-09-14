from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"as_of", "symbol"}
INTERVAL_COLUMNS = {"symbol", "start_date", "end_date"}


def load_snapshots(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["as_of"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    return load_snapshots_from_frame(df)


def load_snapshots_from_frame(snapshots: pd.DataFrame) -> pd.DataFrame:
    clean = snapshots.copy()
    missing = REQUIRED - set(clean.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    clean["as_of"] = pd.to_datetime(clean["as_of"], errors="coerce").dt.normalize()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    if clean["as_of"].isna().any() or clean["symbol"].eq("").any():
        raise ValueError("Snapshots contain invalid dates or blank symbols")
    if clean.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Snapshots contain duplicate as_of/symbol rows")
    return clean.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def validate_membership_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    missing = INTERVAL_COLUMNS - set(intervals.columns)
    if missing:
        raise ValueError(f"Missing interval columns: {sorted(missing)}")
    clean = intervals[list(INTERVAL_COLUMNS)].copy()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    clean["start_date"] = pd.to_datetime(clean["start_date"], errors="coerce").dt.normalize()
    clean["end_date"] = pd.to_datetime(clean["end_date"], errors="coerce").dt.normalize()
    if clean["symbol"].eq("").any() or clean["start_date"].isna().any():
        raise ValueError("Universe intervals contain blank symbols or invalid start dates")
    if (clean["end_date"].notna() & clean["start_date"].ge(clean["end_date"].fillna(pd.Timestamp.max))).any():
        raise ValueError("Universe intervals must have end_date after start_date")
    clean = clean.sort_values(["symbol", "start_date", "end_date"], na_position="last").reset_index(drop=True)
    for symbol, group in clean.groupby("symbol", sort=False):
        starts = group["start_date"].tolist(); ends = group["end_date"].tolist()
        for idx in range(1, len(starts)):
            if pd.isna(ends[idx - 1]) or starts[idx] < ends[idx - 1]:
                raise ValueError(f"Overlapping open-ended or dated intervals for symbol {symbol}")
    return clean


def build_membership_intervals(snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshots = load_snapshots_from_frame(snapshots)
    dates = sorted(snapshots["as_of"].unique())
    rows: list[dict] = []
    for i, start in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else pd.NaT
        for symbol in snapshots.loc[snapshots["as_of"] == start, "symbol"]:
            rows.append({"symbol": symbol, "start_date": start, "end_date": end})
    return validate_membership_intervals(pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"]))


def universe_for_date(intervals: pd.DataFrame, as_of: str | pd.Timestamp) -> set[str]:
    intervals = validate_membership_intervals(intervals)
    if intervals.empty:
        return set()
    d = pd.Timestamp(as_of).normalize()
    start = intervals["start_date"]; end = intervals["end_date"]
    mask = start.le(d) & (end.isna() | end.gt(d))
    return set(intervals.loc[mask, "symbol"].astype(str))


def apply_point_in_time_universe(prices: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(required - set(prices.columns))}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.strip().str.upper()
    if p["date"].isna().any() or p["symbol"].eq("").any():
        raise ValueError("Prices contain invalid dates or blank symbols")
    i = validate_membership_intervals(intervals)
    keep = pd.Series(False, index=p.index)
    for _, row in i.iterrows():
        mask = p["date"].ge(row.start_date)
        if pd.notna(row.end_date): mask &= p["date"].lt(row.end_date)
        keep |= mask & p["symbol"].eq(row.symbol)
    return p.loc[keep].copy()


def membership_counts(intervals: pd.DataFrame) -> pd.DataFrame:
    """Return constituent counts at each supplied interval start date."""
    clean = validate_membership_intervals(intervals)
    if clean.empty:
        return pd.DataFrame(columns=["as_of", "constituents"])
    return (clean.groupby("start_date")["symbol"].nunique().rename("constituents").reset_index().rename(columns={"start_date": "as_of"}))


def validate_membership_count_range(intervals: pd.DataFrame, *, minimum: int | None = None, maximum: int | None = None) -> pd.DataFrame:
    """Reject supplied PIT history outside an explicitly requested count range."""
    counts = membership_counts(intervals)
    if minimum is not None and (counts["constituents"] < minimum).any():
        raise ValueError("PIT universe contains a snapshot below the configured minimum")
    if maximum is not None and (counts["constituents"] > maximum).any():
        raise ValueError("PIT universe contains a snapshot above the configured maximum")
    return counts


def build_universe_history(snapshot_path: str | Path, output_path: str | Path) -> Path:
    snapshots = load_snapshots(snapshot_path)
    intervals = build_membership_intervals(snapshots)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output, index=False)
    return output
