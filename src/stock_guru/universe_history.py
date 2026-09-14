from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"as_of", "symbol"}


def load_snapshots(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["as_of"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    if df["as_of"].isna().any() or df["symbol"].eq("").any():
        raise ValueError("Snapshots contain invalid dates or blank symbols")
    if df.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Snapshots contain duplicate as_of/symbol rows")
    return df.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def build_membership_intervals(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Convert dated constituent observations into membership intervals.

    A snapshot is authoritative from its observation date until the next snapshot.
    The final interval remains open-ended. This does not fabricate membership before
    the first supplied snapshot.
    """
    snapshots = snapshots.copy()
    required = REQUIRED
    if not required.issubset(snapshots.columns):
        raise ValueError(f"Missing snapshot columns: {sorted(required - set(snapshots.columns))}")
    snapshots["as_of"] = pd.to_datetime(snapshots["as_of"]).dt.normalize()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.strip()
    snapshots = snapshots.drop_duplicates(["as_of", "symbol"])
    dates = sorted(snapshots["as_of"].unique())
    rows: list[dict] = []
    for i, start in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else pd.NaT
        members = snapshots.loc[snapshots["as_of"] == start, "symbol"]
        for symbol in members:
            rows.append({"symbol": symbol, "start_date": start, "end_date": end})
    return pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])


def universe_for_date(intervals: pd.DataFrame, as_of: str | pd.Timestamp) -> set[str]:
    """Return only membership supported by supplied historical snapshots."""
    if intervals.empty:
        return set()
    d = pd.Timestamp(as_of).normalize()
    start = pd.to_datetime(intervals["start_date"]).dt.normalize()
    end = pd.to_datetime(intervals["end_date"], errors="coerce").dt.normalize()
    mask = start.le(d) & (end.isna() | end.gt(d))
    return set(intervals.loc[mask, "symbol"].astype(str))


def apply_point_in_time_universe(prices: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(required - set(prices.columns))}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    if p["date"].isna().any():
        raise ValueError("Prices contain invalid dates")
    i = intervals.copy()
    i["start_date"] = pd.to_datetime(i["start_date"]).dt.normalize()
    i["end_date"] = pd.to_datetime(i["end_date"], errors="coerce").dt.normalize()
    keep = pd.Series(False, index=p.index)
    for _, row in i.iterrows():
        mask = p["date"].ge(row.start_date)
        if pd.notna(row.end_date):
            mask &= p["date"].lt(row.end_date)
        keep |= mask & p["symbol"].eq(row.symbol)
    return p.loc[keep].copy()


def build_universe_history(snapshot_path: str | Path, output_path: str | Path) -> Path:
    snapshots = load_snapshots(snapshot_path)
    intervals = build_membership_intervals(snapshots)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output, index=False)
    return output
