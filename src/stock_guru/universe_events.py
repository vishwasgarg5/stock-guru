from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_EVENT_COLUMNS = {"effective_date", "symbol", "action", "source", "source_id"}
VALID_ACTIONS = {"include", "exclude"}


def load_events(path: str | Path) -> pd.DataFrame:
    """Load a provenance-bearing inclusion/exclusion event ledger.

    Events are facts supplied by an external source; this loader never invents
    membership before the first supplied event or baseline snapshot.
    """
    df = pd.read_csv(path)
    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.normalize()
    for column in ("symbol", "action", "source", "source_id"):
        df[column] = df[column].astype(str).str.strip()
    if df["effective_date"].isna().any():
        raise ValueError("Events contain invalid effective dates")
    if df["symbol"].eq("").any() or df["source"].eq("").any() or df["source_id"].eq("").any():
        raise ValueError("Events contain blank symbol or provenance fields")
    df["action"] = df["action"].str.lower()
    if (~df["action"].isin(VALID_ACTIONS)).any():
        raise ValueError("Events contain unsupported actions")
    if df.duplicated(["effective_date", "symbol"]).any():
        raise ValueError("Events contain conflicting duplicate effective_date/symbol rows")
    return df.sort_values(["effective_date", "symbol"]).reset_index(drop=True)


def apply_events_to_baseline(baseline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Produce dated constituent snapshots from a supplied baseline and event ledger.

    ``baseline`` is authoritative at its ``as_of`` date. Events take effect on
    their effective date. The function refuses an exclusion for a symbol that is
    not currently a member and refuses an inclusion for an existing member, which
    catches malformed event files rather than silently changing history.
    """
    if not {"as_of", "symbol"}.issubset(baseline.columns):
        raise ValueError("Baseline must contain as_of and symbol")
    b = baseline[["as_of", "symbol"]].copy()
    b["as_of"] = pd.to_datetime(b["as_of"], errors="coerce").dt.normalize()
    b["symbol"] = b["symbol"].astype(str).str.strip()
    if b["as_of"].isna().any() or b["symbol"].eq("").any():
        raise ValueError("Baseline contains invalid dates or blank symbols")
    if b.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Baseline contains duplicate as_of/symbol rows")
    if b["as_of"].nunique() != 1:
        raise ValueError("Baseline must contain exactly one authoritative as_of date")

    e = events.copy()
    missing = REQUIRED_EVENT_COLUMNS - set(e.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    e["effective_date"] = pd.to_datetime(e["effective_date"], errors="coerce").dt.normalize()
    for column in ("symbol", "action", "source", "source_id"):
        e[column] = e[column].astype(str).str.strip()
    e["action"] = e["action"].str.lower()
    if e["effective_date"].isna().any() or (~e["action"].isin(VALID_ACTIONS)).any():
        raise ValueError("Events contain invalid dates or actions")
    if e[["symbol", "source", "source_id"]].eq("").any().any():
        raise ValueError("Events contain blank symbol or provenance fields")
    if e.duplicated(["effective_date", "symbol"]).any():
        raise ValueError("Events contain conflicting duplicate effective_date/symbol rows")

    baseline_date = b["as_of"].iloc[0]
    if (e["effective_date"] <= baseline_date).any():
        raise ValueError("Events must occur after the baseline as_of date")

    members = set(b["symbol"])
    rows = [{"as_of": baseline_date, "symbol": symbol} for symbol in sorted(members)]
    for date, group in e.sort_values(["effective_date", "symbol"]).groupby("effective_date", sort=True):
        for row in group.itertuples(index=False):
            if row.action == "include":
                if row.symbol in members:
                    raise ValueError(f"Cannot include existing member {row.symbol} on {date.date()}")
                members.add(row.symbol)
            else:
                if row.symbol not in members:
                    raise ValueError(f"Cannot exclude non-member {row.symbol} on {date.date()}")
                members.remove(row.symbol)
        rows.extend({"as_of": date, "symbol": symbol} for symbol in sorted(members))
    return pd.DataFrame(rows, columns=["as_of", "symbol"])


def build_universe_history_from_events(baseline_path: str | Path, events_path: str | Path,
                                       output_path: str | Path) -> Path:
    """Build point-in-time membership intervals from an authoritative baseline and events."""
    baseline = pd.read_csv(baseline_path, parse_dates=["as_of"])
    events = load_events(events_path)
    from .universe_history import build_membership_intervals

    snapshots = apply_events_to_baseline(baseline, events)
    intervals = build_membership_intervals(snapshots)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output, index=False)
    return output
