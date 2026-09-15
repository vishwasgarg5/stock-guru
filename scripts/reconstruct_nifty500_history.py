from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stock_guru.free_pit_universe import build_reverse_event_derived_intervals


REQUIRED_EVENT_COLUMNS = {"effective_date", "symbol", "action", "source_id"}
REQUIRED_SOURCE_COLUMNS = {"source_id", "status"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return set(next(csv.reader(handle), []))


def load_verified_events(event_ledger: Path, source_log: Path) -> list[dict[str, str]]:
    events = _read_csv(event_ledger)
    sources = _read_csv(source_log)
    event_columns = set(events[0]) if events else _csv_header(event_ledger)
    source_columns = set(sources[0]) if sources else _csv_header(source_log)
    if not REQUIRED_EVENT_COLUMNS <= event_columns:
        raise ValueError("Event ledger is missing required columns")
    if not REQUIRED_SOURCE_COLUMNS <= source_columns:
        raise ValueError("Historical source log is missing required columns")
    verified = {
        row["source_id"]
        for row in sources
        if row.get("status") in {"verified", "correction-preserved-in-file"}
    }
    unknown = sorted({row["source_id"] for row in events} - verified)
    if unknown:
        raise ValueError(f"Event ledger contains unverified source IDs: {', '.join(unknown)}")
    return events


def load_anchor(snapshot: Path) -> tuple[str, set[str]]:
    rows = _read_csv(snapshot)
    if not rows or "as_of" not in rows[0] or "symbol" not in rows[0]:
        raise ValueError("Snapshot must contain as_of and symbol columns")
    dates = {row["as_of"].strip() for row in rows}
    if len(dates) != 1 or not next(iter(dates)):
        raise ValueError("Snapshot must contain exactly one non-empty as_of date")
    symbols = {row["symbol"].strip().upper() for row in rows if row["symbol"].strip()}
    if len(rows) != 500 or len(symbols) != 500:
        raise ValueError(f"Expected 500 unique snapshot rows, got {len(rows)} rows and {len(symbols)} symbols")
    return next(iter(dates)), symbols


def reconstruct(snapshot: Path, event_ledger: Path, source_log: Path, *, start_date: str | None = None) -> list[dict[str, object]]:
    anchor_date, symbols = load_anchor(snapshot)
    events = load_verified_events(event_ledger, source_log)
    return build_reverse_event_derived_intervals(
        symbols,
        events,
        anchor_date=anchor_date,
        start_date=start_date,
    )


def write_intervals(path: Path, intervals: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["symbol", "start_date", "end_date", "evidence_tier", "source_ids"],
        )
        writer.writeheader()
        for row in intervals:
            writer.writerow(
                {
                    **row,
                    "source_ids": ";".join(str(source) for source in row["source_ids"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct free-public NIFTY 500 PIT membership from a captured anchor snapshot and verified primary events")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--events", default="data/nifty500_events.csv")
    parser.add_argument("--source-log", default="data/nifty500_historical_source_log.csv")
    parser.add_argument("--start-date")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    intervals = reconstruct(
        Path(args.snapshot),
        Path(args.events),
        Path(args.source_log),
        start_date=args.start_date,
    )
    write_intervals(Path(args.output), intervals)
    print(f"Wrote {len(intervals)} membership intervals to {args.output}")


if __name__ == "__main__":
    main()
