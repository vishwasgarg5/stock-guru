from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Iterable

from .event_chain import apply_events


@dataclass(frozen=True)
class MembershipInterval:
    symbol: str
    start_date: str
    end_date: str | None
    evidence_tier: str
    source_ids: tuple[str, ...]


def _ordered_events(events: Iterable[dict[str, str]], *, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, str]]:
    rows = sorted(events, key=lambda r: (r["effective_date"], r["source_id"], r["action"], r["symbol"]))
    if start_date is not None:
        rows = [r for r in rows if r["effective_date"] >= start_date]
    if end_date is not None:
        rows = [r for r in rows if r["effective_date"] <= end_date]
    return rows


def build_event_derived_intervals(
    initial_symbols: set[str],
    events: Iterable[dict[str, str]],
    *,
    anchor_date: str,
    end_date: str | None = None,
) -> list[dict[str, object]]:
    """Reconstruct PIT membership from a public anchor snapshot and verified events.

    The function is deliberately fail-closed: an invalid transition raises rather
    than inventing a missing constituent. Every generated interval is marked
    EVENT_DERIVED because membership after the anchor depends on primary events.
    """
    rows = _ordered_events(events, start_date=anchor_date, end_date=end_date)
    active = {s.strip().upper() for s in initial_symbols if s.strip()}
    open_intervals: dict[str, tuple[str, set[str]]] = {
        symbol: (anchor_date, set()) for symbol in active
    }
    completed: list[MembershipInterval] = []

    for event in rows:
        effective = event["effective_date"]
        symbol = event["symbol"].strip().upper()
        source_id = event["source_id"].strip()
        if event["action"] == "exclude":
            if symbol not in active or symbol not in open_intervals:
                raise ValueError(f"Cannot exclude {symbol}: unsupported or invalid membership state at {effective}")
            start, sources = open_intervals.pop(symbol)
            sources = set(sources)
            sources.add(source_id)
            completed.append(MembershipInterval(symbol, start, effective, "EVENT_DERIVED", tuple(sorted(sources))))
            active.remove(symbol)
        elif event["action"] == "include":
            if symbol in active:
                raise ValueError(f"Cannot include {symbol}: already active at {effective}")
            active.add(symbol)
            open_intervals[symbol] = (effective, {source_id})
        else:
            raise ValueError(f"Unsupported action: {event['action']}")

    for symbol, (start, sources) in open_intervals.items():
        completed.append(MembershipInterval(symbol, start, None, "EVENT_DERIVED", tuple(sorted(sources))))

    return [asdict(interval) for interval in sorted(completed, key=lambda x: (x.symbol, x.start_date))]


def membership_on_date(
    intervals: Iterable[dict[str, object]],
    target_date: str,
) -> set[str]:
    """Return membership for a date using only explicitly reconstructed intervals."""
    target = date.fromisoformat(target_date)
    result: set[str] = set()
    for interval in intervals:
        start = date.fromisoformat(str(interval["start_date"]))
        end_value = interval.get("end_date")
        end = date.fromisoformat(str(end_value)) if end_value else None
        if start <= target and (end is None or target < end):
            result.add(str(interval["symbol"]).upper())
    return result


def audit_membership_cardinality(intervals: Iterable[dict[str, object]], dates: Iterable[str], *, expected_size: int = 500) -> dict[str, object]:
    """Audit reconstructed membership without filling or repairing gaps."""
    results: list[dict[str, object]] = []
    for target in dates:
        members = membership_on_date(intervals, target)
        results.append({"as_of": target, "count": len(members), "expected": expected_size, "status": "PASS" if len(members) == expected_size else "BLOCKED"})
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "BLOCKED",
        "expected_size": expected_size,
        "dates": results,
    }
