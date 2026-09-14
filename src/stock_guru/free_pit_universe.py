from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Iterable


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


def _validate_events(events: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    rows = _ordered_events(events)
    seen: set[tuple[str, str, str, str]] = set()
    required = {"effective_date", "symbol", "action", "source_id"}
    for row in rows:
        if not required <= row.keys() or any(not str(row[key]).strip() for key in required):
            raise ValueError("Event row is missing required provenance fields")
        key = (row["effective_date"], row["symbol"].strip().upper(), row["action"], row["source_id"])
        if key in seen:
            raise ValueError(f"Duplicate event key: {key}")
        if row["action"] not in {"include", "exclude"}:
            raise ValueError(f"Unsupported action: {row['action']}")
        date.fromisoformat(row["effective_date"])
        seen.add(key)
    return rows


def build_event_derived_intervals(initial_symbols: set[str], events: Iterable[dict[str, str]], *, anchor_date: str, end_date: str | None = None) -> list[dict[str, object]]:
    """Build forward intervals from an explicitly known membership anchor."""
    date.fromisoformat(anchor_date)
    if end_date is not None:
        date.fromisoformat(end_date)
    rows = _validate_events(events)
    rows = [r for r in rows if r["effective_date"] >= anchor_date and (end_date is None or r["effective_date"] <= end_date)]
    active = {s.strip().upper() for s in initial_symbols if s.strip()}
    open_intervals: dict[str, tuple[str, set[str]]] = {symbol: (anchor_date, set()) for symbol in active}
    completed: list[MembershipInterval] = []
    for event in rows:
        effective = event["effective_date"]
        symbol = event["symbol"].strip().upper()
        source_id = event["source_id"].strip()
        if event["action"] == "exclude":
            if symbol not in active or symbol not in open_intervals:
                raise ValueError(f"Cannot exclude {symbol}: unsupported or invalid membership state at {effective}")
            start, sources = open_intervals.pop(symbol)
            completed.append(MembershipInterval(symbol, start, effective, "EVENT_DERIVED", tuple(sorted(set(sources) | {source_id}))))
            active.remove(symbol)
        else:
            if symbol in active:
                raise ValueError(f"Cannot include {symbol}: already active at {effective}")
            active.add(symbol)
            open_intervals[symbol] = (effective, {source_id})
    for symbol, (start, sources) in open_intervals.items():
        completed.append(MembershipInterval(symbol, start, end_date, "EVENT_DERIVED", tuple(sorted(sources))))
    return [asdict(i) for i in sorted(completed, key=lambda x: (x.symbol, x.start_date))]


def build_reverse_event_derived_intervals(anchor_symbols: set[str], events: Iterable[dict[str, str]], *, anchor_date: str, start_date: str | None = None) -> list[dict[str, object]]:
    """Reconstruct historical membership backwards from a known public snapshot.

    The anchor is the post-event membership on ``anchor_date``. Reverse excludes
    add a member back; reverse includes remove a member. Same-date releases are
    reversed as one deterministic batch. Contradictory state fails closed.
    """
    anchor = date.fromisoformat(anchor_date)
    lower = date.fromisoformat(start_date) if start_date is not None else None
    if lower is not None and lower > anchor:
        raise ValueError("start_date cannot be after anchor_date")
    rows = [r for r in _validate_events(events) if date.fromisoformat(r["effective_date"]) <= anchor]
    if lower is not None:
        rows = [r for r in rows if date.fromisoformat(r["effective_date"]) >= lower]

    active = {s.strip().upper() for s in anchor_symbols if s.strip()}
    event_dates = sorted({r["effective_date"] for r in rows}, reverse=True)
    post_state: dict[str, set[str]] = {anchor_date: set(active)}
    pre_state: dict[str, set[str]] = {}
    transition_sources: dict[tuple[str, str], set[str]] = {}

    for effective in event_dates:
        post_state[effective] = set(active)
        batch = [r for r in rows if r["effective_date"] == effective]
        for event in reversed(batch):
            symbol = event["symbol"].strip().upper()
            source_id = event["source_id"].strip()
            if event["action"] == "exclude":
                if symbol in active:
                    raise ValueError(f"Cannot reverse exclude {symbol}: it is active after {effective}")
                active.add(symbol)
            else:
                if symbol not in active:
                    raise ValueError(f"Cannot reverse include {symbol}: it is inactive after {effective}")
                active.remove(symbol)
            transition_sources.setdefault((effective, symbol), set()).add(source_id)
        pre_state[effective] = set(active)

    # For ascending event dates, the membership immediately before each date is
    # the reconstructed pre-state; immediately after it is the post-state. This
    # preserves effective-date semantics even when the anchor itself is an event.
    dates = sorted(set(event_dates) | {anchor_date})
    intervals: list[MembershipInterval] = []
    for idx, left in enumerate(dates):
        right = dates[idx + 1] if idx + 1 < len(dates) else None
        if left == anchor_date:
            members = post_state[anchor_date]
        else:
            members = post_state[left]
        for symbol in sorted(members):
            sources = transition_sources.get((left, symbol), set())
            tier = "EVENT_DERIVED" if sources else "BLOCKED"
            intervals.append(MembershipInterval(symbol, left, right, tier, tuple(sorted(sources))))

    # Add the historical segment before the oldest event. Its left edge is not
    # established by the supplied evidence, so it is intentionally BLOCKED.
    if event_dates:
        oldest = min(event_dates)
        oldest_pre = pre_state[oldest]
        prior_end = oldest
        prior_start = start_date or "0001-01-01"
        for symbol in sorted(oldest_pre):
            intervals.append(MembershipInterval(symbol, prior_start, prior_end, "BLOCKED", tuple()))

    return [asdict(i) for i in intervals]


def membership_on_date(intervals: Iterable[dict[str, object]], target_date: str) -> set[str]:
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
    return {"status": "PASS" if all(row["status"] == "PASS" for row in results) else "BLOCKED", "expected_size": expected_size, "dates": results}
