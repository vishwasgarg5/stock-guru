import pytest

from stock_guru.free_pit_universe import (
    audit_membership_cardinality,
    build_event_derived_intervals,
    membership_on_date,
)


def test_event_derived_intervals_preserve_primary_source_ids():
    intervals = build_event_derived_intervals(
        {"AAA", "BBB"},
        [
            {"effective_date": "2025-02-01", "symbol": "AAA", "action": "exclude", "source_id": "nse-1"},
            {"effective_date": "2025-02-01", "symbol": "CCC", "action": "include", "source_id": "nse-1"},
        ],
        anchor_date="2025-01-01",
    )
    assert membership_on_date(intervals, "2025-01-15") == {"AAA", "BBB"}
    assert membership_on_date(intervals, "2025-02-01") == {"BBB", "CCC"}
    assert all(row["evidence_tier"] == "EVENT_DERIVED" for row in intervals)
    assert {"nse-1"} <= {source for row in intervals for source in row["source_ids"]}


def test_invalid_exclusion_fails_closed():
    with pytest.raises(ValueError, match="unsupported or invalid"):
        build_event_derived_intervals(
            {"AAA"},
            [{"effective_date": "2025-02-01", "symbol": "BBB", "action": "exclude", "source_id": "nse-1"}],
            anchor_date="2025-01-01",
        )


def test_cardinality_audit_blocks_without_filling():
    intervals = build_event_derived_intervals(
        {"AAA", "BBB"},
        [],
        anchor_date="2025-01-01",
    )
    report = audit_membership_cardinality(intervals, ["2025-01-01"], expected_size=500)
    assert report["status"] == "BLOCKED"
    assert report["dates"][0]["count"] == 2
