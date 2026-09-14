import pandas as pd
import pytest

from stock_guru.universe_events import apply_events_to_baseline, load_events


def _baseline():
    return pd.DataFrame({"as_of": pd.to_datetime(["2020-01-01"]), "symbol": ["A"]})


def _events():
    return pd.DataFrame({
        "effective_date": pd.to_datetime(["2021-01-01", "2022-01-01"]),
        "symbol": ["B", "A"],
        "action": ["include", "exclude"],
        "source": ["official", "official"],
        "source_id": ["inc-1", "exc-1"],
    })


def test_events_build_authoritative_snapshots_without_fabricating_history():
    out = apply_events_to_baseline(_baseline(), _events())
    assert set(out.loc[out["as_of"] == pd.Timestamp("2020-01-01"), "symbol"]) == {"A"}
    assert set(out.loc[out["as_of"] == pd.Timestamp("2021-01-01"), "symbol"]) == {"A", "B"}
    assert set(out.loc[out["as_of"] == pd.Timestamp("2022-01-01"), "symbol"]) == {"B"}


def test_events_require_provenance_columns():
    with pytest.raises(ValueError, match="Missing event columns"):
        load_events(pd.io.common.StringIO("effective_date,symbol,action\n2021-01-01,B,include\n"))


def test_events_reject_duplicate_symbol_date():
    events = _events().copy()
    events.loc[1, "effective_date"] = events.loc[0, "effective_date"]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        apply_events_to_baseline(_baseline(), events)


def test_events_reject_excluding_non_member():
    events = _events().copy()
    events.loc[1, "symbol"] = "C"
    with pytest.raises(ValueError, match="non-member"):
        apply_events_to_baseline(_baseline(), events)


def test_events_reject_event_on_or_before_baseline():
    events = _events().copy()
    events.loc[0, "effective_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="after the baseline"):
        apply_events_to_baseline(_baseline(), events)
