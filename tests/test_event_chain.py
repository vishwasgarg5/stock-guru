import pytest

from stock_guru.event_chain import apply_events, load_event_rows


def test_load_event_rows_preserves_provenance_and_rejects_duplicates(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "effective_date,symbol,action,source,source_id\n"
        "2024-01-02,AAA,exclude,https://example.test/a,source-a\n",
        encoding="utf-8",
    )
    rows = load_event_rows([path])
    assert rows[0]["source_id"] == "source-a"
    duplicate = path.read_text(encoding="utf-8") + "2024-01-02,AAA,exclude,https://example.test/a,source-a\n"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate historical event key"):
        load_event_rows([path])


def test_apply_events_is_strict_about_state_transitions():
    events = [
        {"effective_date": "2024-01-02", "symbol": "BBB", "action": "exclude", "source": "x", "source_id": "x"},
        {"effective_date": "2024-01-03", "symbol": "CCC", "action": "include", "source": "x", "source_id": "y"},
    ]
    assert apply_events({"BBB", "AAA"}, events) == {"AAA", "CCC"}

    with pytest.raises(ValueError, match="not present"):
        apply_events({"AAA"}, events[:1])

    with pytest.raises(ValueError, match="already present"):
        apply_events({"AAA", "CCC"}, events[1:])
