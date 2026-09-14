from pathlib import Path
import csv
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "nifty500_event_validation_targets.csv"
FILES = {
    "ind_prs15092021": ROOT / "data" / "nifty500_events_2021_09.csv",
    "ind_prs24022022_1": ROOT / "data" / "nifty500_events_2022_02.csv",
    "ind_prs01092022": ROOT / "data" / "nifty500_events_2022_09_periodic.csv",
    "ind_prs16092022": ROOT / "data" / "nifty500_events_2022_09_amalgamation.csv",
    "ind_prs17022023_1": ROOT / "data" / "nifty500_events_2023_03.csv",
    "ind_prs17082023": ROOT / "data" / "nifty500_events_2023_09.csv",
    "ind_prs17102023": ROOT / "data" / "nifty500_events_2023_10.csv",
    "ind_prs28022024": ROOT / "data" / "nifty500_events_2024_03.csv",
    "ind_prs23082024": ROOT / "data" / "nifty500_events_2024_09_final.csv",
    "ind_prs25092024": ROOT / "data" / "nifty500_events_2024_09_final.csv",
    "ind_prs21022025": ROOT / "data" / "nifty500_events_2025_2024.csv",
    "ind_prs22082025": ROOT / "data" / "nifty500_events_2025_09.csv",
    "ind_prs15092025_1": ROOT / "data" / "nifty500_events_2025_09.csv",
    "ind_prs11122025": ROOT / "data" / "nifty500_events_2025_12.csv",
    "ind_prs23022026": ROOT / "data" / "nifty500_events.csv",
}


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _targets():
    with TARGETS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_validation_target_has_a_source_mapping():
    source_ids = {target["source_id"] for target in _targets()}
    assert source_ids <= set(FILES), f"unmapped validation sources: {sorted(source_ids - set(FILES))}"
    for source_id, path in FILES.items():
        assert path.is_file(), f"missing event evidence file for {source_id}: {path}"


def test_verified_event_count_targets():
    for target in _targets():
        source_id = target["source_id"]
        path = FILES[source_id]
        rows = [row for row in _rows(path) if row["source_id"] == source_id]
        counts = Counter(row["action"] for row in rows)
        assert counts["exclude"] == int(target["expected_excludes"])
        assert counts["include"] == int(target["expected_includes"])


def test_verified_event_keys_are_unique():
    seen = set()
    for path in set(FILES.values()):
        for row in _rows(path):
            key = (row["effective_date"], row["symbol"], row["action"], row["source_id"])
            assert key not in seen, f"duplicate historical event key: {key}"
            seen.add(key)
