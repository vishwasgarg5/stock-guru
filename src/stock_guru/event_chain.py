from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {"effective_date", "symbol", "action", "source", "source_id"}
VALID_ACTIONS = {"include", "exclude"}


def load_event_rows(paths: list[str | Path]) -> list[dict[str, str]]:
    """Load dated NIFTY 500 event evidence while preserving source provenance."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path_value in paths:
        path = Path(path_value)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
                raise ValueError(f"Event file missing required columns: {path}")
            for row in reader:
                row = {key: str(value or "").strip() for key, value in row.items()}
                if not row["effective_date"] or not row["symbol"] or not row["source_id"]:
                    raise ValueError(f"Event row has blank required value: {path}")
                if row["action"] not in VALID_ACTIONS:
                    raise ValueError(f"Invalid event action {row['action']!r} in {path}")
                key = (row["effective_date"], row["symbol"], row["action"], row["source_id"])
                if key in seen:
                    raise ValueError(f"Duplicate historical event key: {key}")
                seen.add(key)
                rows.append(row)
    return sorted(rows, key=lambda row: (row["effective_date"], row["source_id"], row["action"], row["symbol"]))


def apply_events(initial_symbols: set[str], events: list[dict[str, str]]) -> set[str]:
    """Apply an ordered event chain to a supplied snapshot without inventing missing constituents."""
    symbols = {symbol.strip().upper() for symbol in initial_symbols if symbol.strip()}
    for event in sorted(events, key=lambda row: (row["effective_date"], row["source_id"], row["action"], row["symbol"])):
        symbol = event["symbol"].strip().upper()
        if event["action"] == "exclude":
            if symbol not in symbols:
                raise ValueError(f"Exclusion is not present in reconstructed universe: {symbol}")
            symbols.remove(symbol)
        else:
            if symbol in symbols:
                raise ValueError(f"Inclusion is already present in reconstructed universe: {symbol}")
            symbols.add(symbol)
    return symbols
