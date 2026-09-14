import pandas as pd
import pytest

from stock_guru.universe_reconciliation import reconcile_snapshots


def frame(rows):
    return pd.DataFrame(rows, columns=["as_of", "symbol"])


def test_reconciliation_passes_matching_common_dates():
    a = frame([["2024-01-31", "A"], ["2024-01-31", "B"]])
    b = frame([["2024-01-31", "A"], ["2024-01-31", "B"]])
    result = reconcile_snapshots(a, b)
    assert result["status"] == "PASS"
    assert result["common_snapshot_dates"] == 1


def test_reconciliation_reports_symbol_differences():
    a = frame([["2024-01-31", "A"], ["2024-01-31", "B"]])
    b = frame([["2024-01-31", "A"], ["2024-01-31", "C"]])
    result = reconcile_snapshots(a, b)
    assert result["status"] == "MISMATCH"
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["only_supplied"] == ["B"]
    assert result["mismatches"][0]["only_reconstructed"] == ["C"]


def test_reconciliation_can_require_reconstructed_dates():
    a = frame([["2024-01-31", "A"]])
    b = frame([["2024-01-31", "A"], ["2024-02-29", "A"]])
    with pytest.raises(ValueError, match="missing reconstructed dates"):
        reconcile_snapshots(a, b, require_all_reconstructed_dates=True)
