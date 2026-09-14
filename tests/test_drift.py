import pandas as pd

from stock_guru.drift import drift_summary, feature_drift


def test_stable_feature_has_low_psi():
    reference = pd.DataFrame({"x": range(100)})
    current = pd.DataFrame({"x": range(100)})
    out = feature_drift(reference, current, ["x"])
    assert out["x"]["psi"] < 1e-9
    assert not out["x"]["drifted"]


def test_drift_summary_counts_drifted_features():
    reference = pd.DataFrame({"x": range(100), "y": range(100)})
    current = pd.DataFrame({"x": [0] * 100, "y": range(100)})
    diagnostics = feature_drift(reference, current, ["x", "y"])
    summary = drift_summary(diagnostics)
    assert summary["features"] >= 1
    assert summary["max_psi"] >= 0.0


def test_invalid_bin_count_rejected():
    try:
        feature_drift(pd.DataFrame({"x": range(10)}), pd.DataFrame({"x": range(10)}), ["x"], bins=1)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("Expected invalid-bin validation")
