import pandas as pd
import pytest

from stock_guru.calibration import confidence_calibration_table, uncertainty_scale_for_coverage, apply_uncertainty_scale


def test_confidence_calibration_reports_directional_accuracy():
    frame = pd.DataFrame({
        "forecast_confidence": [0.2, 0.2, 0.9, 0.9],
        "base_close": [100, 100, 100, 100],
        "pred_close": [101, 99, 101, 101],
        "actual_close": [102, 98, 102, 99],
    })
    result = confidence_calibration_table(frame)
    assert result["[0.0, 0.5)"]["observed_direction_accuracy"] == 1.0
    assert result["[0.9, 1.0)"]["observed_direction_accuracy"] == 0.5


def test_uncertainty_scale_hits_requested_empirical_quantile():
    frame = pd.DataFrame({
        "base_close": [100] * 10,
        "pred_close": [100] * 10,
        "actual_close": [100 + x for x in range(1, 11)],
        "pred_close_uncertainty_pct": [0.01] * 10,
    })
    scale = uncertainty_scale_for_coverage(frame, target_coverage=0.8)
    assert scale > 0
    adjusted = apply_uncertainty_scale(frame, scale)
    assert (adjusted["pred_close_uncertainty_pct"] > frame["pred_close_uncertainty_pct"]).all()


def test_invalid_calibration_target_is_rejected():
    with pytest.raises(ValueError):
        uncertainty_scale_for_coverage(pd.DataFrame(), target_coverage=1.0)
