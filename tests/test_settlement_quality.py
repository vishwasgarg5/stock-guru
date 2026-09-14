import pandas as pd
import pytest
from stock_guru.settlement_quality import validate_settlement_frame


def test_settlement_quality_accepts_valid_frame():
    report = validate_settlement_frame(pd.DataFrame({"prediction_date": ["2026-09-14"], "symbol": ["ABC"], "actual_close": [101.0], "pred_close": [100.0]}))
    assert report["validated"] is True


def test_settlement_quality_rejects_duplicate_key():
    frame = pd.DataFrame({"prediction_date": ["2026-09-14", "2026-09-14"], "symbol": ["ABC", "ABC"], "actual_close": [101, 102], "pred_close": [100, 100]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_settlement_frame(frame)
