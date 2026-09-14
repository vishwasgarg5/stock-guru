import pandas as pd
import pytest
from stock_guru.prediction_contract import validate_prediction_frame


def test_prediction_contract_accepts_valid_frame():
    frame = pd.DataFrame({"symbol": ["ABC"], "pred_open": [100], "pred_high": [102], "pred_low": [98], "pred_close": [101]})
    assert validate_prediction_frame(frame)["validated"] is True


def test_prediction_contract_rejects_duplicate_symbols():
    frame = pd.DataFrame({"symbol": ["ABC", "ABC"], "pred_open": [100, 100], "pred_high": [102, 102], "pred_low": [98, 98], "pred_close": [101, 101]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_frame(frame)
