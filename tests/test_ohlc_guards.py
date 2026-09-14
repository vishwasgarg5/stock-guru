import pandas as pd
import pytest

from stock_guru.ohlc import OHLCForecaster


def test_predict_rejects_empty_frame():
    forecaster = OHLCForecaster()
    with pytest.raises(ValueError, match="No rows"):
        forecaster.predict(pd.DataFrame())


def test_fit_rejects_no_complete_training_rows():
    forecaster = OHLCForecaster()
    frame = pd.DataFrame({"feature": [1.0], "target_open": [float("nan")], "target_high": [1.0], "target_low": [1.0], "target_close": [1.0]})
    with pytest.raises(ValueError, match="No complete rows"):
        forecaster.fit(frame, ["feature"])
