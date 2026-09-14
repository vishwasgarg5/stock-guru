import numpy as np
import pytest

from stock_guru.temporal_challenger import LSTMChallenger, LSTMConfig, temporal_challenger_metrics


def test_lstm_challenger_rejects_wrong_shapes_before_optional_dependency():
    challenger = LSTMChallenger()
    with pytest.raises(ValueError, match="sequences must be"):
        challenger.fit(np.zeros((4, 3)), np.zeros((4, 1)))


def test_lstm_challenger_requires_fit_before_prediction():
    challenger = LSTMChallenger()
    with pytest.raises(RuntimeError, match="Fit"):
        challenger.predict(np.zeros((1, 20, 2), dtype=np.float32))


def test_lstm_challenger_validates_config():
    with pytest.raises(ValueError, match="lookback"):
        LSTMChallenger(LSTMConfig(lookback=0))
    with pytest.raises(ValueError, match="hidden_size"):
        LSTMChallenger(LSTMConfig(hidden_size=0))
    with pytest.raises(ValueError, match="epochs"):
        LSTMChallenger(LSTMConfig(epochs=0))
    with pytest.raises(ValueError, match="learning_rate"):
        LSTMChallenger(LSTMConfig(learning_rate=0))


def test_lstm_challenger_rejects_non_finite_input_before_optional_dependency():
    challenger = LSTMChallenger()
    x = np.zeros((2, 20, 3), dtype=np.float32)
    y = np.zeros((2, 4), dtype=np.float32)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        challenger.fit(x, y)


def test_lstm_challenger_enforces_configured_lookback_before_optional_dependency():
    challenger = LSTMChallenger(LSTMConfig(lookback=10))
    with pytest.raises(ValueError, match="lookback"):
        challenger.fit(np.zeros((2, 20, 3), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))


def test_temporal_challenger_metrics_are_dependency_free():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.array([[2.0, 2.0], [3.0, 6.0]])
    metrics = temporal_challenger_metrics(actual, predicted)
    assert metrics["samples"] == 2
    assert metrics["outputs"] == 2
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.5))


def test_temporal_challenger_metrics_reject_invalid_arrays():
    with pytest.raises(ValueError, match="matching non-empty 2D"):
        temporal_challenger_metrics(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        temporal_challenger_metrics(np.array([[np.nan]]), np.array([[0.0]]))
