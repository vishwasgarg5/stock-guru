import numpy as np
import pytest

from stock_guru.temporal_challenger import LSTMChallenger


def test_lstm_challenger_rejects_wrong_shapes_before_optional_dependency():
    challenger = LSTMChallenger()
    with pytest.raises(ValueError, match="sequences must be"):
        challenger.fit(np.zeros((4, 3)), np.zeros((4, 1)))


def test_lstm_challenger_requires_fit_before_prediction():
    challenger = LSTMChallenger()
    with pytest.raises(RuntimeError, match="Fit"):
        challenger.predict(np.zeros((1, 20, 2), dtype=np.float32))
