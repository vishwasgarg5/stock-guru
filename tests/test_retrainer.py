from stock_guru.retrainer import should_accept


def test_retraining_accepts_only_validation_improvement():
    decision = should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 1.9})
    assert decision.accepted
    assert not should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 2.0}).accepted
