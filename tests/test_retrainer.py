from stock_guru.retrainer import should_accept
from stock_guru.model_selection import should_promote


def test_retraining_accepts_only_validation_improvement():
    decision = should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 1.9})
    assert decision.accepted
    assert not should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 2.0}).accepted


def test_promotion_rejects_ranking_regression():
    old = {
        "pred_close_rmse": 2.0,
        "close_direction_accuracy": 0.60,
        "top_k_excess_return": 0.004,
        "precision_at_k": 0.55,
    }
    new = {
        "pred_close_rmse": 1.9,
        "close_direction_accuracy": 0.61,
        "top_k_excess_return": 0.003,
        "precision_at_k": 0.56,
    }
    assert not should_promote(old, new)


def test_promotion_accepts_forecast_and_ranking_improvement():
    old = {
        "pred_close_rmse": 2.0,
        "close_direction_accuracy": 0.60,
        "top_k_excess_return": 0.004,
        "precision_at_k": 0.55,
    }
    new = {
        "pred_close_rmse": 1.9,
        "close_direction_accuracy": 0.61,
        "top_k_excess_return": 0.005,
        "precision_at_k": 0.57,
    }
    assert should_promote(old, new)


def test_promotion_backwards_compatible_without_ranking_metrics():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61}
    assert should_promote(old, new)
