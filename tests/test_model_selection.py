from stock_guru.model_selection import select_champion


def _metrics(rmse, direction):
    return {"validation_folds": 20, "pred_close_rmse": rmse, "close_direction_accuracy": direction,
            "regime_metrics": {"bear": {"samples": 20, "close_direction_accuracy": 0.50},
                               "high_vol_bear": {"samples": 20, "close_direction_accuracy": 0.50}}}


def test_select_champion_keeps_incumbent_when_no_candidate_passes():
    candidates = {"old": _metrics(0.03, 0.50), "new": _metrics(0.04, 0.52)}
    out = select_champion(candidates, incumbent="old")
    assert out["champion"] == "old"
    assert out["promoted"] is False


def test_select_champion_promotes_better_candidate():
    candidates = {"old": _metrics(0.03, 0.50), "new": _metrics(0.02, 0.55)}
    out = select_champion(candidates, incumbent="old")
    assert out["champion"] == "new"
    assert out["promoted"] is True
