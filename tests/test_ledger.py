import pandas as pd
from stock_guru.ledger import append_predictions, load_pending
from stock_guru.feedback import settle_prediction_feedback


def test_prediction_ledger_is_idempotent(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = pd.DataFrame({"prediction_date": ["2026-09-10", "2026-09-10"], "symbol": ["ABC", "ABC"], "model_version": ["v1", "v1"], "pred_close": [101, 102]})
    append_predictions(path, rows.iloc[[0]])
    append_predictions(path, rows)
    out = pd.read_csv(path)
    assert len(out) == 1
    assert out.loc[0, "pred_close"] == 102


def test_load_pending_filters_prediction_date(tmp_path):
    path = tmp_path / "predictions.csv"
    append_predictions(path, pd.DataFrame({"prediction_date": ["2026-09-09", "2026-09-12"], "symbol": ["AAA", "BBB"], "pred_close": [101, 202]}))
    out = load_pending(path, as_of="2026-09-10")
    assert out["symbol"].tolist() == ["AAA"]


def test_settlement_is_idempotent(tmp_path):
    predictions = tmp_path / "predictions.csv"
    feedback = tmp_path / "feedback.csv"
    append_predictions(predictions, pd.DataFrame({"prediction_date": ["2026-09-10"], "symbol": ["AAA"], "model_version": ["v1"], "close": [100.0], "pred_close": [101.0]}))
    market = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-11"]), "symbol": ["AAA", "AAA"], "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [100, 102]})
    first = settle_prediction_feedback(predictions, feedback, market, as_of="2026-09-11")
    second = settle_prediction_feedback(predictions, feedback, market, as_of="2026-09-11")
    assert len(first) == 1
    assert second.empty
    assert len(pd.read_csv(feedback)) == 1
