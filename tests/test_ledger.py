import pandas as pd
from stock_guru.ledger import append_predictions, load_pending


def test_prediction_ledger_is_idempotent(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = pd.DataFrame({
        "prediction_date": ["2026-09-10", "2026-09-10"],
        "symbol": ["ABC", "ABC"],
        "model_version": ["v1", "v1"],
        "pred_close": [101, 102],
    })
    append_predictions(path, rows.iloc[[0]])
    append_predictions(path, rows)
    out = pd.read_csv(path)
    assert len(out) == 1
    assert out.loc[0, "pred_close"] == 102


def test_load_pending_filters_prediction_date(tmp_path):
    path = tmp_path / "predictions.csv"
    append_predictions(path, pd.DataFrame({
        "prediction_date": ["2026-09-09", "2026-09-12"],
        "symbol": ["AAA", "BBB"],
        "pred_close": [101, 202],
    }))
    out = load_pending(path, as_of="2026-09-10")
    assert out["symbol"].tolist() == ["AAA"]
