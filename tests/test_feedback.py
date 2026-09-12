import pandas as pd

from stock_guru.feedback import append_feedback


def _row(prediction_date, symbol, model_version, error):
    return pd.DataFrame(
        {
            "prediction_date": [prediction_date],
            "symbol": [symbol],
            "model_version": [model_version],
            "return_error": [error],
        }
    )


def test_append_feedback_deduplicates_prediction_identity(tmp_path):
    store = tmp_path / "feedback.csv"

    append_feedback(str(store), _row("2026-01-05", "RELIANCE", "adaptive-v1", 0.10))
    append_feedback(str(store), _row("2026-01-05", "RELIANCE", "adaptive-v1", 0.20))
    append_feedback(str(store), _row("2026-01-05", "TCS", "adaptive-v1", 0.30))

    saved = pd.read_csv(store)
    assert len(saved) == 2
    assert saved.loc[saved["symbol"] == "RELIANCE", "return_error"].iloc[0] == 0.20


def test_append_feedback_keeps_distinct_model_versions(tmp_path):
    store = tmp_path / "feedback.csv"

    append_feedback(str(store), _row("2026-01-05", "RELIANCE", "adaptive-v1", 0.10))
    append_feedback(str(store), _row("2026-01-05", "RELIANCE", "adaptive-v2", 0.20))

    saved = pd.read_csv(store)
    assert len(saved) == 2
    assert set(saved["model_version"]) == {"adaptive-v1", "adaptive-v2"}
