from stock_guru.retrain_contract import validate_retrain_gate


def test_retrain_gate_blocks_insufficient_feedback():
    result = validate_retrain_gate(feedback_rows=5, min_feedback_rows=20, candidate_metrics={"rmse": 1.0})
    assert result["eligible"] is False


def test_retrain_gate_allows_candidate_with_no_champion():
    result = validate_retrain_gate(feedback_rows=20, min_feedback_rows=20, candidate_metrics={"rmse": 1.0})
    assert result["eligible"] is True
