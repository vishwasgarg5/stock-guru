from stock_guru import load_trading_calendar, next_session_date, previous_session_date, validate_model_artifact, validate_trading_calendar


def test_public_api_exports_are_callable():
    assert callable(load_trading_calendar)
    assert callable(next_session_date)
    assert callable(previous_session_date)
    assert callable(validate_model_artifact)
    assert callable(validate_trading_calendar)
