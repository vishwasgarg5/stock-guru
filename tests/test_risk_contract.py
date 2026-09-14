import pytest
from stock_guru.risk_contract import validate_position_limits


def test_risk_contract_accepts_positions():
    assert validate_position_limits({"ABC": 0.10})["validated"] is True


def test_risk_contract_rejects_oversized_position():
    with pytest.raises(ValueError, match="limit exceeded"):
        validate_position_limits({"ABC": 0.11})
