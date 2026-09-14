import pytest
from stock_guru.monitoring_contract import validate_monitoring_snapshot


def test_monitoring_contract_accepts_healthy():
    report = validate_monitoring_snapshot({"status": "HEALTHY", "model_version": "v1", "timestamp": "2026-09-15T00:00:00Z"})
    assert report["status"] == "HEALTHY"


def test_monitoring_contract_rejects_unknown_status():
    with pytest.raises(ValueError, match="HEALTHY"):
        validate_monitoring_snapshot({"status": "OK", "model_version": "v1", "timestamp": "now"})
