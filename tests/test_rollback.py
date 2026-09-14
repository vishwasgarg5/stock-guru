from pathlib import Path

from stock_guru.rollback import decide_recovery


def _artifact(root: Path, version: str) -> Path:
    root.mkdir()
    (root / "ranker.joblib").write_bytes(b"ranker")
    (root / "ohlc.joblib").write_bytes(b"ohlc")
    (root / "features.csv").write_text("feature\nclose\n", encoding="utf-8")
    (root / "model_metadata.json").write_text(
        '{"model_version":"' + version + '","features":["close"],'
        '"pit_context":{"fundamentals_supplied":true,"universe_intervals_supplied":true}}',
        encoding="utf-8",
    )
    return root


def test_healthy_keeps_current_artifact(tmp_path):
    current = _artifact(tmp_path / "current", "current")
    out = decide_recovery(health_status="HEALTHY", current_artifact=current, last_known_good=None)
    assert out == {"action": "KEEP", "reason": "current health is HEALTHY", "target": str(current)}


def test_degraded_rolls_back_only_to_valid_target(tmp_path):
    good = _artifact(tmp_path / "good", "good")
    out = decide_recovery(health_status="DEGRADED", current_artifact=None, last_known_good=good)
    assert out["action"] == "ROLLBACK"
    assert out["target"] == str(good)
    assert "good" in out["reason"]


def test_unknown_without_target_blocks(tmp_path):
    out = decide_recovery(health_status="UNKNOWN", current_artifact=None, last_known_good=None)
    assert out["action"] == "BLOCK"


def test_invalid_rollback_target_blocks(tmp_path):
    target = tmp_path / "invalid"
    target.mkdir()
    out = decide_recovery(health_status="DEGRADED", current_artifact=None, last_known_good=target)
    assert out["action"] == "BLOCK"


def test_invalid_health_status_rejected():
    try:
        decide_recovery(health_status="BROKEN", current_artifact=None, last_known_good=None)
    except ValueError as exc:
        assert "HEALTHY" in str(exc)
    else:
        raise AssertionError("invalid health status must be rejected")
