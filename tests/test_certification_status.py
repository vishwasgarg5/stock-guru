from pathlib import Path
import json


def test_certification_status_is_fail_closed():
    path = Path("data/nifty500_certification_status.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["certification"] == "BLOCKED"
    assert data["status"] == "blocked"
    assert data["completed_development_steps"] == 30
    assert data["remaining_certification_tasks"]
    assert "synthetic" in data["evidence_policy"]
