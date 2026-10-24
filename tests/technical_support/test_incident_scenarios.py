"""Stage-1 evaluation fixtures: evidence diversity, isolation and repeatability."""

import json

import pytest

from technical_support import incident_scenarios as incidents
from technical_support import store


@pytest.fixture
def incident_db(tmp_path, monkeypatch):
    path = tmp_path / "incidents.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", path)
    return path


def _read(case_id, view="overview", path=None):
    app = incidents.SCENARIOS[case_id]["app_id"]
    code = "DEMO-BETA-7394" if app == "APP_DEMO_002" else "DEMO-ALPHA-4821"
    return incidents.get_incident_evidence(case_id, app, code, view, path)


@pytest.mark.parametrize("case_id", sorted(incidents.GROUND_TRUTH))
@pytest.mark.parametrize("view", incidents.VIEWS)
def test_all_evidence_views_are_available_without_answer_leak(incident_db, case_id, view):
    result = _read(case_id, view, incident_db)
    assert result["ok"] and result["is_demo_snapshot"]
    assert result["incident_id"] == case_id
    assert result["view"] == view
    assert "ground_truth" not in result and "root_cause" not in result
    serialized = json.dumps(result)
    assert incidents.GROUND_TRUTH[case_id] not in serialized


def test_same_symptom_can_have_different_traffic_root_causes(incident_db):
    a, b = _read("F01", path=incident_db), _read("F02", path=incident_db)
    assert a["evidence"] == b["evidence"]
    ea, eb = _read("F01", "traffic", incident_db), _read("F02", "traffic", incident_db)
    assert ea["evidence"]["window_used"] == ea["evidence"]["window_limit"]
    assert eb["evidence"]["window_used"] < eb["evidence"]["window_limit"]
    assert eb["evidence"]["peak_rps"] > eb["evidence"]["rps_limit"]


def test_same_401_symptom_needs_auth_evidence(incident_db):
    assert _read("F05", path=incident_db)["evidence"] == _read("F06", path=incident_db)["evidence"]
    expired = _read("F05", "auth", incident_db)["evidence"]
    mismatch = _read("F06", "auth", incident_db)["evidence"]
    assert expired["token_expired"]
    assert not mismatch["token_expired"]
    assert mismatch["credential_environment"] != mismatch["request_environment"]


def test_export_causes_are_individually_distinguishable(incident_db):
    denied = _read("F03", "export", incident_db)["evidence"]
    over_limit = _read("F04", "export", incident_db)["evidence"]
    unsupported = _read("F07", "export", incident_db)["evidence"]
    assert not denied["resource_access"] and denied["requested_records"] <= 5000
    assert over_limit["resource_access"] and over_limit["requested_records"] > 5000
    assert unsupported["version"] == "1.0" and unsupported["requested_records"] <= 5000


def test_dependency_latency_evidence(incident_db):
    result = _read("F08", "dependency", incident_db)
    assert result["evidence"]["downstream_p95_ms"] > 5000
    assert result["evidence"]["downstream_error_rate"] > 0.5


def test_verification_and_app_isolation(incident_db):
    invalid = incidents.get_incident_evidence(
        "F01", "APP_DEMO_001", "DEMO-BETA-7394", db_path=incident_db
    )
    unknown = incidents.get_incident_evidence(
        "F99", "APP_DEMO_001", "DEMO-ALPHA-4821", db_path=incident_db
    )
    cross_app = incidents.get_incident_evidence(
        "F05", "APP_DEMO_001", "DEMO-ALPHA-4821", db_path=incident_db
    )
    assert invalid["error_code"] == "APP_NOT_FOUND_OR_VERIFICATION_FAILED"
    assert unknown["error_code"] == cross_app["error_code"] == "INCIDENT_NOT_FOUND"
    assert not incidents.list_incident_scenarios("APP_DEMO_001", "bad", incident_db)["ok"]
    a = incidents.list_incident_scenarios("APP_DEMO_001", "DEMO-ALPHA-4821", incident_db)
    b = incidents.list_incident_scenarios("APP_DEMO_002", "DEMO-BETA-7394", incident_db)
    assert len(a["incidents"]) == 6 and len(b["incidents"]) == 2
    assert not set(x["incident_id"] for x in a["incidents"]) & set(
        x["incident_id"] for x in b["incidents"]
    )


def test_seeding_is_idempotent_and_does_not_overwrite_changes(incident_db):
    incidents.seed_incident_scenarios(incident_db)
    with store._connect(incident_db) as conn:
        conn.execute(
            "UPDATE demo_incident_scenarios SET symptom = ? WHERE incident_id = ?",
            ("已人工标记", "F01"),
        )
    incidents.seed_incident_scenarios(incident_db)
    with store._connect(incident_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM demo_incident_scenarios").fetchone()[0] == 8
    assert _read("F01", path=incident_db)["evidence"]["symptom"] == "已人工标记"


def test_invalid_view_is_rejected(incident_db):
    result = incidents.get_incident_evidence(
        "F01", "APP_DEMO_001", "DEMO-ALPHA-4821", "ground_truth", incident_db
    )
    assert result["error_code"] == "INVALID_EVIDENCE_VIEW"
