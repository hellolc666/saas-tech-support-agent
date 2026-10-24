"""Hypothesis verification and reproducible non-LLM baseline checks."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from technical_support import diagnosis, incident_scenarios as data, store


@pytest.fixture
def incident_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "diagnosis3.db")
    return tmp_path / "diagnosis3.db"


def evidence(case, view, db):
    app = data.SCENARIOS[case]["app_id"]
    code = "DEMO-ALPHA-4821" if app == "APP_DEMO_001" else "DEMO-BETA-7394"
    return data.get_incident_evidence(case, app, code, view, db)


def observed(case, db):
    first = evidence(case, "overview", db)
    dimension = diagnosis.detail_for_overview(first["evidence"])
    second = evidence(case, dimension, db)
    values = [SimpleNamespace(name="get_incident_evidence", content=json.dumps(p))
              for p in (first, second)]
    return diagnosis.collected_views(values, case)


@pytest.mark.parametrize("case,expected", sorted(data.GROUND_TRUTH.items()))
def test_hypotheses_mark_single_supported_cause(incident_db, case, expected):
    viewset = observed(case, incident_db)
    overview = viewset["overview"]["evidence"]
    candidates = diagnosis.hypotheses_for_overview(overview)
    assert len(candidates) >= 1
    assert all(item["status"] == "unverified" for item in candidates)
    report = diagnosis.assess(viewset)
    supported = [x["root_cause"] for x in report["hypotheses"] if x["status"] == "supported"]
    assert supported == [expected]
    assert report["root_cause"] == expected
    assert report["verdict"] == "supported"
    assert report["observed_at"] == data.OBSERVED_AT
    assert report["next_step"]
    assert report["is_demo_snapshot"] is True
    assert "ground_truth" not in json.dumps(report)


def test_missing_evidence_must_abstain(incident_db):
    values = observed("F01", incident_db)
    del values["traffic"]["evidence"]["window_limit"]
    verdict = diagnosis.assess(values)
    assert verdict["root_cause"] == "UNDETERMINED"
    assert verdict["verdict"] == "insufficient"
    assert any(x["status"] == "unknown" for x in verdict["hypotheses"])


def test_conflicting_evidence_must_abstain(incident_db):
    values = observed("F01", incident_db)
    values["traffic"]["evidence"]["peak_rps"] = 42
    result = diagnosis.assess(values)
    assert result["root_cause"] == "CONFLICTING_EVIDENCE"
    assert result["verdict"] == "conflicting"
    assert [x["status"] for x in result["hypotheses"]] == ["supported", "supported"]


def test_unknown_evidence_not_treated_as_false(incident_db):
    values = observed("F05", incident_db)
    values["auth"]["evidence"]["credential_environment"] = None
    result = diagnosis.assess(values)
    assert result["root_cause"] == "UNDETERMINED"
    assert any(x["status"] == "unknown" for x in result["hypotheses"])


def test_unauthorized_payload_not_accepted():
    msg = SimpleNamespace(name="get_incident_evidence", content=json.dumps({
        "ok": False, "incident_id": "F01", "view": "overview", "evidence": {"http_status": 429},
    }))
    assert diagnosis.collected_views([msg], "F01") == {}


def test_baseline_evaluator_runs_in_isolated_database():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_saas_diagnosis.py"
    result = runpy.run_path(str(script))["evaluate"]()
    assert result["cases"] == 8
    assert result["correct"] == 8
    assert result["mean_evidence_calls"] == 2
    assert "NOT a Qwen" in result["scope"]
