"""Deterministic, tenant-scoped SaaS incident fixtures for diagnosis evaluation.

All observations are synthetic historical snapshots, not production telemetry.
Ground truth stays outside read APIs; the Agent must infer it from evidence.
"""

from __future__ import annotations

import copy
import json
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from technical_support import store

EvidenceView = Literal["overview", "traffic", "auth", "export", "dependency"]
VIEWS = ("overview", "traffic", "auth", "export", "dependency")
OBSERVED_AT = "2026-01-01T00:00:00+00:00"

# Used only by offline grading. Never included in user-facing evidence.
GROUND_TRUTH = {
    "F01": "QUOTA_EXHAUSTED",
    "F02": "BURST_RATE_LIMIT",
    "F03": "RESOURCE_PERMISSION_DENIED",
    "F04": "EXPORT_COUNT_LIMIT",
    "F05": "TOKEN_EXPIRED",
    "F06": "CREDENTIAL_ENV_MISMATCH",
    "F07": "VERSION_UNSUPPORTED",
    "F08": "DOWNSTREAM_DEPENDENCY_DEGRADED",
}


def _fixture(
    app_id: str,
    category: str,
    symptom: str,
    http_status: int,
    error_code: str,
    **changes: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "traffic": {
            "window_used": 20,
            "window_limit": 100,
            "peak_rps": 4,
            "rps_limit": 20,
        },
        "auth": {
            "token_expired": False,
            "credential_environment": "test",
            "request_environment": "test",
        },
        "export": {
            "version": "2.0",
            "role": "编辑者",
            "resource_access": True,
            "requested_records": 100,
            "max_records": 5000,
        },
        "dependency": {
            "api_p95_ms": 120,
            "downstream_p95_ms": 60,
            "downstream_error_rate": 0.0,
        },
    }
    for view, updates in changes.items():
        if view not in evidence:
            raise ValueError(f"Unknown fixture field: {view}")
        evidence[view].update(updates)
    return {
        "app_id": app_id,
        "category": category,
        "symptom": symptom,
        "http_status": http_status,
        "error_code": error_code,
        "evidence": evidence,
    }


# Repeated surface symptoms intentionally have different evidence/root causes.
SCENARIOS = {
    "F01": _fixture(
        "APP_DEMO_001", "api", "API 返回 429", 429, "RATE_LIMIT_EXCEEDED",
        traffic={"window_used": 100, "peak_rps": 7},
    ),
    "F02": _fixture(
        "APP_DEMO_001", "api", "API 返回 429", 429, "RATE_LIMIT_EXCEEDED",
        traffic={"window_used": 24, "peak_rps": 42},
    ),
    "F03": _fixture(
        "APP_DEMO_001", "export", "数据导出失败", 403, "EXPORT_FAILED",
        export={"resource_access": False},
    ),
    "F04": _fixture(
        "APP_DEMO_001", "export", "数据导出失败", 403, "EXPORT_FAILED",
        export={"requested_records": 5001},
    ),
    "F05": _fixture(
        "APP_DEMO_002", "api", "API 返回 401", 401, "AUTH_FAILED",
        auth={"token_expired": True},
    ),
    "F06": _fixture(
        "APP_DEMO_002", "api", "API 返回 401", 401, "AUTH_FAILED",
        auth={"credential_environment": "production", "request_environment": "test"},
    ),
    "F07": _fixture(
        "APP_DEMO_001", "export", "数据导出失败", 403, "EXPORT_FAILED",
        export={"version": "1.0"},
    ),
    "F08": _fixture(
        "APP_DEMO_001", "api", "API 请求异常缓慢", 503, "UPSTREAM_TIMEOUT",
        dependency={
            "api_p95_ms": 8500,
            "downstream_p95_ms": 8200,
            "downstream_error_rate": 0.62,
        },
    ),
}


def seed_incident_scenarios(db_path: str | Path | None = None) -> None:
    """Create additive scenario tables and seed exactly once without resetting user data."""
    store.initialize_database(db_path)
    with closing(store._connect(db_path)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS demo_incident_scenarios (
                incident_id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL REFERENCES applications(app_id),
                category TEXT NOT NULL,
                symptom TEXT NOT NULL,
                http_status INTEGER NOT NULL,
                error_code TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )"""
        )
        conn.executemany(
            """INSERT OR IGNORE INTO demo_incident_scenarios
               (incident_id, app_id, category, symptom, http_status, error_code,
                observed_at, evidence_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    incident_id, row["app_id"], row["category"], row["symptom"],
                    row["http_status"], row["error_code"], OBSERVED_AT,
                    json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True),
                )
                for incident_id, row in SCENARIOS.items()
            ],
        )
        conn.commit()


def list_incident_scenarios(
    app_id: str, access_code: str, db_path: str | Path | None = None,
) -> dict[str, Any]:
    """List only the synthetic incidents belonging to a verified demo application."""
    seed_incident_scenarios(db_path)
    with closing(store._connect(db_path)) as conn:
        app = store._verified_app(conn, app_id, access_code)
        if app is None:
            return store._denied()
        rows = conn.execute(
            """SELECT incident_id, category, symptom, http_status, observed_at
               FROM demo_incident_scenarios WHERE app_id = ? ORDER BY incident_id""",
            (app["app_id"],),
        ).fetchall()
    return {
        "ok": True,
        "app_id": app["app_id"],
        "incidents": [dict(row) for row in rows],
        "is_demo_snapshot": True,
    }


def get_incident_evidence(
    incident_id: str,
    app_id: str,
    access_code: str,
    view: EvidenceView = "overview",
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve one evidence dimension without exposing the expected root cause."""
    if view not in VIEWS:
        return {"ok": False, "error_code": "INVALID_EVIDENCE_VIEW", "allowed_views": list(VIEWS)}
    seed_incident_scenarios(db_path)
    with closing(store._connect(db_path)) as conn:
        app = store._verified_app(conn, app_id, access_code)
        if app is None:
            return store._denied()
        row = conn.execute(
            """SELECT incident_id, app_id, category, symptom, http_status,
                      error_code, observed_at, evidence_json
               FROM demo_incident_scenarios WHERE incident_id = ? AND app_id = ?""",
            (incident_id.strip().upper(), app["app_id"]),
        ).fetchone()
        if row is None:
            return {"ok": False, "error_code": "INCIDENT_NOT_FOUND", "error": "该应用下未找到故障事件。"}
        evidence = json.loads(row["evidence_json"])
    result = {
        "ok": True,
        "incident_id": row["incident_id"],
        "app_id": row["app_id"],
        "view": view,
        "observed_at": row["observed_at"],
        "is_demo_snapshot": True,
    }
    if view == "overview":
        result["evidence"] = {
            "category": row["category"],
            "symptom": row["symptom"],
            "http_status": row["http_status"],
            "error_code": row["error_code"],
        }
    else:
        # Deep-copy protects the fixture shape from callers modifying it.
        result["evidence"] = copy.deepcopy(evidence[view])
    return result
