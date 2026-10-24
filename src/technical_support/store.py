"""Isolated SQLite demo data for SaaS diagnostics and technical tickets.

All data and verification codes here are synthetic. Never use these codes as
production credentials. Production identity must come from trusted auth context.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_DB_PATH = Path(os.getenv("SAAS_SUPPORT_DB_PATH", "data/saas_support.db"))


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _hash_demo_code(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create tables and synthetic sample records without replacing user tickets."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    with closing(_connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS applications (
                app_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                version TEXT NOT NULL,
                environment TEXT NOT NULL,
                access_code_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_quotas (
                app_id TEXT PRIMARY KEY REFERENCES applications(app_id),
                window_label TEXT NOT NULL,
                rate_limit INTEGER NOT NULL,
                used INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_request_logs (
                request_id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL REFERENCES applications(app_id),
                http_status INTEGER NOT NULL,
                error_code TEXT,
                occurred_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS export_access (
                app_id TEXT PRIMARY KEY REFERENCES applications(app_id),
                role TEXT NOT NULL,
                can_access INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS technical_tickets (
                ticket_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                app_id TEXT NOT NULL REFERENCES applications(app_id),
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            );
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO tenants (tenant_id, name) VALUES (?, ?)",
            [("TENANT_ALPHA", "演示企业甲"), ("TENANT_BETA", "演示企业乙")],
        )
        connection.executemany(
            """INSERT OR IGNORE INTO applications
               (app_id, tenant_id, version, environment, access_code_hash)
               VALUES (?, ?, ?, ?, ?)""",
            [
                ("APP_DEMO_001", "TENANT_ALPHA", "2.0", "演示环境", _hash_demo_code("DEMO-ALPHA-4821")),
                ("APP_DEMO_002", "TENANT_BETA", "1.0", "演示环境", _hash_demo_code("DEMO-BETA-7394")),
            ],
        )
        connection.executemany(
            """INSERT OR IGNORE INTO api_quotas
               (app_id, window_label, rate_limit, used, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                ("APP_DEMO_001", "静态演示窗口", 100, 100, "2026-01-01T00:00:00+00:00"),
                ("APP_DEMO_002", "静态演示窗口", 100, 3, "2026-01-01T00:00:00+00:00"),
            ],
        )
        connection.executemany(
            """INSERT OR IGNORE INTO api_request_logs
               (request_id, app_id, http_status, error_code, occurred_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                ("REQ_DEMO_429", "APP_DEMO_001", 429, "RATE_LIMIT_EXCEEDED", "2026-01-01T00:00:00+00:00"),
                ("REQ_DEMO_401", "APP_DEMO_002", 401, "TOKEN_EXPIRED", "2026-01-01T00:00:00+00:00"),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO export_access VALUES (?, ?, ?)",
            [("APP_DEMO_001", "编辑者", 1), ("APP_DEMO_002", "访客", 1)],
        )
        connection.commit()
    return path


def _verified_app(connection: sqlite3.Connection, app_id: str, access_code: str) -> sqlite3.Row | None:
    row = connection.execute(
        "SELECT app_id, tenant_id, version, environment, access_code_hash FROM applications WHERE app_id = ?",
        (app_id.strip().upper(),),
    ).fetchone()
    if row is None or not hmac.compare_digest(row["access_code_hash"], _hash_demo_code(access_code.strip())):
        return None
    return row


def _denied() -> dict[str, Any]:
    return {"ok": False, "error_code": "APP_NOT_FOUND_OR_VERIFICATION_FAILED", "error": "应用不存在或演示验证码不匹配。"}


def get_api_usage(app_id: str, access_code: str, db_path: str | Path | None = None) -> dict[str, Any]:
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        app = _verified_app(connection, app_id, access_code)
        if app is None:
            return _denied()
        row = connection.execute("SELECT * FROM api_quotas WHERE app_id = ?", (app["app_id"],)).fetchone()
    if row is None:
        return {"ok": False, "error_code": "NO_USAGE_DATA", "error": "该应用没有演示配额记录。"}
    return {
        "ok": True, "app_id": app["app_id"], "version": app["version"],
        "environment": app["environment"], "window_label": row["window_label"],
        "rate_limit": row["rate_limit"], "used": row["used"], "updated_at": row["updated_at"],
        "is_demo_snapshot": True,
    }


def get_request_log(
    request_id: str, app_id: str, access_code: str, db_path: str | Path | None = None,
) -> dict[str, Any]:
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        app = _verified_app(connection, app_id, access_code)
        if app is None:
            return _denied()
        row = connection.execute(
            "SELECT request_id, app_id, http_status, error_code, occurred_at "
            "FROM api_request_logs WHERE request_id = ? AND app_id = ?",
            (request_id.strip().upper(), app["app_id"]),
        ).fetchone()
    if row is None:
        return {"ok": False, "error_code": "REQUEST_NOT_FOUND", "error": "该应用下未找到对应请求记录。"}
    return {"ok": True, **dict(row), "is_demo_snapshot": True}


def create_technical_ticket(
    app_id: str,
    access_code: str,
    category: str,
    summary: str,
    idempotency_key: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist a ticket only after caller's HITL approval; key scopes to tenant."""
    if not category.strip() or not summary.strip() or not idempotency_key.strip():
        return {"ok": False, "error_code": "INVALID_TICKET_INPUT", "error": "工单类别、摘要和提交标识不能为空。"}
    if len(summary) > 1000 or len(category) > 50 or len(idempotency_key) > 128:
        return {"ok": False, "error_code": "INVALID_TICKET_INPUT", "error": "工单字段长度超出限制。"}
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        app = _verified_app(connection, app_id, access_code)
        if app is None:
            return _denied()
        tenant_id = app["tenant_id"]
        key = idempotency_key.strip()
        # BEGIN IMMEDIATE serializes same-key writers before SELECT/INSERT.
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT * FROM technical_tickets WHERE tenant_id = ? AND idempotency_key = ?",
            (tenant_id, key),
        ).fetchone()
        if existing is not None:
            if (existing["app_id"], existing["category"], existing["summary"]) != (
                app["app_id"], category.strip(), summary.strip()
            ):
                return {"ok": False, "error_code": "IDEMPOTENCY_CONFLICT", "error": "提交标识已用于不同的工单内容。"}
            return {
                "ok": True, "ticket_id": existing["ticket_id"], "status": existing["status"],
                "duplicate": True,
            }
        ticket_id = f"ST-{uuid4().hex[:10].upper()}"
        connection.execute(
            """INSERT INTO technical_tickets
               (ticket_id, tenant_id, app_id, category, summary, status, idempotency_key, created_at)
               VALUES (?, ?, ?, ?, ?, '待处理', ?, ?)""",
            (ticket_id, tenant_id, app["app_id"], category.strip(), summary.strip(), key,
             datetime.now(UTC).isoformat()),
        )
        connection.commit()
    return {"ok": True, "ticket_id": ticket_id, "status": "待处理", "duplicate": False}


def get_ticket_status(
    ticket_id: str, app_id: str, access_code: str, db_path: str | Path | None = None,
) -> dict[str, Any]:
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        app = _verified_app(connection, app_id, access_code)
        if app is None:
            return _denied()
        row = connection.execute(
            "SELECT ticket_id, app_id, category, summary, status, created_at "
            "FROM technical_tickets WHERE ticket_id = ? AND app_id = ? AND tenant_id = ?",
            (ticket_id.strip().upper(), app["app_id"], app["tenant_id"]),
        ).fetchone()
    if row is None:
        return {"ok": False, "error_code": "TICKET_NOT_FOUND", "error": "该应用下未找到对应工单。"}
    return {"ok": True, **dict(row)}


def check_export_eligibility(app_id: str, access_code: str, record_count: int,
                             db_path: str | Path | None = None) -> dict[str, Any]:
    """Diagnose export against verified synthetic app version and access data; never export."""
    if type(record_count) is not int or record_count <= 0:
        return {"ok": False, "error_code": "INVALID_RECORD_COUNT", "error": "记录数量必须为正整数。"}
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        app = _verified_app(connection, app_id, access_code)
        if app is None:
            return _denied()
        access = connection.execute("SELECT role, can_access FROM export_access WHERE app_id = ?",
                                    (app["app_id"],)).fetchone()
    if access is None:
        return {"ok": False, "error_code": "NO_EXPORT_ACCESS_DATA", "error": "缺少导出权限数据，无法判断。"}
    reasons = []
    if app["version"] != "2.0":
        reasons.append("VERSION_UNSUPPORTED")
    if not access["can_access"] or access["role"] not in {"管理员", "编辑者"}:
        reasons.append("EXPORT_PERMISSION_DENIED")
    if record_count > 5000:
        reasons.append("EXPORT_LIMIT_EXCEEDED")
    return {"ok": True, "app_id": app["app_id"], "version": app["version"],
            "role": access["role"], "record_count": record_count, "max_records": 5000,
            "eligible": not reasons, "reasons": reasons, "is_demo_snapshot": True,
            "message": "满足已知导出条件；仍失败请提供筛选条件、发生时间和请求 ID。" if not reasons
                       else "；".join({"VERSION_UNSUPPORTED": "当前版本不支持批量导出",
                                     "EXPORT_PERMISSION_DENIED": "当前角色或资源权限不允许导出",
                                     "EXPORT_LIMIT_EXCEEDED": "超过单次5000条限制，请筛选后分批导出"}[r] for r in reasons)}
