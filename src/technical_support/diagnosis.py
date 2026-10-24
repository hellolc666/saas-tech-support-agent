"""Evidence-driven incident triage without looking at evaluation answer labels.

The inference rules form a reproducible *baseline*, not a general LLM judge.
They operate on two incident evidence observations returned by authenticated tools.
"""

from __future__ import annotations

import json
from typing import Any

REQUIRED_DETAIL = {
    429: "traffic",
    401: "auth",
    503: "dependency",
}

CAUSES = {
    "QUOTA_EXHAUSTED": ("时间窗口配额耗尽", "等待下一配额窗口，检查配额或减少请求量"),
    "BURST_RATE_LIMIT": ("瞬时请求速率超过限制", "实施客户端限速、退避与抖动重试"),
    "RESOURCE_PERMISSION_DENIED": ("当前资源缺少导出权限", "核实项目成员身份及资源授权"),
    "EXPORT_COUNT_LIMIT": ("导出记录数超过单次限制", "使用筛选条件分批导出"),
    "TOKEN_EXPIRED": ("访问令牌已过期", "通过正规渠道刷新令牌，不上传密钥"),
    "CREDENTIAL_ENV_MISMATCH": ("请求环境与凭证环境不匹配", "核对目标环境、凭证来源及请求地址"),
    "VERSION_UNSUPPORTED": ("当前版本不支持批量导出", "核对并升级至支持批量导出的版本"),
    "DOWNSTREAM_DEPENDENCY_DEGRADED": ("下游依赖异常导致服务变慢", "检查模拟依赖指标及超时、降级和重试策略"),
}


def _json_tool_payload(message: Any) -> dict[str, Any] | None:
    """Accept LangChain ToolMessage duck-typed values for isolated unit testing."""
    if getattr(message, "name", None) != "get_incident_evidence":
        return None
    try:
        payload = json.loads(message.content) if isinstance(message.content, str) else message.content
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload.get("ok") is True else None


def collected_views(messages: list[Any], incident_id: str | None) -> dict[str, dict[str, Any]]:
    """Keep only observations for one incident; never join different cases."""
    rows: dict[str, dict[str, Any]] = {}
    if not incident_id:
        return rows
    for message in messages:
        payload = _json_tool_payload(message)
        if payload and payload.get("incident_id") == incident_id.upper():
            view = payload.get("view")
            if isinstance(view, str) and isinstance(payload.get("evidence"), dict):
                rows[view] = payload
    return rows


def detail_for_overview(overview: dict[str, Any]) -> str | None:
    """Choose next evidence dimension *from observed symptom*, not case ID."""
    if overview.get("category") == "export":
        return "export"
    return REQUIRED_DETAIL.get(overview.get("http_status"))


def next_evidence_view(views: dict[str, dict[str, Any]]) -> str | None:
    """Return the next required dimension or None when sufficient evidence exists."""
    if "overview" not in views:
        return "overview"
    detail = detail_for_overview(views["overview"].get("evidence", {}))
    return detail if detail and detail not in views else None


def assess(views: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Infer cause from observations, returning None until relevant evidence is read."""
    if "overview" not in views:
        return None
    overview = views["overview"]
    detail_view = detail_for_overview(overview.get("evidence", {}))
    if not detail_view or detail_view not in views:
        return None
    details = views[detail_view]
    if (details.get("incident_id") != overview.get("incident_id")
            or details.get("app_id") != overview.get("app_id")):
        return None
    evidence = details["evidence"]
    cause = None
    if detail_view == "traffic":
        if evidence.get("window_used", 0) >= evidence.get("window_limit", float("inf")):
            cause = "QUOTA_EXHAUSTED"
        elif evidence.get("peak_rps", 0) > evidence.get("rps_limit", float("inf")):
            cause = "BURST_RATE_LIMIT"
    elif detail_view == "auth":
        if evidence.get("token_expired") is True:
            cause = "TOKEN_EXPIRED"
        elif evidence.get("credential_environment") != evidence.get("request_environment"):
            cause = "CREDENTIAL_ENV_MISMATCH"
    elif detail_view == "export":
        if evidence.get("version") != "2.0":
            cause = "VERSION_UNSUPPORTED"
        elif not evidence.get("resource_access") or evidence.get("role") not in {"管理员", "编辑者"}:
            cause = "RESOURCE_PERMISSION_DENIED"
        elif evidence.get("requested_records", 0) > evidence.get("max_records", float("inf")):
            cause = "EXPORT_COUNT_LIMIT"
    elif detail_view == "dependency":
        if (evidence.get("downstream_p95_ms", 0) > 2000
                and evidence.get("downstream_error_rate", 0) > 0.2):
            cause = "DOWNSTREAM_DEPENDENCY_DEGRADED"
    label, remediation = CAUSES.get(cause, ("现有证据不足以确定根因", "请求人工排查或补充其他维度数据"))
    return {
        "incident_id": overview["incident_id"],
        "root_cause": cause or "UNDETERMINED",
        "reason": label,
        "next_step": remediation,
        "evidence_views": ["overview", detail_view],
        "evidence": {detail_view: evidence},
        "is_demo_snapshot": True,
    }
