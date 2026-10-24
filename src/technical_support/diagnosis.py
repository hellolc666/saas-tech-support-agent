"""Evidence-driven synthetic incident diagnosis (no offline labels in inference).

The deterministic reasoner is an inspectable baseline, not an LLM evaluator.
It compares competing hypotheses and abstains on insufficient/conflicting evidence.
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

# These are *candidate* causes for a symptom, not the offline ground truth.
CANDIDATES_BY_VIEW = {
    "traffic": ("QUOTA_EXHAUSTED", "BURST_RATE_LIMIT"),
    "auth": ("TOKEN_EXPIRED", "CREDENTIAL_ENV_MISMATCH"),
    "export": ("VERSION_UNSUPPORTED", "RESOURCE_PERMISSION_DENIED", "EXPORT_COUNT_LIMIT"),
    "dependency": ("DOWNSTREAM_DEPENDENCY_DEGRADED",),
}


def _json_tool_payload(message: Any) -> dict[str, Any] | None:
    """Parse a successful evidence observation from a LangChain ToolMessage."""
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
    """Choose evidence from symptom, not from incident ID or answer labels."""
    if overview.get("category") == "export":
        return "export"
    return REQUIRED_DETAIL.get(overview.get("http_status"))


def hypotheses_for_overview(overview: dict[str, Any]) -> list[dict[str, str]]:
    """Construct competing hypotheses without reading the hidden answer key."""
    view = detail_for_overview(overview)
    return [
        {"root_cause": code, "reason": CAUSES[code][0], "status": "unverified"}
        for code in CANDIDATES_BY_VIEW.get(view, ())
    ]


def next_evidence_view(views: dict[str, dict[str, Any]]) -> str | None:
    """Select the next needed view; stop once required evidence is present."""
    if "overview" not in views:
        return "overview"
    detail = detail_for_overview(views["overview"].get("evidence", {}))
    return detail if detail and detail not in views else None


def _number(evidence: dict[str, Any], key: str) -> float | int | None:
    value = evidence.get(key)
    # A missing field, boolean or malformed number is not negative evidence.
    return value if type(value) in (float, int) else None


def _compare(evidence: dict[str, Any], left: str, right: str, op: str) -> bool | None:
    a, b = _number(evidence, left), _number(evidence, right)
    if a is None or b is None:
        return None
    return a >= b if op == ">=" else a > b


def _test_hypothesis(code: str, evidence: dict[str, Any]) -> tuple[bool | None, list[str]]:
    """Return (supported/rejected/unknown, consulted field names)."""
    if code == "QUOTA_EXHAUSTED":
        return _compare(evidence, "window_used", "window_limit", ">="), ["window_used", "window_limit"]
    if code == "BURST_RATE_LIMIT":
        return _compare(evidence, "peak_rps", "rps_limit", ">"), ["peak_rps", "rps_limit"]
    if code == "TOKEN_EXPIRED":
        v = evidence.get("token_expired")
        return (v if type(v) is bool else None), ["token_expired"]
    if code == "CREDENTIAL_ENV_MISMATCH":
        fields = ["credential_environment", "request_environment"]
        a, b = (evidence.get(k) for k in fields)
        if not all(isinstance(x, str) and x for x in (a, b)):
            return None, fields
        return a != b, fields
    if code == "VERSION_UNSUPPORTED":
        v = evidence.get("version")
        return (v != "2.0" if isinstance(v, str) and v else None), ["version"]
    if code == "RESOURCE_PERMISSION_DENIED":
        fields = ["resource_access", "role"]
        access, role = (evidence.get(k) for k in fields)
        if type(access) is not bool or not isinstance(role, str) or not role:
            return None, fields
        return (not access or role not in {"管理员", "编辑者"}), fields
    if code == "EXPORT_COUNT_LIMIT":
        return _compare(evidence, "requested_records", "max_records", ">"), ["requested_records", "max_records"]
    if code == "DOWNSTREAM_DEPENDENCY_DEGRADED":
        fields = ["downstream_p95_ms", "downstream_error_rate"]
        p95, rate = (_number(evidence, key) for key in fields)
        if p95 is None or rate is None:
            return None, fields
        return p95 > 2000 and rate > 0.2, fields
    raise ValueError(f"Unknown cause: {code}")


def assess(views: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Evaluate competing causes using one incident's authenticated observations.

    Returns None while collecting evidence. Returns UNDETERMINED for incomplete
    evidence or CONFLICTING_EVIDENCE when multiple causes are simultaneously supported.
    """
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
    candidates = CANDIDATES_BY_VIEW.get(detail_view, ())
    hypotheses = []
    supported = []
    unknown = []
    for code in candidates:
        value, fields = _test_hypothesis(code, evidence)
        status = "supported" if value is True else "rejected" if value is False else "unknown"
        hypotheses.append({
            "root_cause": code,
            "reason": CAUSES[code][0],
            "status": status,
            "evidence_fields": fields,
        })
        if value is True:
            supported.append(code)
        elif value is None:
            unknown.append(code)
    if len(supported) > 1:
        root = "CONFLICTING_EVIDENCE"
        reason = "多项根因假设同时得到支持，无法唯一定位根因"
        next_step = "核对异常证据或补充可区分的观测"
        verdict = "conflicting"
    elif len(supported) == 1 and not unknown:
        root = supported[0]
        reason, next_step = CAUSES[root]
        verdict = "supported"
    else:
        root = "UNDETERMINED"
        reason = "现有证据不足以唯一确定根因"
        next_step = "补充缺失或可区分的诊断证据，必要时请求人工排查"
        verdict = "insufficient"
    return {
        "incident_id": overview["incident_id"],
        "root_cause": root,
        "reason": reason,
        "next_step": next_step,
        "verdict": verdict,
        "hypotheses": hypotheses,
        "evidence_views": ["overview", detail_view],
        "evidence": {detail_view: evidence},
        "observed_at": overview.get("observed_at"),
        "is_demo_snapshot": True,
    }
