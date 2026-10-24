"""LLM-callable read and write tools for synthetic SaaS support data."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from langchain_core.tools import tool

from technical_support.incident_scenarios import (
    get_incident_evidence,
    list_incident_scenarios,
)
from technical_support.store import (
    check_export_eligibility,
    create_technical_ticket,
    get_api_usage,
    get_request_log,
    get_ticket_status,
)


@tool("get_api_usage")
def get_api_usage_tool(app_id: str, access_code: str) -> dict:
    """Verify a demo app ID/code and read its synthetic quota snapshot.

    `access_code` is the user's demo code, never a real API key.
    """
    return get_api_usage(app_id, access_code)


@tool("get_request_log")
def get_request_log_tool(request_id: str, app_id: str, access_code: str) -> dict:
    """Read a demo request's HTTP status within the verified application only."""
    return get_request_log(request_id, app_id, access_code)


@tool("create_technical_ticket")
def create_technical_ticket_tool(
    app_id: str, access_code: str, category: str, summary: str,
    submission_id: str = "",
) -> dict:
    """Create a technical ticket for a verified demo app; graph approval is required.

    A submission ID identifies one requested write. Repeating that ID is a retry;
    a different ID is a new ticket even if the content is identical. Do not put
    codes, passwords or secrets into the ticket summary.
    """
    clean_category = category.strip()
    clean_summary = summary.strip()
    if access_code and access_code.lower() in (clean_category + clean_summary).lower():
        return {"ok": False, "error_code": "INVALID_TICKET_INPUT", "error": "工单内容中不能包含验证码。"}
    # Direct invocations without an ID represent independent operations.
    # In the online graph, report_tool_failure injects a stable ID derived
    # from the persisted tool-call ID, rather than trusting model-supplied IDs.
    effective_id = submission_id or uuid4().hex
    fingerprint = hashlib.sha256(effective_id.encode("utf-8")).hexdigest()[:32]
    return create_technical_ticket(
        app_id, access_code, clean_category, clean_summary, f"submission-{fingerprint}"
    )


@tool("get_ticket_status")
def get_ticket_status_tool(ticket_id: str, app_id: str, access_code: str) -> dict:
    """Read a persisted ticket only after verifying its owning demo app."""
    return get_ticket_status(ticket_id, app_id, access_code)


@tool("check_export_eligibility")
def check_export_eligibility_tool(app_id: str, access_code: str, record_count: int) -> dict:
    """Diagnose export failure using demo app version, role, permission and requested record count.

    Ask for missing app_id, demo access_code and positive record_count first. Does not export files.
    """
    return check_export_eligibility(app_id, access_code, record_count)


@tool("list_incident_scenarios")
def list_incident_scenarios_tool(app_id: str, access_code: str) -> dict:
    """List synthetic historical incident IDs for a verified demo application.

    Returns metadata, never the offline benchmark ground-truth root-cause labels.
    """
    return list_incident_scenarios(app_id, access_code)


@tool("get_incident_evidence")
def get_incident_evidence_tool(
    incident_id: str, app_id: str, access_code: str, view: str = "overview",
) -> dict:
    """Investigate a synthetic SaaS incident via one evidence dimension at a time.

    Start with view='overview'. Then select traffic/auth/export/dependency
    based on the observed error and symptom. An ID like F01 is a synthetic
    historical case, NOT an active production incident. Never request a real key.
    """
    return get_incident_evidence(incident_id, app_id, access_code, view)


TECHNICAL_DATA_TOOLS = [
    get_api_usage_tool, get_request_log_tool, check_export_eligibility_tool,
    create_technical_ticket_tool, get_ticket_status_tool,
    list_incident_scenarios_tool, get_incident_evidence_tool,
]
