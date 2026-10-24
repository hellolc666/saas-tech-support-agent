import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from test_knowledge_agent import FakeToolModel

from technical_support import knowledge_agent as agent
from technical_support import store


@pytest.mark.asyncio
async def test_unverified_ticket_success_blocked_and_code_redacted():
    request = ModelRequest(
        model=FakeToolModel(responses=[AIMessage(content="unused")]),
        messages=[],
        state={"messages": [HumanMessage(content="创建工单")]},
        tools=agent.TECHNICAL_SUPPORT_TOOLS,
    )

    async def handler(request):
        return ModelResponse(result=[AIMessage(content="已创建工单 ST-1234567890 DEMO-ALPHA-4821")])

    result = await agent.trace_and_verify_response.awrap_model_call(request, handler)
    assert "无法确认" in result.result[0].content
    assert "ST-1234567890" not in result.result[0].content


@pytest.mark.asyncio
async def test_approval_timeout_forces_rejection(monkeypatch):
    from schema import UserInput
    from service.service import _handle_input

    monkeypatch.setenv("SAAS_APPROVAL_TTL_SECONDS", "1")
    graph = SimpleNamespace(
        aget_state=AsyncMock(
            return_value=SimpleNamespace(
                created_at=(datetime.now(UTC) - timedelta(seconds=5)).isoformat(),
                tasks=[
                    SimpleNamespace(
                        interrupts=[
                            SimpleNamespace(
                                value={
                                    "action_requests": [
                                        {"name": "create_technical_ticket", "args": {}}
                                    ]
                                }
                            )
                        ]
                    )
                ],
            )
        )
    )
    kwargs, _ = await _handle_input(UserInput(message="确认", thread_id="expired"), graph)
    choice = kwargs["input"].resume["decisions"][0]
    assert choice["type"] == "reject" and "超时" in choice["message"]


@pytest.mark.parametrize(
    "version,role,access",
    [("1.0", "访客", 1), ("2.0", "编辑者", 1), ("2.0", "管理员", 1), ("2.0", "编辑者", 0)],
)
@pytest.mark.parametrize("count", [1, 4999, 5000, 5001, 10000])
def test_business_matrix(tmp_path, version, role, access, count):
    path = tmp_path / "matrix.db"
    store.initialize_database(path)
    with store._connect(path) as con:
        con.execute("UPDATE applications SET version=? WHERE app_id='APP_DEMO_001'", (version,))
        con.execute(
            "UPDATE export_access SET role=?,can_access=? WHERE app_id='APP_DEMO_001'",
            (role, access),
        )
    result = store.check_export_eligibility("APP_DEMO_001", "DEMO-ALPHA-4821", count, path)
    assert result["eligible"] == (
        version == "2.0" and role in {"管理员", "编辑者"} and access == 1 and count <= 5000
    )


def test_trace_has_no_prompt_or_credentials(tmp_path, monkeypatch):
    from technical_support.trace import emit

    p = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SAAS_TRACE_PATH", str(p))
    emit(
        "test",
        tool="get_api_usage",
        status="ok",
        access_code="sensitive",
        prompt="secret",
        duration_ms=1,
    )
    text = p.read_text(encoding="utf-8")
    assert "sensitive" not in text and "secret" not in text
    assert json.loads(text)["tool"] == "get_api_usage"


@pytest.mark.asyncio
async def test_local_feedback_persists_without_comment(tmp_path, monkeypatch):
    from schema import Feedback
    from service.service import saas_feedback

    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "feedback.db")
    value = Feedback(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="stars",
        score=0.8,
        kwargs={"comment": "private-token"},
    )
    await saas_feedback(value)
    await saas_feedback(value)
    with store._connect() as con:
        rows = con.execute("SELECT * FROM feedback").fetchall()
    assert len(rows) == 1 and rows[0]["score"] == 0.8
    assert "private-token" not in str([dict(row) for row in rows])


@pytest.mark.asyncio
@pytest.mark.parametrize("historical", [False, True])
async def test_success_guard_rejects_mixed_or_historical_ticket_ids(historical):
    observation = ToolMessage(
        content=json.dumps({"ok": True, "ticket_id": "ST-1111111111"}),
        name="create_technical_ticket",
        tool_call_id="t",
    )
    human = HumanMessage(content="创建工单")
    messages = [observation, human] if historical else [human, observation]
    request = ModelRequest(
        model=FakeToolModel(responses=[AIMessage(content="unused")]),
        messages=messages,
        state={"messages": messages},
        tools=agent.TECHNICAL_SUPPORT_TOOLS,
    )

    async def handler(request):
        content = (
            "已创建工单 ST-1111111111"
            if historical
            else "已创建工单 ST-1111111111 和 ST-2222222222"
        )
        return ModelResponse(result=[AIMessage(content=content)])

    result = await agent.trace_and_verify_response.awrap_model_call(request, handler)
    assert "无法确认" in result.result[0].content
