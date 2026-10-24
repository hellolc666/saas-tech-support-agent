import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError
from test_knowledge_agent import FakeToolModel

from technical_support import knowledge_agent, profile, store, trace, workflow
from technical_support.inspector_api import router


def plan(*tools):
    return {
        "independent_query": "APP_DEMO_001导出5001条失败",
        "tasks": [
            {
                "id": str(i),
                "title": name,
                "tool": name,
                "depends_on": [],
                "query": "5001条导出限制",
                "conditional": False,
            }
            for i, name in enumerate(tools)
        ],
    }


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "business.db")
    monkeypatch.setenv("SAAS_TRACE_PATH", str(tmp_path / "trace.jsonl"))


@pytest.mark.parametrize(
    "query,expected",
    [("那5001条呢", "5001"), ("改为5000条", "5000"), ("不是5001条，是4999条", "4999")],
)
def test_context_preserves_latest_quantity(query, expected):
    facts = workflow.context_from(
        [HumanMessage(content="APP_DEMO_001导出100条失败"), HumanMessage(content=query)]
    )
    assert facts["record_count"] == expected and facts["app_id"] == "APP_DEMO_001"


def test_switch_app_clears_stale_request_and_quantity():
    facts = workflow.context_from(
        [
            HumanMessage(content="APP_DEMO_001 REQ_DEMO_429 导出5001条"),
            HumanMessage(content="现在查APP_DEMO_002"),
        ]
    )
    assert facts == {"app_id": "APP_DEMO_002"}


def test_summary_is_redacted_and_recent_tool_pairs_remain():
    messages = []
    for i in range(10):
        messages += [
            HumanMessage(content=f"问题{i} DEMO-ALPHA-4821"),
            AIMessage(content="", tool_calls=[{"name": "get_api_usage", "args": {}, "id": str(i)}]),
            ToolMessage(content='{"used":99}', tool_call_id=str(i), name="get_api_usage"),
            AIMessage(content="结果"),
        ]
    recent, summary = workflow.recent_context(messages, keep_turns=3)
    assert isinstance(recent[0], HumanMessage) and len(recent) == 12
    assert "DEMO-ALPHA-4821" not in summary and "used" not in summary


@pytest.mark.parametrize("dependencies", [["0"], ["unknown"]])
def test_invalid_task_dependencies_rejected(dependencies):
    value = plan("get_api_usage")
    value["tasks"][0]["depends_on"] = dependencies
    with pytest.raises(ValidationError):
        workflow.Understanding.model_validate(value)


def test_partial_failure_blocks_dependent_ticket():
    value = plan("get_api_usage", "create_technical_ticket")
    value["tasks"][1]["depends_on"] = ["0"]
    state = {
        "understanding": value,
        "messages": [
            HumanMessage(content="APP_DEMO_001 DEMO-ALPHA-4821"),
            ToolMessage(
                content='{"ok":false,"error_code":"TOOL_FAILED"}',
                tool_call_id="t",
                name="get_api_usage",
            ),
        ],
    }
    statuses = workflow.task_statuses(state)
    assert [s["status"] for s in statuses] == ["failed", "blocked"]


@pytest.mark.asyncio
async def test_graph_persists_plan_and_evidence(isolated, monkeypatch):
    monkeypatch.setattr(
        knowledge_agent,
        "retrieve_docs",
        lambda q: {
            "found": True,
            "references": [{"chunk_id": "doc", "content": "5000条", "source": "guide.md"}],
        },
    )
    planner = FakeToolModel(
        responses=[
            AIMessage(content=json.dumps(plan("check_export_eligibility", "search_technical_docs")))
        ]
    )
    model = FakeToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_export_eligibility",
                        "args": {
                            "app_id": "APP_DEMO_001",
                            "access_code": "DEMO-ALPHA-4821",
                            "record_count": 5001,
                        },
                        "id": "e",
                    },
                    {"name": "search_technical_docs", "args": {"query": "导出"}, "id": "d"},
                ],
            ),
            AIMessage(content="单次上限5000条"),
        ]
    )
    graph = knowledge_agent.build_technical_support_graph(model, planner=planner)
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "planned", "trace_id": "test-trace"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="APP_DEMO_001 导出5001条失败 DEMO-ALPHA-4821")]}, cfg
    )
    assert len(result["task_results"]) == 2
    assert all(t["status"] == "completed" for t in result["task_results"])
    assert "任务进度" in result["messages"][-1].content
    snapshot = await graph.aget_state(cfg)
    assert snapshot.values["business_context"]["record_count"] == "5001"
    events = trace.read_trace("planned")
    assert {"rewrite", "tool_execution", "task_summary", "request_complete"} <= {
        e["event"] for e in events
    }
    assert "DEMO-ALPHA-4821" not in json.dumps(events)


@pytest.mark.asyncio
async def test_planner_failure_keeps_original_request(isolated, monkeypatch):
    monkeypatch.setattr(workflow, "get_config", lambda: {"configurable": {}})
    middleware = workflow.SupportWorkflow(
        SimpleNamespace(ainvoke=AsyncMock(side_effect=TimeoutError))
    )
    result = await middleware.abefore_agent({"messages": [HumanMessage(content="401怎么办")]}, None)
    assert result["understanding"]["independent_query"] == "401怎么办"
    assert result["understanding"]["planner_error"] == "TimeoutError"


def test_profile_persistence_and_trace_isolation(isolated):
    token = profile.create_profile()
    other = profile.create_profile()
    owner = profile.authenticate(token)
    profile.save_preferences(owner, {"answer_style": "简洁中文", "environment_note": "测试环境"})
    assert profile.get_preferences(profile.authenticate(token))["answer_style"] == "简洁中文"
    profile.claim_thread(owner, "private-thread")
    with pytest.raises(ValueError):
        profile.claim_thread(profile.authenticate(other), "private-thread")
    trace.emit("model", thread_id="private-thread", status="ok")
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        assert (
            c.get("/saas/traces/private-thread", headers={"X-SaaS-Token": other}).status_code == 404
        )
        assert c.get("/saas/traces/private-thread", headers={"X-SaaS-Token": token}).json()[
            "events"
        ]
        assert c.get("/saas/memory").status_code == 401
        assert c.delete("/saas/memory", headers={"X-SaaS-Token": token}).status_code == 200
    assert profile.get_preferences(owner) == {}


def test_sensitive_profile_rejected(isolated):
    owner = profile.authenticate(profile.create_profile())
    with pytest.raises(ValueError):
        profile.save_preferences(owner, {"environment_note": "验证码 DEMO-ALPHA-4821"})


def test_nested_trace_parent_relationship(isolated):
    with trace.span("outer"):
        with trace.span("inner"):
            trace.emit("ranking", ranking=[{"id": "doc", "score": 0.9}])
    rows = [json.loads(line) for line in trace.trace_path().read_text().splitlines()]
    inner = next(r for r in rows if r["event"] == "inner")
    outer = next(r for r in rows if r["event"] == "outer")
    assert inner["parent_span_id"] == outer["span_id"]


def test_conditional_ticket_stays_blocked_after_rejected_attempt():
    value = plan("create_technical_ticket")
    value["tasks"][0]["conditional"] = True
    state = {
        "understanding": value,
        "messages": [
            HumanMessage(content="APP_DEMO_001 DEMO-ALPHA-4821"),
            ToolMessage(
                content='{"ok":false,"error_code":"TASK_NOT_READY"}',
                tool_call_id="write",
                name="create_technical_ticket",
                status="error",
            ),
        ],
    }
    assert workflow.task_statuses(state)[0]["status"] == "waiting_input"


def test_trace_preserves_numeric_token_usage():
    assert trace.redact({"input_tokens": 123, "api_key": "secret"}) == {
        "input_tokens": 123,
        "api_key": "[hidden]",
    }


@pytest.mark.asyncio
async def test_planner_cannot_drop_export_document_evidence(isolated, monkeypatch):
    monkeypatch.setattr(workflow, "get_config", lambda: {"configurable": {}})
    planner = FakeToolModel(
        responses=[AIMessage(content=json.dumps(plan("check_export_eligibility")))]
    )
    result = await workflow.SupportWorkflow(planner).abefore_agent(
        {"messages": [HumanMessage(content="APP_DEMO_001导出5001条失败")]}, None
    )
    assert {t["tool"] for t in result["understanding"]["tasks"]} == {
        "check_export_eligibility",
        "search_technical_docs",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_enhanced_plan_survives_checkpoint_restart(isolated, tmp_path, decision):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    value = plan("create_technical_ticket")
    value["independent_query"] = "APP_DEMO_001创建工单"
    planner = FakeToolModel(responses=[AIMessage(content=json.dumps(value))])
    cfg = {"configurable": {"thread_id": "durable-plan"}}
    owner = profile.authenticate(profile.create_profile())
    profile.save_preferences(owner, {"answer_style": "简洁中文"})
    cfg["configurable"]["_memory_owner"] = owner
    store.initialize_database()
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_technical_ticket",
                "id": "write",
                "args": {
                    "app_id": "APP_DEMO_001",
                    "access_code": "DEMO-ALPHA-4821",
                    "category": "接口异常",
                    "summary": "增强流程恢复测试",
                },
            }
        ],
    )
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        graph = knowledge_agent.build_technical_support_graph(
            FakeToolModel(responses=[call]), planner=planner
        )
        graph.checkpointer = saver
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="APP_DEMO_001创建工单，验证码DEMO-ALPHA-4821，类别接口异常，摘要增强流程恢复测试"
                    )
                ]
            },
            cfg,
        )
        assert result["__interrupt__"]
        snapshot = await graph.aget_state(cfg)
        assert snapshot.values["profile_preferences"]["answer_style"] == "简洁中文"
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        graph = knowledge_agent.build_technical_support_graph(
            FakeToolModel(responses=[AIMessage(content="操作完成")]), planner=planner
        )
        graph.checkpointer = saver
        result = await graph.ainvoke(Command(resume={"decisions": [{"type": decision}]}), cfg)
        assert result["task_results"][0]["status"] == (
            "completed" if decision == "approve" else "cancelled"
        )
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == (
            decision == "approve"
        )


@pytest.mark.asyncio
async def test_early_answer_cannot_skip_planned_read_only_step(isolated, monkeypatch):
    monkeypatch.setattr(
        knowledge_agent,
        "retrieve_docs",
        lambda q: {"found": True, "references": [{"chunk_id": "proof"}]},
    )
    planner = FakeToolModel(
        responses=[AIMessage(content=json.dumps(plan("search_technical_docs")))]
    )
    model = FakeToolModel(
        responses=[AIMessage(content="提前作答"), AIMessage(content="根据实际证据回答")]
    )
    graph = knowledge_agent.build_technical_support_graph(model, planner=planner)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="导出5001条失败")]},
        {"configurable": {"thread_id": "early"}},
    )
    assert any(
        isinstance(m, ToolMessage) and m.name == "search_technical_docs" for m in result["messages"]
    )
    assert result["task_results"][0]["status"] == "completed"
