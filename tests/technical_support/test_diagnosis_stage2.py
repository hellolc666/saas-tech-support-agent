"""Stage 2: evidence routing, cause inference and LangGraph task readiness."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from technical_support import diagnosis, incident_scenarios as fixtures, store, tools, workflow


@pytest.fixture
def incident_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "stage2.db")
    return tmp_path / "stage2.db"


def _payload(case, view, db):
    app = fixtures.SCENARIOS[case]["app_id"]
    code = "DEMO-ALPHA-4821" if app == "APP_DEMO_001" else "DEMO-BETA-7394"
    return fixtures.get_incident_evidence(case, app, code, view, db)


def _message(payload):
    return SimpleNamespace(name="get_incident_evidence", content=json.dumps(payload))


@pytest.mark.parametrize("case,expected", sorted(fixtures.GROUND_TRUTH.items()))
def test_two_step_evidence_infers_all_eight_causes(incident_db, case, expected):
    overview = _payload(case, "overview", incident_db)
    first = diagnosis.collected_views([_message(overview)], case)
    assert diagnosis.assess(first) is None
    view = diagnosis.next_evidence_view(first)
    assert view in ("traffic", "auth", "export", "dependency")
    second = diagnosis.collected_views(
        [_message(overview), _message(_payload(case, view, incident_db))], case
    )
    assert diagnosis.next_evidence_view(second) is None
    result = diagnosis.assess(second)
    assert result["root_cause"] == expected
    assert result["is_demo_snapshot"] is True
    assert result["evidence_views"] == ["overview", view]
    assert "ground_truth" not in json.dumps(result)


def test_missing_evidence_and_different_case_cannot_prove_a_cause(incident_db):
    overview = _payload("F01", "overview", incident_db)
    wrong = _payload("F02", "traffic", incident_db)
    assert diagnosis.assess(diagnosis.collected_views([_message(overview)], "F01")) is None
    mixed = diagnosis.collected_views([_message(overview), _message(wrong)], "F01")
    assert diagnosis.assess(mixed) is None
    assert diagnosis.next_evidence_view(mixed) == "traffic"
    # A detail from a different application must never satisfy the current case.
    mixed_app = dict(_payload("F01", "traffic", incident_db), app_id="APP_DEMO_002")
    mixed_app_observations = diagnosis.collected_views(
        [_message(overview), _message(mixed_app)], "F01"
    )
    assert diagnosis.assess(mixed_app_observations) is None


def test_new_tools_read_evidence_without_leaking_labels(incident_db):
    listing = tools.list_incident_scenarios_tool.invoke({
        "app_id": "APP_DEMO_001", "access_code": "DEMO-ALPHA-4821"
    })
    assert listing["ok"] and any(x["incident_id"] == "F01" for x in listing["incidents"])
    args = {
        "incident_id": "F01", "app_id": "APP_DEMO_001", "access_code": "DEMO-ALPHA-4821",
        "view": "traffic",
    }
    valid = tools.get_incident_evidence_tool.invoke(args)
    assert valid["ok"] and valid["evidence"]["window_used"] == 100
    assert "root_cause" not in valid
    forbidden = tools.get_incident_evidence_tool.invoke({**args, "access_code": "DEMO-BETA-7394"})
    assert not forbidden["ok"]
    outside = tools.get_incident_evidence_tool.invoke({**args, "incident_id": "F05"})
    assert outside["error_code"] == "INCIDENT_NOT_FOUND"


def test_context_extracts_case_id():
    facts = workflow.context_from([
        HumanMessage(content="诊断 APP_DEMO_001 的 F01 故障；演示验证码 DEMO-ALPHA-4821")
    ])
    assert facts["incident_id"] == "F01" and facts["app_id"] == "APP_DEMO_001"
    assert workflow.context_from([HumanMessage(content="诊断F01怎么办")])["incident_id"] == "F01"


def test_task_is_pending_after_overview_and_completed_after_detail(incident_db):
    app, code, case = "APP_DEMO_001", "DEMO-ALPHA-4821", "F01"
    instruction = HumanMessage(content=f"请诊断 {app} 的 {case}，演示验证码 {code}")
    item = {"id": "incident", "title": "故障取证", "tool": "get_incident_evidence",
            "depends_on": [], "query": case, "conditional": False}
    state = {"understanding": {"tasks": [item]}, "messages": [instruction]}
    first = workflow.task_statuses(state)
    assert first[0]["status"] == "pending"
    state["messages"].append(ToolMessage(
        name="get_incident_evidence", content=json.dumps(_payload(case, "overview", incident_db)),
        tool_call_id="overview",
    ))
    assert workflow.task_statuses(state)[0]["status"] == "pending"
    state["messages"].append(ToolMessage(
        name="get_incident_evidence", content=json.dumps(_payload(case, "traffic", incident_db)),
        tool_call_id="traffic",
    ))
    result = workflow.task_statuses(state)[0]
    assert result["status"] == "completed"
    assert result["root_cause"] == "QUOTA_EXHAUSTED"


@pytest.mark.asyncio
async def test_early_model_answer_forces_two_evidence_reads(incident_db):
    """The existing planner/middleware must not stop at a plausible answer."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from test_knowledge_agent import FakeToolModel
    from technical_support.knowledge_agent import build_technical_support_graph

    task = {
        "id": "incident", "title": "调查 F01", "tool": "get_incident_evidence",
        "depends_on": [], "query": "F01", "conditional": False,
    }
    planner = FakeToolModel(responses=[AIMessage(content=json.dumps({
        "independent_query": "调查 APP_DEMO_001 的 F01 429 根因",
        "missing_information": [], "clarification": "", "tasks": [task],
    }, ensure_ascii=False))])
    # Model repeatedly tries to finish before looking at evidence.
    model = FakeToolModel(responses=[
        AIMessage(content="这是配额故障。"),
        AIMessage(content="应当核实。"),
        AIMessage(content="根据已经核实的模拟配额窗口数据，根因是时间窗口配额耗尽。"),
    ])
    graph = build_technical_support_graph(model, planner=planner)
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "stage2-f01"}}
    result = await graph.ainvoke({"messages": [HumanMessage(
        content="请诊断 APP_DEMO_001 的合成故障 F01，演示验证码 DEMO-ALPHA-4821"
    )]}, cfg)
    observations = [m for m in result["messages"] if isinstance(m, ToolMessage)
                    and m.name == "get_incident_evidence"]
    assert {json.loads(x.content)["view"] for x in observations} == {"overview", "traffic"}
    assert result["task_results"][0]["status"] == "completed"
    assert result["task_results"][0]["root_cause"] == "QUOTA_EXHAUSTED"
