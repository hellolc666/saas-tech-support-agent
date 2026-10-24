from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from technical_support import knowledge_agent as agent
from technical_support import store, tools


@pytest.fixture
def demo_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "saas.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", path)
    return path


def test_read_data_requires_matching_demo_code_and_app(demo_db):
    good = store.get_api_usage("APP_DEMO_001", "DEMO-ALPHA-4821")
    assert good["ok"] and good["used"] == 100 and good["is_demo_snapshot"]
    wrong = store.get_api_usage("APP_DEMO_001", "DEMO-BETA-7394")
    unknown = store.get_api_usage("APP_DEMO_999", "DEMO-BETA-7394")
    assert wrong == unknown
    assert not wrong["ok"]
    assert store.get_request_log("REQ_DEMO_429", "APP_DEMO_001", "DEMO-ALPHA-4821")["http_status"] == 429
    assert store.get_request_log("REQ_DEMO_429", "APP_DEMO_002", "DEMO-BETA-7394")["error_code"] == "REQUEST_NOT_FOUND"


def test_ticket_cannot_be_read_from_other_app_or_tenant(demo_db):
    created = store.create_technical_ticket(
        "APP_DEMO_001", "DEMO-ALPHA-4821", "接口异常", "请求持续429", "submission-1"
    )
    assert created["ok"] and created["status"] == "待处理"
    ticket_id = created["ticket_id"]
    own = store.get_ticket_status(ticket_id, "APP_DEMO_001", "DEMO-ALPHA-4821")
    assert own["ticket_id"] == ticket_id
    other = store.get_ticket_status(ticket_id, "APP_DEMO_002", "DEMO-BETA-7394")
    assert not other["ok"] and other["error_code"] == "TICKET_NOT_FOUND"
    assert not store.get_ticket_status(ticket_id, "APP_DEMO_001", "DEMO-BETA-7394")["ok"]


def test_idempotency_retries_and_payload_conflicts(demo_db):
    args = ("APP_DEMO_001", "DEMO-ALPHA-4821", "接口异常", "请求持续429", "submission-1")
    first = store.create_technical_ticket(*args)
    second = store.create_technical_ticket(*args)
    assert second["ticket_id"] == first["ticket_id"] and second["duplicate"]
    changed = store.create_technical_ticket(*args[:3], "另一个问题", args[4])
    assert changed["error_code"] == "IDEMPOTENCY_CONFLICT"
    denied = store.create_technical_ticket("APP_DEMO_001", "DEMO-BETA-7394", "a", "b", "submission-2")
    assert not denied["ok"]


def test_concurrent_retries_create_exactly_one_ticket(demo_db):
    store.initialize_database()
    args = ("APP_DEMO_001", "DEMO-ALPHA-4821", "接口异常", "请求持续429", "same-key")
    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda _: store.create_technical_ticket(*args), range(4)))
    assert len({item["ticket_id"] for item in outcomes}) == 1
    assert sorted(item["duplicate"] for item in outcomes) == [False, True, True, True]


def test_agent_tool_deduplicates_submission_not_ticket_content(demo_db):
    args = {"app_id": "APP_DEMO_001", "access_code": "DEMO-ALPHA-4821",
            "category": "接口异常", "summary": "请求持续429"}
    first = tools.create_technical_ticket_tool.invoke({**args, "submission_id": "request-A"})
    second = tools.create_technical_ticket_tool.invoke({**args, "submission_id": "request-A"})
    assert first["ok"] and second["duplicate"] and first["ticket_id"] == second["ticket_id"]
    another = tools.create_technical_ticket_tool.invoke({**args, "submission_id": "request-B"})
    assert another["ok"] and not another["duplicate"]
    assert another["ticket_id"] != first["ticket_id"]
    conflict = tools.create_technical_ticket_tool.invoke(
        {**args, "summary": "这是不同故障", "submission_id": "request-A"}
    )
    assert conflict["error_code"] == "IDEMPOTENCY_CONFLICT"
    # Without an explicit submission ID, every direct tool call is new.
    direct = tools.create_technical_ticket_tool.invoke(args)
    assert direct["ticket_id"] not in {first["ticket_id"], another["ticket_id"]}
    args["summary"] = "验证码 DEMO-ALPHA-4821"
    assert not tools.create_technical_ticket_tool.invoke(args)["ok"]


def test_new_mock_ticket_with_same_content_is_not_deduplicated():
    first = agent._pending_ticket("问题是：请求持续429", "APP_DEMO_001", "DEMO-ALPHA-4821")
    second = agent._pending_ticket("问题是：请求持续429", "APP_DEMO_001", "DEMO-ALPHA-4821")
    assert first is not None and second is not None
    assert first["summary"] == second["summary"]
    assert first["idempotency_key"] != second["idempotency_key"]


@pytest.mark.asyncio
async def test_mock_requires_explicit_confirmation_and_can_query_status(demo_db):
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "mock-ticket"}}
    first = await graph.ainvoke({"messages": [HumanMessage(content=(
        "为 APP_DEMO_001 创建技术支持工单，演示验证码 DEMO-ALPHA-4821，"
        "问题是：请求持续429"
    ))]}, config)
    assert "请回复“确认”" in first["messages"][-1].content
    assert "DEMO-ALPHA-4821" not in first["messages"][-1].content
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == 0
    completed = await graph.ainvoke({"messages": [HumanMessage(content="确认")]}, config)
    ticket_id = completed["messages"][-1].content.split("：")[1].split("，")[0]
    assert store.get_ticket_status(ticket_id, "APP_DEMO_001", "DEMO-ALPHA-4821")["ok"]
    status = await graph.ainvoke({"messages": [HumanMessage(content=(
        f"查询工单 {ticket_id} 状态，应用 APP_DEMO_001，演示验证码 DEMO-ALPHA-4821"
    ))]}, config)
    assert "待处理" in status["messages"][-1].content


@pytest.mark.asyncio
async def test_mock_rejection_does_not_write(demo_db):
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "reject-ticket"}}
    await graph.ainvoke({"messages": [HumanMessage(content=(
        "为 APP_DEMO_001 创建工单，演示验证码 DEMO-ALPHA-4821，问题是：登录错误"
    ))]}, config)
    answer = await graph.ainvoke({"messages": [HumanMessage(content="取消")]}, config)
    assert "没有创建" in answer["messages"][-1].content
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == 0


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,should_write", [("approve", True), ("reject", False)])
async def test_online_graph_hitl_prevents_unapproved_writes(demo_db, decision, should_write):
    store.initialize_database()
    model = FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "create_technical_ticket", "args": {
            "app_id": "APP_DEMO_001", "access_code": "DEMO-ALPHA-4821",
            "category": "接口异常", "summary": "请求持续429",
        }, "id": "ticket-call"}]),
        AIMessage(content="写入操作处理完成。"),
    ])
    graph = agent.build_technical_support_graph(model)
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": f"online-{decision}"}}
    interrupted = await graph.ainvoke({"messages": [HumanMessage(content="创建技术支持工单")]}, config)
    assert interrupted["__interrupt__"]
    assert "create_technical_ticket" in str(interrupted["__interrupt__"][0].value)
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == 0
    choice = {"type": "approve"} if should_write else {"type": "reject", "message": "已取消"}
    result = await graph.ainvoke(Command(resume={"decisions": [choice]}), config)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == int(should_write)
