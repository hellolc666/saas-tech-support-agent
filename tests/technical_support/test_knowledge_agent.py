"""No paid API calls or embedding downloads: real graphs with injected tool/model data."""

import sys
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from technical_support import knowledge_agent as agent

REFERENCE = {
    "title": "401 登录失效与令牌过期",
    "content": "接口返回401时先重新登录，并检查令牌是否过期。",
    "source": "data/policies/saas_support_guide.md",
    "chunk_id": "example-chunk",
}


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_retrieval_adapter_uses_existing_search_entrypoint(monkeypatch):
    captured = []

    def search(query):
        captured.append(query)
        return {"references": [REFERENCE]}

    monkeypatch.setitem(sys.modules, "customer_support.policy", SimpleNamespace(search_policy=search))
    result = agent.retrieve_docs(" 401怎么办？ ")
    assert captured == ["401怎么办？"]
    assert result == {"found": True, "references": [REFERENCE]}


def test_missing_hits_do_not_expose_old_policy_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "customer_support.policy", SimpleNamespace(
        search_policy=lambda query: {"references": [], "content": "旧政策提示"}
    ))
    result = agent.retrieve_docs("unknown")
    assert result["found"] is False
    assert result["references"] == []
    assert "旧政策" not in str(result)


def test_blank_query_is_not_searched(monkeypatch):
    def unexpected(query):
        pytest.fail("Blank input must not call the retriever")
    monkeypatch.setitem(sys.modules, "customer_support.policy", SimpleNamespace(search_policy=unexpected))
    assert agent.retrieve_docs("  ")["found"] is False


def test_business_tools_are_registered_without_ecommerce_actions():
    assert [tool.name for tool in agent.TECHNICAL_SUPPORT_TOOLS] == [
        "search_technical_docs", "get_api_usage", "get_request_log", "check_export_eligibility",
        "create_technical_ticket", "get_ticket_status",
        "list_incident_scenarios", "get_incident_evidence",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("message,forced", [
    (HumanMessage(content="访问项目出现403怎么办？"), True),
    (HumanMessage(content="那1.0呢？"), True),
    (HumanMessage(content="你好！"), False),
    (HumanMessage(content="查询应用APP_DEMO_001的配额"), False),
    (HumanMessage(content="帮我创建工单"), False),
    (ToolMessage(content="检索结果", tool_call_id="call-1"), False),
])
async def test_retrieval_is_forced_only_on_substantive_human_turns(message, forced):
    request = ModelRequest(
        model=FakeToolModel(responses=[AIMessage(content="unused")]),
        messages=[], state={"messages": [message]}, tools=agent.TECHNICAL_SUPPORT_TOOLS,
    )
    captured = {}

    async def handler(updated):
        captured["choice"] = updated.tool_choice
        return ModelResponse(result=[AIMessage(content="ok")])

    await agent.require_document_lookup.awrap_model_call(request, handler)
    assert (captured["choice"] == "search_technical_docs") is forced


@pytest.mark.asyncio
async def test_online_graph_executes_retrieval_and_finishes(monkeypatch):
    queries = []

    def retrieve(query):
        queries.append(query)
        return {"found": True, "references": [REFERENCE]}

    monkeypatch.setattr(agent, "retrieve_docs", retrieve)
    model = FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_technical_docs", "args": {"query": "401登录失效"}, "id": "lookup-1"
        }]),
        AIMessage(content="请重新登录。来源：data/policies/saas_support_guide.md > 401 登录失效与令牌过期"),
    ])
    graph = agent.build_technical_support_graph(model)
    result = await graph.ainvoke({"messages": [HumanMessage(content="401登录失效怎么办？")]})
    assert queries == ["401登录失效"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "example-chunk" in tool_messages[0].content
    assert "saas_support_guide.md" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_mock_answer_quotes_actual_tool_data(monkeypatch):
    monkeypatch.setattr(agent, "retrieve_docs", lambda query: {"found": True, "references": [REFERENCE]})
    graph = agent.build_technical_support_mock()
    result = await graph.ainvoke({"messages": [HumanMessage(content="401怎么办？")]})
    answer = result["messages"][-1].content
    assert REFERENCE["content"] in answer
    assert f"{REFERENCE['source']} > {REFERENCE['title']}" in answer
    assert "离线检索演示" in answer


@pytest.mark.asyncio
async def test_mock_missing_document_does_not_invent_source(monkeypatch):
    monkeypatch.setattr(agent, "retrieve_docs", lambda query: {"found": False, "references": []})
    result = await agent.build_technical_support_mock().ainvoke({
        "messages": [HumanMessage(content="未收录的功能怎么办？")]
    })
    answer = result["messages"][-1].content
    assert "没有足够相关的资料" in answer
    assert "来源" not in answer


@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected", [
    ("帮我创建技术支持工单", "请补充应用 ID"),
    ("查询工单状态", "请补充应用 ID"),
    ("查询订单ORD1001", "暂不处理电商"),
    ("你好", "星河协作平台"),
])
async def test_mock_handles_scope_and_unavailable_actions(monkeypatch, question, expected):
    def unexpected(query):
        pytest.fail("This branch must not perform retrieval")
    monkeypatch.setattr(agent, "retrieve_docs", unexpected)
    result = await agent.build_technical_support_mock().ainvoke({"messages": [HumanMessage(content=question)]})
    assert expected in result["messages"][-1].content


@pytest.mark.asyncio
async def test_mock_follow_up_uses_previous_question_with_checkpoint(monkeypatch):
    captured = []

    def retrieve(query):
        captured.append(query)
        return {"found": True, "references": [REFERENCE]}

    monkeypatch.setattr(agent, "retrieve_docs", retrieve)
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "saas-follow-up"}}
    await graph.ainvoke({"messages": [HumanMessage(content="2.0支持批量导出吗？")]}, config)
    await graph.ainvoke({"messages": [HumanMessage(content="那1.0呢？")]}, config)
    assert "2.0支持批量导出" in captured[-1]
    assert "那1.0呢" in captured[-1]


@pytest.mark.asyncio
async def test_current_production_status_cannot_be_inferred_from_503_doc(monkeypatch):
    def unexpected(query):
        pytest.fail("A static document must not serve as a live status check")

    monkeypatch.setattr(agent, "retrieve_docs", unexpected)
    result = await agent.build_technical_support_mock().ainvoke({
        "messages": [HumanMessage(content="目前生产环境是否正在维护？")]
    })
    answer = result["messages"][-1].content
    assert "无法核实" in answer
    assert "503" not in answer


@pytest.mark.asyncio
async def test_online_model_does_not_use_static_docs_for_live_status():
    request = ModelRequest(
        model=FakeToolModel(responses=[AIMessage(content="unused")]),
        messages=[], state={"messages": [HumanMessage(content="目前生产环境是否正在维护？")]},
        tools=agent.TECHNICAL_SUPPORT_TOOLS,
    )

    async def unexpected(updated):
        pytest.fail("Live status has no authorized tool or document answer")

    response = await agent.require_document_lookup.awrap_model_call(request, unexpected)
    assert "无法核实" in response.result[0].content


@pytest.mark.asyncio
async def test_usage_followup_remembers_only_public_app_id(monkeypatch):
    monkeypatch.setattr(agent, "retrieve_docs", lambda query: {"found": False, "references": []})
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "quota-follow-up"}}
    first = await graph.ainvoke({
        "messages": [HumanMessage(content="查询 APP_DEMO_001 的配额")]
    }, cfg)
    assert "演示验证码" in first["messages"][-1].content
    assert first["slots"] == {"app_id": "APP_DEMO_001"}
    assert "access_code" not in first["slots"]
    second = await graph.ainvoke({
        "messages": [HumanMessage(content="DEMO-ALPHA-4821")]
    }, cfg)
    assert "演示配额快照" in second["messages"][-1].content
    assert second["awaiting"] is None
    assert second["slots"] == {}


@pytest.mark.asyncio
async def test_request_followup_does_not_turn_missing_id_into_usage(monkeypatch):
    monkeypatch.setattr(agent, "retrieve_docs", lambda query: {"found": False, "references": []})
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "request-follow-up"}}
    first = await graph.ainvoke({
        "messages": [HumanMessage(content="查询 APP_DEMO_001 请求日志，DEMO-ALPHA-4821")]
    }, cfg)
    assert "请求 ID" in first["messages"][-1].content
    second = await graph.ainvoke({
        "messages": [HumanMessage(content="REQ_DEMO_429，DEMO-ALPHA-4821")]
    }, cfg)
    assert "演示请求快照" in second["messages"][-1].content


@pytest.mark.asyncio
async def test_tool_failure_is_safe_observation_not_success(monkeypatch):
    from technical_support import tools

    def unavailable(*args, **kwargs):
        raise RuntimeError("private-db-connection-secret")

    monkeypatch.setattr(tools, "get_api_usage", unavailable)
    model = FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "get_api_usage", "args": {
            "app_id": "APP_DEMO_001", "access_code": "DEMO-ALPHA-4821",
        }, "id": "failed-query"}]),
        AIMessage(content="工具执行失败，无法确认结果。"),
    ])
    graph = agent.build_technical_support_graph(model)
    result = await graph.ainvoke({"messages": [HumanMessage(content="查询应用配额")]})
    observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert observation.status == "error"
    assert "TOOL_EXECUTION_FAILED" in observation.content
    assert "private-db-connection-secret" not in str(result)
