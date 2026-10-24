import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from technical_support import knowledge_agent as agent
from technical_support import store


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_durable_approval_survives_graph_and_connection_restart(
    tmp_path, monkeypatch, decision
):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "business.db")
    store.initialize_database()
    path = str(tmp_path / "checkpoint.db")
    cfg = {"configurable": {"thread_id": "persistent-ticket"}}
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_technical_ticket",
                "id": "create",
                "args": {
                    "app_id": "APP_DEMO_001",
                    "access_code": "DEMO-ALPHA-4821",
                    "category": "接口异常",
                    "summary": "持续429",
                },
            }
        ],
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = agent.build_technical_support_graph(Model(responses=[call]))
        graph.checkpointer = saver
        result = await graph.ainvoke({"messages": [HumanMessage(content="创建工单")]}, cfg)
        assert result["__interrupt__"]
        with store._connect() as con:
            assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == 0
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = agent.build_technical_support_graph(
            Model(responses=[AIMessage(content="操作结束")])
        )
        graph.checkpointer = saver
        await graph.ainvoke(Command(resume={"decisions": [{"type": decision}]}), cfg)
        with store._connect() as con:
            assert con.execute("SELECT COUNT(*) FROM technical_tickets").fetchone()[0] == (
                decision == "approve"
            )
