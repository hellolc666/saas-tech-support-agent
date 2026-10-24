
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from technical_support import knowledge_agent as agent
from technical_support import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "export.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", path)
    store.initialize_database()
    return path


@pytest.mark.parametrize("count,eligible", [(1, True), (5000, True), (5001, False)])
def test_export_boundary(db, count, eligible):
    result = store.check_export_eligibility("APP_DEMO_001", "DEMO-ALPHA-4821", count)
    assert result["eligible"] is eligible
    assert ("EXPORT_LIMIT_EXCEEDED" in result["reasons"]) is (not eligible)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "5001"])
def test_export_invalid_count(db, count):
    assert store.check_export_eligibility("APP_DEMO_001", "DEMO-ALPHA-4821", count)["error_code"] == "INVALID_RECORD_COUNT"


def test_export_rejects_unverified_identity(db):
    denied = store.check_export_eligibility("APP_DEMO_001", "DEMO-BETA-7394", 100)
    assert not denied["ok"] and "version" not in denied


def test_export_version_and_permission_come_from_database(db):
    result = store.check_export_eligibility("APP_DEMO_002", "DEMO-BETA-7394", 100)
    assert set(result["reasons"]) == {"VERSION_UNSUPPORTED", "EXPORT_PERMISSION_DENIED"}
    with store._connect() as con:
        con.execute("UPDATE applications SET version = '2.0' WHERE app_id = 'APP_DEMO_002'")
    result = store.check_export_eligibility("APP_DEMO_002", "DEMO-BETA-7394", 100)
    assert result["reasons"] == ["EXPORT_PERMISSION_DENIED"]
    with store._connect() as con:
        con.execute("UPDATE export_access SET role = '编辑者', can_access = 0 WHERE app_id = 'APP_DEMO_002'")
    assert store.check_export_eligibility("APP_DEMO_002", "DEMO-BETA-7394", 100)["reasons"] == ["EXPORT_PERMISSION_DENIED"]


@pytest.mark.asyncio
async def test_export_followup_keeps_count_and_app(db, monkeypatch):
    monkeypatch.setattr(agent, "retrieve_docs", lambda q: {"found": False, "references": []})
    graph = agent.build_technical_support_mock()
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "export"}}
    for text in ["为什么导出5001条失败？", "APP_DEMO_001"]:
        result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, cfg)
        assert result["awaiting"] == "导出诊断"
    result = await graph.ainvoke({"messages": [HumanMessage(content="DEMO-ALPHA-4821")]}, cfg)
    assert "5000条限制" in result["messages"][-1].content
    assert "2.0" in result["messages"][-1].content
    assert result["awaiting"] is None
    assert "DEMO-ALPHA" not in result["messages"][-1].content
