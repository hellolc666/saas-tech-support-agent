import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from schema import ChatMessage
from service.service import _safe_saas_message
from technical_support import profile, store
from technical_support.inspector_api import router
from technical_support.memory_extractor import extract_and_store


class Extractor:
    def __init__(self, *actions, error=None):
        self.actions = list(actions) or [{"operation": "noop"}]
        self.error = error
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.error:
            raise self.error
        return AIMessage(content=json.dumps({"actions": self.actions}, ensure_ascii=False))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", tmp_path / "business.db")
    monkeypatch.setenv("SAAS_TRACE_PATH", str(tmp_path / "trace.jsonl"))


def identity():
    token = profile.create_profile()
    return token, profile.authenticate(token)


@pytest.mark.asyncio
async def test_memory_add_duplicate_update_delete_and_restart(isolated):
    _token, owner = identity()
    first = Extractor(
        {
            "operation": "add",
            "memory_key": "environment.os",
            "memory_value": "Windows 11",
            "category": "environment",
        }
    )
    changed = await extract_and_store(first, owner, "thread-a", "我主要使用 Windows 11")
    assert changed == [{"operation": "add", "memory_key": "environment.os"}]
    assert profile.list_memories(owner)[0]["memory_value"] == "Windows 11"

    # Retrying the same user turn cannot call the model or duplicate the row.
    assert await extract_and_store(first, owner, "thread-a", "我主要使用 Windows 11") == []
    assert first.calls == 1 and len(profile.list_memories(owner)) == 1

    update = Extractor(
        {
            "operation": "add",
            "memory_key": "environment.os",
            "memory_value": "Ubuntu 24.04",
            "category": "environment",
        }
    )
    changed = await extract_and_store(update, owner, "thread-a", "我已经改用 Ubuntu 24.04")
    assert changed[0]["operation"] == "update"
    assert profile.select_memories(owner, "如何部署", limit=8)[0]["memory_value"] == "Ubuntu 24.04"

    # Every profile call opens a fresh SQLite connection, proving disk persistence.
    assert profile.authenticate(_token) == owner
    forgetting = Extractor(
        {
            "operation": "delete",
            "memory_key": "environment.os",
            "memory_value": "",
            "category": "environment",
        }
    )
    await extract_and_store(forgetting, owner, "thread-b", "请忘记我的操作系统记忆")
    assert profile.list_memories(owner) == []


@pytest.mark.asyncio
async def test_identical_text_in_separate_turns_can_be_relearned(isolated):
    _token, owner = identity()
    extractor = Extractor(
        {
            "operation": "add",
            "memory_key": "environment.os",
            "memory_value": "Windows",
            "category": "environment",
        }
    )
    statement = "我主要使用 Windows"
    await extract_and_store(extractor, owner, "thread-a", statement, message_id="message-1")
    profile.clear_memories(owner)
    await extract_and_store(extractor, owner, "thread-a", statement, message_id="message-2")
    assert extractor.calls == 2
    assert profile.list_memories(owner)[0]["memory_value"] == "Windows"
    assert (
        await extract_and_store(extractor, owner, "thread-a", statement, message_id="message-2")
        == []
    )
    assert extractor.calls == 2


@pytest.mark.asyncio
async def test_memory_filters_dynamic_secret_noop_and_failure(isolated):
    _token, owner = identity()
    unsafe = Extractor(
        {
            "operation": "add",
            "memory_key": "background.api_key",
            "memory_value": "sk-secret",
            "category": "background",
        }
    )
    assert await extract_and_store(unsafe, owner, "a", "以后记住我的 API KEY 是 sk-secret") == []

    dynamic = Extractor(
        {
            "operation": "add",
            "memory_key": "background.error",
            "memory_value": "当前接口 429",
            "category": "background",
        }
    )
    assert await extract_and_store(dynamic, owner, "b", "我现在遇到 429 错误") == []
    noop = Extractor({"operation": "noop"})
    assert await extract_and_store(noop, owner, "c", "你好") == []
    failed = Extractor(error=RuntimeError("model unavailable"))
    assert await extract_and_store(failed, owner, "d", "我以后偏好简短回答") == []
    assert profile.list_memories(owner) == []


@pytest.mark.asyncio
async def test_memory_database_failure_does_not_fail_answer(isolated, monkeypatch):
    _token, owner = identity()

    def unavailable(_owner):
        raise OSError("database temporarily unavailable")

    monkeypatch.setattr(profile, "get_auto_memory_enabled", unavailable)
    extractor = Extractor()
    assert await extract_and_store(extractor, owner, "thread-a", "我喜欢简短回答") == []
    assert extractor.calls == 0


def test_memory_and_conversation_owner_isolation(isolated):
    token_a, owner_a = identity()
    token_b, owner_b = identity()
    profile.claim_thread(owner_a, "thread-a")
    profile.ensure_conversation(owner_a, "thread-a", "请帮我排查 Docker 部署")
    profile.apply_memory_actions(
        owner_a,
        "thread-a",
        [
            {
                "operation": "add",
                "memory_key": "environment.os",
                "memory_value": "Windows",
                "category": "environment",
            }
        ],
    )
    assert profile.list_memories(owner_b) == []
    with pytest.raises(ValueError, match="不属于"):
        profile.claim_thread(owner_b, "thread-a")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    headers_a, headers_b = {"X-SaaS-Token": token_a}, {"X-SaaS-Token": token_b}
    assert len(client.get("/saas/conversations", headers=headers_a).json()["items"]) == 1
    assert client.get("/saas/conversations", headers=headers_b).json()["items"] == []
    assert (
        client.patch(
            "/saas/conversations/thread-a", headers=headers_b, json={"title": "越权修改"}
        ).status_code
        == 404
    )
    assert client.delete("/saas/conversations/thread-a", headers=headers_b).status_code == 404


def test_conversation_list_search_rename_and_soft_delete(isolated):
    token, owner = identity()
    for thread_id, message in [("old", "导出失败如何处理"), ("new", "Docker 部署问题")]:
        profile.claim_thread(owner, thread_id)
        profile.ensure_conversation(owner, thread_id, message)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    headers = {"X-SaaS-Token": token}
    assert {
        item["thread_id"]
        for item in client.get("/saas/conversations", headers=headers).json()["items"]
    } == {"old", "new"}
    renamed = client.patch(
        "/saas/conversations/new", headers=headers, json={"title": "生产环境部署"}
    )
    assert renamed.status_code == 200 and renamed.json()["title"] == "生产环境部署"
    result = client.get("/saas/conversations", headers=headers, params={"search": "生产"})
    assert [item["thread_id"] for item in result.json()["items"]] == ["new"]

    assert client.delete("/saas/conversations/new", headers=headers).status_code == 204
    assert not profile.owns_thread(owner, "new")
    assert profile.owns_thread(owner, "new", include_deleted=True)
    with pytest.raises(ValueError, match="已删除"):
        profile.claim_thread(owner, "new")
    assert (
        client.get("/saas/conversations", headers=headers, params={"search": "生产"}).json()[
            "items"
        ]
        == []
    )


def test_auto_memory_switch_and_manual_memory_management(isolated):
    token, owner = identity()
    profile.apply_memory_actions(
        owner,
        "source",
        [
            {
                "operation": "add",
                "memory_key": "preference.style",
                "memory_value": "简短",
                "category": "preference",
            }
        ],
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    headers = {"X-SaaS-Token": token}

    assert (
        client.patch(
            "/saas/memory-settings", headers=headers, json={"auto_memory_enabled": False}
        ).json()["auto_memory_enabled"]
        is False
    )
    item = client.get("/saas/memories", headers=headers).json()["items"][0]
    assert client.delete(f"/saas/memories/{item['id']}", headers=headers).status_code == 204
    assert client.get("/saas/memories", headers=headers).json()["items"] == []


def test_saas_tool_details_are_redacted_without_mutating_checkpoint_message():
    original = ChatMessage(
        type="ai",
        content="准备查询",
        tool_calls=[
            {
                "name": "get_api_usage",
                "args": {"access_code": "DEMO-ALPHA-4821", "app_id": "APP_DEMO_001"},
                "id": "call-1",
            }
        ],
    )
    safe = _safe_saas_message(original)
    assert safe.tool_calls[0]["args"]["access_code"] == "[hidden]"
    assert original.tool_calls[0]["args"]["access_code"] == "DEMO-ALPHA-4821"
    tool = _safe_saas_message(ChatMessage(type="tool", content='{"access_code":"DEMO-ALPHA-4821"}'))
    assert "DEMO-ALPHA-4821" not in tool.content


def test_existing_manual_preference_migrates_and_can_be_deleted(isolated):
    token, owner = identity()
    with profile.connect() as con:
        con.execute(
            "UPDATE support_profiles SET preferences=? WHERE owner=?",
            (json.dumps({"answer_style": "简短回答"}, ensure_ascii=False), owner),
        )
        con.commit()
    profile.authenticate(token)
    item = profile.list_memories(owner)[0]
    assert (item["memory_key"], item["memory_value"]) == (
        "preference.answer_style",
        "简短回答",
    )
    assert profile.delete_memory(owner, item["id"])
    profile.authenticate(token)
    assert profile.list_memories(owner) == []
    assert profile.get_preferences(owner) == {}


def test_saas_only_requires_workspace_for_invoke_stream_and_history(isolated, monkeypatch):
    from service.service import app

    monkeypatch.setenv("SAAS_ONLY", "1")
    with patch("service.service.get_agent", return_value=AsyncMock()):
        client = TestClient(app)
        payload = {"message": "你好", "thread_id": "unbound-thread"}
        assert client.post("/saas-support/invoke", json=payload).status_code == 401
        assert client.post("/saas-support/stream", json=payload).status_code == 401
        assert client.post("/history", json={"thread_id": "unbound-thread"}).status_code == 401
        token, _owner = identity()
        assert (
            client.post(
                "/history",
                headers={"X-SaaS-Token": token},
                json={"thread_id": "unbound-thread"},
            ).status_code
            == 404
        )
