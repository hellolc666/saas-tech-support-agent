"""SaaS identity, conversation navigation, automatic memory and trace controls."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import streamlit as st

from client import AgentClientError
from client.browser_identity import persist_browser_token

COOKIE_NAME = "saas_support_token"


def _reset_chat_state(thread_id: str | None = None, messages=None) -> None:
    st.session_state.thread_id = thread_id or uuid.uuid4().hex
    st.session_state.messages = list(messages or [])
    for key in (
        "last_audio",
        "last_message",
        "last_feedback",
        "support_events",
        "support_events_thread",
    ):
        st.session_state.pop(key, None)


def initialize_identity(client) -> str:
    """Create or restore a stable local-demo capability before loading history."""
    token = st.session_state.get("support_token")
    if not token:
        try:
            token = st.context.cookies.get(COOKIE_NAME)
        except Exception:
            token = None
    if token:
        client.profile_token = token
        try:
            client.get_memories()
        except AgentClientError:
            token = None
    if not token:
        token = client.create_workspace()
        client.profile_token = token
    st.session_state.support_token = token
    client.profile_token = token
    persist_browser_token(token)
    return token


def _group_label(value: str) -> str:
    local_tz = ZoneInfo("Asia/Shanghai")
    updated = datetime.fromisoformat(value).astimezone(local_tz).date()
    today = datetime.now(UTC).astimezone(local_tz).date()
    if updated == today:
        return "今天"
    if (today - updated).days == 1:
        return "昨天"
    return "更早"


def _switch_conversation(client, thread_id: str) -> None:
    history = client.get_history(thread_id=thread_id)
    _reset_chat_state(thread_id, history.messages)
    st.query_params["thread_id"] = thread_id


def conversation_sidebar(client) -> None:
    if st.button("＋ 新建会话", use_container_width=True, type="primary"):
        _reset_chat_state()
        st.query_params.pop("thread_id", None)
        st.rerun()

    search = st.text_input("搜索历史会话", key="conversation_search", placeholder="输入标题关键词")
    try:
        conversations = client.list_conversations(search=search, limit=50)
    except AgentClientError:
        st.warning("历史会话暂时无法读取。")
        return
    if not conversations:
        st.caption("暂无历史会话，发送第一条消息后会自动保存。")
        return

    grouped = {"今天": [], "昨天": [], "更早": []}
    for conversation in conversations:
        grouped[_group_label(conversation["updated_at"])].append(conversation)

    for group, rows in grouped.items():
        if not rows:
            continue
        st.caption(group)
        for conversation in rows:
            thread_id = conversation["thread_id"]
            title = conversation["title"]
            selected = thread_id == st.session_state.get("thread_id")
            left, right = st.columns([8, 2], gap="small")
            with left:
                label = ("● " if selected else "") + (
                    title[:24] + "…" if len(title) > 25 else title
                )
                if st.button(label, key=f"open-{thread_id}", use_container_width=True):
                    try:
                        _switch_conversation(client, thread_id)
                        st.rerun()
                    except AgentClientError:
                        st.error("无法打开该会话。")
            with right, st.popover("⋯", use_container_width=True):
                new_title = st.text_input(
                    "重命名", value=title, max_chars=80, key=f"title-{thread_id}"
                )
                if st.button("保存标题", key=f"rename-{thread_id}"):
                    try:
                        client.rename_conversation(thread_id, new_title)
                        st.rerun()
                    except AgentClientError:
                        st.error("标题保存失败，请移除敏感信息后重试。")
                if st.button("删除会话", key=f"delete-{thread_id}", type="secondary"):
                    try:
                        client.delete_conversation(thread_id)
                        if selected:
                            _reset_chat_state()
                            st.query_params.pop("thread_id", None)
                        st.rerun()
                    except AgentClientError:
                        st.error("会话删除失败。")


def panel(client):
    with st.expander("自动长期记忆", expanded=False):
        st.caption(
            "系统只自动保存稳定的开发环境、回答偏好和长期习惯；动态配额、故障、验证码和密钥不会保存。"
        )
        try:
            data = client.get_memories()
            enabled = st.toggle(
                "启用自动记忆",
                value=bool(data["auto_memory_enabled"]),
                key="auto_memory_enabled",
            )
            if enabled != bool(data["auto_memory_enabled"]):
                client.set_auto_memory(enabled)
                st.rerun()
            if not data.get("extractor_available"):
                st.info("当前为Mock模式：可以查看已有记忆，但不会假装执行Qwen自动提取。")
            items = data.get("items", [])
            if not items:
                st.caption("还没有自动记忆。正常聊天中表达长期偏好后会显示在这里。")
            for item in items:
                with st.container(border=True):
                    st.markdown(f"**{item['memory_key']}** · {item['category']}")
                    st.write(item["memory_value"])
                    st.caption("更新于 " + item["updated_at"])
                    if st.button("删除此记忆", key=f"memory-{item['id']}"):
                        client.delete_memory(item["id"])
                        st.rerun()
            if items and st.button("清空全部长期记忆"):
                client.clear_memories()
                st.rerun()
        except AgentClientError:
            st.error("长期记忆读取失败。")

        st.download_button(
            "导出身份恢复凭证",
            st.session_state.support_token,
            file_name="support-memory-token.txt",
            help="清理浏览器数据或更换设备后可用它恢复身份。",
        )
        restored = st.text_input("恢复身份凭证", type="password", key="restore_support_token")
        if st.button("恢复这个身份") and restored:
            previous = client.profile_token
            client.profile_token = restored
            try:
                client.get_memories()
                st.session_state.support_token = restored
                persist_browser_token(restored)
                _reset_chat_state()
                st.query_params.pop("thread_id", None)
                st.rerun()
            except AgentClientError:
                client.profile_token = previous
                st.error("凭证无效。")

        st.caption(
            "本地演示凭证保存在浏览器可读Cookie/localStorage中；生产环境应改用正式登录和HttpOnly Secure Cookie。"
        )

    with st.expander("执行记录 / Trace", expanded=False):
        st.caption("查看本会话的查询重写、任务、检索排名、耗时和错误。")
        if st.button("刷新本会话执行记录"):
            try:
                result = client._saas_request("GET", "/traces/" + st.session_state.thread_id)
                st.session_state.support_events = result["events"]
                st.session_state.support_events_thread = st.session_state.thread_id
            except AgentClientError:
                st.info("当前会话尚无可读取的记录。")
        if st.session_state.get("support_events_thread") == st.session_state.thread_id:
            events = st.session_state.get("support_events", [])
            st.dataframe(
                [
                    {
                        key: event.get(key)
                        for key in (
                            "at",
                            "event",
                            "status",
                            "duration_ms",
                            "trace_id",
                        )
                    }
                    for event in events
                ],
                hide_index=True,
            )
            with st.expander("完整事件"):
                st.json(events)
