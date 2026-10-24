"""Streamlit controls for explicit memory and operational traces."""

import uuid

import httpx
import streamlit as st


def panel(client):
    def request(method, path, **kwargs):
        response = httpx.request(
            method, client.base_url + path, headers=client._headers, timeout=15, **kwargs
        )
        response.raise_for_status()
        return response.json()

    with st.expander("记忆空间", expanded=False):
        st.caption("保存的偏好可跨会话使用；应用权限和配额会重新查询。访问凭证不放入网页链接。")
        if not st.session_state.get("support_token"):
            if st.button("创建我的记忆空间"):
                try:
                    st.session_state.support_token = request("POST", "/saas/workspace")["token"]
                    st.session_state.thread_id = uuid.uuid4().hex
                    st.session_state.messages = []
                    st.query_params.pop("thread_id", None)
                    st.rerun()
                except httpx.HTTPError:
                    st.error("创建失败，请检查服务。")
        restored = st.text_input("恢复访问凭证", type="password", key="restore_support_token")
        if st.button("恢复记忆空间") and restored:
            previous = getattr(client, "profile_token", None)
            client.profile_token = restored
            try:
                request("GET", "/saas/memory")
                st.session_state.support_token = restored
                st.session_state.thread_id = uuid.uuid4().hex
                st.session_state.messages = []
                st.rerun()
            except httpx.HTTPError:
                client.profile_token = previous
                st.error("凭证无效。")
        if st.session_state.get("support_token"):
            client.profile_token = st.session_state.support_token
            st.download_button(
                "保存访问凭证以便重启后恢复",
                st.session_state.support_token,
                file_name="support-memory-token.txt",
            )
            try:
                preferences = request("GET", "/saas/memory")
                with st.form("memory_preferences"):
                    style = st.text_input(
                        "回答偏好", value=preferences.get("answer_style", ""), max_chars=500
                    )
                    environment = st.text_area(
                        "环境说明（不要填写验证码或密钥）",
                        value=preferences.get("environment_note", ""),
                        max_chars=500,
                    )
                    if st.form_submit_button("保存到长期记忆"):
                        request(
                            "PUT",
                            "/saas/memory",
                            json={"answer_style": style, "environment_note": environment},
                        )
                        st.success("已保存，下次提问生效。")
                if st.button("清空长期记忆"):
                    request("DELETE", "/saas/memory")
                    st.rerun()
                st.json(preferences)
            except httpx.HTTPError:
                st.error("记忆读取或保存失败，请检查凭证及输入中是否含有敏感信息。")
    with st.expander("执行记录 / Trace", expanded=False):
        st.caption("创建记忆空间后，新会话的重写、任务、检索排名、耗时和错误可在这里查看。")
        if st.button("刷新本会话执行记录"):
            try:
                events = request("GET", "/saas/traces/" + st.session_state.thread_id)["events"]
                st.session_state.support_events = events
                st.session_state.support_events_thread = st.session_state.thread_id
            except httpx.HTTPError:
                st.info("当前会话尚无可读取的记录，请先创建记忆空间并发起新对话。")
        if st.session_state.get("support_events_thread") == st.session_state.thread_id:
            events = st.session_state.get("support_events", [])
            st.dataframe(
                [
                    {
                        k: event.get(k)
                        for k in (
                            "at",
                            "event",
                            "status",
                            "duration_ms",
                            "trace_id",
                            "span_id",
                            "parent_span_id",
                        )
                    }
                    for event in events
                ],
                hide_index=True,
            )
            st.json(events)
