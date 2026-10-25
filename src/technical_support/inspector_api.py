"""Capability-protected memory, conversation and trace APIs."""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from core import settings
from technical_support import profile
from technical_support.trace import read_trace

router = APIRouter(prefix="/saas")


def owner(token):
    try:
        return profile.authenticate(token)
    except ValueError as error:
        raise HTTPException(401, str(error)) from error


class Preferences(BaseModel):
    answer_style: str = Field(default="", max_length=500)
    environment_note: str = Field(default="", max_length=500)


class MemorySetting(BaseModel):
    auto_memory_enabled: bool


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def _extractor_available() -> bool:
    mode = settings.CUSTOMER_SUPPORT_MODE.lower()
    configured = bool(
        settings.DASHSCOPE_API_KEY and settings.DASHSCOPE_BASE_URL and settings.DASHSCOPE_MODEL
    )
    return mode == "qwen" or (mode == "auto" and configured)


@router.post("/workspace")
def create_workspace():
    return {"token": profile.create_profile()}


# Backward-compatible explicit preferences. Existing values are migrated into memories.
@router.get("/memory")
def memory(x_saas_token: Annotated[str | None, Header()] = None):
    return profile.get_preferences(owner(x_saas_token))


@router.put("/memory")
def save_memory(value: Preferences, x_saas_token: Annotated[str | None, Header()] = None):
    try:
        return profile.save_preferences(owner(x_saas_token), value.model_dump())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.delete("/memory")
def clear_legacy_memory(x_saas_token: Annotated[str | None, Header()] = None):
    return profile.save_preferences(owner(x_saas_token), {})


@router.get("/memories")
def memories(
    limit: int = Query(default=100, ge=1, le=200),
    x_saas_token: Annotated[str | None, Header()] = None,
):
    identity = owner(x_saas_token)
    return {
        "items": profile.list_memories(identity, limit=limit),
        "auto_memory_enabled": profile.get_auto_memory_enabled(identity),
        "extractor_available": _extractor_available(),
    }


@router.patch("/memory-settings")
def update_memory_settings(
    value: MemorySetting, x_saas_token: Annotated[str | None, Header()] = None
):
    identity = owner(x_saas_token)
    return {
        "auto_memory_enabled": profile.set_auto_memory_enabled(identity, value.auto_memory_enabled),
        "extractor_available": _extractor_available(),
    }


@router.delete("/memories/{memory_id}", status_code=204)
def remove_memory(memory_id: int, x_saas_token: Annotated[str | None, Header()] = None):
    if not profile.delete_memory(owner(x_saas_token), memory_id):
        raise HTTPException(404, "未找到该记忆")
    return Response(status_code=204)


@router.delete("/memories")
def remove_all_memories(x_saas_token: Annotated[str | None, Header()] = None):
    return {"deleted": profile.clear_memories(owner(x_saas_token))}


@router.get("/conversations")
def conversations(
    search: str = Query(default="", max_length=100),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    x_saas_token: Annotated[str | None, Header()] = None,
):
    items = profile.list_conversations(
        owner(x_saas_token), search=search, limit=limit, offset=offset
    )
    return {"items": items, "limit": limit, "offset": offset}


@router.patch("/conversations/{thread_id}")
def rename_conversation(
    thread_id: str,
    value: ConversationRename,
    x_saas_token: Annotated[str | None, Header()] = None,
):
    try:
        return profile.rename_conversation(owner(x_saas_token), thread_id, value.title)
    except ValueError as error:
        raise HTTPException(404 if "没有" in str(error) else 422, str(error)) from error


@router.delete("/conversations/{thread_id}", status_code=204)
def delete_conversation(thread_id: str, x_saas_token: Annotated[str | None, Header()] = None):
    try:
        profile.delete_conversation(owner(x_saas_token), thread_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    return Response(status_code=204)


@router.get("/traces/{thread_id}")
def traces(thread_id: str, x_saas_token: Annotated[str | None, Header()] = None):
    identity = owner(x_saas_token)
    if not profile.owns_thread(identity, thread_id):
        raise HTTPException(404, "此记忆空间中没有该会话")
    return {"events": read_trace(thread_id), "limit": 1000}
