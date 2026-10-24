"""Capability-protected local memory and trace inspection API."""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

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


@router.post("/workspace")
def create_workspace():
    return {"token": profile.create_profile()}


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
def clear_memory(x_saas_token: Annotated[str | None, Header()] = None):
    return profile.save_preferences(owner(x_saas_token), {})


@router.get("/traces/{thread_id}")
def traces(thread_id: str, x_saas_token: Annotated[str | None, Header()] = None):
    identity = owner(x_saas_token)
    if not profile.owns_thread(identity, thread_id):
        raise HTTPException(404, "此记忆空间中没有该会话")
    return {"events": read_trace(thread_id), "limit": 1000}
