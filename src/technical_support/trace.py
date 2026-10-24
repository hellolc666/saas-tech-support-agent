"""Local structured spans with bounded, redacted operational evidence."""

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langgraph.config import get_config

_lock = threading.Lock()
_parent = ContextVar("support_span", default=None)
FIELDS = {
    "tool",
    "status",
    "duration_ms",
    "error_code",
    "documents",
    "intent",
    "trace_id",
    "span_id",
    "parent_span_id",
    "thread_id",
    "query",
    "result",
    "ranking",
    "usage",
    "tasks",
    "summary",
    "decision",
    "stage",
}


def redact(value):
    if isinstance(value, dict):
        return {
            k: (
                "[hidden]"
                if isinstance(v, str)
                and any(
                    t in k.lower()
                    for t in ("token", "secret", "password", "access_code", "api_key")
                )
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value[:100]]
    if isinstance(value, str):
        value = re.sub(
            r"DEMO-(?:ALPHA|BETA)-\d{4}|sk-[\w-]+|Bearer\s+[\w.\-]+", "[hidden]", value, flags=re.I
        )
        value = re.sub(
            r"(?i)(验证码|密码|密钥|token|api[_ -]?key|access[_ -]?code)\s*[:：=为是]?\s*[^\s，,。;；]+",
            r"\1[hidden]",
            value,
        )
        return value[:8000]
    return value


def trace_path():
    return Path(os.getenv("SAAS_TRACE_PATH", "data/logs/saas_trace.jsonl"))


def emit(event, **fields):
    try:
        try:
            config = get_config().get("configurable", {})
        except RuntimeError:
            config = {}
        record = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "thread_id": config.get("thread_id"),
            "trace_id": config.get("trace_id"),
            "span_id": uuid4().hex,
            "parent_span_id": _parent.get() or config.get("trace_id"),
        }
        record.update(redact({k: v for k, v in fields.items() if k in FIELDS}))
        path = trace_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).warning("Trace persistence unavailable")


@contextmanager
def span(event, **fields):
    parent = _parent.get()
    sid = uuid4().hex
    token = _parent.set(sid)
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException as error:
        status = "interrupted" if "Interrupt" in type(error).__name__ else "error"
        raise
    finally:
        _parent.reset(token)
        emit(
            event,
            span_id=sid,
            **({"parent_span_id": parent} if parent else {}),
            status=status,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            **fields,
        )


def read_trace(thread_id):
    path = trace_path()
    if not path.exists():
        return []
    # Bounded tail; never return other conversations.
    with path.open("rb") as file:
        file.seek(0, 2)
        start = max(0, file.tell() - 8_000_000)
        file.seek(start)
        if start:
            file.readline()
        lines = file.read().decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        try:
            record = json.loads(line)
            if record.get("thread_id") == thread_id:
                rows.append(record)
        except ValueError:
            continue
    return rows[-1000:]
