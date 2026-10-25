"""Capability-protected profiles, durable memories and conversation metadata."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from contextlib import closing
from datetime import UTC, datetime

from technical_support import store


def _now() -> str:
    return datetime.now(UTC).isoformat()


def connect():
    con = store._connect()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS support_profiles (
          owner TEXT PRIMARY KEY,
          preferences TEXT NOT NULL DEFAULT '{}',
          auto_memory_enabled INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS support_threads (
          thread_id TEXT PRIMARY KEY,
          owner TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS support_memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          owner TEXT NOT NULL,
          memory_key TEXT NOT NULL,
          memory_value TEXT NOT NULL,
          category TEXT NOT NULL,
          source_thread_id TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(owner, memory_key)
        );
        CREATE INDEX IF NOT EXISTS idx_support_memories_owner_updated
          ON support_memories(owner, updated_at DESC);
        CREATE TABLE IF NOT EXISTS support_memory_turns (
          owner TEXT NOT NULL,
          thread_id TEXT NOT NULL,
          message_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          processed_at TEXT NOT NULL,
          PRIMARY KEY(owner, thread_id, message_hash)
        );
        CREATE TABLE IF NOT EXISTS support_conversations (
          thread_id TEXT PRIMARY KEY,
          owner TEXT NOT NULL,
          title TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          deleted_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_support_conversations_owner_updated
          ON support_conversations(owner, updated_at DESC);
        """
    )
    columns = {row["name"] for row in con.execute("PRAGMA table_info(support_profiles)")}
    if "auto_memory_enabled" not in columns:
        con.execute(
            "ALTER TABLE support_profiles ADD COLUMN auto_memory_enabled INTEGER NOT NULL DEFAULT 1"
        )
        con.commit()
    return con


def create_profile():
    token = secrets.token_urlsafe(32)
    owner = hashlib.sha256(token.encode()).hexdigest()
    with closing(connect()) as con:
        con.execute("INSERT INTO support_profiles(owner) VALUES (?)", (owner,))
        con.commit()
    return token


def authenticate(token):
    if not token or len(token) > 200:
        raise ValueError("无效的记忆访问凭证")
    owner = hashlib.sha256(token.encode()).hexdigest()
    with closing(connect()) as con:
        if not con.execute("SELECT 1 FROM support_profiles WHERE owner=?", (owner,)).fetchone():
            raise ValueError("无效的记忆访问凭证")
    _migrate_legacy_preferences(owner)
    return owner


def _migrate_legacy_preferences(owner: str) -> None:
    """Expose pre-upgrade manual preferences in the new memory list once per key."""
    with closing(connect()) as con:
        row = con.execute(
            "SELECT preferences FROM support_profiles WHERE owner=?", (owner,)
        ).fetchone()
        preferences = json.loads(row["preferences"]) if row else {}
        if not any(preferences.get(field) for field in ("answer_style", "environment_note")):
            return
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT preferences FROM support_profiles WHERE owner=?", (owner,)
        ).fetchone()
        preferences = json.loads(row["preferences"]) if row else {}
        now = _now()
        for key, field, category in (
            ("preference.answer_style", "answer_style", "preference"),
            ("environment.note", "environment_note", "environment"),
        ):
            value = preferences.get(field)
            if value:
                con.execute(
                    """INSERT INTO support_memories
                       (owner,memory_key,memory_value,category,source_thread_id,created_at,updated_at)
                       VALUES (?,?,?,?,NULL,?,?) ON CONFLICT(owner,memory_key) DO NOTHING""",
                    (owner, key, value, category, now, now),
                )
        con.commit()


def claim_thread(owner, thread_id, existing=False):
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT owner FROM support_threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if row and row["owner"] != owner:
            raise ValueError("当前会话不属于此记忆空间")
        deleted = con.execute(
            "SELECT deleted_at FROM support_conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if deleted and deleted["deleted_at"]:
            raise ValueError("该会话已删除")
        if not row:
            if existing:
                raise ValueError("旧会话未绑定记忆空间，请新建对话")
            con.execute("INSERT INTO support_threads VALUES (?,?)", (thread_id, owner))
        con.commit()


def owns_thread(owner, thread_id, *, include_deleted=False):
    with closing(connect()) as con:
        row = con.execute(
            """SELECT t.owner, c.deleted_at
               FROM support_threads t
               LEFT JOIN support_conversations c ON c.thread_id=t.thread_id
               WHERE t.thread_id=? AND t.owner=?""",
            (thread_id, owner),
        ).fetchone()
    return bool(row and (include_deleted or not row["deleted_at"]))


def thread_owner(thread_id):
    with closing(connect()) as con:
        row = con.execute(
            "SELECT owner FROM support_threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
    return row["owner"] if row else None


def conversation_deleted(thread_id: str) -> bool:
    with closing(connect()) as con:
        row = con.execute(
            "SELECT deleted_at FROM support_conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
    return bool(row and row["deleted_at"])


def get_preferences(owner):
    with closing(connect()) as con:
        row = con.execute(
            "SELECT preferences FROM support_profiles WHERE owner=?", (owner,)
        ).fetchone()
    return json.loads(row["preferences"]) if row else {}


def save_preferences(owner, preferences):
    """Keep the old explicit preference API for safe, lossless migration."""
    from technical_support.trace import redact

    allowed = {"answer_style", "environment_note"}
    if set(preferences) - allowed or any(
        not isinstance(v, str) or len(v) > 500 for v in preferences.values()
    ):
        raise ValueError("仅支持回答偏好和环境说明，每项最多500字")
    if any(redact(v) != v for v in preferences.values()):
        raise ValueError("请移除验证码、令牌或密钥后再保存")
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "UPDATE support_profiles SET preferences=? WHERE owner=?",
            (json.dumps(preferences, ensure_ascii=False), owner),
        )
        now = _now()
        legacy = {
            "preference.answer_style": ("preference", preferences.get("answer_style", "")),
            "environment.note": ("environment", preferences.get("environment_note", "")),
        }
        for key, (category, value) in legacy.items():
            if value:
                con.execute(
                    """INSERT INTO support_memories
                       (owner,memory_key,memory_value,category,source_thread_id,created_at,updated_at)
                       VALUES (?,?,?,?,NULL,?,?)
                       ON CONFLICT(owner,memory_key) DO UPDATE SET
                         memory_value=excluded.memory_value, category=excluded.category,
                         updated_at=excluded.updated_at""",
                    (owner, key, value, category, now, now),
                )
            else:
                con.execute(
                    "DELETE FROM support_memories WHERE owner=? AND memory_key=?", (owner, key)
                )
        con.commit()
    return preferences


def get_auto_memory_enabled(owner: str) -> bool:
    with closing(connect()) as con:
        row = con.execute(
            "SELECT auto_memory_enabled FROM support_profiles WHERE owner=?", (owner,)
        ).fetchone()
    return bool(row and row["auto_memory_enabled"])


def set_auto_memory_enabled(owner: str, enabled: bool) -> bool:
    with closing(connect()) as con:
        con.execute(
            "UPDATE support_profiles SET auto_memory_enabled=? WHERE owner=?",
            (int(enabled), owner),
        )
        con.commit()
    return enabled


def list_memories(owner: str, *, limit: int = 100) -> list[dict]:
    with closing(connect()) as con:
        rows = con.execute(
            """SELECT id,memory_key,memory_value,category,source_thread_id,created_at,updated_at
               FROM support_memories WHERE owner=? ORDER BY updated_at DESC LIMIT ?""",
            (owner, min(max(limit, 1), 200)),
        ).fetchall()
    return [dict(row) for row in rows]


def select_memories(owner: str, query: str, *, limit: int = 8, max_chars: int = 2400) -> list[dict]:
    """Select bounded relevant memories without another model or vector database."""
    rows = list_memories(owner, limit=200)
    query_terms = set(re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]{2}", query.lower()))

    def score(row: dict) -> tuple[int, str]:
        text = f"{row['memory_key']} {row['memory_value']} {row['category']}".lower()
        overlap = sum(term in text for term in query_terms)
        general = 2 if row["category"] in {"environment", "preference"} else 0
        return overlap * 4 + general, row["updated_at"]

    selected, total = [], 0
    for row in sorted(rows, key=score, reverse=True):
        size = len(row["memory_key"]) + len(row["memory_value"])
        if len(selected) >= limit or total + size > max_chars:
            break
        selected.append(row)
        total += size
    return selected


def claim_memory_turn(
    owner: str, thread_id: str, message: str, *, message_id: str | None = None
) -> tuple[bool, str]:
    # LangGraph keeps a stable HumanMessage ID across a retry or HITL resume.
    # Distinct turns may contain identical text and must remain independently eligible.
    identity = message_id or message
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            """SELECT status FROM support_memory_turns
               WHERE owner=? AND thread_id=? AND message_hash=?""",
            (owner, thread_id, digest),
        ).fetchone()
        if row:
            con.commit()
            return False, digest
        con.execute(
            "INSERT INTO support_memory_turns VALUES (?,?,?,?,?)",
            (owner, thread_id, digest, "processing", _now()),
        )
        con.commit()
    return True, digest


def finish_memory_turn(owner: str, thread_id: str, digest: str, status: str) -> None:
    with closing(connect()) as con:
        con.execute(
            """UPDATE support_memory_turns SET status=?,processed_at=?
               WHERE owner=? AND thread_id=? AND message_hash=?""",
            (status, _now(), owner, thread_id, digest),
        )
        con.commit()


def apply_memory_actions(owner: str, thread_id: str, actions: list[dict]) -> list[dict]:
    """Apply validated extractor actions atomically and idempotently."""
    now = _now()
    changed: list[dict] = []
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        for action in actions:
            operation = action["operation"]
            key = action.get("memory_key", "").strip()
            if operation == "noop":
                continue
            if operation == "delete":
                cursor = con.execute(
                    "DELETE FROM support_memories WHERE owner=? AND memory_key=?", (owner, key)
                )
                _sync_legacy_preference(con, owner, key, "")
                if cursor.rowcount:
                    changed.append({"operation": "delete", "memory_key": key})
                continue
            value = action["memory_value"].strip()
            category = action["category"]
            con.execute(
                """INSERT INTO support_memories
                   (owner,memory_key,memory_value,category,source_thread_id,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(owner,memory_key) DO UPDATE SET
                     memory_value=excluded.memory_value, category=excluded.category,
                     source_thread_id=excluded.source_thread_id, updated_at=excluded.updated_at""",
                (owner, key, value, category, thread_id, now, now),
            )
            _sync_legacy_preference(con, owner, key, value)
            changed.append({"operation": operation, "memory_key": key})
        con.commit()
    return changed


def _sync_legacy_preference(con, owner: str, key: str, value: str) -> None:
    field = {
        "preference.answer_style": "answer_style",
        "environment.note": "environment_note",
    }.get(key)
    if not field:
        return
    row = con.execute("SELECT preferences FROM support_profiles WHERE owner=?", (owner,)).fetchone()
    preferences = json.loads(row["preferences"]) if row else {}
    if value:
        preferences[field] = value
    else:
        preferences.pop(field, None)
    con.execute(
        "UPDATE support_profiles SET preferences=? WHERE owner=?",
        (json.dumps(preferences, ensure_ascii=False), owner),
    )


def delete_memory(owner: str, memory_id: int) -> bool:
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT memory_key FROM support_memories WHERE owner=? AND id=?", (owner, memory_id)
        ).fetchone()
        cursor = con.execute(
            "DELETE FROM support_memories WHERE owner=? AND id=?", (owner, memory_id)
        )
        if row:
            _sync_legacy_preference(con, owner, row["memory_key"], "")
        con.commit()
    return bool(cursor.rowcount)


def clear_memories(owner: str) -> int:
    with closing(connect()) as con:
        cursor = con.execute("DELETE FROM support_memories WHERE owner=?", (owner,))
        con.execute("UPDATE support_profiles SET preferences='{}' WHERE owner=?", (owner,))
        con.commit()
    return cursor.rowcount


def make_title(message: str) -> str:
    from technical_support.trace import redact

    title = str(redact(message)).replace("[hidden]", "")
    title = re.sub(r"APP_DEMO_\d{3}", "", title, flags=re.I)
    title = re.sub(r"^(请问|请帮我|帮我|麻烦|为什么我的|为什么)\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip(" ，,。！？?!：:")
    return (title[:30] + "…") if len(title) > 31 else (title or "新会话")


def ensure_conversation(owner: str, thread_id: str, first_message: str) -> dict:
    now = _now()
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM support_conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if row and row["owner"] != owner:
            raise ValueError("当前会话不属于此记忆空间")
        if row and row["deleted_at"]:
            raise ValueError("该会话已删除")
        if not row:
            con.execute(
                "INSERT INTO support_conversations VALUES (?,?,?,?,?,NULL)",
                (thread_id, owner, make_title(first_message), now, now),
            )
        con.commit()
    return get_conversation(owner, thread_id)


def touch_conversation(owner: str, thread_id: str) -> None:
    with closing(connect()) as con:
        con.execute(
            """UPDATE support_conversations SET updated_at=?
               WHERE owner=? AND thread_id=? AND deleted_at IS NULL""",
            (_now(), owner, thread_id),
        )
        con.commit()


def get_conversation(owner: str, thread_id: str, *, include_deleted=False) -> dict:
    sql = "SELECT * FROM support_conversations WHERE owner=? AND thread_id=?"
    params = (owner, thread_id)
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    with closing(connect()) as con:
        row = con.execute(sql, params).fetchone()
    if not row:
        raise ValueError("此记忆空间中没有该会话")
    return dict(row)


def list_conversations(
    owner: str, *, search: str = "", limit: int = 50, offset: int = 0
) -> list[dict]:
    sql = """SELECT thread_id,title,created_at,updated_at
             FROM support_conversations WHERE owner=? AND deleted_at IS NULL"""
    params: list = [owner]
    if search.strip():
        sql += " AND title LIKE ? ESCAPE '\\'"
        escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped}%")
    sql += " ORDER BY updated_at DESC, thread_id LIMIT ? OFFSET ?"
    params.extend([min(max(limit, 1), 100), max(offset, 0)])
    with closing(connect()) as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def rename_conversation(owner: str, thread_id: str, title: str) -> dict:
    clean = re.sub(r"\s+", " ", title).strip()
    if not clean or len(clean) > 80:
        raise ValueError("标题长度必须为1到80个字符")
    from technical_support.trace import redact

    if redact(clean) != clean:
        raise ValueError("标题不能包含验证码、令牌或密钥")
    with closing(connect()) as con:
        cursor = con.execute(
            """UPDATE support_conversations SET title=?,updated_at=?
               WHERE owner=? AND thread_id=? AND deleted_at IS NULL""",
            (clean, _now(), owner, thread_id),
        )
        con.commit()
    if not cursor.rowcount:
        raise ValueError("此记忆空间中没有该会话")
    return get_conversation(owner, thread_id)


def delete_conversation(owner: str, thread_id: str) -> None:
    """Soft-delete metadata; LangGraph checkpoints remain retained but inaccessible."""
    now = _now()
    with closing(connect()) as con:
        cursor = con.execute(
            """UPDATE support_conversations SET deleted_at=?,updated_at=?
               WHERE owner=? AND thread_id=? AND deleted_at IS NULL""",
            (now, now, owner, thread_id),
        )
        con.commit()
    if not cursor.rowcount:
        raise ValueError("此记忆空间中没有该会话")
