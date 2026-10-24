"""Explicit persistent preferences protected by a random capability, not user_id."""

import hashlib
import json
import secrets
from contextlib import closing

from technical_support import store


def connect():
    con = store._connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS support_profiles (
      owner TEXT PRIMARY KEY, preferences TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS support_threads (
      thread_id TEXT PRIMARY KEY, owner TEXT NOT NULL);
    """)
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
    return owner


def claim_thread(owner, thread_id, existing=False):
    with closing(connect()) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT owner FROM support_threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if row and row["owner"] != owner:
            raise ValueError("当前会话不属于此记忆空间")
        if not row:
            if existing:
                raise ValueError("旧会话未绑定记忆空间，请新建对话")
            con.execute("INSERT INTO support_threads VALUES (?,?)", (thread_id, owner))
        con.commit()


def owns_thread(owner, thread_id):
    with closing(connect()) as con:
        return bool(
            con.execute(
                "SELECT 1 FROM support_threads WHERE thread_id=? AND owner=?", (thread_id, owner)
            ).fetchone()
        )


def thread_owner(thread_id):
    with closing(connect()) as con:
        row = con.execute(
            "SELECT owner FROM support_threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
    return row["owner"] if row else None


def get_preferences(owner):
    with closing(connect()) as con:
        row = con.execute(
            "SELECT preferences FROM support_profiles WHERE owner=?", (owner,)
        ).fetchone()
    return json.loads(row["preferences"]) if row else {}


def save_preferences(owner, preferences):
    # Explicit, bounded fields: no credentials or live business facts.
    from technical_support.trace import redact

    allowed = {"answer_style", "environment_note"}
    if set(preferences) - allowed or any(
        not isinstance(v, str) or len(v) > 500 for v in preferences.values()
    ):
        raise ValueError("仅支持回答偏好和环境说明，每项最多500字")
    if any(redact(v) != v for v in preferences.values()):
        raise ValueError("请移除验证码、令牌或密钥后再保存")
    with closing(connect()) as con:
        con.execute(
            "UPDATE support_profiles SET preferences=? WHERE owner=?",
            (json.dumps(preferences, ensure_ascii=False), owner),
        )
        con.commit()
    return preferences
