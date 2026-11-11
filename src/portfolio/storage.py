"""Owned, durable SQLite records for the supported single-instance deployment."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .config import Principal


class NotFound(LookupError):
    pass


class Conflict(ValueError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "portfolio.sqlite"
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
                    role TEXT NOT NULL, content TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS preferences (
                    tenant TEXT, owner TEXT, key TEXT, value TEXT NOT NULL,
                    PRIMARY KEY(tenant,owner,key));
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    session_id TEXT REFERENCES sessions(id), name TEXT NOT NULL, kind TEXT NOT NULL,
                    content BLOB NOT NULL, sha256 TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    document_id TEXT NOT NULL, parent_id TEXT NOT NULL,
                    text TEXT NOT NULL, parent_text TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    session_id TEXT NOT NULL, name TEXT NOT NULL, media_type TEXT NOT NULL,
                    content BLOB NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL, session_id TEXT NOT NULL,
                    request_key TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', version INTEGER NOT NULL DEFAULT 1,
                    artifact_id TEXT, reviewer TEXT, UNIQUE(tenant,owner,request_key));
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, actor TEXT NOT NULL,
                    action TEXT NOT NULL, object_id TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS chunks_scope ON chunks(tenant,owner);
                CREATE INDEX IF NOT EXISTS messages_thread ON messages(session_id,id);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def new_session(self, principal: Principal, title="New analysis") -> dict:
        session = {"id": uuid4().hex, "tenant": principal.tenant, "owner": principal.user,
                   "title": str(title)[:100]}
        with self.connection() as db:
            db.execute("INSERT INTO sessions(id,tenant,owner,title) VALUES(:id,:tenant,:owner,:title)", session)
        return session

    def session(self, principal: Principal, session_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=? AND tenant=? AND owner=?",
                             (session_id, principal.tenant, principal.user)).fetchone()
        if row is None:
            raise NotFound("Session not found")
        return dict(row)

    def sessions(self, principal: Principal) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM sessions WHERE tenant=? AND owner=? ORDER BY created DESC,id DESC LIMIT 100",
                (principal.tenant, principal.user))]

    def history(self, principal: Principal, session_id: str, limit=40) -> list[dict]:
        self.session(principal, session_id)
        with self.connection() as db:
            rows = db.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                              (session_id, min(limit, 100))).fetchall()
        return [dict(row) for row in reversed(rows)]

    def message(self, principal: Principal, session_id: str, role: str, content: str):
        self.session(principal, session_id)
        if role not in {"user", "assistant"} or len(content) > 30000:
            raise ValueError("Invalid message")
        with self.connection() as db:
            db.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)", (session_id, role, content))

    def preference(self, principal: Principal, key: str, value: str | None = None):
        if key not in {"language", "verbosity", "chart_style"}:
            raise ValueError("Only language, verbosity and chart_style preferences are permitted")
        with self.connection() as db:
            if value is not None:
                if not 0 < len(value) <= 120:
                    raise ValueError("Preference must contain 1–120 characters")
                db.execute("INSERT INTO preferences VALUES(?,?,?,?) ON CONFLICT(tenant,owner,key) "
                           "DO UPDATE SET value=excluded.value", (principal.tenant, principal.user, key, value))
            row = db.execute("SELECT value FROM preferences WHERE tenant=? AND owner=? AND key=?",
                             (principal.tenant, principal.user, key)).fetchone()
        return row["value"] if row else None

    def preferences(self, principal: Principal) -> dict:
        with self.connection() as db:
            return {row["key"]: row["value"] for row in db.execute(
                "SELECT key,value FROM preferences WHERE tenant=? AND owner=?", (principal.tenant, principal.user))}

    def save_upload(self, principal: Principal, session_id: str, name: str, kind: str, content: bytes) -> dict:
        self.session(principal, session_id)
        record = {"id": uuid4().hex, "tenant": principal.tenant, "owner": principal.user,
                  "session_id": session_id, "name": Path(name).name[:120], "kind": kind,
                  "content": content, "sha256": hashlib.sha256(content).hexdigest()}
        with self.connection() as db:
            db.execute("INSERT INTO uploads(id,tenant,owner,session_id,name,kind,content,sha256) "
                       "VALUES(:id,:tenant,:owner,:session_id,:name,:kind,:content,:sha256)", record)
        return {key: value for key, value in record.items() if key != "content"}

    def upload(self, principal: Principal, upload_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=? AND tenant=? AND owner=?",
                             (upload_id, principal.tenant, principal.user)).fetchone()
        if row is None:
            raise NotFound("Upload not found")
        return dict(row)

    def uploads(self, principal: Principal, session_id: str) -> list[dict]:
        self.session(principal, session_id)
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,name,kind,sha256 FROM uploads WHERE session_id=? ORDER BY created DESC,id DESC", (session_id,))]

    def add_chunks(self, principal: Principal, chunks: list[dict]):
        with self.connection() as db:
            for chunk in chunks:
                db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                           (chunk["id"], principal.tenant, principal.user, chunk["document_id"],
                            chunk["parent_id"], chunk["text"], chunk["parent_text"], json.dumps(chunk["metadata"])))

    def chunks(self, principal: Principal) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM chunks WHERE tenant=? AND owner=? ORDER BY id LIMIT 5000",
                              (principal.tenant, principal.user)).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]

    def artifact(self, principal: Principal, artifact_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=? AND tenant=? AND owner=?",
                             (artifact_id, principal.tenant, principal.user)).fetchone()
        if row is None:
            raise NotFound("Artifact not found")
        return dict(row)

    def save_artifact(self, principal, session_id, name, media_type, content: bytes) -> str:
        self.session(principal, session_id)
        artifact_id = uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO artifacts(id,tenant,owner,session_id,name,media_type,content) VALUES(?,?,?,?,?,?,?)",
                       (artifact_id, principal.tenant, principal.user, session_id, Path(name).name,
                        media_type, content))
        return artifact_id

    def new_review(self, principal, session_id, content: str, request_key: str) -> dict:
        self.session(principal, session_id)
        if not 1 <= len(content) <= 20000 or not re_key(request_key):
            raise ValueError("Invalid review content or idempotency key")
        fingerprint = digest({"session_id": session_id, "content": content})
        review_id = uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM reviews WHERE tenant=? AND owner=? AND request_key=?",
                             (principal.tenant, principal.user, request_key)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise Conflict("Idempotency key already belongs to another proposal")
                return dict(old)
            db.execute("INSERT INTO reviews(id,tenant,owner,session_id,request_key,fingerprint,payload) "
                       "VALUES(?,?,?,?,?,?,?)", (review_id, principal.tenant, principal.user, session_id,
                                                request_key, fingerprint, json.dumps({"content": content})))
            db.execute("INSERT INTO audit(tenant,actor,action,object_id) VALUES(?,?,?,?)",
                       (principal.tenant, principal.user, "review_proposed", review_id))
        return self.review(principal, review_id)

    def review(self, principal: Principal, review_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM reviews WHERE id=? AND tenant=?", (review_id, principal.tenant)).fetchone()
        if row is None or (row["owner"] != principal.user and principal.role not in {"reviewer", "admin"}):
            raise NotFound("Review not found")
        return dict(row)

    def pending_reviews(self, principal) -> list[dict]:
        if principal.role not in {"reviewer", "admin"}:
            raise PermissionError("Reviewer role required")
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM reviews WHERE tenant=? AND status='pending' ORDER BY id LIMIT 100", (principal.tenant,))]

    def decide(self, principal, review_id: str, approved: bool, version: int) -> dict:
        if principal.role not in {"reviewer", "admin"}:
            raise PermissionError("Reviewer role required")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM reviews WHERE id=? AND tenant=?", (review_id, principal.tenant)).fetchone()
            if row is None:
                raise NotFound("Review not found")
            desired = "approved" if approved else "rejected"
            if row["status"] != "pending":
                if row["status"] != desired:
                    raise Conflict("Review already has a different terminal decision")
                return dict(row)
            if row["version"] != version:
                raise Conflict("Stale review version")
            artifact_id = None
            if approved:
                artifact_id = uuid4().hex
                db.execute("INSERT INTO artifacts(id,tenant,owner,session_id,name,media_type,content) VALUES(?,?,?,?,?,?,?)",
                           (artifact_id, row["tenant"], row["owner"], row["session_id"], "approved-report.txt",
                            "text/plain", json.loads(row["payload"])["content"].encode()))
            db.execute("UPDATE reviews SET status=?,version=version+1,artifact_id=?,reviewer=? WHERE id=?",
                       (desired, artifact_id, principal.user, review_id))
            db.execute("INSERT INTO audit(tenant,actor,action,object_id) VALUES(?,?,?,?)",
                       (principal.tenant, principal.user, desired, review_id))
        return self.review(principal, review_id)


def re_key(key: str) -> bool:
    return isinstance(key, str) and 1 <= len(key) <= 100 and key.isascii()
