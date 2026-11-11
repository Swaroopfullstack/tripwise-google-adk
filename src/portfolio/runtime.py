"""Application-owned resources shared by authenticated API/UI/CLI entrypoints."""
from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from threading import Lock, Semaphore
from weakref import WeakValueDictionary

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import Generation, Principal, Settings
from .data import infer_target, profile, read_csv
from .insight import AgentContext, build_insight
from .models import LLM
from .reliability import RateLimiter, TTLCache
from .retrieval import EvidenceDesk
from .review import build_review
from .storage import Store, digest


class Services:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.store = Store(self.settings.data_dir)
        self.connection = sqlite3.connect(self.settings.data_dir / "checkpoints.sqlite", check_same_thread=False, timeout=10)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()
        self.llm = LLM(self.settings)
        self.rag = EvidenceDesk(self.settings, self.store, self.llm)
        self.training_slots, self.indexing_slots = Semaphore(1), Semaphore(1)
        self.admission = RateLimiter(self.settings.requests_per_minute)
        self.cache = TTLCache()
        self._locks, self._locks_guard = WeakValueDictionary(), Lock()
        self.insight, self.review = build_insight(self), build_review(self)

    def close(self):
        self.rag.close()
        self.connection.close()

    @contextmanager
    def thread_lock(self, key):
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = Lock()
                self._locks[key] = lock
        if not lock.acquire(timeout=1):
            raise RuntimeError("This session already has a request in progress")
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def config(principal, session_id, kind="insight"):
        return {"configurable": {"thread_id": f"{kind}:{principal.tenant}:{principal.user}:{session_id}"},
                "recursion_limit": 20}

    def upload_csv(self, principal, session_id, name, content):
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        frame = read_csv(content, self.settings)
        record = self.store.save_upload(principal, session_id, name, "csv", content)
        return {**record, "profile": profile(frame), "target_inference": infer_target(frame)}

    def ingest(self, principal, session_id, name, content):
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        if not self.indexing_slots.acquire(blocking=False):
            raise RuntimeError("Another index update is in progress")
        try:
            return self.rag.ingest(principal, session_id, name, content)
        finally:
            self.indexing_slots.release()

    def chat(self, principal: Principal, session_id: str, query: str, dataset_id=None,
             generation: Generation | None = None, target=None, problem_type=None):
        self.store.session(principal, session_id)
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        if not 1 <= len(query.strip()) <= 2000:
            raise ValueError("Question must contain 1–2000 characters")
        spec = self.settings.validate_override(generation or self.settings.generation())
        with self.thread_lock((principal, session_id)):
            self.store.message(principal, session_id, "user", query)
            result = self.insight.invoke({"query": query, "dataset_id": dataset_id,
                                          "generation": spec.model_dump(), "target": target,
                                          "problem_type": problem_type,
                                          "messages": [HumanMessage(content=query)]},
                                         self.config(principal, session_id),
                                         context=AgentContext(principal, session_id))
            self.store.message(principal, session_id, "assistant", result["answer"])
        return {"session_id": session_id, "route": result["decision"], "answer": result["answer"],
                "payload": result["payload"], "mode": self.settings.run_mode,
                "model": spec.model if self.settings.run_mode == "live" else "deterministic demo (no inference)",
                "generation": spec.model_dump()}

    def propose(self, principal, session_id, content, request_key):
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        row = self.store.new_review(principal, session_id, content, request_key)
        config = self.config(principal, row["id"], "review")
        with self.thread_lock(("review", row["id"])):
            snapshot = self.review.get_state(config)
            if not snapshot.values:
                self.review.invoke({"review_id": row["id"]}, config, context=AgentContext(principal, session_id))
        return {key: row[key] for key in ("id", "status", "version", "artifact_id", "session_id", "owner")}

    def stream_chat(self, principal, session_id, query, dataset_id=None, generation=None, target=None, problem_type=None):
        """Yield bounded progress, then the public result; never expose raw checkpoint state."""
        self.store.session(principal, session_id)
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        if not 1 <= len(query.strip()) <= 2000:
            raise ValueError("Question must be 1-2000 characters")
        spec = self.settings.validate_override(generation or self.settings.generation())
        with self.thread_lock((principal, session_id)):
            self.store.message(principal, session_id, "user", query)
            config = self.config(principal, session_id)
            for update in self.insight.stream({"query": query, "dataset_id": dataset_id,
                    "generation": spec.model_dump(), "target": target, "problem_type": problem_type,
                    "messages": [HumanMessage(content=query)]}, config,
                    context=AgentContext(principal, session_id), stream_mode="updates"):
                yield {"event": "progress", "data": {"completed_nodes": list(update)}}
            result = self.insight.get_state(config).values
            self.store.message(principal, session_id, "assistant", result["answer"])
            yield {"event": "result", "data": {"session_id": session_id, "route": result["decision"],
                "answer": result["answer"], "payload": result["payload"], "mode": self.settings.run_mode}}

    def decide(self, principal, review_id, approved, version):
        with self.thread_lock(("review", review_id)):
            row = self.store.decide(principal, review_id, approved, version)
            owner = Principal(row["tenant"], row["owner"])
            config = self.config(owner, review_id, "review")
            snapshot = self.review.get_state(config)
            if snapshot.next:
                self.review.invoke(Command(resume={"status": row["status"]}), config,
                                   context=AgentContext(owner, row["session_id"]))
        return {key: row[key] for key in ("id", "status", "version", "artifact_id")}

    def ask_evidence(self, principal, session_id, question, strategy=None, generation=None, k=None, fetch_k=None):
        self.store.session(principal, session_id)
        spec = self.settings.validate_override(generation or self.settings.generation())
        history = self.store.history(principal, session_id, limit=8)
        index_version = digest([row["id"] for row in self.store.chunks(principal)])
        key = digest([principal.tenant, principal.user, session_id, history, index_version,
                      self.settings.embedding_backend, self.settings.embedding_model, "rag-prompt-v1", question,
                      strategy, k, fetch_k, spec.model_dump(), self.settings.run_mode])
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        result = self.rag.answer(principal, question, strategy, generation=spec, history=history, k=k, fetch_k=fetch_k)
        if result.get("supported"):
            self.cache.put(key, result)
        return {**result, "cached": False}
