"""One init_chat_model provider boundary for all seven projects."""
from __future__ import annotations

import re
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock, Semaphore
from time import monotonic
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel, Field

from .config import Generation, Settings
from .reliability import CircuitBreaker


@dataclass
class ModelBudget:
    """Reserve approximate tokens before each call; actual usage is telemetry only."""
    calls: int = 0
    reserved_tokens: int = 0
    started: float = field(default_factory=monotonic)
    lock: Lock = field(default_factory=Lock)

    def reserve(self, tokens):
        with self.lock:
            if monotonic() - self.started > 150:
                raise TimeoutError("Model workflow deadline reached")
            if self.calls >= 12 or self.reserved_tokens + tokens > 60000:
                raise ValueError("Model workflow call/token reservation budget reached")
            self.calls += 1
            self.reserved_tokens += tokens


_budget: ContextVar[ModelBudget | None] = ContextVar("portfolio_model_budget", default=None)


@contextmanager
def model_budget(existing: ModelBudget | None = None):
    token = _budget.set(existing or ModelBudget())
    try:
        yield
    finally:
        _budget.reset(token)


def redact(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return re.sub(r"\b(?:sk-|AIza)[A-Za-z0-9_-]{12,}\b", "[SECRET]", text)


def build_model(settings: Settings, generation: Generation | None = None):
    """Construct a provider client; no inference is issued here."""
    spec = generation or settings.generation()
    credential = settings.provider_key(spec.provider)
    if not credential:
        raise ValueError(f"No credential configured for {spec.provider}; use demo mode or configure it")
    kwargs = {"api_key": credential, "max_tokens": spec.max_tokens,
              "timeout": settings.llm_timeout_seconds, "max_retries": settings.llm_max_retries}
    for name in ("temperature", "top_p", "top_k"):
        if (value := getattr(spec, name)) is not None:
            kwargs[name] = value
    if spec.provider == "google_genai":
        kwargs["vertexai"] = False  # Explicit Developer API; do not infer ADC/Vertex from the host.
    return init_chat_model(model=spec.model, model_provider=spec.provider, **kwargs)


class RouteDecision(BaseModel):
    route: Literal["data", "rag", "ml", "tools", "memory"]
    operation: Literal["summary", "missing", "correlation", "chart", "train", "question", "calculate", "remember", "recall"]
    column: str | None = None
    target: str | None = None
    problem_type: Literal["regression", "classification"] | None = None
    reason: str = Field(max_length=300)


class GroundedAnswer(BaseModel):
    answer: str = Field(max_length=10000)
    citations: list[str] = Field(max_length=12)
    supported: bool


class QueryVariants(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


class ReviewScore(BaseModel):
    score: float = Field(ge=0, le=1)
    feedback: str = Field(max_length=1000)


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.breakers: dict[str, CircuitBreaker] = {}
        self.lock = Lock()
        self.slots = Semaphore(4)

    def structured(self, schema, system: str, user: str, generation: Generation | None = None):
        if self.settings.run_mode != "live":
            raise RuntimeError("Demo workflows must use explicit fixtures, not pretend to call a provider")
        messages = [SystemMessage(content=redact(system)), HumanMessage(content=redact(user))]
        if count_tokens_approximately(messages) > 9000 or len(user.encode("utf-8")) > 36000:
            raise ValueError("Model input exceeds the request budget; narrow the question or evidence")
        spec = generation or self.settings.generation()
        reservation = int(count_tokens_approximately(messages)) + spec.max_tokens
        if budget := _budget.get():
            budget.reserve(reservation)
        key = f"{spec.provider}:{spec.model}"
        with self.lock:
            breaker = self.breakers.setdefault(key, CircuitBreaker())
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Model concurrency limit reached")
        try:
            client = build_model(self.settings, spec)
            output = breaker.call(lambda: client.with_structured_output(schema, include_raw=True).invoke(messages))
            if output.get("parsing_error") or output.get("parsed") is None:
                raise ValueError("Provider output did not satisfy the required schema")
            raw = output.get("raw")
            logging.getLogger("portfolio.models").info({"provider": spec.provider, "model": spec.model,
                "usage": getattr(raw, "usage_metadata", None), "reserved_tokens": reservation})
            return output["parsed"]
        finally:
            self.slots.release()

    def text(self, system: str, user: str, generation: Generation | None = None) -> str:
        class TextResult(BaseModel):
            text: str = Field(max_length=12000)
        return self.structured(TextResult, system, user, generation).text
