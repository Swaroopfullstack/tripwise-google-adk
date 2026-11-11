"""Actual A2A discovery, durable task storage and an allowlisted PDF agent."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.tasks import DatabaseTaskStore
from a2a.types import Message, Part, SendMessageRequest
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.genai import types
from google.protobuf.json_format import MessageToDict
from pydantic import Field
from sqlalchemy.ext.asyncio import create_async_engine

from .adk_project import session_service
from .config import Principal, Settings
from .protocol import BearerBoundary
from .public_data import public_file
from .runtime import Services


class PublicEvidenceAgent(BaseAgent):
    services: Services = Field(exclude=True)

    async def _run_async_impl(self, ctx):
        question = " ".join(part.text or "" for part in ctx.user_content.parts or [])
        if not 1 <= len(question) <= 1000:
            raise ValueError("Public evidence question must be 1-1000 characters")
        def answer():
            principal = Principal("public", "a2a_catalog")
            if not self.services.store.chunks(principal):
                session = self.services.store.new_session(principal, "Public A2A catalog")["id"]
                path = public_file("nist_ai_rmf")
                self.services.ingest(principal, session, path.name, path.read_bytes())
            return self.services.rag.answer(principal, question)
        result = await asyncio.to_thread(answer)
        import json
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=json.dumps(result))]))


def create_a2a_app(settings=None, host="127.0.0.1", port=8020):
    settings = settings or Settings()
    service = Services(settings)
    sessions = session_service(settings)
    agent = PublicEvidenceAgent(name="public_evidence", description="Read-only Q&A from NIST AI RMF 1.0 with page citations.", services=service)
    runner = Runner(app_name="public_evidence", agent=agent, session_service=sessions)
    engine = create_async_engine("sqlite+aiosqlite:///" + (settings.data_dir / "a2a-tasks.sqlite").as_posix())
    tasks = DatabaseTaskStore(engine)
    @asynccontextmanager
    async def lifespan(app):
        await tasks.initialize()
        try:
            yield
        finally:
            await runner.close()
            await sessions.close()
            await engine.dispose()
            service.close()
    app = to_a2a(agent, runner=runner, task_store=tasks, host=host, port=port, lifespan=lifespan)
    # One credential represents one public-catalog operator. This service exposes no tenant uploads.
    return BearerBoundary(app, settings.mcp_http_token.get_secret_value())


async def a2a_demo(url="http://127.0.0.1:8020", question="What are Govern, Map, Measure and Manage?", token=None):
    parsed = urlparse(url)
    # Card URLs are untrusted. This sample permits only a trusted local service.
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http":
        raise ValueError("This example client only accepts the local service")
    token = token or Settings().mcp_http_token.get_secret_value()
    async with httpx.AsyncClient(headers={"Authorization": "Bearer " + token}, timeout=90) as http:
        card = await A2ACardResolver(http, url).get_agent_card()
        for interface in card.supported_interfaces:
            if interface.url.rstrip("/") != url.rstrip("/"):
                raise ValueError("Agent card advertises an unexpected endpoint")
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
        try:
            message = Message(message_id=uuid4().hex, role="ROLE_USER", parts=[Part(text=question)])
            events = [MessageToDict(event) async for event in client.send_message(SendMessageRequest(message=message))]
            return {"protocol": "A2A JSON-RPC", "card_name": card.name, "events": events}
        finally:
            await client.close()


def run_a2a():
    import uvicorn
    host, port = os.getenv("A2A_HOST", "127.0.0.1"), int(os.getenv("A2A_PORT", "8020"))
    uvicorn.run(create_a2a_app(host=host, port=port), host=host, port=port)
