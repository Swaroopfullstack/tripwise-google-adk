import asyncio
import json
import subprocess
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import SecretStr
import pytest

from portfolio.config import ROOT, Principal
from portfolio.data import read_csv
from portfolio.ml import train_models
from portfolio.models import ModelBudget, build_model
from portfolio.protocol import BearerBoundary, create_mcp_server, decode_tool, mcp_demo
from portfolio.public_data import public_file
from portfolio.retrieval import load_pages
from portfolio.runtime import Services


@pytest.mark.parametrize("source,problem,count", [("uci_iris", "classification", 5), ("sklearn_diabetes", "regression", 6)])
def test_actual_bounded_ml_public_fixtures(settings, source, problem, count):
    frame = read_csv(public_file(source).read_bytes(), settings)
    result = train_models(frame, settings)
    assert result["problem_type"] == problem
    assert len(result["leaderboard"]) == count
    assert result["train_rows"] + result["test_rows"] == len(frame)
    assert result["winner"] == result["leaderboard"][0]["model"]
    assert "training data only" in result["selection_method"]
    assert result["baseline_validation_mean"] < result["leaderboard"][0]["validation_mean"]
    metric = "accuracy" if problem == "classification" else "r2"
    assert result["held_out_test_metrics"][metric] > (.8 if problem == "classification" else .2)


def test_training_deadline_stops_worker(settings):
    with patch("portfolio.ml.subprocess.run", side_effect=subprocess.TimeoutExpired("training", 1)):
        with pytest.raises(TimeoutError, match="worker stopped"):
            train_models(read_csv(public_file("uci_iris").read_bytes(), settings), settings)


def test_public_pdf_real_pages_and_invalid_content():
    pdf = public_file("nist_ai_rmf")
    pages = load_pages(pdf.name, pdf.read_bytes())
    assert len(pages) > 40 and pages[0]["page"] == 1
    assert "NIST" in pages[0]["text"]
    with pytest.raises(ValueError, match="Invalid"):
        load_pages("bad.pdf", b"not a PDF")
    with pytest.raises(ValueError, match="Unknown catalog"):
        public_file("../../private.env")


@pytest.mark.protocol
def test_actual_mcp_stdio_and_agent_adapter():
    direct = asyncio.run(mcp_demo())
    assert direct["iris"]["rows"] == 150
    assert direct["pdf"]["pages"][0]["page"] == 1
    agent = asyncio.run(mcp_demo(use_agent=True))
    assert "tool" in agent["message_types"]


@pytest.mark.protocol
def test_mcp_resources_and_path_rejection():
    import sys
    async def exercise():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-X", "utf8", "-m", "portfolio.protocol"], cwd=str(ROOT))
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                resources = (await client.list_resources()).resources
                assert any(str(item.uri) == "catalog://public" for item in resources)
                resource = await client.read_resource("catalog://public")
                assert "nist_ai_rmf" in resource.contents[0].text
                bad = await client.call_tool("load_public_pdf", {"source_id": "../../private.pdf"})
                assert bad.isError
                search = decode_tool(await client.call_tool("search_public_pdf", {"query": "Govern Map Measure Manage", "k": 2}))
                assert len(search["evidence"]) <= 2 and search["evidence"]
    asyncio.run(exercise())


def test_http_mcp_requires_credential(settings):
    server = create_mcp_server(settings)
    protected = BearerBoundary(server.streamable_http_app(), "a-test-credential-at-least-24-chars")
    with TestClient(protected, base_url="http://127.0.0.1:8010") as client:
        assert client.post("/mcp", json={}).status_code == 401
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
            headers={"Authorization": "Bearer a-test-credential-at-least-24-chars", "Accept": "application/json, text/event-stream"})
        assert response.status_code == 200 and "protocolVersion" in response.text


def test_review_pending_checkpoint_survives_restart(settings):
    first = Services(settings)
    owner = Principal("demo", "alice")
    session = first.store.new_session(owner, "Recovery")["id"]
    row = first.propose(owner, session, "Review this exact report", "restart-safe")
    first.close()
    second = Services(settings)
    try:
        result = second.decide(Principal("demo", "reviewer", "reviewer"), row["id"], True, 1)
        assert second.store.artifact(owner, result["artifact_id"])["content"] == b"Review this exact report"
        state = second.review.get_state(second.config(owner, row["id"], "review"))
        assert not state.next
    finally:
        second.close()


def test_actual_provider_client_construction_without_network(settings):
    configured = settings.model_copy(update={"google_api_key": SecretStr("dummy-local-test"),
        "openai_api_key": SecretStr("dummy-local-test"), "anthropic_api_key": SecretStr("dummy-local-test")})
    from portfolio.config import Generation
    for provider, model in [("google_genai", "gemini-3.8-flash"), ("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet-4-6")]:
        client = build_model(configured, Generation(provider=provider, model=model, max_tokens=512))
        assert client is not None  # Constructs actual SDK adapters; deliberately no inference.


def test_model_budget_bounds_calls_and_reservation():
    budget = ModelBudget()
    budget.reserve(59000)
    with pytest.raises(ValueError, match="budget"):
        budget.reserve(1001)


def test_adk_real_workflow_and_durable_state(settings):
    pytest.importorskip("google.adk")
    from portfolio.adk_project import run_trip
    first = asyncio.run(run_trip(settings=settings))
    second = asyncio.run(run_trip(session_id=first["session_id"], settings=settings))
    assert first["state"]["runs"] == 1 and second["state"]["runs"] == 2
    assert first["state"]["budget_details"]["estimate_usd"] == 360
    assert first["state"]["callback_applied"] and first["state"]["review_approved"]
    assert {item["author"] for item in first["events"]} >= {"counter", "destination", "budget", "writer", "critic"}


def test_a2a_real_discovery_and_sdk_message_exchange(settings):
    pytest.importorskip("google.adk")
    from portfolio.a2a_project import create_a2a_app
    from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
    from a2a.types import Message, Part, SendMessageRequest
    from google.protobuf.json_format import MessageToDict
    token = "local-a2a-test-token-at-least-24-characters"
    settings = settings.model_copy(update={"mcp_http_token": SecretStr(token)})
    app = create_a2a_app(settings)
    # TestClient drives real ASGI lifespan. Async SDK uses ASGITransport for deterministic local networking.
    with TestClient(app) as probe:
        assert probe.get("/.well-known/agent-card.json").status_code == 401
        async def exchange():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8020",
                     headers={"Authorization": "Bearer " + token}, timeout=90) as http:
                card = await A2ACardResolver(http, "http://127.0.0.1:8020").get_agent_card()
                assert card.name == "public_evidence"
                client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
                try:
                    request = SendMessageRequest(message=Message(message_id="public-test", role="ROLE_USER",
                        parts=[Part(text="What are Govern Map Measure Manage?")]))
                    events = [MessageToDict(event) async for event in client.send_message(request)]
                    text = json.dumps(events)
                    assert "nist-ai-rmf" in text and "page" in text
                    assert "TASK_STATE_COMPLETED" in text
                finally:
                    await client.close()
        asyncio.run(exchange())
