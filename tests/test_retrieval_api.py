from fastapi.testclient import TestClient
import pytest

from portfolio.api import create_app
from portfolio.config import Principal
from portfolio.insight import AgentContext
from portfolio.research import build_research
from portfolio.runtime import Services


def test_chroma_evidence_is_owned_persistent_and_no_answer_is_honest(services, settings, alice):
    session = services.store.new_session(alice)["id"]
    services.ingest(alice, session, "policy.txt", b"Employees receive 20 days of annual leave. Reimbursement requires a receipt.")
    answer = services.ask_evidence(alice, session, "How many days of annual leave?")
    assert answer["supported"] and "20" in answer["answer"]
    assert answer["citations"] and answer["evidence"][0]["source"] == "policy.txt"
    assert services.ask_evidence(alice, session, "How many days of annual leave?")["cached"]
    assert services.rag.retrieve(Principal("demo", "bob"), "annual leave") == []
    absent = services.ask_evidence(alice, session, "quasar pineapple orbital velocity")
    assert not absent["supported"]
    restarted = Services(settings)
    try:
        assert restarted.ask_evidence(alice, session, "annual leave")["supported"]
    finally:
        restarted.close()


@pytest.mark.parametrize("strategy", ["hybrid", "vector", "mmr", "multi_query", "parent", "compression", "self_query"])
def test_retrieval_strategies_keep_provenance(services, alice, strategy):
    session = services.store.new_session(alice)["id"]
    services.ingest(alice, session, "manual.md", b"Risk governance assigns accountability. Risk measurement evaluates evidence. Manage risk with documented controls.")
    result = services.rag.retrieve(alice, "risk governance", strategy=strategy)
    assert result and all(item["page"] == 1 and item["source"] == "manual.md" for item in result)


def test_parallel_research_deduplicates_and_does_not_approve_exhaustion(services, alice):
    session = services.store.new_session(alice)["id"]
    graph = build_research(services)
    initial = {"topic": "risk governance", "findings": [], "attempts": 0, "max_attempts": 2,
               "feedback": "", "generation": services.settings.generation().model_dump()}
    result = graph.invoke(initial, services.config(alice, "empty-research"), context=AgentContext(alice, session))
    assert result["status"] == "exhausted" and result["attempts"] == 2
    services.ingest(alice, session, "risk.txt", b"Risk governance establishes accountability. Risk measurement evaluates reliable evidence.")
    result = graph.invoke(initial, services.config(alice, "evidence-research"), context=AgentContext(alice, session))
    assert result["status"] == "accepted"
    assert len({item["id"] for item in result["evidence"]}) == len(result["evidence"])


def test_api_auth_upload_chat_cross_user_and_review(settings):
    headers = {"Authorization": "Bearer demo-alice-key"}
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").json()["ready"]
        assert client.get("/v1/sessions").status_code == 401
        created = client.post("/v1/sessions", json={"title": "API test"}, headers=headers)
        assert created.status_code == 201
        session = created.json()["id"]
        assert client.get(f"/v1/sessions/{session}/messages", headers={"Authorization": "Bearer demo-bob-key"}).status_code == 404
        upload = client.post(f"/v1/sessions/{session}/uploads", files={"file": ("table.csv", b"age,target\n20,0\n21,1\n")}, headers=headers)
        assert upload.status_code == 201
        result = client.post("/v1/chat", json={"session_id": session, "message": "show missing values", "dataset_id": upload.json()["id"]}, headers=headers)
        assert result.status_code == 200 and result.json()["route"]["route"] == "data"
        assert client.post("/v1/chat", json={"session_id": session, "message": "calculate 2+2", "tenant": "admin"}, headers=headers).status_code == 422
        proposed = client.post("/v1/reviews", json={"session_id": session, "content": "Exact report", "idempotency_key": "api-one"}, headers=headers)
        assert proposed.status_code == 201
        review_id = proposed.json()["id"]
        assert client.post(f"/v1/reviews/{review_id}/decision", json={"approved": True, "expected_version": 1}, headers=headers).status_code == 403
        approved = client.post(f"/v1/reviews/{review_id}/decision", json={"approved": True, "expected_version": 1}, headers={"Authorization": "Bearer demo-reviewer-key"})
        assert approved.status_code == 200 and approved.json()["artifact_id"]
