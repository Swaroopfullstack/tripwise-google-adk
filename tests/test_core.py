from unittest.mock import patch

import pytest

from portfolio.config import Generation, Principal, Settings
from portfolio.data import calculate, infer_target, read_csv
from portfolio.foundations import middleware_demo, runnable_demo
from portfolio.models import build_model
from portfolio.reliability import CircuitBreaker, CircuitOpen, TTLCache
from portfolio.storage import Conflict, NotFound, Store


def test_provider_generation_mapping(settings):
    configured = settings.model_copy(update={"google_api_key": __import__("pydantic").SecretStr("test-placeholder")})
    with patch("portfolio.models.init_chat_model") as initialize:
        build_model(configured, Generation(top_k=20, top_p=.8, temperature=1, max_tokens=800))
    assert initialize.call_args.kwargs["model_provider"] == "google_genai"
    assert initialize.call_args.kwargs["top_k"] == 20
    with pytest.raises(ValueError, match="top_k"):
        Generation(provider="openai", model="gpt-4o-mini", top_k=20)


def test_production_rejects_demo_configuration(tmp_path):
    with pytest.raises(ValueError, match="Production requires"):
        Settings(_env_file=None, app_env="production", data_dir=tmp_path)


def test_csv_limits_and_target_inference(settings):
    frame = read_csv(b"age,target\n20,0\n21,1\n22,0\n23,1\n24,0\n25,1\n", settings)
    assert infer_target(frame)["problem_type"] == "classification"
    with pytest.raises(ValueError, match="unique"):
        read_csv(b"a,a\n1,2\n", settings)


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "2**100", "(2+3).__class__", "1/0"])
def test_safe_calculator_rejects_unsafe_or_unbounded_input(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


def test_numeric_calculator():
    assert calculate("(12+3)*4/2") == 30


def test_preferences_and_sessions_survive_restart_and_are_isolated(tmp_path):
    alice, bob = Principal("demo", "alice"), Principal("demo", "bob")
    store = Store(tmp_path)
    session = store.new_session(alice)
    store.message(alice, session["id"], "user", "My first question")
    store.preference(alice, "language", "Telugu")
    restarted = Store(tmp_path)
    assert restarted.preference(alice, "language") == "Telugu"
    assert restarted.preference(bob, "language") is None
    assert restarted.history(alice, session["id"])[0]["content"] == "My first question"
    with pytest.raises(NotFound):
        restarted.history(bob, session["id"])
    with pytest.raises(ValueError):
        restarted.preference(alice, "permanent_approval", "yes")


def test_durable_review_authorization_idempotency_and_conflict(services, alice):
    session = services.store.new_session(alice)["id"]
    proposal = services.propose(alice, session, "Reviewed analysis report", "request-one")
    assert proposal == services.propose(alice, session, "Reviewed analysis report", "request-one")
    with pytest.raises(PermissionError):
        services.decide(alice, proposal["id"], True, 1)
    reviewer = Principal("demo", "manager", "reviewer")
    with pytest.raises(Conflict, match="Stale"):
        services.decide(reviewer, proposal["id"], True, 99)
    result = services.decide(reviewer, proposal["id"], True, 1)
    assert result == services.decide(reviewer, proposal["id"], True, 1)
    assert services.store.artifact(alice, result["artifact_id"])["content"] == b"Reviewed analysis report"
    with pytest.raises(Conflict):
        services.decide(reviewer, proposal["id"], False, 2)
    with services.store.connection() as db:
        assert db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1


def test_rejected_review_has_no_export(services, alice):
    session = services.store.new_session(alice)["id"]
    proposal = services.propose(alice, session, "Do not publish", "reject-one")
    result = services.decide(Principal("demo", "manager", "reviewer"), proposal["id"], False, 1)
    assert result["status"] == "rejected" and result["artifact_id"] is None


def test_cache_copy_and_expiry():
    clock = [0]
    cache = TTLCache(ttl=3, clock=lambda: clock[0])
    cache.put("alice:doc:v1", {"answer": [1]})
    cache.get("alice:doc:v1")["answer"].append(2)
    assert cache.get("alice:doc:v1") == {"answer": [1]}
    assert cache.get("bob:doc:v1") is None
    clock[0] = 3
    assert cache.get("alice:doc:v1") is None


def test_breaker_counts_consecutive_failures_and_recovery():
    clock = [0]
    breaker = CircuitBreaker(threshold=2, cooldown=3, clock=lambda: clock[0])
    def fail():
        raise ConnectionError("fixture outage")
    with pytest.raises(ConnectionError):
        breaker.call(fail)
    assert breaker.call(lambda: "ok") == "ok"
    for _ in range(2):
        with pytest.raises(ConnectionError):
            breaker.call(fail)
    with pytest.raises(CircuitOpen):
        breaker.call(lambda: "should not run")
    clock[0] = 4
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.failures == 0


def test_real_runnables_and_middleware_gate():
    assert runnable_demo("Explain checkpoints")["branch"] == "concise"
    assert middleware_demo("approve")["effects_after_review"] == 1
    assert middleware_demo("reject")["effects_after_review"] == 0


def test_five_specialist_routes_and_owned_thread(services, alice):
    session = services.store.new_session(alice)["id"]
    remembered = services.chat(alice, session, "remember language=Telugu")
    assert remembered["route"]["route"] == "memory"
    second = services.store.new_session(alice)["id"]
    assert "Telugu" in services.chat(alice, second, "Recall my preferences")["answer"]
    assert services.chat(alice, session, "calculate 12*7")["payload"]["result"] == 84
    data = services.upload_csv(alice, session, "sample.csv", b"age,target\n20,0\n21,1\n22,0\n")
    assert services.chat(alice, session, "show missing values", data["id"])["route"]["route"] == "data"
    assert services.chat(alice, session, "ask my PDF a question")["route"]["route"] == "rag"
    assert services.store.history(alice, session)
