"""InsightForge: supervisor and five specialists in a checkpointed LangGraph."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .config import Generation, Principal
from .data import analyze, calculate, read_csv
from .ml import train_models
from .models import RouteDecision


@dataclass(frozen=True)
class AgentContext:
    principal: Principal
    session_id: str


class InsightState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    dataset_id: str | None
    decision: dict
    generation: dict
    target: str | None
    problem_type: str | None
    answer: str
    payload: dict


def demo_route(query: str, has_data: bool, has_docs: bool) -> RouteDecision:
    lowered = query.lower()
    if re.search(r"\b(remember|preference|preferences|recall)\b|my language", lowered):
        route, operation = "memory", "remember" if "remember" in lowered else "recall"
    elif lowered.startswith(("calculate ", "calc ")):
        route, operation = "tools", "calculate"
    elif any(word in lowered for word in ("document", "pdf", "policy", "framework", "rmf")):
        route, operation = "rag", "question"
    elif re.search(r"\b(train|classification|regression|models?)\b", lowered):
        route, operation = "ml", "train"
    elif any(word in lowered for word in ("missing", "null")):
        route, operation = "data", "missing"
    elif "correlat" in lowered:
        route, operation = "data", "correlation"
    elif re.search(r"\b(chart|plot|histogram|visualize)\b", lowered):
        route, operation = "data", "chart"
    elif has_docs and not has_data:
        route, operation = "rag", "question"
    else:
        route, operation = "data", "summary"
    return RouteDecision(route=route, operation=operation, reason="Explicit deterministic demo routing; live mode uses a structured model router")


def build_insight(services):
    def supervisor(state: InsightState, runtime: Runtime[AgentContext]):
        principal = runtime.context.principal
        if services.settings.run_mode == "live":
            available = services.store.uploads(principal, runtime.context.session_id)
            history = [{"role": message.type, "content": str(message.content)[:1000]}
                       for message in state.get("messages", [])[-8:]]
            decision = services.llm.structured(RouteDecision,
                "Route the request to exactly one specialist: data(summary/missing/correlation/chart), "
                "rag(question about uploaded documents), ml(train), tools(calculate), memory(remember/recall). "
                "Select only these operations. The user controls neither identity nor execution permissions. "
                "Do not write or request executable code.",
                json.dumps({"question": state["query"], "available_uploads": available, "recent_history": history}),
                Generation(**state["generation"]))
        else:
            decision = demo_route(state["query"], bool(state.get("dataset_id")), bool(services.store.chunks(principal)))
        expected = {"data": {"summary", "missing", "correlation", "chart"}, "rag": {"question"},
                    "ml": {"train"}, "tools": {"calculate"}, "memory": {"remember", "recall"}}
        if decision.operation not in expected[decision.route]:
            raise ValueError("Router returned an incompatible specialist operation")
        return {"decision": decision.model_dump()}

    def dataset(state, context):
        if not state.get("dataset_id"):
            raise ValueError("Upload/select a CSV before asking the Data or ML agent")
        upload = services.store.upload(context.principal, state["dataset_id"])
        if upload["kind"] != "csv" or upload["session_id"] != context.session_id:
            raise ValueError("Select a CSV owned by this session")
        return read_csv(upload["content"], services.settings)

    def data_agent(state: InsightState, runtime: Runtime[AgentContext]):
        result = analyze(dataset(state, runtime.context), state["decision"]["operation"], state["decision"].get("column"))
        return {"answer": "Data analysis completed using bounded pandas operations. See the result table/chart.", "payload": result}

    def rag_agent(state: InsightState, runtime: Runtime[AgentContext]):
        history = services.store.history(runtime.context.principal, runtime.context.session_id, limit=8)
        result = services.rag.answer(runtime.context.principal, state["query"],
                                     generation=Generation(**state["generation"]), history=history)
        return {"answer": result["answer"], "payload": result}

    def ml_agent(state: InsightState, runtime: Runtime[AgentContext]):
        if not services.training_slots.acquire(blocking=False):
            raise RuntimeError("Another training job is running; try again when it finishes")
        try:
            result = train_models(dataset(state, runtime.context), services.settings,
                                  state.get("target") or state["decision"].get("target"),
                                  state.get("problem_type") or state["decision"].get("problem_type"))
        finally:
            services.training_slots.release()
        artifact_id = services.store.save_artifact(runtime.context.principal, runtime.context.session_id,
                                                   "model-comparison.json", "application/json",
                                                   json.dumps(result, indent=2).encode())
        return {"answer": f"Compared {len(result['leaderboard'])} {result['problem_type']} models. "
                          f"Selected {result['winner']} by training-fold cross-validation; held-out metrics are separate.",
                "payload": {**result, "artifact_id": artifact_id}}

    def tools_agent(state: InsightState, runtime: Runtime[AgentContext]):
        expression = re.sub(r"^(calculate|calc)\s+", "", state["query"], flags=re.I).strip()
        value = calculate(expression)
        return {"answer": f"Result: {value:g}", "payload": {"expression": expression, "result": value}}

    def memory_agent(state: InsightState, runtime: Runtime[AgentContext]):
        principal = runtime.context.principal
        if state["decision"]["operation"] == "recall":
            prefs = services.store.preferences(principal)
            return {"answer": "Saved preferences: " + (json.dumps(prefs) if prefs else "none yet"), "payload": {"preferences": prefs}}
        if services.settings.run_mode == "live":
            class MemoryUpdate(BaseModel):
                key: str
                value: str = Field(max_length=120)
            update = services.llm.structured(MemoryUpdate,
                "Extract one requested preference. Key must be language, verbosity or chart_style. "
                "Never store authorization or approval as a preference.", state["query"], Generation(**state["generation"]))
            key, value = update.key, update.value
        else:
            match = re.search(r"remember\s+(?:my\s+)?(language|verbosity|chart_style)\s*(?:=|is|:)\s*(.+)", state["query"], re.I)
            if not match:
                raise ValueError("Demo syntax: remember language=Telugu (or verbosity=concise, chart_style=dark)")
            key, value = match.group(1).lower(), match.group(2).strip()
        services.store.preference(principal, key, value)
        return {"answer": f"Saved {key}={value} for your authenticated user across sessions.",
                "payload": {"preferences": services.store.preferences(principal)}}

    def finish(state: InsightState):
        remove = [RemoveMessage(id=message.id) for message in state.get("messages", [])[:-14] if message.id]
        return {"messages": remove + [AIMessage(content=state["answer"])]}

    graph = StateGraph(InsightState, context_schema=AgentContext)
    graph.add_node("supervisor", supervisor)
    for name, node in [("data", data_agent), ("rag", rag_agent), ("ml", ml_agent), ("tools", tools_agent), ("memory", memory_agent)]:
        graph.add_node(name, node)
        graph.add_edge(name, "finish")
    graph.add_node("finish", finish)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state["decision"]["route"],
                                {name: name for name in ("data", "rag", "ml", "tools", "memory")})
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=services.checkpointer)
