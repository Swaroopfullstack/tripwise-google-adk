"""ReviewOps: durable interrupt plus authoritative transactional review records."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from .insight import AgentContext


class ReviewState(TypedDict, total=False):
    review_id: str
    status: str
    artifact_id: str | None


def build_review(services):
    def gate(state: ReviewState, runtime: Runtime[AgentContext]):
        proposal = services.store.review(runtime.context.principal, state["review_id"])
        decision = interrupt({"review_id": proposal["id"], "version": proposal["version"],
                              "action": "export_report", "payload": proposal["payload"],
                              "allowed_decisions": ["approve", "reject"]})
        # The resume payload is not authorization. Re-read the authoritative,
        # reviewer-approved record before reporting any effect.
        resolved = services.store.review(runtime.context.principal, state["review_id"])
        if resolved["status"] == "pending":
            raise PermissionError("An authenticated reviewer must commit a decision first")
        if decision.get("status") != resolved["status"]:
            raise ValueError("Resume outcome differs from the committed decision")
        return {"status": resolved["status"], "artifact_id": resolved["artifact_id"]}

    graph = StateGraph(ReviewState, context_schema=AgentContext)
    graph.add_node("review", gate)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=services.checkpointer)
