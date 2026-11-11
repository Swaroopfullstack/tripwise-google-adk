"""Small executable concept labs complementing the seven application workflows."""
from __future__ import annotations

import asyncio
import math
import sqlite3
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .foundations import ScriptedToolModel


def sampling_lab(temperature=.7, top_k=3, top_p=.8):
    """Numerical teaching example; provider samplers may use a different ordering."""
    if not .01 <= temperature <= 2 or not 1 <= top_k <= 5 or not 0 < top_p <= 1:
        raise ValueError("Sampling values outside the teaching bounds")
    logits = {"agent": 2., "chain": 1.5, "graph": 1., "tool": .5, "banana": -1.}
    weights = {key: math.exp(value/temperature) for key, value in logits.items()}
    probabilities = {key: value/sum(weights.values()) for key, value in weights.items()}
    candidates = sorted(probabilities, key=probabilities.get, reverse=True)[:top_k]
    total, nucleus, accumulated = sum(probabilities[key] for key in candidates), [], 0
    for key in candidates:
        nucleus.append(key)
        accumulated += probabilities[key]/total
        if accumulated >= top_p:
            break
    denominator = sum(probabilities[key] for key in nucleus)
    return {"temperature": temperature, "top_k": top_k, "top_p": top_p,
            "probabilities": probabilities, "normalized_candidates": {key: probabilities[key]/denominator for key in nucleus},
            "explanation": "Apply temperature, retain k, then nucleus cutoff; this is an illustrative sampler, not a provider guarantee"}


def cosine_lab(left=(1., 2., 0.), right=(2., 1., 0.)):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    denominator = math.sqrt(sum(x*x for x in left)*sum(y*y for y in right))
    if denominator == 0:
        raise ValueError("Cosine is undefined for a zero vector")
    similarity = sum(x*y for x, y in zip(left, right))/denominator
    return {"dimensions": len(left), "cosine_similarity": similarity, "cosine_distance": 1-similarity,
            "probability": False}


def owned_sql_lab(query="SELECT item, amount FROM owned_orders ORDER BY amount DESC"):
    """Query only an owned snapshot; authorizer, progress budget and row bound enforce access.

    A public SQL tool would receive the principal from trusted runtime context, never a
    model-chosen owner. The input database here contains only Alice's authorized snapshot.
    """
    if len(query) > 1000:
        raise ValueError("Query too large")
    database = sqlite3.connect(":memory:")
    try:
        database.execute("CREATE TABLE owned_orders(item TEXT, amount REAL)")
        database.executemany("INSERT INTO owned_orders VALUES (?,?)", [("Training", 120), ("Books", 35)])
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, first, second, db, trigger):
            if action not in allowed:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and first != "owned_orders":
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and second not in {"sum", "avg", "count", "min", "max", "round", "coalesce"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        database.set_authorizer(authorize)
        steps = 0
        def progress():
            nonlocal steps
            steps += 1
            return int(steps > 200)
        database.set_progress_handler(progress, 100)
        cursor = database.execute(query)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(101)
        if len(rows) > 100:
            raise ValueError("Result row budget reached")
        return {"tables": ["owned_orders"], "columns": columns, "rows": rows, "scope": "Alice-only teaching snapshot"}
    finally:
        database.close()


class AddArguments(BaseModel):
    left: int = Field(ge=-10000, le=10000)
    right: int = Field(ge=-10000, le=10000)


@tool(args_schema=AddArguments)
def add_numbers(left: int, right: int) -> int:
    """Add two bounded integers; the tool runtime supplies an actual observation."""
    get_stream_writer()({"progress": "Executing bounded addition"})
    return left + right


class ToolState(TypedDict):
    messages: Annotated[list, add_messages]


def tool_graph_lab():
    model = ScriptedToolModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "add_numbers", "args": {"left": 12, "right": 7}, "id": "addition-observation"}]), AIMessage(content="19")])
    def agent(state):
        return {"messages": [model.bind_tools([add_numbers]).invoke(state["messages"])]}
    graph = StateGraph(ToolState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([add_numbers]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 8}
    events = list(app.stream({"messages": [HumanMessage(content="Add 12 and 7")]}, config,
                            stream_mode=["updates", "custom"]))
    snapshot = app.get_state(config)
    history = list(app.get_state_history(config))
    messages = snapshot.values["messages"]
    # update_state creates a branch from a saved checkpoint. It does not copy a
    # session into a new identity or repeat any irreversible external action.
    branch = app.update_state(history[-1].config, {"messages": [HumanMessage(content="Historical branch input")]}, as_node="agent")
    return {"message_types": [message.type for message in messages],
            "observation_ids": [message.tool_call_id for message in messages if message.type == "tool"],
            "answer": messages[-1].content, "stream_modes": [mode for mode, _ in events],
            "checkpoints": len(history), "next": list(snapshot.next), "branch_config": branch,
            "mermaid": app.get_graph().draw_mermaid()}


class FormState(TypedDict, total=False):
    name: str
    purpose: str


def sequential_interrupt_lab():
    def form(state):
        name = interrupt({"field": "name", "question": "What name should be shown?"})
        purpose = interrupt({"field": "purpose", "question": "What is the report purpose?"})
        if not all(isinstance(value, str) and 1 <= len(value) <= 100 for value in (name, purpose)):
            raise ValueError("Invalid form values")
        return {"name": name, "purpose": purpose}
    graph = StateGraph(FormState)
    graph.add_node("form", form)
    graph.add_edge(START, "form")
    graph.add_edge("form", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid4().hex}}
    first = app.invoke({}, config)
    second = app.invoke(Command(resume="Learner"), config)
    final = app.invoke(Command(resume="Portfolio practice"), config)
    return {"first": first["__interrupt__"][0].value, "second": second["__interrupt__"][0].value, "final": final}


def prompts_and_history_lab():
    plain = PromptTemplate.from_template("{style}: explain {topic}").partial(style="Concise")
    prompt = ChatPromptTemplate.from_messages([("system", "Teach clearly."), ("human", "{question}")])
    # Actual pipe composition and batch/async interface without paid inference.
    chain = prompt | RunnableLambda(lambda value: AIMessage(content="Prompt roles: " + ",".join(m.type for m in value.to_messages()))) | StrOutputParser()
    history = [HumanMessage(content="Old question"), AIMessage(content="Old answer"),
               HumanMessage(content="Recent question"), AIMessage(content="Recent answer")]
    trimmed = trim_messages(history, max_tokens=2, token_counter=len, strategy="last", start_on="human", allow_partial=False)
    return {"plain_prompt": plain.invoke({"topic": "runnables"}).to_string(),
            "batch": chain.batch([{"question": "one"}, {"question": "two"}]),
            "async": asyncio.run(chain.ainvoke({"question": "async"})),
            "trimmed_roles": [message.type for message in trimmed], "trimmed_count": len(trimmed)}


def main():
    from .cli import emit
    emit({"sampling": sampling_lab(), "cosine": cosine_lab(), "owned_sql": owned_sql_lab(),
          "tool_graph": tool_graph_lab(), "interrupts": sequential_interrupt_lab(), "prompts": prompts_and_history_lab()})


if __name__ == "__main__":
    main()
