"""TripWise: real ADK agents, durable sessions and a LangChain model adapter.

Prices are explicit teaching estimates, not live quotes. No booking tool exists.
"""
from __future__ import annotations

import asyncio
import json
from typing import Literal
from uuid import uuid4

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field

from .config import Settings
from .models import build_model, redact


class LangChainADKModel(BaseLlm):
    """Text and function-call bridge; every live call uses init_chat_model.

    Images/audio, provider-native grounding and ADK response_schema are deliberately
    rejected instead of silently being dropped. Token streaming is in Foundation Studio.
    """
    settings: Settings = Field(exclude=True)

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        if llm_request.config.response_schema or llm_request.config.response_json_schema:
            raise ValueError("The ADK bridge supports text/function calls; use LangChain schemas for typed output")
        system = llm_request.config.system_instruction
        if not isinstance(system, str):
            system = "\n".join(part.text or "" for part in getattr(system, "parts", []) or [])
        if self.settings.run_mode == "demo":
            yield self.fixture(llm_request, system or "")
            return
        messages, call_ids = [SystemMessage(content=redact(system or ""))], {}
        for content in llm_request.contents:
            texts, calls, responses = [], [], []
            for part in content.parts or []:
                if part.text:
                    texts.append(redact(part.text))
                elif part.function_call:
                    call = part.function_call
                    call_id = call.id or uuid4().hex
                    call_ids.setdefault(call.name, []).append(call_id)
                    calls.append({"id": call_id, "name": call.name, "args": dict(call.args or {})})
                elif part.function_response:
                    response = part.function_response
                    candidates = call_ids.get(response.name, [])
                    call_id = response.id or (candidates.pop(0) if candidates else None)
                    if not call_id:
                        raise ValueError("Function response has no matching call")
                    responses.append(ToolMessage(content=redact(json.dumps(response.response)), tool_call_id=call_id))
                else:
                    raise ValueError("Only text and function calls are supported by the ADK model adapter")
            if texts or calls:
                messages.append(AIMessage(content="\n".join(texts), tool_calls=calls) if content.role == "model"
                                else HumanMessage(content="\n".join(texts)))
            messages.extend(responses)
        if sum(len(str(message.content)) for message in messages) > 30000:
            raise ValueError("ADK model context exceeds the bounded input budget")
        declarations = []
        for tool in llm_request.config.tools or []:
            for definition in tool.function_declarations or []:
                parameters = definition.parameters_json_schema
                if parameters is None and definition.parameters:
                    parameters = definition.parameters.model_dump(mode="json", exclude_none=True)
                declarations.append({"name": definition.name, "description": definition.description or definition.name,
                                     "parameters": parameters or {"type": "object", "properties": {}}})
        client = build_model(self.settings)
        if declarations:
            client = client.bind_tools(declarations)
        async with asyncio.timeout(self.settings.llm_timeout_seconds * (self.settings.llm_max_retries + 1) + 5):
            result = await client.ainvoke(messages)
        parts = []
        if isinstance(result.content, str) and result.content:
            parts.append(types.Part(text=result.content))
        elif isinstance(result.content, list):
            parts.extend(types.Part(text=block["text"]) for block in result.content
                         if isinstance(block, dict) and block.get("type") == "text" and block.get("text"))
        parts.extend(types.Part(function_call=types.FunctionCall(name=call["name"], args=call["args"], id=call["id"]))
                     for call in result.tool_calls)
        yield LlmResponse(content=types.Content(role="model", parts=parts), partial=False, turn_complete=True)

    @staticmethod
    def fixture(request, system):
        has_result = any(part.function_response for content in request.contents for part in content.parts or [])
        if "ROLE:budget" in system and not has_result:
            part = types.Part(function_call=types.FunctionCall(name="estimate_budget",
                                  args={"destination": "Lisbon", "days": 3}, id=uuid4().hex))
        elif "ROLE:critic" in system and not has_result:
            part = types.Part(function_call=types.FunctionCall(name="finish_review", args={"approved": True}, id=uuid4().hex))
        else:
            text = ("Lisbon: walkable neighborhoods, museums and a day by the river." if "ROLE:destination" in system else
                    "Budget checked against the actual estimate_budget tool result." if "ROLE:budget" in system else
                    "Reviewed: estimate limitations and transport assumptions are explicit." if "ROLE:critic" in system else
                    "DEMO itinerary: Day 1 Alfama; Day 2 museums; Day 3 river walk. Budget is an educational estimate; verify current prices.")
            part = types.Part(text=text)
        return LlmResponse(content=types.Content(role="model", parts=[part]))


def estimate_budget(destination: Literal["Lisbon", "Jaipur", "Tokyo"], days: int, tool_context: ToolContext) -> dict:
    """Return a teaching estimate in USD, excluding flights. These are not current market prices."""
    if not 1 <= days <= 14:
        raise ValueError("Trip length must be one to fourteen days")
    amount = {"Lisbon": 120, "Jaipur": 55, "Tokyo": 165}[destination] * days
    result = {"destination": destination, "days": days, "estimate_usd": amount,
              "assumptions": "Illustrative lodging, local transport and meals; excludes flights; verify prices"}
    tool_context.state["budget_details"] = result
    return result


def finish_review(approved: bool, tool_context: ToolContext) -> dict:
    """Exit the review loop only when the critic explicitly accepts the draft."""
    tool_context.state["review_approved"] = approved
    tool_context.actions.escalate = approved
    return {"approved": approved, "external_action": False}


def input_boundary(callback_context, llm_request):
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.text:
                if len(part.text) > 6000:
                    raise ValueError("ADK input is too large")
                part.text = redact(part.text)
    callback_context.state["callback_applied"] = True
    return None


class InvocationCounter(BaseAgent):
    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, actions=EventActions(state_delta={
            "runs": int(ctx.session.state.get("runs", 0)) + 1, "review_approved": False}))


def make_agent(settings: Settings | None = None):
    settings = settings or Settings()
    model = LangChainADKModel(model=settings.llm_model, settings=settings)
    def specialist(name, instruction, output, tools=None):
        return LlmAgent(name=name, model=model, instruction=instruction, output_key=output,
                        tools=tools or [], before_model_callback=input_boundary,
                        disallow_transfer_to_parent=True, disallow_transfer_to_peers=True)
    destination = specialist("destination", "ROLE:destination. Suggest an itinerary for the user's city. State uncertainty.", "destination_notes")
    budget = specialist("budget", "ROLE:budget. Call estimate_budget for Lisbon, Jaipur or Tokyo and 1-14 days. Other cities unsupported.",
                        "budget_notes", [estimate_budget])
    writer = specialist("writer", "ROLE:writer. Use {destination_notes} and {budget_notes}. Explain limitations. Do not claim bookings.", "draft")
    critic = specialist("critic", "ROLE:critic. Review {draft}. Call finish_review(true) only if estimates and uncertainty are clear; otherwise false.",
                        "critique", [finish_review])
    refiner = specialist("refiner", "ROLE:refiner. Improve {draft} using {critique}; keep price limitations explicit.", "draft")
    return SequentialAgent(name="tripwise", description="Bounded travel-planning teaching workflow; no purchases or live fares.", sub_agents=[
        InvocationCounter(name="counter"), ParallelAgent(name="planning", sub_agents=[destination, budget]), writer,
        LoopAgent(name="review_loop", max_iterations=2, sub_agents=[critic, refiner])])


def session_service(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return DatabaseSessionService(db_url="sqlite+aiosqlite:///" + (settings.data_dir / "adk-sessions.sqlite").as_posix())


async def run_trip(question="Plan three days in Lisbon", session_id=None, settings: Settings | None = None):
    settings = settings or Settings()
    if not 1 <= len(question) <= 2000:
        raise ValueError("Question must be 1-2000 characters")
    sessions = session_service(settings)
    runner = Runner(app_name="tripwise", agent=make_agent(settings), session_service=sessions)
    # This CLI is a trusted local operator surface, not a public user-ID selector.
    session = await sessions.get_session(app_name="tripwise", user_id="local_operator", session_id=session_id) if session_id else None
    if session_id and not session:
        raise ValueError("Unknown local ADK session")
    session = session or await sessions.create_session(app_name="tripwise", user_id="local_operator")
    events = []
    try:
        async with asyncio.timeout(180):
            async for event in runner.run_async(user_id=session.user_id, session_id=session.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=question)]),
                    run_config=RunConfig(max_llm_calls=12)):
                events.append({"author": event.author, "text": " ".join(part.text or "" for part in
                    (event.content.parts if event.content else []) or []), "final": event.is_final_response()})
        saved = await sessions.get_session(app_name="tripwise", user_id=session.user_id, session_id=session.id)
        return {"mode": settings.run_mode, "session_id": session.id, "state": saved.state, "events": events}
    finally:
        await runner.close()
        await sessions.close()
