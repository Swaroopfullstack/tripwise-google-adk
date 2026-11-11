"""Small operator CLI; user-facing data access is through the authenticated API."""
from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
import json
from uuid import uuid4

from .config import Principal, Settings


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def demo(project):
    if project == "foundation":
        from .foundations import middleware_demo, runnable_demo
        return {"runnables": runnable_demo("Explain checkpoints and reducers with an example"),
                "approved": middleware_demo("approve"), "rejected": middleware_demo("reject")}
    if project == "mcp":
        from .protocol import mcp_demo
        return asyncio.run(mcp_demo())
    if project == "mcp-agent":
        from .protocol import mcp_demo
        return asyncio.run(mcp_demo(use_agent=True))
    if project == "adk":
        from .adk_project import run_trip
        return asyncio.run(run_trip())
    from .runtime import Services
    from .public_data import public_file
    from .research import build_research
    from .insight import AgentContext
    service = Services()
    principal = Principal("demo", "cli_operator")
    try:
        session = service.store.new_session(principal, "CLI " + project)["id"]
        if project == "review":
            proposal = service.propose(principal, session, "Publish a report only after an authenticated review.", uuid4().hex)
            result = service.decide(Principal("demo", "cli_reviewer", "reviewer"), proposal["id"], True, proposal["version"])
            return {"proposal": proposal, "decision": result, "artifact": service.store.artifact(principal, result["artifact_id"])["name"]}
        if project in {"evidence", "research"}:
            path = public_file("nist_ai_rmf")
            service.ingest(principal, session, path.name, path.read_bytes())
            if project == "evidence":
                return service.ask_evidence(principal, session, "What are the four functions Govern Map Measure Manage?")
            result = build_research(service).invoke({"topic": "AI risk governance and measurement", "findings": [],
                "attempts": 0, "max_attempts": 2, "feedback": "", "generation": service.settings.generation().model_dump()},
                service.config(principal, uuid4().hex, "research"), context=AgentContext(principal, session))
            return {key: result[key] for key in ("report", "status", "attempts", "score", "evidence")}
        path = public_file("uci_iris")
        dataset = service.upload_csv(principal, session, path.name, path.read_bytes())
        return {"summary": service.chat(principal, session, "Summarize the data", dataset["id"]),
                "ml": service.chat(principal, session, "Train and compare models", dataset["id"]),
                "memory": service.chat(principal, session, "remember language=English"),
                "calculator": service.chat(principal, session, "calculate 12 * (3 + 4)")}
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description="Seven AI engineering portfolio projects")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="Download attributed public PDFs/CSV fixtures")
    commands.add_parser("doctor", help="Report versions and configured mode; never prints credentials")
    sample = commands.add_parser("demo", help="Run an actual local workflow")
    sample.add_argument("project", choices=["insight", "evidence", "review", "research", "mcp", "mcp-agent", "adk", "foundation"])
    mcp = commands.add_parser("mcp", help="Run public catalog MCP server")
    mcp.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    a2a = commands.add_parser("a2a", help="Run protected public-evidence A2A server")
    a2a.add_argument("--client", action="store_true")
    adk = commands.add_parser("adk", help="Run TripWise; optionally continue a durable local session")
    adk.add_argument("--question", default="Plan three days in Lisbon")
    adk.add_argument("--session")
    commands.add_parser("evaluate", help="Evaluate attributed public fixtures; writes reports/public-evaluation.json")
    args = parser.parse_args()
    if args.command == "bootstrap":
        from .public_data import bootstrap
        emit(bootstrap())
    elif args.command == "doctor":
        settings = Settings()
        emit({"mode": settings.run_mode, "environment": settings.app_env,
              "generation": settings.generation().model_dump(), "embedding_backend": settings.embedding_backend,
              "packages": {name: version(name) for name in ["langchain", "langgraph", "fastapi", "chromadb", "mcp", "scikit-learn"]},
              "live_inference_tested": False})
    elif args.command == "demo":
        emit(demo(args.project))
    elif args.command == "mcp":
        from .protocol import run_mcp
        run_mcp(args.transport)
    elif args.command == "a2a":
        from .a2a_project import a2a_demo, run_a2a
        if args.client:
            emit(asyncio.run(a2a_demo()))
        else:
            run_a2a()
    elif args.command == "adk":
        from .adk_project import run_trip
        emit(asyncio.run(run_trip(args.question, args.session)))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        emit(evaluate())


if __name__ == "__main__":
    main()
