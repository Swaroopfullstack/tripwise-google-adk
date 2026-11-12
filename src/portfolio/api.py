"""Authenticated FastAPI boundary. Run a single worker with durable local storage."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import logging
import re
from time import monotonic
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .config import Generation, Settings, sampling_controls
from .insight import AgentContext
from .models import ModelBudget, model_budget
from .research import build_research
from .runtime import Services
from .storage import Conflict, NotFound

logger = logging.getLogger("portfolio.requests")
bearer = HTTPBearer(auto_error=False)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionBody(StrictBody):
    title: str = Field(default="Analysis", min_length=1, max_length=100)


class ChatBody(StrictBody):
    session_id: str = Field(min_length=1, max_length=40)
    message: str = Field(min_length=1, max_length=2000)
    dataset_id: str | None = Field(default=None, max_length=40)
    generation: Generation | None = None
    target: str | None = Field(default=None, max_length=100)
    problem_type: str | None = Field(default=None, pattern="^(regression|classification)$")


class EvidenceBody(StrictBody):
    session_id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=2000)
    strategy: str = Field(default="hybrid", max_length=20)
    generation: Generation | None = None
    k: int | None = Field(default=None, ge=1, le=8)
    fetch_k: int | None = Field(default=None, ge=1, le=40)


class ProposalBody(StrictBody):
    session_id: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=20000)
    idempotency_key: str = Field(min_length=1, max_length=100)


class DecisionBody(StrictBody):
    approved: bool
    expected_version: int = Field(ge=1)


class ResearchBody(StrictBody):
    session_id: str = Field(min_length=1, max_length=40)
    topic: str = Field(min_length=1, max_length=500)
    generation: Generation | None = None
    max_attempts: int = Field(default=2, ge=1, le=3)


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.services = Services(settings)
        app.state.research = build_research(app.state.services)
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(title="TripWise ADK API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_audit(request: Request, call_next):
        start = monotonic()
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", supplied) else uuid4().hex
        request.state.request_id = request_id
        with model_budget():
            response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(json.dumps({"request_id": request_id, "method": request.method, "path": request.url.path,
                                "status": response.status_code, "duration_ms": round((monotonic()-start)*1000, 2)}))
        return response

    def services(request: Request):
        return request.app.state.services

    def identity(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                 service=Depends(services)):
        principal = None
        for token, candidate in service.settings.identities().items():
            if credentials and hmac.compare_digest(credentials.credentials, token):
                principal = candidate
        if principal is None:
            raise HTTPException(401, "Valid bearer credential required", headers={"WWW-Authenticate": "Bearer"})
        if not service.admission.admit((principal.tenant, principal.user)):
            raise HTTPException(429, "Request rate exceeded", headers={"Retry-After": "60"})
        return principal

    @app.exception_handler(NotFound)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(PermissionError)
    async def forbidden(request, error):
        return JSONResponse(status_code=403, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(status_code=400, content={"detail": str(error)[:500]})

    @app.exception_handler(TimeoutError)
    async def deadline(request, error):
        return JSONResponse(status_code=504, content={"detail": "Operation exceeded its deadline",
                                                    "request_id": request.state.request_id})

    @app.exception_handler(RuntimeError)
    async def unavailable(request, error):
        logger.warning(json.dumps({"request_id": request.state.request_id, "error_type": type(error).__name__}))
        return JSONResponse(status_code=503, content={"detail": "Operation unavailable; retry or check configuration",
                                                    "request_id": request.state.request_id})

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        logger.error(json.dumps({"request_id": getattr(request.state, "request_id", "unknown"),
                                 "error_type": type(error).__name__}))
        return JSONResponse(status_code=500, content={"detail": "Request failed", "request_id": getattr(request.state, "request_id", "unknown")})

    @app.get("/health/live")
    def live():
        return {"alive": True}

    @app.get("/health/ready")
    def ready(service=Depends(services)):
        with service.store.connection() as db:
            db.execute("SELECT 1").fetchone()
        return {"ready": True, "mode": service.settings.run_mode, "version": "1.0.0",
                "inference_verified_by_probe": False}

    @app.get("/v1/config")
    def config(principal=Depends(identity), service=Depends(services)):
        return {"mode": service.settings.run_mode, "generation": service.settings.generation().model_dump(),
                "allowed_models": service.settings.allowed_models(),
                "model_controls": {provider: {model: sampling_controls(provider, model) for model in models}
                                   for provider, models in service.settings.allowed_models().items()},
                "identity": {"tenant": principal.tenant,
                "user": principal.user, "role": principal.role}, "embedding_backend": service.settings.embedding_backend,
                "retrieval_k": service.settings.retrieval_k, "retrieval_fetch_k": service.settings.retrieval_fetch_k,
                "strategies": ["hybrid", "vector", "mmr", "multi_query", "parent", "compression", "self_query"]}

    @app.get("/v1/sessions")
    def sessions(principal=Depends(identity), service=Depends(services)):
        return service.store.sessions(principal)

    @app.post("/v1/sessions", status_code=201)
    def new_session(body: SessionBody, principal=Depends(identity), service=Depends(services)):
        return service.store.new_session(principal, body.title)

    @app.get("/v1/sessions/{session_id}/messages")
    def messages(session_id: str, principal=Depends(identity), service=Depends(services)):
        return service.store.history(principal, session_id)

    @app.get("/v1/sessions/{session_id}/uploads")
    def uploads(session_id: str, principal=Depends(identity), service=Depends(services)):
        return service.store.uploads(principal, session_id)

    @app.post("/v1/sessions/{session_id}/uploads", status_code=201)
    async def upload(session_id: str, file: UploadFile, principal=Depends(identity), service=Depends(services)):
        try:
            content = await file.read(service.settings.max_upload_mb*1024*1024+1)
            name = file.filename or "upload.txt"
            if name.lower().endswith(".csv"):
                return await run_in_threadpool(service.upload_csv, principal, session_id, name, content)
            return await run_in_threadpool(service.ingest, principal, session_id, name, content)
        finally:
            await file.close()

    @app.post("/v1/chat")
    def chat(body: ChatBody, principal=Depends(identity), service=Depends(services)):
        return service.chat(principal, body.session_id, body.message, body.dataset_id, body.generation,
                            body.target, body.problem_type)

    @app.get("/v1/datasets/{dataset_id}/profile")
    def dataset_profile(dataset_id: str, principal=Depends(identity), service=Depends(services)):
        from .data import infer_target, profile, read_csv
        upload = service.store.upload(principal, dataset_id)
        if upload["kind"] != "csv":
            raise ValueError("This upload is not a CSV")
        frame = read_csv(upload["content"], service.settings)
        return {"profile": profile(frame), "target_inference": infer_target(frame)}

    @app.post("/v1/chat/stream")
    def stream(body: ChatBody, principal=Depends(identity), service=Depends(services)):
        # Authenticate and validate ownership before sending HTTP 200 headers.
        service.store.session(principal, body.session_id)
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        service.settings.validate_override(body.generation or service.settings.generation())
        def events():
            budget = ModelBudget()
            iterator = service.stream_chat(principal, body.session_id, body.message, body.dataset_id,
                                           body.generation, body.target, body.problem_type)
            try:
                while True:
                    # Starlette may advance each iteration in a different copied
                    # thread context. Do not keep a ContextVar token across yield.
                    with model_budget(budget):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                    yield "event: " + item["event"] + "\ndata: " + json.dumps(item["data"]) + "\n\n"
            except Exception as error:
                # After headers, errors are explicit events rather than a fake result.
                yield "event: error\ndata: " + json.dumps({"detail": "Workflow failed", "error_type": type(error).__name__}) + "\n\n"
            finally:
                iterator.close()
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/evidence/query")
    def evidence(body: EvidenceBody, principal=Depends(identity), service=Depends(services)):
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        return service.ask_evidence(principal, body.session_id, body.question, body.strategy, body.generation, body.k, body.fetch_k)

    @app.post("/v1/reviews", status_code=201)
    def propose(body: ProposalBody, principal=Depends(identity), service=Depends(services)):
        return service.propose(principal, body.session_id, body.content, body.idempotency_key)

    @app.get("/v1/reviews")
    def pending(principal=Depends(identity), service=Depends(services)):
        return service.store.pending_reviews(principal)

    @app.get("/v1/reviews/{review_id}")
    def review(review_id: str, principal=Depends(identity), service=Depends(services)):
        return service.store.review(principal, review_id)

    @app.post("/v1/reviews/{review_id}/decision")
    def decide(review_id: str, body: DecisionBody, principal=Depends(identity), service=Depends(services)):
        return service.decide(principal, review_id, body.approved, body.expected_version)

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str, principal=Depends(identity), service=Depends(services)):
        item = service.store.artifact(principal, artifact_id)
        return Response(item["content"], media_type=item["media_type"],
                        headers={"Content-Disposition": f'attachment; filename="{item["name"]}"'})

    @app.post("/v1/research")
    def research(body: ResearchBody, request: Request, principal=Depends(identity), service=Depends(services)):
        service.store.session(principal, body.session_id)
        if principal.role not in {"analyst", "admin"}:
            raise PermissionError("Analyst role required")
        generation = service.settings.validate_override(body.generation or service.settings.generation())
        with service.thread_lock((principal, body.session_id)):
            result = request.app.state.research.invoke({"topic": body.topic, "findings": [], "attempts": 0,
                "max_attempts": body.max_attempts, "status": "pending", "feedback": "", "generation": generation.model_dump()},
                service.config(principal, uuid4().hex, "research"), context=AgentContext(principal, body.session_id))
        return {key: result[key] for key in ("report", "status", "attempts", "score", "feedback", "evidence")}

    return app


app = create_app()
