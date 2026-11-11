"""Streamlit portfolio UI. All user-data operations go through the authenticated API."""
from __future__ import annotations

import json
import os
from uuid import uuid4

import httpx
import plotly.graph_objects as go
import streamlit as st


PROJECTS = ['TripWise ADK']


def run(project='TripWise ADK'):
    st.set_page_config(page_title=f"{project} | AI Engineering Portfolio", page_icon="◈", layout="wide")
    st.markdown("""<style>.block-container{padding-top:2rem;max-width:1280px}
    h1{letter-spacing:-.035em}div[data-testid=stMetric]{background:#132235;padding:14px;border-radius:10px}
    </style>""", unsafe_allow_html=True)  # Static developer CSS only.
    st.sidebar.title("◈ AI Engineering")
    selected_project = project
    st.sidebar.caption(project)
    production = os.getenv("APP_ENV", "development") == "production"
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    st.sidebar.caption("Authenticated API: " + base_url)
    token = st.sidebar.text_input("API bearer credential", type="password", value="" if production else "demo-alice-key")
    if not production:
        st.sidebar.caption("Local demo: demo-alice-key · demo-bob-key · demo-reviewer-key. Never use these in production.")

    def api(method, path, **kwargs):
        response = httpx.request(method, base_url + path, headers={"Authorization": "Bearer " + token},
                                 timeout=150, follow_redirects=False, **kwargs)
        if response.is_error:
            try:
                detail = response.json().get("detail", "Request failed")
            except ValueError:
                detail = "Request failed"
            raise ValueError(f"HTTP {response.status_code}: {detail}")
        return response.json()

    try:
        config = api("GET", "/v1/config")
        current_identity = config["identity"]
        identity_key = (current_identity["tenant"], current_identity["user"])
        if st.session_state.get("identity_key") != identity_key:
            for key in ("session_id", "dataset_id", "last_result", "last_evidence", "last_research", "proposal"):
                st.session_state.pop(key, None)
            st.session_state.identity_key = identity_key
        st.sidebar.success(f"{current_identity['user']} · {current_identity['role']}")
        st.sidebar.caption(f"Mode: {config['mode']} · Embeddings: {config['embedding_backend']}")
        if config["mode"] == "demo":
            st.info("Offline demo: real graphs, databases, retrieval and ML. Routing/writing use explicit deterministic fixtures; no LLM inference.")
        sessions = api("GET", "/v1/sessions")
        if st.sidebar.button("＋ New session", use_container_width=True) or not sessions:
            created = api("POST", "/v1/sessions", json={"title": "Analysis " + uuid4().hex[:6]})
            st.session_state.session_id = created["id"]
            st.rerun()
        choices = {session["id"]: session["title"] for session in sessions}
        if st.session_state.get("session_id") not in choices:
            st.session_state.session_id = next(iter(choices))
        session_id = st.sidebar.selectbox("Conversation", list(choices), key="session_id", format_func=choices.get)
        if st.session_state.get("last_session_id") != session_id:
            for key in ("dataset_id", "last_result", "last_evidence", "last_research", "proposal", "proposal_key"):
                st.session_state.pop(key, None)
            st.session_state.last_session_id = session_id
        generation = dict(config["generation"])
        with st.sidebar.expander("Provider and generation settings"):
            providers = list(config["allowed_models"])
            provider = st.selectbox("Provider", providers, index=providers.index(generation["provider"]))
            allowed = config["allowed_models"][provider]
            model = st.selectbox("Model", allowed, index=allowed.index(generation["model"]) if generation["model"] in allowed else 0)
            temperature = st.slider("Temperature", 0.0, 2.0, float(generation["temperature"] or 0), .1)
            top_p = st.slider("Top-p", .05, 1.0, float(generation["top_p"] or .95), .05) if st.checkbox("Override top-p") else None
            top_k = st.number_input("Top-k", 1, 1000, int(generation["top_k"] or 40)) if st.checkbox("Override top-k", disabled=provider == "openai") and provider != "openai" else None
            max_tokens = st.number_input("Maximum output tokens", 64, 8192, int(generation["max_tokens"]), 64)
            generation.update(provider=provider, model=model, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)
            st.caption("Only server-enabled models appear. Unsupported model-specific sampling settings return an error; they are not silently ignored.")
        uploads = api("GET", f"/v1/sessions/{session_id}/uploads")
        csv_choices = {item["id"]: item["name"] for item in uploads if item["kind"] == "csv"}
        if st.session_state.get("dataset_id") not in csv_choices:
            st.session_state.dataset_id = next(iter(csv_choices), None)
        dataset_id = st.sidebar.selectbox("Active CSV", list(csv_choices), key="dataset_id", format_func=csv_choices.get) if csv_choices else None
        if current_identity["role"] in {"analyst", "admin"}:
            uploaded = st.sidebar.file_uploader("Upload CSV, PDF or text", type=["csv", "pdf", "md", "txt"])
            if st.sidebar.button("Save / index upload", disabled=uploaded is None):
                with st.spinner("Validating and storing your upload…"):
                    result = api("POST", f"/v1/sessions/{session_id}/uploads",
                                 files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                st.sidebar.success("Saved " + result["name"])
                st.rerun()
        st.sidebar.caption("Public fixtures: run `uv run portfolio bootstrap`, then upload a file from assets/public.")
        st.title(selected_project)
        descriptions = {
            "InsightForge": "A supervisor routes your request to Data, RAG, ML, Tools or Memory specialists.",
            "EvidenceDesk": "Compare retrieval strategies and inspect the exact evidence behind an answer.",
            "ReviewOps": "Propose an export, pause the graph, and let an authenticated reviewer approve or reject it.",
            "ResearchSwarm": "Plan, retrieve in parallel, combine evidence, draft and critique with bounded revisions.",
            "ProtocolBridge": "MCP tools and remote-agent interoperability over attributed public data.",
            "TripWise ADK": "Session state, callbacks and deterministic/LLM workflow composition in Google ADK.",
            "Foundation Studio": "Prompts, runnables, structured results, streaming and tool-review middleware.",
        }
        st.caption(descriptions[selected_project])
        if selected_project == "InsightForge":
            for message in api("GET", f"/v1/sessions/{session_id}/messages"):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            target = problem_type = None
            train_requested = False
            if dataset_id:
                with st.expander("ML training controls"):
                    details = api("GET", f"/v1/datasets/{dataset_id}/profile")
                    inferred = details["target_inference"]
                    st.caption(f"Detected target: {inferred['target']} · {inferred['problem_type']} · {inferred['reason']}")
                    if not st.checkbox("Use automatic target and problem detection", value=True):
                        target = st.selectbox("Target column", details["profile"]["columns"])
                        problem_type = st.selectbox("Problem type", ["regression", "classification"])
                    train_requested = st.button("Train and compare models")
            question = st.chat_input("Summarize data, show missing values, train models, ask about a PDF, calculate 12*7…")
            if train_requested:
                question = "Train and compare models"
            if question:
                with st.spinner("Running the selected specialist…"):
                    st.session_state.last_result = api("POST", "/v1/chat", json={"session_id": session_id,
                        "message": question, "dataset_id": dataset_id, "generation": generation,
                        "target": target, "problem_type": problem_type})
                st.rerun()
            if result := st.session_state.get("last_result"):
                st.caption(f"Route: {result['route']['route']} · {result['route']['reason']}")
                payload = result["payload"]
                if "chart" in payload:
                    chart = payload["chart"]
                    st.plotly_chart(go.Figure(go.Bar(x=chart["x"], y=chart["y"])).update_layout(title=chart["column"]), use_container_width=True)
                if "leaderboard" in payload:
                    st.subheader("Cross-validation comparison")
                    st.dataframe(payload["leaderboard"], use_container_width=True, hide_index=True)
                    st.subheader("Held-out test: selected model only")
                    st.json(payload["held_out_test_metrics"])
                    st.download_button("Download model comparison", json.dumps(payload, indent=2), "model-comparison.json", "application/json")
                with st.expander("Structured specialist result", expanded="leaderboard" not in payload):
                    st.json(payload)
        elif selected_project == "EvidenceDesk":
            strategy = st.selectbox("Retrieval strategy", config["strategies"])
            retrieval_k = st.slider("Evidence chunks (retrieval k)", 1, 8, config["retrieval_k"])
            fetch_k = st.slider("Candidate pool (fetch-k)", retrieval_k, 40, max(retrieval_k, config["retrieval_fetch_k"]))
            question = st.text_input("Document question", "What are the four functions of the AI risk management framework?")
            if st.button("Ask with evidence"):
                with st.spinner("Retrieving and validating citations…"):
                    st.session_state.last_evidence = api("POST", "/v1/evidence/query", json={
                        "session_id": session_id, "question": question, "strategy": strategy, "generation": generation,
                        "k": retrieval_k, "fetch_k": fetch_k})
            if result := st.session_state.get("last_evidence"):
                st.markdown(result["answer"])
                st.caption(f"Supported: {result['supported']} · Cached: {result.get('cached', False)}")
                for item in result["evidence"]:
                    with st.expander(f"{item['source']} · page {item['page']} · {item['id'][:12]}"):
                        st.text(item["text"])
        elif selected_project == "ReviewOps":
            if current_identity["role"] in {"reviewer", "admin"}:
                pending = api("GET", "/v1/reviews")
                if not pending:
                    st.success("No pending reviews")
                for item in pending:
                    with st.container(border=True):
                        st.write(f"Review {item['id']} · owner {item['owner']} · version {item['version']}")
                        st.text(json.loads(item["payload"])["content"])
                        left, right = st.columns(2)
                        approve = left.button("Approve export", key="approve-" + item["id"])
                        reject = right.button("Reject export", key="reject-" + item["id"])
                        if approve or reject:
                            api("POST", f"/v1/reviews/{item['id']}/decision", json={"approved": approve, "expected_version": item["version"]})
                            st.rerun()
            else:
                content = st.text_area("Exact report proposed for export", "Dataset analysis complete. Review the evidence and model limitations before sharing.")
                key = st.text_input("Idempotency key", value=st.session_state.setdefault("proposal_key", uuid4().hex))
                if st.button("Request human review"):
                    st.session_state.proposal = api("POST", "/v1/reviews", json={"session_id": session_id,
                        "content": content, "idempotency_key": key})
                if proposal := st.session_state.get("proposal"):
                    current = api("GET", f"/v1/reviews/{proposal['id']}")
                    st.json({key: current[key] for key in ("id", "status", "version", "artifact_id")})
                    if current["artifact_id"]:
                        response = httpx.get(base_url + "/v1/artifacts/" + current["artifact_id"],
                                             headers={"Authorization": "Bearer " + token}, timeout=30)
                        response.raise_for_status()
                        st.download_button("Download approved report", response.content, "approved-report.txt")
                st.caption("Switch to a configured reviewer credential to inspect and decide this exact proposal.")
        elif selected_project == "ResearchSwarm":
            topic = st.text_input("Research topic", "AI risk governance and measurement")
            if st.button("Run research team"):
                with st.spinner("Planning, parallel retrieval, writing and critique…"):
                    st.session_state.last_research = api("POST", "/v1/research", json={"session_id": session_id,
                        "topic": topic, "max_attempts": 2, "generation": generation})
            if result := st.session_state.get("last_research"):
                st.markdown(result["report"])
                st.caption(f"Outcome: {result['status']} · attempts: {result['attempts']} · rubric score: {result['score']}")
                st.text(result["feedback"])
                st.download_button("Download research brief", result["report"], "research-brief.md", "text/markdown")
        elif selected_project == "ProtocolBridge":
            st.code("uv run portfolio mcp --transport stdio\nuv run portfolio demo mcp", language="powershell")
            st.markdown("See the ProtocolBridge README for the authenticated HTTP transport and A2A service. The public-data catalog constrains downloads and PDF loading.")
        elif selected_project == "TripWise ADK":
            st.code("uv run --extra adk portfolio demo adk\nuv run --extra adk adk web agents --port 8011", language="powershell")
            st.caption("The CLI demo uses an actual ADK runner with an explicitly labeled fixture model. Live mode uses the configured Gemini model/provider adapter.")
        else:
            from .foundations import runnable_demo
            st.json(runnable_demo("Explain checkpoint replay"))
            st.code("uv run --extra gradio python -m portfolio.gradio_app\nuv run portfolio demo foundation", language="powershell")
        st.divider()
        st.caption("Portfolio engineering lab · inspect the validation report and deployment contract before describing it as a deployed production service.")
    except (ValueError, httpx.HTTPError) as error:
        st.error(str(error)[:500])
        st.caption("Start the backend with: uv run uvicorn portfolio.api:app --host 127.0.0.1 --port 8000")
