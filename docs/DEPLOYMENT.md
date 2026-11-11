# Deployment contract and operations

## Supported shape

One API instance, **one Uvicorn worker**, persistent local SSD/volume for `DATA_DIR`, and a Streamlit UI that talks to that API. SQLite and local Chroma are authoritative on that instance. Rate limits, per-session locks, cache, circuit breakers and semaphore admission are process-local. Starting a second worker/replica would bypass those shared-work assumptions; this configuration does not support it.

For horizontal scale, replace business SQLite with transactional shared PostgreSQL, use a managed/shared vector service and shared object storage, use a supported distributed checkpoint store, and move training to an authenticated durable job queue with distributed admission. Adding only a PostgreSQL checkpointer does not migrate business records or locks. The optional postgres dependency is an extension, not an enabled deployment mode.

## Docker

```powershell
docker compose build
docker compose up
```

The supplied Compose profile is a **local demo** with named durable volumes. It binds ports to loopback. The image installs frozen dependencies, runs as UID 10001, uses one API worker, and has an HTTP liveness probe. Streamlit uses the same image and points to the API container. Container build/run was not executed on the development host because Docker was unavailable; validate it on your Docker host before relying on it.

For live deployment, inject the production settings in [configuration](CONFIGURATION.md) using your secret manager. Terminate HTTPS at a reverse proxy/managed ingress. Configure proxy upload limits (at most the app's upload bound plus multipart overhead) and sensible request timeouts. Restrict Streamlit access using your organizational authentication or an authenticated reverse proxy. The backend still enforces per-user credentials. Do not expose development credentials or a Gradio demo publicly.

The HF download cache has a separate mounted volume. Pre-warm the configured model before accepting production traffic; runtime model downloads require network access and can delay readiness of the first RAG request. Restrict egress to the required model/provider/source hosts at your infrastructure boundary. The public loader accepts only catalog IDs and fixed source URLs; it does not proxy arbitrary user URLs.

## Managed service example

Use a Python/Docker web service with persistent disk attached to `/app/runtime`. Run the API with:

```sh
uv run --frozen uvicorn portfolio.api:app --host 0.0.0.0 --port "$PORT" --workers 1
```

Deploy Streamlit separately or with a supported sidecar, set `API_BASE_URL` to the protected backend origin, and keep the API's disk attached across releases. An ephemeral filesystem loses history, reviews, indexes and checkpoints. Managed services that cannot attach the necessary persistent disk require the shared-store redesign above. No cloud resource is provisioned by these files.

## Health and monitoring

- `/health/live`: process liveness; no paid inference.
- `/health/ready`: local store initialization/query check; returns mode/version and `inference_verified_by_probe=false`.
- HTTP logs: request ID, method/path, status and elapsed time; no question/body/credential logging.
- Model logs: provider/model, approximate reservation and provider usage metadata when available; no prompts.
- Watch 401/403, 429 admission, 503 unavailable, 504 deadlines, training timeouts, review backlog, disk growth and provider error rates.

Readiness does not test DNS/provider entitlement, embedding availability, or response quality. Add a controlled scheduled live-provider probe and an evaluated smoke set in your environment. Set provider account spending caps. Traces exported to third-party tools need explicit retention/redaction/access policies; no trace exporter is automatically enabled here.

## Back up, restore, release, roll back

1. Stop writes and shut down the API/protocol/ADK processes cleanly.
2. Copy the entire data root (business DB, WAL-related state after shutdown, checkpoints, Chroma, ADK/A2A DBs) to an encrypted versioned backup.
3. Restore to a separate data directory and start the **same pinned application version** against it.
4. Verify login, known session/history, a document question and a pending approval before accepting new writes.
5. Preserve the old data snapshot when changing schemas or embedding models. This initial release has create-if-missing schemas, not a general migration engine; write explicit migrations for future incompatible changes.

Release through CI, keep the frozen lockfile and an image digest, run a local smoke test on the built image, then promote the same image. Roll back code and its compatible data snapshot together when required. Never clear the live database to “fix” a migration. Test a restore drill before making recovery promises.

## Security and resource boundaries

Static tokens are stored in deployment secrets; rotate through a controlled restart. Owner checks run in SQL and retrieval filters. Reviewer role is server-derived. Uploads are bounded, CSV operations are allowlisted, arithmetic uses an AST interpreter, and no user-supplied Python/SQL/shell runs in the main assistant. PDF and ML tasks have parent deadlines. These are useful bounds, not an OS sandbox for arbitrary code. Untrusted hostile PDFs need additional container/process isolation and platform resource limits.

Per-user chunk quota is 3,000, per-document 1,200, and PDF extraction is bounded to 100 pages/500,000 characters. Session/upload/history storage can still grow over time: define product retention, deletion, disk quotas, encryption and backup access before accepting real customer data. A single API process is a deliberate architecture constraint, not a high-availability system.

## Required environment-specific release evidence

Run live provider tool/structured-output tests for every enabled model, real HF retrieval/no-answer quality tests, your user-auth integration checks, container smoke tests, concurrent request/load experiments, a backup/restore rehearsal and your data-retention controls. Record outcomes in a release-specific validation report. The repository's local validation is a reproducible starting point, not a substitute for these deployment checks.
