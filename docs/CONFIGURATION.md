# Configuration: change providers without rewriting agents

## One model factory

[models.py](../src/portfolio/models.py) is the live provider boundary. It calls `init_chat_model(model=..., model_provider=..., ...)`. FastAPI, Streamlit-driven requests, RAG, graph specialists, MCP agent clients, ADK's adapter, and Gradio obtain provider configuration here. Deterministic ML algorithms and read-only loader tools do not need an LLM call.

`.env` loads from the repository root. An explicitly exported environment variable takes precedence. `PORTFOLIO_ROOT` can point a packaged/container install to its assets and configuration root. Provider keys stay on the backend. Streamlit gets model names and allowed settings, never provider keys.

| Setting | Meaning | Default / bound |
|---|---|---|
| `RUN_MODE` | Explicit demo fixture or actual inference | `demo` / `live` |
| `LLM_PROVIDER` | LangChain provider identifier | `google_genai`; also `openai`, `anthropic` |
| `LLM_MODEL` | Provider-specific model ID | `gemini-3.8-flash`; verify account availability |
| `LLM_TEMPERATURE` | Sampling randomness | 1.0; blank omits; 0–2 allowed locally |
| `LLM_TOP_P` | Nucleus probability mass | blank omits; (0,1] |
| `LLM_TOP_K` | Number of candidate tokens in sampling | blank omits; 1–1000; OpenAI rejected |
| `LLM_MAX_TOKENS` | Maximum generated output | 1600; 64–8192 |
| `LLM_TIMEOUT_SECONDS` | Provider call timeout | 30; 1–120 |
| `LLM_MAX_RETRIES` | Provider adapter retries | 1; 0–3 |
| `EMBEDDING_BACKEND` | Retrieval representation | `hashing` offline or `huggingface` semantic |
| `EMBEDDING_MODEL` | HF model identity and index namespace | all-MiniLM-L6-v2 |
| `EMBEDDING_LOCAL_ONLY` | Require cached model files, no Hub lookups | false; set true after warming weights |
| `RETRIEVAL_K` | Evidence documents returned | 4; 1–8 |
| `RETRIEVAL_FETCH_K` | Candidate count before diversity/ranking | 16; at least k, at most 40 |
| `RETRIEVAL_STRATEGY` | Retrieval algorithm | `hybrid` |
| `MAX_UPLOAD_MB` | Request file bytes | 8; at most 20 |
| `MAX_DATA_ROWS` / `MAX_DATA_COLUMNS` | CSV bounds | 5000 / 50 |
| `ML_TIMEOUT_SECONDS` | Parent terminates training worker | 90 |
| `REQUESTS_PER_MINUTE` | Process-local per-user admission | 30 |
| `DATA_DIR` | SQLite/Chroma durable root | `runtime` |

**Generation top-k is not retrieval k.** Changing sampling k affects token selection at a provider; changing retrieval k changes the number of evidence chunks given to the answerer. MMR fetch-k controls its candidate pool. Keep these experiments separate.

Provider capabilities differ by model and version. Blank optional sampling settings are omitted. OpenAI `top_k` fails validation. Google and Anthropic model-specific rejections are returned as failures; controls are never silently dropped. Some reasoning models restrict temperature/top-p too. Start with defaults, test one parameter at a time, and retain a passing evaluation before changing a production default. A configured model is not a promise of availability.

## Switch to another provider

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key
LLM_TOP_K=
```

Or set `LLM_PROVIDER=anthropic`, your account's model ID and `ANTHROPIC_API_KEY`. When enabling UI choices, configure an explicit server allowlist:

```dotenv
ALLOWED_MODELS_JSON={"google_genai":["gemini-3.8-flash"],"openai":["gpt-4o-mini"],"anthropic":["claude-sonnet-4-6"]}
```

The deployment's default is automatically included. The server rejects choices outside this list. Set a credential for each provider that you enable. For ADK, the adapter deliberately supports **text and function calls**, rejecting unsupported multimodal/schema inputs. ADK uses the same default configuration; its local operator CLI does not expose arbitrary per-request provider changes.

## Identities and production mode

```dotenv
APP_ENV=production
RUN_MODE=live
EMBEDDING_BACKEND=huggingface
API_IDENTITIES_JSON={"replace-with-random-analyst-secret-24-plus-chars":{"tenant":"acme","user":"alice","role":"analyst"},"replace-with-random-reviewer-secret-24-plus-chars":{"tenant":"acme","user":"manager","role":"reviewer"}}
MCP_HTTP_TOKEN=replace-with-separate-random-protocol-secret
```

Generate random values with a password manager or `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. Do not use the illustrative values above. Token lookup establishes tenant/user/role; caller bodies cannot supply authority. Put real secrets in your hosting secret store. Rotation requires replacing the configuration and restarting. Static bearer identities are intentionally simple; an organizational deployment should map verified OIDC claims to the same Principal boundary.

Production refuses demo mode, hash embeddings, absent API identities and an absent default provider key. These startup checks establish configuration requirements; they do not run paid inference or verify live-model behavior.

## Memory, indexes and cache invalidation

Business records/history/preferences are in `app.sqlite`; graph checkpoints are in `checkpoints.sqlite`; ADK sessions and A2A tasks have their own databases. Chroma vectors are under the same data root. An embedding model/backend change selects a different collection; re-ingest documents. RAG cache keys include tenant, owner, session/history, index chunk IDs, prompt version, embedding identity, question, strategy, generation options and run mode. The cache is bounded and expires after 120 seconds; it is process-local, exact, and not a semantic cache.

## Request budgets

HTTP workflows reserve approximate input+maximum output tokens before model calls, at most 60,000 total and 12 calls. Inputs also have 9,000 approximate-token/36 KB limits per structured call. Provider SDK retries are bounded; workflow graph recursion is 20. The shared LLM service admits four simultaneous calls. Training admits one child process, indexing one operation, and repeated work on the same session conflicts. The 150-second budget prevents starting more model calls after the deadline; it does not forcibly stop a synchronous provider call already executing. That call has its own timeout/retries. ADK has a separate 12-call/180-second outer limit. A paid API account should also enforce its own project-level monetary quotas.

Usage metadata is logged without prompts or credentials when returned by the provider. Approximate reservations are not a billing ledger. No hard dollar-price estimate is embedded because provider rates and models change.
