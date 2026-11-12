# Calling the actual API

Start the backend, then run this Python example from the repository root using `uv run python YOUR_SCRIPT.py`. The demonstration credential is for local development only.

```python
from pathlib import Path
import json
import httpx

with httpx.Client(
    base_url="http://127.0.0.1:8000",
    headers={"Authorization": "Bearer demo-alice-key"},
    timeout=150,
) as client:
    response = client.post("/v1/sessions", json={"title": "My experiment"})
    response.raise_for_status()
    session_id = response.json()["id"]

    path = Path("assets/public/iris.csv")
    response = client.post(
        f"/v1/sessions/{session_id}/uploads",
        files={"file": (path.name, path.read_bytes(), "text/csv")},
    )
    response.raise_for_status()
    dataset_id = response.json()["id"]

    response = client.post("/v1/chat", json={
        "session_id": session_id,
        "dataset_id": dataset_id,
        "message": "Train and compare models",
    })
    response.raise_for_status()
    result = response.json()
    print(result["route"], result["payload"]["held_out_test_metrics"])

    with client.stream("POST", "/v1/chat/stream", json={
        "session_id": session_id, "message": "calculate 6*7"
    }) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if line.startswith("event:"):
                print(line)
            elif line.startswith("data:"):
                print(json.loads(line.removeprefix("data:")))
```

Upload a PDF to the same session before calling `/v1/evidence/query`:

```json
{
  "session_id": "YOUR_SESSION_ID",
  "question": "What are Govern Map Measure and Manage?",
  "strategy": "hybrid",
  "k": 4,
  "fetch_k": 16,
  "generation": {
    "provider": "google_genai",
    "model": "gemini-3.8-flash",
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "max_tokens": 1600
  }
}
```

`k` and `fetch_k` are retrieval controls. Nested generation `top_k` controls token sampling where the model supports it. Omit generation to use the deployment default. A model choice must be server-allowlisted.

## Endpoint map

| Route | Purpose |
|---|---|
| GET `/v1/config` | Safe runtime settings + authenticated identity |
| GET/POST `/v1/sessions` | Owned sessions |
| GET `/v1/sessions/{id}/messages` | Owned conversation history |
| GET/POST `/v1/sessions/{id}/uploads` | Owned listing and bounded upload |
| GET `/v1/datasets/{id}/profile` | Profile an owned CSV |
| POST `/v1/chat` | Main supervisor workflow |
| POST `/v1/chat/stream` | SSE graph progress and final result |
| POST `/v1/evidence/query` | Configurable evidence-grounded Q&A |
| POST `/v1/research` | Bounded research workflow |
| GET/POST `/v1/reviews` | Reviewer queue / analyst proposal |
| GET `/v1/reviews/{id}` | Authorized review details |
| POST `/v1/reviews/{id}/decision` | Same-tenant reviewer decision + expected version |
| GET `/v1/artifacts/{id}` | Owner-only artifact bytes |

See `/docs` for the generated OpenAPI contract. Unknown body fields are rejected. Failure codes distinguish missing auth (401), forbidden role (403), missing/other-owner data (404), stale/conflicting decisions (409), typed validation (422), admission (429), availability (503) and deadline (504). Streaming failures after headers use SSE error events, so clients must handle both HTTP and stream outcomes.
