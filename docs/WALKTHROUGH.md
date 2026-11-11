# TripWise ADK walkthrough

## What to demonstrate

A real ADK Runner processes events with a durable DatabaseSessionService. An InvocationCounter BaseAgent emits state deltas, destination and budget specialists run in parallel, a writer consumes output keys, and a bounded critic/refiner loop can escalate to terminate.

### Run and continue a session

```powershell
uv run --extra adk portfolio adk --question "Plan three days in Lisbon"
uv run --extra adk portfolio adk --session COPY_SESSION_ID --question "Review the assumptions"
uv run --extra adk adk web agents --port 8011
```

The CLI prints event authors, state and a session ID. Repeat using that ID and observe the persistent runs counter. The web UI is a separate ADK developer surface; its own session/lifecycle settings should not be confused with the CLI's durable operator session.

### Model and tool boundaries

[adk_project.py](../src/portfolio/adk_project.py) supplies a BaseLlm adapter that converts ADK text/function messages to LangChain messages and delegates live calls to the same init_chat_model factory as the other projects. It rejects unsupported modalities and response schemas explicitly. The fixture model does not issue inference, but its actual function calls execute real tool logic in the ADK Runner.

estimate_budget validates a supported destination and 1–14 days, writes ToolContext state and returns an educational estimate. It is not a current hotel/flight quote. finish_review controls loop escalation without making any booking. before_model_callback redacts a narrow set of patterns and enforces input bounds; this is distinct from the operation's tool-level validation.

### What is taught here

Runner app/user/session identity; async Event iteration; output_key flow; instruction placeholders; parallel agents with distinct state keys; bounded loops; EventActions/state_delta; deterministic BaseAgent; callbacks; function calls/responses; persistence and cleanup. Session state has no global mutable user dictionary.

The locked ADK 2.14 release deprecates SequentialAgent, ParallelAgent and LoopAgent in favor of its newer Workflow API. They remain executable here to teach the supplied course's patterns, and tests capture their behavior. Before adopting a future major ADK release, migrate deliberately and replay the same workflow tests; do not suppress incompatibilities.

### Domain extensions

The original course's travel policies, employee records, booking/expense flows and manager budget rules are separate business requirements. This project does not pretend to book travel, pay reimbursements or implement the full supplied enterprise TripWise system. Use ReviewOps for the authorization/idempotency pattern, then add date schemas, employee ownership, policy versions, outbox delivery and real provider adapters as separate tested features. The original course coverage matrix is in the linked portfolio hub.
