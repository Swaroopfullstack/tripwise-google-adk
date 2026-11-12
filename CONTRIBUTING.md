# Contributing

This project is open source under the [MIT license](LICENSE). Issues and pull requests are welcome.

## Run locally

Use Python 3.11–3.13 and uv, from the repository root:

```powershell
uv sync --frozen --extra adk --extra gradio
Copy-Item .env.example .env
uv run portfolio doctor
uv run ruff check src tests
uv run pytest -q
```

On macOS/Linux, use `cp .env.example .env`. The default demo uses local fixtures;
hosted-model credentials and paid inference are optional. Semantic embeddings need
the `embeddings` extra and a model download; see [configuration](docs/CONFIGURATION.md).

## Submit a change

1. Fork the repository and create a branch with a focused change.
2. Describe the behavior, reproduction steps and expected result in your issue or PR.
3. Add a regression check for a bug or a meaningful behavior change. Document whether
   verification used a fixture, a live integration or a proposed extension.
4. Run the relevant checks. Keep the dependency lock when changing dependencies.
5. Retain copyright, license notices and [public-data provenance](assets/public/PROVENANCE.json).

Do not include private API keys, bearer credentials, uploaded user data, runtime
databases or a real `.env` in a contribution. Use the tracked example templates and
redacted reproduction data. Contributions to the original code are under the same
MIT license. Dependencies, datasets and model weights retain their own licenses;
the repository's MIT license does not relicense those materials.
