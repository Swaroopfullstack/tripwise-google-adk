# Exactly what your private .env should look like

Choose a template, copy it to `.env` **on your machine**, then replace the placeholders privately. No real credential belongs in a GitHub file, chat message, screenshot or commit.

## 1. Try everything offline first

```powershell
Copy-Item .env.example .env
uv run portfolio doctor
```

This runs explicit demo model fixtures with real databases, tools and ML. No hosted LLM key is required.

## 2. Use Gemini and Hugging Face locally

```powershell
Copy-Item .env.gemini.example .env
```

Open your private `.env` in your editor, replace `REPLACE_LOCALLY_WITH_YOUR_GOOGLE_API_KEY` with your actual key, and save. The essential settings are:

```dotenv
APP_ENV=development
RUN_MODE=live
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-3.8-flash
GOOGLE_API_KEY=REPLACE_LOCALLY_WITH_YOUR_GOOGLE_API_KEY
LLM_TEMPERATURE=1.0
LLM_TOP_P=
LLM_TOP_K=
LLM_MAX_TOKENS=1600
EMBEDDING_BACKEND=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=4
RETRIEVAL_FETCH_K=16
```

Use a model enabled for your provider account; the string is configurable. Restart the backend after changing settings. The initial HF run downloads model weights. After warming the deployment cache, `EMBEDDING_LOCAL_ONLY=true` requires cached files and avoids Hub lookups. Re-upload documents after changing the embedding backend/model.

Blank top-p/top-k means use provider defaults. Sampling `LLM_TOP_K` affects candidate tokens where supported. `RETRIEVAL_K` affects evidence chunks. OpenAI chat configuration rejects sampling top-k; leave it blank when switching providers.

## 3. Production

Use [.env.production.example](../.env.production.example) as the variable list. In a hosted deployment, populate your secret manager rather than committing a production `.env` file. Replace **every** placeholder with a distinct appropriate value. Do not use the sample analyst/reviewer/protocol strings as real passwords.

Generate independent random application/protocol credentials locally:

```powershell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Run it separately for each credential, then store those values privately. The Google API key comes from your provider account; generating a random string does not create a provider key. `API_IDENTITIES_JSON` maps your application bearer tokens to trusted tenant/user/role identities. Reviewers need a separate role/credential. The Streamlit user enters an application token, **not** the Google API key.

The sample production URL `http://api:8000` is for the private Docker network. Use your real protected backend origin for other hosting. Read the [deployment contract](DEPLOYMENT.md): production startup checks do not perform cloud deployment or live-model evaluation.

## Before a commit

```powershell
git check-ignore .env
git status --short
git diff --cached --name-only
```

The three `*.example` templates are safe to publish because they contain blanks/placeholders only. Private `.env` variants, key files, OAuth credentials, runtime databases, user uploads and caches are excluded. A public example must stay an example after publication. If a real key is ever exposed, revoke/rotate it at the provider; deleting a file alone does not remove it from Git history.
