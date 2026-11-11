FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM python:3.13-slim-bookworm
COPY --from=uv /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 UV_LINK_MODE=copy PORTFOLIO_ROOT=/app HF_HOME=/app/model-cache
WORKDIR /app
RUN groupadd --gid 10001 portfolio && useradd --uid 10001 --gid 10001 --create-home portfolio
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra embeddings --extra adk --extra gradio --no-install-project
COPY src ./src
COPY app.py demo.py ./
COPY agents ./agents
COPY assets ./assets
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev --extra embeddings --extra adk --extra gradio && mkdir -p /app/runtime /app/model-cache && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live',timeout=3)"]
CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "portfolio.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
