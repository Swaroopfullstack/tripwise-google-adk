"""Validated configuration and server-derived identity shared by every project."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(os.getenv("PORTFOLIO_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
Provider = Literal["google_genai", "openai", "anthropic"]


@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str
    role: str = "analyst"

    def __post_init__(self):
        if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in (self.tenant, self.user)):
            raise ValueError("Identity IDs must be bounded letters, digits, underscores or hyphens")
        if self.role not in {"analyst", "reviewer", "admin"}:
            raise ValueError("Unsupported role")


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Provider = "google_genai"
    model: str = Field(default="gemini-3.8-flash", min_length=1, max_length=120, pattern=r"^[\w./:-]+$")
    temperature: float | None = Field(default=1.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    max_tokens: int = Field(default=1600, ge=64, le=8192)

    @model_validator(mode="after")
    def compatible_options(self):
        if self.provider == "openai" and self.top_k is not None:
            raise ValueError("OpenAI chat models do not accept top_k; leave it blank when switching")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore", case_sensitive=False)
    app_env: Literal["development", "test", "production"] = "development"
    run_mode: Literal["demo", "live"] = "demo"
    data_dir: Path = ROOT / "runtime"
    llm_provider: Provider = "google_genai"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float | None = Field(default=1.0, ge=0, le=2)
    llm_top_p: float | None = Field(default=None, gt=0, le=1)
    llm_top_k: int | None = Field(default=None, ge=1, le=1000)
    llm_max_tokens: int = Field(default=1600, ge=64, le=8192)
    llm_timeout_seconds: float = Field(default=30, ge=1, le=120)
    llm_max_retries: int = Field(default=1, ge=0, le=3)
    google_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    api_identities_json: SecretStr = SecretStr("")
    mcp_http_token: SecretStr = SecretStr("")
    mcp_port: int = Field(default=8010, ge=1024, le=65535)
    api_base_url: str = "http://127.0.0.1:8000"
    embedding_backend: Literal["hashing", "huggingface"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_only: bool = False
    retrieval_k: int = Field(default=4, ge=1, le=8)
    retrieval_fetch_k: int = Field(default=16, ge=4, le=40)
    retrieval_strategy: Literal["hybrid", "vector", "mmr", "multi_query", "parent", "compression", "self_query"] = "hybrid"
    max_upload_mb: int = Field(default=8, ge=1, le=20)
    max_data_rows: int = Field(default=5000, ge=30, le=20000)
    max_data_columns: int = Field(default=50, ge=2, le=100)
    ml_timeout_seconds: int = Field(default=90, ge=5, le=300)
    requests_per_minute: int = Field(default=30, ge=1, le=300)
    # Explicitly authorized additional provider/model choices for client settings.
    allowed_models_json: str = "{}"

    @field_validator("llm_top_p", "llm_top_k", "llm_temperature", mode="before")
    @classmethod
    def blank_optional(cls, value):
        return None if value == "" else value

    @field_validator("data_dir")
    @classmethod
    def resolved_data_dir(cls, value):
        return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()

    @model_validator(mode="after")
    def validate_deployment(self):
        self.generation()
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be at least RETRIEVAL_K")
        if self.app_env == "production":
            if self.run_mode != "live" or self.embedding_backend != "huggingface":
                raise ValueError("Production requires RUN_MODE=live and EMBEDDING_BACKEND=huggingface")
            if not self.api_identities_json.get_secret_value():
                raise ValueError("Production requires explicit API_IDENTITIES_JSON credentials")
            if not self.provider_key(self.llm_provider):
                raise ValueError("Production requires the configured provider's credential")
        self.identities()
        self.allowed_models()
        return self

    def generation(self) -> Generation:
        return Generation(provider=self.llm_provider, model=self.llm_model, temperature=self.llm_temperature,
                          top_p=self.llm_top_p, top_k=self.llm_top_k, max_tokens=self.llm_max_tokens)

    def provider_key(self, provider: Provider) -> str:
        field = {"google_genai": self.google_api_key, "openai": self.openai_api_key,
                 "anthropic": self.anthropic_api_key}[provider]
        return field.get_secret_value()

    def identities(self) -> dict[str, Principal]:
        raw = self.api_identities_json.get_secret_value()
        if not raw:
            return {"demo-alice-key": Principal("demo", "alice"),
                    "demo-bob-key": Principal("demo", "bob"),
                    "demo-reviewer-key": Principal("demo", "manager", "reviewer")}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("API_IDENTITIES_JSON must be a nonempty token-to-identity object")
        result = {}
        for token, identity in parsed.items():
            if not isinstance(token, str) or len(token) < 24:
                raise ValueError("Configured credentials must contain at least 24 characters")
            result[token] = Principal(**identity)
        return result

    def allowed_models(self) -> dict[str, list[str]]:
        choices = json.loads(self.allowed_models_json)
        if not isinstance(choices, dict):
            raise ValueError("ALLOWED_MODELS_JSON must be an object")
        for provider, models in choices.items():
            if provider not in {"google_genai", "openai", "anthropic"} or not isinstance(models, list):
                raise ValueError("Unsupported provider or model list")
            if not all(isinstance(model, str) and re.fullmatch(r"[\w./:-]{1,120}", model) for model in models):
                raise ValueError("Invalid configured model ID")
        choices.setdefault(self.llm_provider, [])
        if self.llm_model not in choices[self.llm_provider]:
            choices[self.llm_provider].append(self.llm_model)
        return choices

    def validate_override(self, generation: Generation) -> Generation:
        if generation.model not in self.allowed_models().get(generation.provider, []):
            raise ValueError("That provider/model is not enabled by this deployment")
        return generation
