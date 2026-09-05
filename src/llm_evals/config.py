"""Environment/config loading.

Security rules (see OPENROUTER_GUIDE_FOR_AGENTS.md):
- The API key lives in a shared .env OUTSIDE this repo.
- Never print/log/commit the key value. Only its presence (bool).

Architecture choices:
- The project's local `.env` holds ONLY non-sensitive config
  (including ``SHARED_ENV_PATH`` pointing at the shared secret file).
- Only reads `OPENROUTER_*` vars. No OpenAI-style keys are used.
- enforces a `:free`-suffix rule so every model used is zero-cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"


@dataclass(frozen=True)
class OpenRouterConfig:
    """Resolved OpenRouter settings loaded from the shared .env."""

    api_key: str
    default_model: str
    fallback_models: list[str]
    judge_model: str = "nemotron-3-super-120b-a12b:free"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 120
    max_retries: int = 3

    @property
    def all_models(self) -> list[str]:
        """Primary model first, then fallbacks."""
        return [self.default_model, *self.fallback_models]

    @property
    def all_ctrl_models(self) -> list[str]:
        """Every model this config can control (chat + judge)."""
        return [*self.all_models, self.judge_model]

    def assert_free_only(self) -> None:
        """Guard: refuse any non-`:free` model (cost safety)."""
        banned = [m for m in self.all_ctrl_models if not m.endswith(":free")]
        if banned:
            raise ValueError(f"Non-free models are not allowed: {banned}")


def load_project_env() -> None:
    """Load the repo-local non-sensitive `.env` (SHARED_ENV_PATH etc.)."""
    load_dotenv(dotenv_path=_PROJECT_ENV, override=False)


def load_shared_env() -> None:
    """Load the shared `.env` (contains the secret key) via python-dotenv."""
    load_project_env()
    shared_path = os.environ.get("SHARED_ENV_PATH", "")
    if shared_path and Path(shared_path).exists():
        load_dotenv(dotenv_path=shared_path, override=True)


def get_openrouter_config() -> OpenRouterConfig:
    """Build the OpenRouter config from environment.

    Keys are loaded from the shared .env. Only presence of the key is
    ever exposed (never its value).
    """
    load_shared_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    default_model = os.environ.get("OPENROUTER_DEFAULT_MODEL", "minimax/minimax-m3:free").strip()
    fallback_raw = os.environ.get("OPENROUTER_FALLBACK_MODELS", "").strip()
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    cfg = OpenRouterConfig(
        api_key=api_key,
        default_model=default_model,
        fallback_models=fallback_models,
        judge_model=os.environ.get(
            "OPENROUTER_JUDGE_MODEL", "nemotron-3-super-120b-a12b:free"
        ).strip(),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        timeout=int(os.environ.get("OPENROUTER_TIMEOUT", "120")),
        max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", "3")),
    )
    cfg.assert_free_only()
    return cfg


def api_key_present() -> bool:
    """Safe check: report ONLY whether the key is set (never the value)."""
    load_shared_env()
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
