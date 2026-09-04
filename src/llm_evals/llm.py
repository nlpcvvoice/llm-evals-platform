"""OpenRouter LLM client with automatic fallback rotation.

- Uses only `:free` models (enforced by config.assert_free_only()).
- Rotates through fallbacks when the primary free endpoint fails
  (402 / 502 / quota / safety rejection) and resets to primary next call.
- Never logs secrets — only model name and rotation reason.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_openai import ChatOpenAI

from .config import OpenRouterConfig, get_openrouter_config

logger = logging.getLogger(__name__)


def _build_chat_client(cfg: OpenRouterConfig, model: str) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at OpenRouter's OpenAI-compatible API."""
    return ChatOpenAI(
        model=model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


def generate_with_rotation(
    prompt: str,
    *,
    config: OpenRouterConfig | None = None,
    models: Sequence[str] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Call the primary OpenRouter free model, rotating to fallbacks on failure.

    Args:
        prompt: The user prompt to send.
        config: Resolved OpenRouter config (defaults to env-based).
        models: Candidate models (defaults to config.all_models).
        response_format: Optional `{"type": "json_object"}` for structured output.

    Returns:
        The model-generated text content.

    Raises:
        RuntimeError: If every candidate model fails.
        ValueError: If any candidate model is not `:free` (cost safety).
    """
    cfg = config or get_openrouter_config()
    candidates = list(models) if models else cfg.all_models
    cfg.assert_free_only()

    last_err: Exception | None = None
    for model in candidates:
        try:
            client = _build_chat_client(cfg, model)
            kwargs: dict[str, Any] = {"response_format": response_format} if response_format else {}
            result = client.invoke(prompt, **kwargs)
            content = result.content if isinstance(result.content, str) else str(result.content)
            logger.info("[llm] ok model=%s key_set=%s", model, bool(cfg.api_key))
            return content
        except Exception as exc:  # noqa: BLE001 - rotation must catch all provider errors
            last_err = exc
            logger.warning(
                "[llm] rotate model=%s reason=%s key_set=%s", model, exc, bool(cfg.api_key)
            )

    raise RuntimeError(f"All OpenRouter free models failed. Last error: {last_err}") from last_err


def llm_complete(
    prompt: str,
    *,
    config: OpenRouterConfig | None = None,
    models: Sequence[str] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Lightweight convenience wrapper around generate_with_rotation."""
    return generate_with_rotation(
        prompt, config=config, models=models, response_format=response_format
    )
