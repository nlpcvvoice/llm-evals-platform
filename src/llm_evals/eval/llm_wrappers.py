"""LLM wrappers used as judges for metric-style evaluators.

All judges go through OpenRouter free models (cost control, see
``config.assert_free_only``).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llm_evals.config import OpenRouterConfig, get_openrouter_config


def build_chat_client(cfg: OpenRouterConfig | None = None) -> ChatOpenAI:
    """A ChatOpenAI client pointed at OpenRouter (primary free model)."""
    cfg = cfg or get_openrouter_config()
    cfg.assert_free_only()
    return _client_for(cfg, cfg.default_model)


def build_judge_client(cfg: OpenRouterConfig | None = None) -> ChatOpenAI:
    """A ChatOpenAI judge (structured-output-friendly free model).

    Judges are chosen separately from the chat fallback chain: RAGAS
    needs reliable JSON-schema adherence (see reports/20260905_2305).
    """
    cfg = cfg or get_openrouter_config()
    cfg.assert_free_only()
    return _client_for(cfg, cfg.judge_model)


def _client_for(cfg: OpenRouterConfig, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(cfg.api_key),
        base_url=cfg.base_url,
        temperature=0,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
