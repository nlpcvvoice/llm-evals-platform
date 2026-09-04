"""Tests for OpenRouter config safety."""
from __future__ import annotations

import pytest

from llm_evals.config import OpenRouterConfig


def test_config_rejects_non_free_model() -> None:
    cfg = OpenRouterConfig(
        api_key="test",
        default_model="gpt-4o",
        fallback_models=["minimax/minimax-m3:free"],
    )
    with pytest.raises(ValueError, match="Non-free models"):
        cfg.assert_free_only()


def test_config_accepts_all_free() -> None:
    cfg = OpenRouterConfig(
        api_key="test",
        default_model="minimax/minimax-m3:free",
        fallback_models=["z-ai/glm-5.2:free"],
    )
    cfg.assert_free_only()  # should not raise
    assert cfg.all_models[0] == "minimax/minimax-m3:free"
    assert len(cfg.all_models) == 2


def test_all_models_orders_primary_first() -> None:
    cfg = OpenRouterConfig(
        api_key="test",
        default_model="a:free",
        fallback_models=["b:free", "c:free"],
    )
    assert cfg.all_models == ["a:free", "b:free", "c:free"]
