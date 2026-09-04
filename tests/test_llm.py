"""Tests for OpenRouter LLM rotation (all mocked, no real API calls)."""
from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from llm_evals.config import OpenRouterConfig
from llm_evals.llm import generate_with_rotation


def _mk_cfg(models: list[str]) -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key="fake-key",  # test-only value, not a real secret
        default_model=models[0],
        fallback_models=models[1:],
    )


class _InvokeResult:
    def __init__(self, text: str) -> None:
        self.content = text


def _patch_client(monkeypatch: MonkeyPatch, impl: object) -> MonkeyPatch:
    monkeypatch.setattr("llm_evals.llm._build_chat_client", lambda cfg, m: impl)
    return monkeypatch


def test_rotation_raises_when_all_fail(monkeypatch: MonkeyPatch) -> None:
    cfg = _mk_cfg(["p:free", "f:free"])
    _patch_client(monkeypatch, _RaisesEveryTime())
    with pytest.raises(RuntimeError, match="All OpenRouter free models failed"):
        generate_with_rotation("hi", config=cfg)


def test_rotation_succeeds_on_fallback(monkeypatch: MonkeyPatch) -> None:
    cfg = _mk_cfg(["p:free", "ok:free"])
    _patch_client(monkeypatch, _FailFirstThenOk())
    out = generate_with_rotation("hi", config=cfg)
    assert out == "from fallback"


def test_rotation_resets_to_primary_each_call(monkeypatch: MonkeyPatch) -> None:
    cfg = _mk_cfg(["p:free", "ok:free"])
    client = _FailFirstThenOk()
    _patch_client(monkeypatch, client)
    # First call: primary fails -> fallback ok.
    assert generate_with_rotation("hi", config=cfg) == "from fallback"
    # Second call: reset to primary -> fails again -> fallback ok.
    assert generate_with_rotation("hi", config=cfg) == "from fallback"


class _RaisesEveryTime:
    def invoke(self, prompt: str, **_: object) -> object:
        raise RuntimeError("402 Insufficient balance")


class _FailFirstThenOk:
    def __init__(self) -> None:
        self._n = 0

    def invoke(self, prompt: str, **_: object) -> object:
        self._n += 1
        if self._n == 1:
            raise RuntimeError("502 Bad Gateway")
        return _InvokeResult("from fallback")
