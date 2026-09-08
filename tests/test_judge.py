"""Unit tests for the self-built LLM-as-judge evaluator.

The actual LLM call is stubbed via monkeypatch (offline CI); `_parse_score_json`
and threshold logic are unit-tested directly.
"""

from __future__ import annotations

import pytest

from llm_evals.eval.data import QAItem
from llm_evals.eval.judge import JudgeLLMEvaluator, _parse_score_json
from llm_evals.eval.runner import EvalRunner

ITEM = QAItem(
    id="j-1",
    question="What is a RAG system?",
    reference_answer="RAG combines retrieval with generation.",
    context=[],
)


def _stub_llm(raw_out: str):
    def fake(prompt: str, **kwargs):
        return raw_out

    return fake


def test_parse_score_json_bare():
    assert _parse_score_json('{"score": 8, "reasoning": "ok"}') == 8.0


def test_parse_score_json_with_prose_wrapper():
    out = 'Here is the result:\n```\n{"score": 7, "reasoning": "complete"}\n```\n'
    assert _parse_score_json(out) == 7.0


def test_parse_score_json_missing_score():
    assert _parse_score_json('{"reasoning": "no score field"}') is None


def test_parse_score_json_not_json():
    assert _parse_score_json("the answer fails to address the question") is None


def test_judge_high_score_passes(monkeypatch):
    monkeypatch.setattr("llm_evals.eval.judge.generate_with_rotation", _stub_llm('{"score": 9}'))
    result = JudgeLLMEvaluator().evaluate(ITEM, "some good answer")
    assert result.score == pytest.approx(0.9)
    assert result.passed is True


def test_judge_low_score_fails(monkeypatch):
    monkeypatch.setattr("llm_evals.eval.judge.generate_with_rotation", _stub_llm('{"score": 2}'))
    result = JudgeLLMEvaluator(threshold=0.5).evaluate(ITEM, "bad answer")
    assert result.score == pytest.approx(0.2)
    assert result.passed is False


def test_judge_unparseable_reports_failure(monkeypatch):
    monkeypatch.setattr("llm_evals.eval.judge.generate_with_rotation", _stub_llm("I cannot judge"))
    result = JudgeLLMEvaluator().evaluate(ITEM, "answer")
    assert result.passed is False
    assert result.score == 0.0
    assert "unparseable" in result.detail


def test_judge_prompt_contains_question_and_answers(monkeypatch):
    captured: dict = {}

    def fake(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return '{"score": 6}'

    monkeypatch.setattr("llm_evals.eval.judge.generate_with_rotation", fake)
    JudgeLLMEvaluator().evaluate(ITEM, "the system answer")
    assert "QUESTION: What is a RAG system?" in captured["prompt"]
    assert "SYSTEM ANSWER: the system answer" in captured["prompt"]
    assert "REFERENCE ANSWER:" in captured["prompt"]


def test_judge_through_eval_runner(monkeypatch):
    def fake(prompt: str, **kwargs):
        return '{"score": 8}'

    monkeypatch.setattr("llm_evals.eval.judge.generate_with_rotation", fake)
    gs = _single_item_set()
    report = EvalRunner(evaluators=[JudgeLLMEvaluator()]).run(
        gs, answer_provider=lambda item: item.reference_answer
    )
    assert report.by_eval["llm_judge_correctness"].avg_score == pytest.approx(0.8)


def _single_item_set():
    from llm_evals.eval.dataset import GoldenSet

    gs = GoldenSet()
    gs.add(ITEM)
    return gs
