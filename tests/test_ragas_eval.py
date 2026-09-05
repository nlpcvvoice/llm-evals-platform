"""Unit tests for RAGAS evaluator wrappers (no real LLM calls in CI).

The expensive `ragas.evaluate` call is stubbed so these run offline.
Real scoring against OpenRouter free models is covered by the manual
smoke path (`python -m llm_evals.eval --ragas`), not by CI.
"""

from __future__ import annotations

import pytest

from llm_evals.eval.data import QAItem
from llm_evals.eval.ragas_eval import RagasEvaluator, build_ragas_evaluators
from llm_evals.eval.runner import EvalRunner

ITEM = QAItem(
    id="it-1",
    question="What causes auroras?",
    reference_answer="Charged solar particles collide with atmosphere, emitting light.",
    context=["Solar wind particles hit Earth's magnetic field and create glowing light."],
)


class _FakeResult:
    """Mirror of the EvaluationResult mapping access used by ragas_eval.

    Real RAGAS returns ``result[name] -> list[float]`` (one score per row).
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def __getitem__(self, key: str) -> list[float]:
        return [self._scores[key]]


def _make_fake_evaluate(scores: float | dict[str, float]):
    """Return a stub ragas.evaluate accepting a Dataset and metrics."""
    if isinstance(scores, dict):
        valued = scores
    else:
        valued = None

    def fake(dataset, metrics=None, **kwargs):
        if valued is not None:
            return _FakeResult({m.name: valued.get(m.name, 0.7) for m in (metrics or [])})
        return _FakeResult({m.name: scores for m in (metrics or [])})

    return fake


@pytest.fixture(autouse=True)
def _mock_judge_client(monkeypatch):
    """Keep unit tests offline: never construct/point at a real judge LLM."""

    class _StubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        "llm_evals.eval.ragas_eval.build_judge_client", lambda *a, **k: _StubClient()
    )


def test_ragas_metric_name():
    ev = RagasEvaluator("context_recall")
    assert ev.name == "ragas_context_recall"


def test_ragas_metric_mapping():
    from llm_evals.eval.ragas_eval import RAGAS_METRICS

    assert set(RAGAS_METRICS) == {"faithfulness", "context_recall"}


def test_ragas_invalid_metric_raises():
    with pytest.raises(ValueError, match="unsupported RAGAS metric"):
        RagasEvaluator("made_up_metric")


def test_ragas_score_above_threshold_passes(monkeypatch):
    ev = RagasEvaluator("faithfulness", threshold=0.5)
    monkeypatch.setattr("llm_evals.eval.ragas_eval.evaluate", _make_fake_evaluate(0.9))
    result = ev.evaluate(ITEM, "charged solar particles create light")
    assert result.score == pytest.approx(0.9)
    assert result.passed is True
    assert "faithfulness" in result.eval_name


def test_ragas_score_below_threshold_fails(monkeypatch):
    ev = RagasEvaluator("faithfulness", threshold=0.8)
    monkeypatch.setattr("llm_evals.eval.ragas_eval.evaluate", _make_fake_evaluate(0.3))
    result = ev.evaluate(ITEM, "nonsense reply")
    assert result.score == pytest.approx(0.3)
    assert result.passed is False


def test_ragas_scores_per_metric(monkeypatch):
    scores = {"faithfulness": 0.6, "context_recall": 1.0}
    monkeypatch.setattr("llm_evals.eval.ragas_eval.evaluate", _make_fake_evaluate(scores))

    ev = RagasEvaluator("faithfulness", threshold=0.5)
    assert ev.evaluate(ITEM, "answer").passed is True

    ev2 = RagasEvaluator("context_recall", threshold=0.5)
    assert ev2.evaluate(ITEM, "answer").score == pytest.approx(1.0)


def test_build_ragas_evaluators_naming():
    evs = build_ragas_evaluators(("faithfulness", "context_recall"))
    assert [e.name for e in evs] == ["ragas_faithfulness", "ragas_context_recall"]


def test_ragas_through_eval_runner(monkeypatch):
    monkeypatch.setattr("llm_evals.eval.ragas_eval.evaluate", _make_fake_evaluate(0.75))
    dataset = _golden_set_single_item()
    runner = EvalRunner(evaluators=build_ragas_evaluators())
    report = runner.run(dataset, answer_provider=lambda item: item.reference_answer)
    assert report.total_eval_count == 2
    assert report.by_eval["ragas_faithfulness"].total == 1
    assert report.by_eval["ragas_context_recall"].total == 1


def test_ragas_dataset_contains_expected_columns(monkeypatch):
    captured: dict = {}

    def fake(dataset, metrics=None, **kwargs):
        captured["columns"] = dataset.column_names
        return _FakeResult({m.name: 0.5 for m in (metrics or [])})

    monkeypatch.setattr("llm_evals.eval.ragas_eval.evaluate", fake)
    RagasEvaluator("faithfulness").evaluate(ITEM, "some answer")
    assert set(captured["columns"]) == {"question", "answer", "contexts", "ground_truth"}


def _golden_set_single_item():
    from llm_evals.eval.dataset import GoldenSet

    gs = GoldenSet()
    gs.add(ITEM)
    return gs
