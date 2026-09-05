"""Tests for EvalRunner / EvalReport and heuristic evaluators."""

from __future__ import annotations

from llm_evals.eval import (
    DEFAULT_HEURISTICS,
    AnswerNonEmpty,
    EvalRunner,
    GoldenSet,
    KeywordRecall,
    QAItem,
)


def _dataset() -> GoldenSet:
    gs = GoldenSet()
    gs.add(
        QAItem(
            "h-1",
            "What is RAG?",
            "Retrieval-Augmented Generation combines retrieval with generation.",
            context=["ctx"],
        )
    )
    gs.add(
        QAItem(
            "h-2",
            "Why grounding?",
            "Grounding reduces hallucination because answers are constrained.",
            context=["ctx"],
        )
    )
    return gs


# --- heuristic evaluators ---


def test_answer_nonempty_pass():
    r = AnswerNonEmpty().evaluate(QAItem("x", "q", "a"), "some answer")
    assert r.passed
    assert r.score == 1.0


def test_answer_nonempty_fail_on_blank():
    r = AnswerNonEmpty().evaluate(QAItem("x", "q", "a"), "   ")
    assert not r.passed
    assert r.score == 0.0


def test_keyword_recall_high_when_answer_matches():
    item = QAItem(
        "x",
        "q",
        "Retrieval-Augmented Generation is a method that combines retrieval "
        "with generation.",
    )
    r = KeywordRecall().evaluate(item, item.reference_answer)
    assert r.score > 0.9
    assert r.passed


def test_keyword_recall_low_when_answer_unrelated():
    item = QAItem(
        "x",
        "q",
        "Retrieval-Augmented Generation is a method that combines retrieval "
        "with generation.",
    )
    r = KeywordRecall().evaluate(
        item, "I don't know the answer to this very complicated question."
    )
    assert r.score < 0.3
    assert not r.passed


# --- runner aggregation over dataset ---


def test_runner_with_reference_answer_provider():
    gs = _dataset()
    runner = EvalRunner(evaluators=DEFAULT_HEURISTICS)
    report = runner.run(gs, answer_provider=lambda item: item.reference_answer)

    assert report.dataset_size == 2
    assert report.total_eval_count == len(DEFAULT_HEURISTICS) * 2
    assert report.pass_rate == 1.0
    assert set(report.by_eval.keys()) == {e.name for e in DEFAULT_HEURISTICS}
    for summary in report.by_eval.values():
        assert summary.total == 2
        assert summary.avg_score > 0.5


def test_runner_reports_failures():
    gs = _dataset()
    runner = EvalRunner(evaluators=[AnswerNonEmpty()])
    report = runner.run(gs, answer_provider=lambda item: "")

    assert report.passed_count == 0
    assert report.pass_rate == 0.0


def test_runner_summary_per_metric():
    gs = _dataset()
    runner = EvalRunner(evaluators=[AnswerNonEmpty()])
    report = runner.run(gs, answer_provider=lambda item: "ok")
    summary = report.by_eval["answer_nonempty"]
    assert summary.avg_score == 1.0
    assert summary.passed == 2
    assert summary.total == 2


def test_runner_empty_dataset_raises():
    runner = EvalRunner()
    try:
        runner.run(GoldenSet(), answer_provider=lambda item: "x")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --- custom evaluator drops in without touching runner ---


def test_custom_evaluator_register():
    from llm_evals.eval.heuristic import EvalResult

    class AlwaysPass:
        name = "always_pass"

        def evaluate(self, item, answer):
            return EvalResult(item.id, self.name, 1.0, True, "always")

    gs = _dataset()
    runner = EvalRunner(evaluators=[AlwaysPass()])
    report = runner.run(gs, answer_provider=lambda item: "anything")
    assert report.by_eval["always_pass"].avg_score == 1.0
    assert report.total_eval_count == 2
