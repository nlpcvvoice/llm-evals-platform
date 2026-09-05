"""EvalRunner: run evaluators over a whole dataset, aggregate results.

The runner is evaluator-agnostic. Anything implementing the `Evaluator`
protocol (heuristic scors, RAGAS wrappers, DeepEval wrappers, LLM-as-judge)
can be registered. This keeps new eval types drop-in without touching the
runner core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from llm_evals.eval.dataset import GoldenSet
from llm_evals.eval.heuristic import DEFAULT_HEURISTICS, EvalResult, Evaluator


@dataclass(frozen=True)
class EvalReport:
    """Aggregated outcome of one full run."""

    generated_at: str
    dataset_size: int
    total_eval_count: int
    passed_count: int
    pass_rate: float
    by_eval: dict[str, EvalSummary]


@dataclass(frozen=True)
class EvalSummary:
    """Per-metric summary over all items."""

    name: str
    avg_score: float
    passed: int
    total: int


class EvalRunner:
    """Batch-evaluates a dataset, returning an aggregated EvalReport."""

    def __init__(self, evaluators: Sequence[Evaluator] | None = None) -> None:
        self.evaluators = list(evaluators if evaluators is not None else DEFAULT_HEURISTICS)

    def run(self, dataset: GoldenSet, answer_provider) -> EvalReport:
        """Score every item in the dataset.

        `answer_provider(item) -> str` supplies the system's answer for a
        given QA item (real RAG output, mocked reply, plain translate, etc).
        """
        if dataset.size == 0:
            raise ValueError("cannot run evaluation on an empty dataset")

        results: list[EvalResult] = []
        for item in dataset.items:
            answer = answer_provider(item)
            for evaluator in self.evaluators:
                results.append(evaluator.evaluate(item, answer))

        return self._summarize(results, dataset.size)

    def _summarize(self, results: list[EvalResult], size: int) -> EvalReport:
        by_eval: dict[str, EvalSummary] = {}
        for evaluator in self.evaluators:
            evals = [r for r in results if r.eval_name == evaluator.name]
            passed = sum(1 for r in evals if r.passed)
            avg = (sum(r.score for r in evals) / len(evals)) if evals else 0.0
            by_eval[evaluator.name] = EvalSummary(
                name=evaluator.name,
                avg_score=round(avg, 4),
                passed=passed,
                total=len(evals),
            )

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        return EvalReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            dataset_size=size,
            total_eval_count=total,
            passed_count=passed_count,
            pass_rate=round(passed_count / total, 4) if total else 0.0,
            by_eval=by_eval,
        )


# Convenience import to keep external callers simple.
__all__ = ["EvalRunner", "EvalReport", "EvalSummary"]
