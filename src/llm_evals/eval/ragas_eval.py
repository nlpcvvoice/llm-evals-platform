"""RAGAS-based evaluators integrated into the EvalRunner framework.

RAGAS scores are produced by LLM judges (OpenRouter free models via
``llm_wrappers``). Two metrics are exposed:
    - faithfulness  (answer claims are supported by context)
    - context_recall (reference answer points are covered)

These wrap RAGAS so results surface as the standard ``EvalResult`` used
by the EvalRunner and report aggregation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from datasets import Dataset  # type: ignore[import-untyped]
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_recall, faithfulness  # type: ignore[attr-defined]
from ragas.metrics.base import Metric

from llm_evals.config import OpenRouterConfig
from llm_evals.eval.data import QAItem
from llm_evals.eval.heuristic import EvalResult, Evaluator
from llm_evals.eval.llm_wrappers import build_judge_client

logger = logging.getLogger(__name__)

RAGAS_METRICS: dict[str, Metric] = {
    "faithfulness": faithfulness,
    "context_recall": context_recall,
}


class RagasEvaluator(Evaluator):
    """Score one QA item with one RAGAS metric using the OpenRouter judge."""

    def __init__(
        self,
        metric_name: str = "faithfulness",
        *,
        config: OpenRouterConfig | None = None,
        threshold: float = 0.5,
    ) -> None:
        if metric_name not in RAGAS_METRICS:
            raise ValueError(f"unsupported RAGAS metric {metric_name!r}")
        self.name = f"ragas_{metric_name}"
        self.metric = RAGAS_METRICS[metric_name]
        self.threshold = threshold
        self._config = config

    def _judge_llm(self):
        """A RAGAS-compatible judge LLM (OpenRouter free model)."""
        client = build_judge_client(self._config)
        return LangchainLLMWrapper(client)

    def evaluate(self, item: QAItem, answer: str) -> EvalResult:
        dataset = Dataset.from_dict(
            {
                "question": [item.question],
                "answer": [answer],
                "contexts": [item.context],
                "ground_truth": [item.reference_answer],
            }
        )
        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                metrics=[self.metric],
                llm=self._judge_llm(),
                show_progress=False,
            ),
        )
        score = float(result[self.name.replace("ragas_", "")][0])
        passed = score >= self.threshold
        return EvalResult(
            item_id=item.id,
            eval_name=self.name,
            score=round(score, 4),
            passed=passed,
            detail=f"{self.name} {score:.4f} (thr {self.threshold})",
        )


def build_ragas_evaluators(
    metric_names: Sequence[str] = ("faithfulness", "context_recall"),
    *,
    config: OpenRouterConfig | None = None,
    threshold: float = 0.5,
) -> Sequence[RagasEvaluator]:
    """One RagasEvaluator per requested metric (drop-in for EvalRunner)."""
    return [RagasEvaluator(m, config=config, threshold=threshold) for m in metric_names]
