"""Core offline/online evaluation modules."""

from llm_evals.eval.data import QAItem, as_dict, from_dict
from llm_evals.eval.dataset import DatasetError, GoldenSet
from llm_evals.eval.heuristic import (
    DEFAULT_HEURISTICS,
    AnswerNonEmpty,
    AnswerWithinLength,
    EvalResult,
    KeywordRecall,
    run_heuristics,
)
from llm_evals.eval.ragas_eval import RAGAS_METRICS, RagasEvaluator, build_ragas_evaluators
from llm_evals.eval.runner import EvalReport, EvalRunner, EvalSummary
from llm_evals.eval.validation import valid_evals, validate_item

__all__ = [
    "QAItem",
    "as_dict",
    "from_dict",
    "DatasetError",
    "GoldenSet",
    "DEFAULT_HEURISTICS",
    "AnswerNonEmpty",
    "AnswerWithinLength",
    "KeywordRecall",
    "EvalResult",
    "run_heuristics",
    "RAGAS_METRICS",
    "RagasEvaluator",
    "build_ragas_evaluators",
    "EvalReport",
    "EvalRunner",
    "EvalSummary",
    "valid_evals",
    "validate_item",
]
