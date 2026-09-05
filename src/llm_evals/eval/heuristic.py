"""Heuristic (programmatic) evaluators — no LLM calls.

These run free and fast, so they can score 100% of traffic in production
and act as a cheap gate before expensive LLM-as-judge sampling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from llm_evals.eval.data import QAItem


@dataclass(frozen=True)
class EvalResult:
    """A single scored evaluation of one QA item."""

    item_id: str
    eval_name: str
    score: float
    passed: bool
    detail: str = ""


class Evaluator(Protocol):
    """An evaluator: given the reference item and the system answer, score."""

    name: str

    def evaluate(self, item: QAItem, answer: str) -> EvalResult: ...


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


class AnswerNonEmpty(Evaluator):
    """Fail if the system returned an empty or whitespace-only answer."""

    name = "answer_nonempty"

    def evaluate(self, item: QAItem, answer: str) -> EvalResult:
        ok = bool(answer.strip())
        return EvalResult(
            item_id=item.id,
            eval_name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            detail="answer ok" if ok else "empty answer",
        )


class AnswerWithinLength(Evaluator):
    """Accept answers within a sane length band relative to the reference."""

    name = "answer_within_length"

    def __init__(self, min_ratio: float = 0.2, max_ratio: float = 3.0) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def evaluate(self, item: QAItem, answer: str) -> EvalResult:
        ref_words = max(1, len(item.reference_answer.split()))
        ans_words = len(answer.split())
        ok = self.min_ratio * ref_words <= ans_words <= self.max_ratio * ref_words
        return EvalResult(
            item_id=item.id,
            eval_name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            detail=f"answer {ans_words} vs ref_floor {int(self.min_ratio * ref_words)}",
        )


class KeywordRecall(Evaluator):
    """Fraction of reference keywords that appear in the system answer."""

    name = "keyword_recall"

    def __init__(self, stop_min_len: int = 5) -> None:
        self.stop_min_len = stop_min_len

    def evaluate(self, item: QAItem, answer: str) -> EvalResult:
        ref_words = {w for w in _words(item.reference_answer) if len(w) >= self.stop_min_len}
        if not ref_words:
            return EvalResult(
                item_id=item.id,
                eval_name=self.name,
                score=0.0,
                passed=False,
                detail="reference too short to score",
            )
        ans_words = _words(answer)
        hits = ref_words & ans_words
        recall = len(hits) / len(ref_words)
        passed = recall >= 0.3
        return EvalResult(
            item_id=item.id,
            eval_name=self.name,
            score=round(recall, 4),
            passed=passed,
            detail=f"{len(hits)}/{len(ref_words)} keywords",
        )


DEFAULT_HEURISTICS: list[Evaluator] = [
    AnswerNonEmpty(),
    AnswerWithinLength(),
    KeywordRecall(),
]


def run_heuristics(item: QAItem, answer: str) -> list[EvalResult]:
    """Run all default heuristic evals for one item."""
    return [e.evaluate(item, answer) for e in DEFAULT_HEURISTICS]
