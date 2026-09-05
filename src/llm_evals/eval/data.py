"""Data structures for golden evaluation datasets.

A golden dataset is a set of (question, reference_answer) pairs used to
measure system quality. Each item also records expected evals so the
runner knows which checks apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

EvalKind = Literal["faithfulness", "recall", "tool_call", "multi_turn", "heuristic"]


def cast_eval_kind(value: str) -> EvalKind:
    """Narrow an arbitrary string to EvalKind for type safety."""
    return cast(EvalKind, value)


@dataclass(frozen=True)
class QAItem:
    """A single golden QA pair with eval metadata."""

    id: str
    question: str
    reference_answer: str
    context: list[str] = field(default_factory=list)
    expected_evals: list[EvalKind] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def as_dict(item: QAItem) -> dict[str, object]:
    """Serialize a QAItem to a plain dict (JSON-safe)."""
    return {
        "id": item.id,
        "question": item.question,
        "reference_answer": item.reference_answer,
        "context": list(item.context),
        "expected_evals": list(item.expected_evals),
        "metadata": dict(item.metadata),
    }


def from_dict(raw: dict[str, object]) -> QAItem:
    """Rehydrate a QAItem from a plain dict (inverse of as_dict)."""
    context_raw = raw.get("context", [])
    context = [str(c) for c in context_raw] if isinstance(context_raw, list) else []
    evals_raw = raw.get("expected_evals", [])
    expected_evals = (
        [cast_eval_kind(str(e)) for e in evals_raw]
        if isinstance(evals_raw, list)
        else []
    )
    metadata_raw = raw.get("metadata", {})
    metadata_items = metadata_raw.items() if isinstance(metadata_raw, dict) else []
    return QAItem(
        id=str(raw["id"]),
        question=str(raw["question"]),
        reference_answer=str(raw["reference_answer"]),
        context=context,
        expected_evals=expected_evals,
        metadata={str(k): str(v) for k, v in metadata_items},
    )
