"""Validation for golden dataset items.

Rules here are deliberately strict: a malformed golden subset silently
corrupts every downstream eval. Fail loudly, fail early.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_evals.eval.data import EvalKind, QAItem

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

VALID_EVALS: set[EvalKind] = {
    "faithfulness",
    "recall",
    "tool_call",
    "multi_turn",
    "heuristic",
}

valid_evals = VALID_EVALS


@dataclass
class ValidationResult:
    """Outcome of validating a single QAItem."""

    item_id: str
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def message(self) -> str:
        if self.ok:
            return f"{self.item_id}: ok"
        return f"{self.item_id}: " + "; ".join(self.errors)


def validate_item(item: QAItem) -> ValidationResult:
    """Check a QAItem against dataset rules. Returns all errors found."""
    errors: list[str] = []

    if not _ID_RE.match(item.id):
        errors.append(
            "id must match ^[a-z0-9][a-z0-9_-]{1,63}$ "
            f"(got {item.id!r})"
        )

    if not item.question.strip():
        errors.append("question must be non-empty")

    if not item.reference_answer.strip():
        errors.append("reference_answer must be non-empty")

    if len(item.question) > 2000:
        errors.append(f"question too long ({len(item.question)} > 2000)")

    if len(item.reference_answer) > 8000:
        errors.append(f"reference_answer too long ({len(item.reference_answer)} > 8000)")

    for ctx in item.context:
        if not ctx.strip():
            errors.append("context entries must be non-empty")

    for ec in item.expected_evals:
        if ec not in VALID_EVALS:
            errors.append(f"unknown eval kind {ec!r} (valid: {sorted(VALID_EVALS)})")

    return ValidationResult(item_id=item.id, errors=errors)
