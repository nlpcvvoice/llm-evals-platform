"""Golden dataset container: load, validate, sample, report.

The dataset is stored as JSONL (one QAItem per line) so it diffs cleanly
in git and is easy to hand-edit.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from llm_evals.eval.data import QAItem, as_dict, from_dict
from llm_evals.eval.validation import ValidationResult, validate_item


class DatasetError(Exception):
    """Raised when a golden dataset fails to load or validate."""


class GoldenSet:
    """An ordered, validated collection of QAItems."""

    def __init__(self) -> None:
        self._items: list[QAItem] = []

    @property
    def items(self) -> list[QAItem]:
        return list(self._items)

    @property
    def size(self) -> int:
        return len(self._items)

    def add(self, item: QAItem) -> None:
        """Add an item, rejecting duplicate IDs."""
        if self._has_id(item.id):
            raise DatasetError(f"duplicate id {item.id!r}")
        self._items.append(item)

    def _has_id(self, item_id: str) -> bool:
        return any(it.id == item_id for it in self._items)

    def validate(self) -> list[ValidationResult]:
        """Return validation results for all items (empty if all good)."""
        return [validate_item(it) for it in self._items]

    def raise_if_invalid(self) -> None:
        """Raise DatasetError if any item fails validation."""
        results = self.validate()
        failed = [r for r in results if not r.ok]
        if failed:
            detail = "; ".join(r.message for r in failed[:5])
            raise DatasetError(f"{len(failed)} invalid item(s): {detail}")

    def sample(self, k: int, *, rng: random.Random | None = None) -> list[QAItem]:
        """Sample k items (no replacement) with attributes weighted by size."""
        if k < 0:
            raise ValueError("k must be >= 0")
        if k >= self.size:
            return self.items
        return (rng or random.Random()).sample(self._items, k)

    def stats(self) -> dict[str, int | set[str]]:
        """Basic dataset statistics for reports/logs."""
        kinds: set[str] = set()
        for it in self._items:
            kinds.update(it.expected_evals)
        with_context = sum(1 for it in self._items if it.context)
        return {"total": self.size, "with_context": with_context, "evals": kinds}

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "GoldenSet":
        """Load a GoldenSet from a JSONL file, validating strictness."""
        p = Path(path)
        if not p.exists():
            raise DatasetError(f"dataset file not found: {p}")
        gs = cls()
        count = 0
        with p.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    gs.add(from_dict(raw))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise DatasetError(f"{p.name}:{line_no}: invalid item: {exc}") from exc
                count += 1
        if count == 0:
            raise DatasetError(f"{p.name}: empty dataset")
        gs.raise_if_invalid()
        return gs

    def to_jsonl(self, path: Path | str) -> None:
        """Write the set to a JSONL file (overwrites)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for it in self._items:
                fh.write(json.dumps(as_dict(it), ensure_ascii=False) + "\n")
