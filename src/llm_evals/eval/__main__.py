"""CLI entrypoint: inspect and validate golden datasets.

Usage:
    python -m llm_evals.eval <path.jsonl>
"""

from __future__ import annotations

import sys

from llm_evals.eval.dataset import DatasetError, GoldenSet

DEFAULT = "data/golden/seed.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    path = args[0] if args else DEFAULT

    try:
        gs = GoldenSet.from_jsonl(path)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stats = gs.stats()
    evals: set[str] = stats["evals"]  # type: ignore[assignment]
    print(f"Loaded {stats['total']} items from {path}")
    print(f"  with context : {stats['with_context']}")
    print(f"  eval kinds   : {sorted(evals)}")
    print("  all items valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
