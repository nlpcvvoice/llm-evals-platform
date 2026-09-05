"""CLI entrypoint: inspect, validate, and run evals on golden datasets.

Usage:
    python -m llm_evals.eval                    # validate default seed set
    python -m llm_evals.eval <path.jsonl>       # validate a specific set
    python -m llm_evals.eval --run <path.jsonl> # run heuristic evals
    python -m llm_evals.eval --ragas <path.jsonl> # run RAGAS metrics (LLM judge)
"""

from __future__ import annotations

import sys

from llm_evals.eval.dataset import DatasetError, GoldenSet
from llm_evals.eval.ragas_eval import build_ragas_evaluators
from llm_evals.eval.runner import EvalRunner

DEFAULT = "data/golden/seed.jsonl"


def _validate(path: str) -> GoldenSet:
    gs = GoldenSet.from_jsonl(path)
    stats = gs.stats()
    evals: set[str] = stats["evals"]  # type: ignore[assignment]
    print(f"Loaded {stats['total']} items from {path}")
    print(f"  with context : {stats['with_context']}")
    print(f"  eval kinds   : {sorted(evals)}")
    print("  all items valid")
    return gs


def _run(path: str) -> int:
    gs = _validate(path)
    runner = EvalRunner()
    # Echo the question's own reference answer as a stand-in; a real
    # system would plug its RAG/agent output here.
    report = runner.run(gs, answer_provider=lambda item: item.reference_answer)
    print(f"Ran {report.total_eval_count} evals on {report.dataset_size} items")
    print(f"  pass rate : {report.pass_rate:.1%} ({report.passed_count}/{report.total_eval_count})")
    for summary in report.by_eval.values():
        print(f"  {summary.name:24s} avg={summary.avg_score:.3f} "
              f"pass={summary.passed}/{summary.total}")
    return 0


def _ragas(path: str) -> int:
    gs = _validate(path)
    runner = EvalRunner(build_ragas_evaluators())
    report = runner.run(gs, answer_provider=lambda item: item.reference_answer)
    print(f"RAGAS ran {report.total_eval_count} evals on {report.dataset_size} items")
    print(f"  pass rate : {report.pass_rate:.1%} ({report.passed_count}/{report.total_eval_count})")
    for summary in report.by_eval.values():
        print(f"  {summary.name:24s} avg={summary.avg_score:.3f} "
              f"pass={summary.passed}/{summary.total}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _validate(DEFAULT)
        return 0
    if args[0] == "--run":
        return _run(args[1] if len(args) > 1 else DEFAULT)
    if args[0] == "--ragas":
        return _ragas(args[1] if len(args) > 1 else DEFAULT)
    try:
        _validate(args[0])
        return 0
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
