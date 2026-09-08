"""CLI entrypoint: inspect, validate, and run evals on golden datasets.

Usage:
    python -m llm_evals.eval                 # validate default seed set
    python -m llm_evals.eval <path.jsonl>    # validate a specific set
    python -m llm_evals.eval --run <path.jsonl>        # heuristic evals
    python -m llm_evals.eval --ragas <path.jsonl>      # RAGAS metrics (LLM judge)
    python -m llm_evals.eval --judge <path.jsonl>      # self-built LLM-as-judge
    python -m llm_evals.eval --ragas 5 <path.jsonl>    # sample 5 items only
    python -m llm_evals.eval --judge 5 <path.jsonl>    # sample 5 items only
"""

from __future__ import annotations

import sys

from llm_evals.eval.dataset import DatasetError, GoldenSet
from llm_evals.eval.judge import DEFAULT_JUDGE
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


def _run_with(runner: EvalRunner, path: str, label: str, sample: int | None = None) -> int:
    gs = _sampled(_validate(path), sample)
    report = runner.run(gs, answer_provider=lambda item: item.reference_answer)
    print(f"{label} ran {report.total_eval_count} evals on {report.dataset_size} items")
    print(f"  pass rate : {report.pass_rate:.1%} ({report.passed_count}/{report.total_eval_count})")
    for summary in report.by_eval.values():
        print(f"  {summary.name:24s} avg={summary.avg_score:.3f} "
              f"pass={summary.passed}/{summary.total}")
    return 0


def _sampled(gs: GoldenSet, sample: int | None) -> GoldenSet:
    if sample is None:
        return gs
    sampled = GoldenSet()
    for item in gs.sample(sample):
        sampled.add(item)
    return sampled


def _run(path: str) -> int:
    return _run_with(EvalRunner(), path, "Heuristic")


def _ragas(path: str, sample: int | None = None) -> int:
    return _run_with(EvalRunner(build_ragas_evaluators()), path, "RAGAS", sample)


def _judge(path: str, sample: int | None = None) -> int:
    return _run_with(EvalRunner(DEFAULT_JUDGE), path, "Judge", sample)


def _split_path_and_sample(
    args: list[str],
) -> tuple[str, int | None]:
    """From `[path]` or `[N, path]`, return (path, sample_count|None)."""
    if len(args) == 2 and args[0].isdigit():
        return args[1], int(args[0])
    return (args[0] if args else DEFAULT), None


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _validate(DEFAULT)
        return 0
    if args[0] in ("--run", "--ragas", "--judge"):
        path, sample = _split_path_and_sample(args[1:])
        if args[0] == "--run":
            return _run(path)
        if args[0] == "--ragas":
            return _ragas(path, sample)
        return _judge(path, sample)
    try:
        _validate(args[0])
        return 0
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
