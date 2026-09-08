"""Self-built LLM-as-judge evaluator.

Own-architecture evaluation (future: third-party in ragas_eval.py):
    - judge takes (question, reference_answer, system_answer)
    - outputs JSON: {"score": 0-10, "reasoning": "..."}
    - score normalized to 0-1, passed if >= threshold

Why build our own judge:
    - RAGAS judges verdicts (binary claims) inside its own prompt structure;
      ours exposes an interpretable scalar + reasoning for the eval report.
    - The free judge model (nemotron) was validated for JSON fidelity first
      (see reports/20260905_2305).
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_evals.config import OpenRouterConfig, get_openrouter_config
from llm_evals.eval.data import QAItem
from llm_evals.eval.heuristic import EvalResult, Evaluator
from llm_evals.llm import generate_with_rotation

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_PROMPT = """You are a strict evaluation judge for a RAG system.
Score how well the system ANSWER addresses the QUESTION, compared to the
REFERENCE ANSWER, on a scale from 0 (totally wrong/irrelevant) to 10 (equivalent
to the reference). Consider factual correctness, completeness, and relevance.

Return ONLY a JSON object, no prose:
{{"score": <int 0-10>, "reasoning": "<1-2 sentence justification>"}}

QUESTION: {question}
REFERENCE ANSWER: {reference_answer}
SYSTEM ANSWER: {answer}
"""


def _parse_score_json(text: str) -> float | None:
    """Extract the "score" field from the judge output (defensive)."""
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        payload: dict[str, Any] = json.loads(match.group(0))
        score = payload.get("score")
        return float(score) if score is not None else None
    except (json.JSONDecodeError, AttributeError):
        return None


class JudgeLLMEvaluator(Evaluator):
    """LLM-as-judge: a scalar correctness score with reasoning."""

    name = "llm_judge_correctness"

    def __init__(
        self,
        *,
        config: OpenRouterConfig | None = None,
        threshold: float = 0.5,
        judge_prompt: str = _JUDGE_PROMPT,
    ) -> None:
        self.threshold = threshold
        self.judge_prompt = judge_prompt
        self._config = config

    def evaluate(self, item: QAItem, answer: str) -> EvalResult:
        prompt = self.judge_prompt.format(
            question=item.question,
            reference_answer=item.reference_answer,
            answer=answer,
        )
        raw = generate_with_rotation(
            prompt,
            config=self._config,
            models=self._resolved().all_judge_models,
        )
        score = _parse_score_json(raw)
        if score is None:
            return EvalResult(
                item_id=item.id,
                eval_name=self.name,
                score=0.0,
                passed=False,
                detail=f"judge output unparseable: {raw[:80]!r}",
            )
        norm = round(score / 10.0, 4)
        return EvalResult(
            item_id=item.id,
            eval_name=self.name,
            score=norm,
            passed=norm >= self.threshold,
            detail=f"raw {score:.0f}/10 ({self.threshold:.2f} thr)",
        )

    def _resolved(self) -> OpenRouterConfig:
        return self._config or get_openrouter_config()


DEFAULT_JUDGE: list[Evaluator] = [JudgeLLMEvaluator()]
