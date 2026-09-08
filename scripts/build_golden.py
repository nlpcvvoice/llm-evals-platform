"""Generate the E2.0 golden benchmark (validated, layered QA set).

This is an authoring script: content is hand-curated, written to a JSONL
golden file so it diffs cleanly in git. Run:
    python scripts/build_golden.py
"""
from __future__ import annotations

from pathlib import Path

from llm_evals.eval.data import QAItem, as_dict
from llm_evals.eval.dataset import GoldenSet

OUT = Path("data/golden/benchmark_v1.jsonl")

ITEMS: list[QAItem] = [
    # ── RAG (10) ──────────────────────────────────────────────────────
    QAItem("rag-001", "What is the main benefit of retrieval-augmented generation over a plain LLM?",
           "RAG grounds answers in retrieved evidence, reducing hallucination and allowing source citations, while keeping the LLM on a trustable memory set.",
           ["RAG retrieves relevant documents and supplies them as context to the LLM.",
            "Grounded answers are consistent with the provided evidence."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "basic"}),
    QAItem("rag-002", "How does chunk size trade off retrieval precision and answer quality?",
           "Small chunks improve retrieval precision but give the model less surrounding context; large chunks give more context at the cost of noisy, less precise retrieval.",
           ["Chunking splits documents into retrievable units.",
            "Smaller chunks = precise hits + less context; larger = more context + noisier."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "intermediate"}),
    QAItem("rag-003", "What is a hybrid search and when is it preferable to vector-only search?",
           "Hybrid search combines dense vector similarity with keyword/lexical matching, so it handles rare terms, names, and exact identifiers better than vector-only search. Use it when queries contain precise tokens.",
           ["Vector search excels at semantic similarity but can miss exact tokens.",
            "Hybrid: dense + sparse/keyword ranking, helps on names, IDs, code identifiers."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "intermediate"}),
    QAItem("rag-004", "Why can a high-context recall still result in a low-quality answer?",
           "Context recall measures whether needed context was retrieved, not whether the model used it faithfully or whether retrieval brought irrelevant chunks; a faithful but irrelevant-selection answer can score fine while being useless.",
           ["Context recall measures coverage of relevant context.",
            "Faithfulness separates retrieval quality from generation quality."],
           ["recall", "tool_call"], {"source": "benchmark", "topic": "rag", "difficulty": "intermediate"}),
    QAItem("rag-005", "What is a reranker and where does it sit in the pipeline?",
           "A reranker reorders the top-N candidates after first-stage retrieval, using cross-attention over query and passage to give much better precision for the last few positions. It sits between retrieval and context injection.",
           ["First stage: cheap recall of many candidates.",
            "Reranker: expensive cross-attention re-ranking of top-N for precision.",
            "Pipeline: retrieve -> rerank -> compose context -> generate."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "intermediate"}),
    QAItem("rag-006", "How do you measure retrieval quality independently from generation quality?",
           "Use retrieval-only metrics: hit rate (is the answer in top-k), MRR (rank of first relevant hit), and nDCG (rank-weighted graded relevance). These isolate retrieval from downstream generation.",
           ["Hit rate: relevant doc present in top-k.",
            "MRR: position of first relevant doc.",
            "nDCG: rank-weighted graded relevance."],
           ["recall", "tool_call"], {"source": "benchmark", "topic": "rag", "difficulty": "advanced"}),
    QAItem("rag-007", "What causes hallucination to persist even with RAG?",
           "When retrieved context is irrelevant, the model can ignore it and fall back to parametric memory; contradictions in source documents also let the model pick wrongly. RAG reduces but cannot eliminate hallucination.",
           ["Irrelevant retrieved content weakens grounding.",
            "Contradictory sources leave room for wrong inference."],
           ["faithfulness"], {"source": "benchmark", "topic": "rag", "difficulty": "advanced"}),
    QAItem("rag-008", "When would you choose BM25 over an embedding-based retriever?",
           "Choose BM25 for exact matches, rare terms, code, or small corpora with precise vocabulary where lexical overlap is the right signal and semantic vectors add noise or cost.",
           ["BM25: sparse/lexical, strong on exact and rare tokens.",
            "Vectors: dense semantic, strong on paraphrase and synonyms."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "intermediate"}),
    QAItem("rag-009", "What does document-level deduplication prevent in a RAG corpus?",
           "Deduplication removes near-identical chunks so retrieval is not biased toward one repeated document, improves indexing cost, and prevents the same answer from being retrieved with different rankings.",
           ["Duplicate chunks skew retrieval frequency.",
            "Dedup lowers index size and retrieval noise."],
           ["faithfulness"], {"source": "benchmark", "topic": "rag", "difficulty": "basic"}),
    QAItem("rag-010", "How does context window growth change the trade-off of chunk size?",
           "Larger context windows let you inject more (and larger) chunks, relaxing the small-chunk constraint for context preservation; retrieval precision and cost become the binding constraints instead of context space.",
           ["Bigger context window fits more chunks.",
            "Precision and token cost become the real limits."],
           ["faithfulness", "recall"], {"source": "benchmark", "topic": "rag", "difficulty": "advanced"}),
    # ── AGENT (8) ─────────────────────────────────────────────────────
    QAItem("agent-001", "When should a system decide NOT to call a tool?",
           "Do not call a tool when the model can answer confidently from context/parameters, when the question is conversational, or when the tool cannot improve accuracy — unnecessary calls add latency, cost, and failure modes.",
           ["Tools externalize capabilities: search, compute, DB lookups.",
            "Unnecessary tool calls add latency, cost, failure points."],
           ["tool_call"], {"source": "benchmark", "topic": "agent", "difficulty": "intermediate"}),
    QAItem("agent-002", "What is tool-calling loop termination and why does it matter?",
           "The agent must stop when the task is complete or clearly unresolvable, preventing infinite call loops that burn tokens and cost. A max-iteration cap plus a clear done condition are the standard guardrails.",
           ["Repeated tool calls can loop forever.",
            "Cap iterations and require an explicit done condition."],
           ["tool_call", "multi_turn"], {"source": "benchmark", "topic": "agent", "difficulty": "basic"}),
    QAItem("agent-003", "How would you evaluate whether the right tool was called, not just whether the final answer is right?",
           "Use a tool-call trace evaluator: check tool name, argument validity, and necessity against the reference. Evaluate per-step, not only the final answer, to catch wrong-but-lucky trajectories.",
           ["Evaluate per-step tool/arg/call-necessity.",
            "Final-answer-only evaluation misses wrong-but-lucky paths."],
           ["tool_call", "multi_turn"], {"source": "benchmark", "topic": "agent", "difficulty": "advanced"}),
    QAItem("agent-004", "What is the failure mode of an agent that over-tools (calls tools excessively)?",
           "Over-tooling adds latency, cost, and compounding failure points, and can derail the answer when tool output is noisy; it hides a model confidence problem behind unnecessary actions.",
           ["Each tool call = latency + cost + potential failure.",
            "Over-tooling often masks low model confidence."],
           ["tool_call"], {"source": "benchmark", "topic": "agent", "difficulty": "intermediate"}),
    QAItem("agent-005", "Why must tool-calling systems have observability of the intermediate steps?",
           "Without step-level traces you cannot debug a bad trajectory (wrong tool, bad arg) or reproduce a failure; observability of each call makes evaluation and root-cause analysis possible.",
           ["Debugging needs per-step traces.",
            "Reproduction and eval need recorded intermediate state."],
           ["tool_call", "multi_turn"], {"source": "benchmark", "topic": "agent", "difficulty": "basic"}),
    QAItem("agent-006", "What distinguishes an agent from a single LLM call?",
           "An agent loops: observe -> decide -> act (tool call) -> observe result, until done. A single LLM call has no action loop and cannot use external tools.",
           ["Agent: observe-decide-act loop until done.",
            "Single call: one inference, no tools."],
           ["tool_call"], {"source": "benchmark", "topic": "agent", "difficulty": "basic"}),
    QAItem("agent-007", "When is multi-turn context relevant to evaluation?", 
           "Multi-turn evals matter when the dialogue accumulates facts, cancels earlier assumptions, or changes intent across turns; a turn-by-turn judge must credit correct use of the conversation, not only the latest utterance.",
           ["Multi-turn: accumulated facts, intent change.",
            "Judge must consider context across turns."],
           ["multi_turn"], {"source": "benchmark", "topic": "agent", "difficulty": "advanced"}),
    QAItem("agent-008", "How do you prevent a tool argument injection attack?",
           "Treat tool arguments as untrusted data: validate against schema, never concatenate user content into shell or SQL, and apply allow-lists for the tools the model may invoke from untrusted input.",
           ["Validate tool args against schema.",
            "No raw interpolation of user content into shell/SQL."],
           ["tool_call", "multi_turn"], {"source": "benchmark", "topic": "agent", "difficulty": "advanced"}),
    # ── EVALS (7) ─────────────────────────────────────────────────────
    QAItem("eval-001", "What is the difference between a golden dataset and ad-hoc test questions?",
           "A golden dataset is versioned, reviewed, covers expected evals, and is the single truth source for regression — ad-hoc questions are unmaintained and cannot protect against regressions.",
           ["Golden set: versioned, curated, maintained.",
            "Ad-hoc tests: not tracked, cannot gate regressions."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "basic"}),
    QAItem("eval-002", "Why round against a baseline (previous release) instead of an absolute threshold?",
           "LLM scores are noisy, so a single absolute threshold produces flaky gates; comparing mean score to a prior baseline with a tolerance band flags true regressions rather than model noise.",
           ["LLM evals have high variance.",
            "Baseline comparison with tolerance catches real regressions."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "advanced"}),
    QAItem("eval-003", "What is LLM-as-judge bias and how do you mitigate it?",
           "Judges can be biased by position (prefer first/last), length (prefer long answers), or self-preference. Mitigate by referencing order swapping, length control in prompts, and calibrating judge agreement (kappa) before trusting it as a gate.",
           ["Position/length/self-preference biases.",
            "Calibrate with kappa; swap order; control length."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "advanced"}),
    QAItem("eval-004", "Why use a cheap deterministic evaluator before an LLM judge?",
           "Deterministic checks (non-empty, length, keyword coverage) run free, instantly, on 100% of traffic; the LLM judge is slow and costs tokens, so reserving it for samples or border cases keeps cost and latency low.",
           ["Heuristics: free, instant, 100% coverage.",
            "Judge: slow, costly, sampled or reserved for edges."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "intermediate"}),
    QAItem("eval-005", "What should a regression gate report to the developer on failure?",
           "It must report which metric dropped, by how much vs baseline, and which golden items regressed, linking to the diff — otherwise a failing gate tells you nothing actionable.",
           ["Metric, delta vs baseline, failing items.",
            "Actionable = locate the regressed cases."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "basic"}),
    QAItem("eval-006", "How often should a golden dataset be refreshed and why?",
           "Quarterly: sample real production traffic into the set and retire obsolete scenarios, so the set tracks real user behavior instead of drifting into stale coverage.",
           ["Refresh samples production traffic.",
            "Obsolete scenarios must be retired."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "intermediate"}),
    QAItem("eval-007", "What is the danger of leaking golden items into training data?",
           "Leaked items inflate scores without real improvement (model memorized the answer); detect via high similarity to training data and treat suspiciously high scores as a red flag.",
           ["Leak = memorized answers inflate scores.",
            "Watch unusually high scores; check similarity."],
           ["heuristic"], {"source": "benchmark", "topic": "evals", "difficulty": "advanced"}),
    # ── LLM ENGINEERING (5) ──────────────────────────────────────────
    QAItem("eng-001", "Why pin `:free` model routing instead of OpenRouter's `openrouter/free` router?",
           "A fixed list of pinned free models keeps behavior consistent and lets you fallback-order known-good providers; the auto router can change composition and stability without notice, making your eval results non-reproducible.",
           ["Pinned models = reproducible evals.",
            "Auto router changes under you silently."],
           ["faithfulness"], {"source": "benchmark", "topic": "eng", "difficulty": "intermediate"}),
    QAItem("eng-002", "What is the standard way to protect an API key in a CI system?",
           "Store it as a repository secret in the CI platform, inject it as an environment variable only in the job, never commit/print it — the platform masks its value in logs.",
           ["Secrets stored encrypted by CI platform.",
            "Only referenced as env, masked in logs."],
           ["heuristic"], {"source": "benchmark", "topic": "eng", "difficulty": "intermediate"}),
    QAItem("eng-003", "How do you perform graceful degradation when the primary model is unavailable?",
           "Rotate through a fallback list ordered by preference or cost; log the rotation reason without secrets so you can detect and fix the primary upstream. The answer path should never fail if any fallback works.",
           ["Ordered fallback list.",
            "Log rotation reason, no secrets."],
           ["faithfulness", "multi_turn"], {"source": "benchmark", "topic": "eng", "difficulty": "basic"}),
    QAItem("eng-004", "Why keep model choice separate from application logic?",
           "Swapping models then becomes a config change, not a code change; a provider abstraction allows A/B testing models and cost/latency control without touching the prompt or call sites.",
           ["Config-driven model choice.",
            "Enables A/B and cost/latency tuning."],
           ["faithfulness", "multi_turn"], {"source": "benchmark", "topic": "eng", "difficulty": "basic"}),
    QAItem("eng-005", "How do you cap runaway LLM cost in a free/finantially constrained deployment?",
           "Use free-only model policy enforcement, per-request timeout and token cap, call sampling for expensive judges, and a max-iteration guard on agent loops.",
           [".free-only policy enforced at config.",
            "Timeout, token caps, judge sampling, loop caps."],
           ["tool_call", "multi_turn"], {"source": "benchmark", "topic": "eng", "difficulty": "intermediate"}),
]


def main() -> None:
    gs = GoldenSet()
    for item in ITEMS:
        gs.add(item)
    gs.raise_if_invalid()
    gs.to_jsonl(OUT)
    stats = gs.stats()
    print(f"Wrote {stats['total']} items -> {OUT}")
    print(f"  with context : {stats['with_context']}")
    print(f"  eval kinds   : {sorted(stats['evals'])}")
    by_topic: dict[str, int] = {}
    for it in ITEMS:
        by_topic[it.metadata["topic"]] = by_topic.get(it.metadata["topic"], 0) + 1
    print(f"  topics       : {by_topic}")


if __name__ == "__main__":
    main()