# LLM Evals Platform

A production-grade **evaluation and monitoring platform** for RAG/Agent systems.
It gives LLM teams measurable quality gates, regression protection, and
production observability — turning a one-off LLM demo into a system you can
trust and maintain.

## Why

Most AI teams can demo an LLM once; few can prove it _stays good_. This platform
closes that gap with three layers of evaluation and a two-state monitoring loop.

## Architecture

```mermaid
flowchart TD
    subgraph OFFLINE["Offline Eval (dev)"]
        DS["Golden Dataset (200+ QA)"] --> RUN["Eval Runner"]
        RUN --> RG["RAGAS: recall@k / faithfulness"]
        RUN --> DE["DeepEval: tool-call / multi-turn"]
        RUN --> LJ["LLM-as-judge"]
        RG --> REP["Eval Report"]
        DE --> REP
        LJ --> REP
        REP --> GATE{"Regression Gate"}
        GATE -->|pass| DEPLOY["Deploy"]
        GATE -->|fail| BLOCK["Block + diff report"]
    end

    subgraph ONLINE["Online Monitor (prod)"]
        PROD["Production traces"] --> H["Heuristic evals (100%)"]
        H --> SAM["LLM-as-judge sample (10-20%)"]
        SAM --> ALERT["Alerts: drift / hallucination / cost"]
        PROD --> COST["Cost tracking"]
    end

    DEPLOY --> PROD
    PROD -->|issues| DS
```

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key (for `LLM-as-judge` and provider calls)
- LangSmith API key (optional, for tracing)

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in your keys
```

### Run tests

```bash
pytest
```

## Project Layout

```
src/llm_evals/
├── eval/        # offline evaluation (RAGAS, DeepEval, LLM-as-judge)
├── monitor/     # production monitoring & observability
└── api/         # FastAPI service
```

## Roadmap

- [ ] Golden dataset builder (200+ QA)
- [ ] Three-layer offline evaluation
- [ ] Regression gate + CI integration
- [ ] Production monitoring & drift detection
- [ ] Cost & latency tracking

## License

MIT
