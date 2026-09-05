# LLM Evals Platform

[![CI](https://github.com/nlpcvvoice/llm-evals-platform/actions/workflows/test.yml/badge.svg)](https://github.com/nlpcvvoice/llm-evals-platform/actions/workflows/test.yml)

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
    subgraph OFFLINE[Offline Eval]
        DS[Golden Dataset 200+ QA] --> RUN[Eval Runner]
        RUN --> RG[RAGAS]
        RUN --> DE[DeepEval]
        RUN --> LJ[LLM-as-judge]
        RG --> REP[Eval Report]
        DE --> REP
        LJ --> REP
        REP --> GATE{Regression Gate}
        GATE -- pass --> DEPLOY[Deploy]
        GATE -- fail --> BLOCK[Block + diff report]
    end

    subgraph ONLINE[Online Monitor]
        PROD[Production traces] --> H[Heuristic evals 100pct]
        H --> SAM[LLM-as-judge sample 10-20pct]
        SAM --> ALERT[Alerts]
        PROD --> COST[Cost tracking]
    end

    DEPLOY --> PROD
    PROD -->|issues| DS
```

## Getting Started

### Prerequisites

- Python 3.11+
- OpenRouter API key (free models, see `config.py`)

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # non-sensitive config; API key loads from shared env
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
