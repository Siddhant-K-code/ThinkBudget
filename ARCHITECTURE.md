# ThinkBudget — Architecture

## What It Does

ThinkBudget is an adaptive inference budget controller for self-hosted LLMs.
It sits between your application and your inference engine (vLLM, SGLang, Ollama),
dynamically controls how much the model "thinks" per query, and tracks GPU cost
in real-time.

## The Problem

Reasoning models (DeepSeek-R1, Qwen-QwQ, Llama-3-Reasoning) generate thousands
of "thinking tokens" per query. Most queries don't need deep reasoning. Without
budget control:

- Simple queries burn 10-50x more tokens than necessary
- GPU costs scale linearly with thinking tokens
- No visibility into cost-per-query at the GPU level
- No way to set per-query or per-user compute budgets

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Application │────▶│   ThinkBudget    │────▶│  Inference       │
│ (any client)│◀────│   Proxy          │◀────│  Engine          │
└─────────────┘     │                  │     │  (vLLM/SGLang/   │
                    │  ┌────────────┐  │     │   Ollama)        │
                    │  │ Classifier │  │     └─────────────────┘
                    │  │ (query     │  │              │
                    │  │ complexity)│  │     ┌─────────────────┐
                    │  └────────────┘  │     │  GPU (RTX 4090/ │
                    │  ┌────────────┐  │     │  5090/PRO 6000) │
                    │  │ Budget     │  │     └─────────────────┘
                    │  │ Controller │  │
                    │  └────────────┘  │
                    │  ┌────────────┐  │
                    │  │ GPU Cost   │  │
                    │  │ Monitor    │  │
                    │  └────────────┘  │
                    │  ┌────────────┐  │
                    │  │ Dashboard  │  │
                    │  │ (FastAPI)  │  │
                    │  └────────────┘  │
                    └──────────────────┘
```

## Components

### 1. Query Complexity Classifier
Classifies incoming queries into complexity tiers without calling the LLM:

| Tier | Budget | Example |
|------|--------|---------|
| TRIVIAL | 32 tokens | "What's 2+2?" |
| SIMPLE | 128 tokens | "Summarize this paragraph" |
| MODERATE | 512 tokens | "Compare REST vs GraphQL" |
| COMPLEX | 2048 tokens | "Debug this race condition" |
| DEEP | 8192+ tokens | "Prove this theorem" |

Classification uses:
- Token count heuristics (short queries → lower tier)
- Keyword signals (math operators, "prove", "debug", "analyze")
- Structural analysis (code blocks, multi-part questions)
- Optional: lightweight embedding similarity to calibration set

### 2. Budget Controller
Enforces the thinking budget at the inference level:
- Sets `max_tokens` for the thinking/reasoning portion
- Implements "budget forcing" — appends stop signals when budget exceeded
- Supports per-user and per-session budgets
- Tracks budget utilization and waste

### 3. GPU Cost Monitor
Real-time GPU metrics collection:
- Power draw (watts) via `nvidia-smi` / `pynvml`
- GPU utilization %
- Memory usage
- Per-query energy cost (joules)
- Dollar cost per query (based on GPU rental rate)

### 4. OpenAI-Compatible Proxy
Drop-in replacement for any OpenAI-compatible endpoint:
- Intercepts `/v1/chat/completions`
- Classifies query → sets budget → forwards to backend → monitors GPU → returns response
- Adds cost headers to response (`X-ThinkBudget-Cost`, `X-ThinkBudget-Tier`)

### 5. Dashboard
FastAPI + HTML dashboard showing:
- Live query stream with tier, budget, actual tokens, GPU cost
- Cost savings vs. unbounded inference
- Tier distribution histogram
- GPU utilization timeline
- Cumulative cost tracker

## Benchmark Suite

Designed to run on CloudRift GPUs:

```
benchmarks/
├── datasets/
│   ├── trivial.jsonl      # 200 trivial queries
│   ├── simple.jsonl       # 200 simple queries
│   ├── moderate.jsonl     # 200 moderate queries
│   ├── complex.jsonl      # 200 complex queries
│   └── deep.jsonl         # 200 deep reasoning queries
├── run_benchmark.py       # Runs with/without ThinkBudget
├── compare.py             # Generates comparison report
└── results/               # Benchmark outputs
```

Measures:
- Tokens generated (with vs. without budget)
- Quality score (LLM-as-judge on answer correctness)
- GPU energy per query
- Dollar cost per query
- Latency (TTFT + generation time)

## File Structure

```
thinkbudget/
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
├── src/
│   └── thinkbudget/
│       ├── __init__.py
│       ├── classifier.py      # Query complexity classifier
│       ├── budget.py           # Budget controller
│       ├── gpu_monitor.py      # GPU cost tracking via pynvml
│       ├── proxy.py            # OpenAI-compatible proxy server
│       ├── dashboard.py        # FastAPI dashboard
│       ├── models.py           # Data models
│       └── config.py           # Configuration
├── benchmarks/
│   ├── datasets/
│   │   └── generate_datasets.py
│   ├── run_benchmark.py
│   └── compare.py
├── dashboard/
│   └── index.html             # Dashboard UI
├── Dockerfile                 # For CloudRift deployment
└── cloudrift/
    └── run.sh                 # CloudRift quick-start script
```
