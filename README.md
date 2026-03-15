# ThinkBudget

A proxy for self-hosted LLMs that caps how much the model thinks per query.

Reasoning models burn thousands of tokens on internal monologue before answering. A greeting gets the same GPU time as a proof. ThinkBudget fixes that.

```
"What is 2+2?"     → TRIVIAL  →   32 thinking tokens  → $0.000002
"Debug this race    → COMPLEX  → 2048 thinking tokens  → $0.000180
 condition..."
```

It classifies the query, sets a token budget, forwards to your backend, and tracks GPU cost. No LLM call for classification. Under a millisecond.

## The problem

DeepSeek-R1, Qwen-QwQ, and similar models wrap their reasoning in `<think>` tags. Without a budget, every query gets the same treatment. "Hello" triggers 4,000 tokens of internal deliberation. You pay for all of it.

Several papers describe adaptive reasoning ([Ares](https://arxiv.org/abs/2603.07915), [AutoThink](https://huggingface.co/blog/codelion/autothink), [Conformal Thinking](https://arxiv.org/abs/2602.03814)). None ship a tool you can deploy.

## How it works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Application │────▶│   ThinkBudget    │────▶│  vLLM / SGLang  │
│ (any client)│◀────│   Proxy          │◀────│  / Ollama       │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Classify.** Heuristic scorer reads the query. No model call. Assigns a tier.
2. **Budget.** Sets a thinking token cap: 32, 128, 512, 2048, or 8192.
3. **Forward.** Sends the request to your backend with the cap applied.
4. **Enforce.** During streaming, injects `</think>` when the budget runs out.
5. **Track.** Samples GPU power via `pynvml`. Records cost per query in dollars and joules.

## Quick start

### Install

```bash
uv pip install thinkbudget
```

For GPU monitoring: `uv pip install thinkbudget[gpu]`

Don't have uv? [Install it](https://docs.astral.sh/uv/getting-started/installation/) or use pip: `pip install thinkbudget`

### Run

```bash
thinkbudget serve \
  --backend-url http://localhost:8000 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu-cost 0.39 \
  --gpu-name "RTX 4090"
```

Point your app at `http://localhost:9100` instead of the backend:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9100/v1", api_key="unused")

# 32-token thinking budget
client.chat.completions.create(
    model="thinkbudget",
    messages=[{"role": "user", "content": "Hello"}],
)

# 2048-token thinking budget
client.chat.completions.create(
    model="thinkbudget",
    messages=[{"role": "user", "content": "Design a distributed task queue with at-least-once delivery"}],
)
```

Every response carries cost headers:

```
X-ThinkBudget-Tier: complex
X-ThinkBudget-Budget: 2048
X-ThinkBudget-Thinking-Tokens: 1847
X-ThinkBudget-Cost: $0.00018200
X-ThinkBudget-Energy-J: 42.3100
```

### Dashboard

`http://localhost:9100/dashboard` shows a live feed: queries, tiers, budgets, tokens used, GPU power, cost per query, cumulative savings.

### Classify only

```bash
thinkbudget classify "What is the capital of France?"
# Tier: SIMPLE | Budget: 128 tokens

thinkbudget classify "Prove the halting problem is undecidable"
# Tier: COMPLEX | Budget: 2,048 tokens
```

## Tiers

| Tier | Budget | Example |
|------|--------|---------|
| TRIVIAL | 32 | "Hello" |
| SIMPLE | 128 | "What is the capital of France?" |
| MODERATE | 512 | "Compare REST vs GraphQL" |
| COMPLEX | 2,048 | "Debug this race condition" |
| DEEP | 8,192 | "Prove the halting problem is undecidable" |

All budgets are configurable. Per tier, per user, per session.

## Deploy on CloudRift

Built for [CloudRift](https://cloudrift.ai) GPU instances.

```bash
git clone https://github.com/Siddhant-K-code/ThinkBudget
cd ThinkBudget
./cloudrift/run.sh
```

This starts vLLM with DeepSeek-R1-Distill-Qwen-7B and ThinkBudget in front of it. One command.

For a different model or GPU:

```bash
MODEL=Qwen/QwQ-32B GPU_COST=0.65 GPU_NAME="RTX 5090" ./cloudrift/run.sh
```

Docker works too:

```bash
docker build -t thinkbudget .
docker run --gpus all -p 9100:9100 -p 8000:8000 thinkbudget
```

## Benchmarks

85 queries across all five tiers. Run them against your backend with and without ThinkBudget:

```bash
python benchmarks/run_benchmark.py --mode compare \
  --backend-url http://localhost:8000 \
  --proxy-url http://localhost:9100 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Generate a markdown report:

```bash
python benchmarks/compare.py \
  benchmarks/results/summary_baseline_*.json \
  benchmarks/results/summary_budgeted_*.json \
  benchmarks/results/report.md
```

## API

### Proxy

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Proxies with budget control |
| `/v1/models` | GET | Lists backend models |
| `/health` | GET | Status and GPU info |

### Dashboard

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/stats` | GET | Totals and distributions |
| `/api/history` | GET | Recent queries |
| `/api/gpu` | GET | Live GPU metrics |
| `/api/classify` | POST | Classify without forwarding |

### Environment variables

| Variable | Default |
|----------|---------|
| `THINKBUDGET_BACKEND_URL` | `http://localhost:8000` |
| `THINKBUDGET_MODEL` | `default` |
| `THINKBUDGET_PORT` | `9100` |
| `THINKBUDGET_GPU_COST_PER_HOUR` | `0.39` |
| `THINKBUDGET_GPU_NAME` | `RTX 4090` |

## Structure

```
src/thinkbudget/
  classifier.py      # Scores query complexity. No LLM call.
  budget.py           # Sets and enforces token budgets.
  gpu_monitor.py      # Reads GPU power via pynvml. Computes cost.
  proxy.py            # OpenAI-compatible proxy. Streams with enforcement.
  dashboard.py        # Serves the live dashboard.
  models.py           # Data types.
  config.py           # Loads config from file or env.
  cli.py              # Entry point.

benchmarks/
  datasets/           # 85 queries, 5 tiers.
  run_benchmark.py    # Runs baseline vs budgeted.
  compare.py          # Generates comparison report.

cloudrift/
  run.sh              # One-command GPU deploy.
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Part of a stack

ThinkBudget fits between [Distill](https://github.com/Siddhant-K-code/Distill) and [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX):

```
Distill       → fewer input tokens (clean context)
ThinkBudget   → fewer thinking tokens (adaptive budget)
LLMTraceFX    → faster inference (GPU kernel profiling)
```

## License

Apache-2.0

[Siddhant Khare](https://siddhantkhare.com)
