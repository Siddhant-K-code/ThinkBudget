# ThinkBudget

**Adaptive inference budget controller for self-hosted LLMs.**

Not every query needs 8,000 thinking tokens. ThinkBudget sits between your app and your inference engine, classifies query complexity, caps reasoning effort per query, and tracks GPU cost in real-time.

```
"What is 2+2?"     → TRIVIAL  →   32 thinking tokens  → $0.000002
"Debug this race    → COMPLEX  → 2048 thinking tokens  → $0.000180
 condition..."
```

## Why

Reasoning models (DeepSeek-R1, Qwen-QwQ, Llama-3-Reasoning) generate thousands of thinking tokens per query. A simple greeting burns the same GPU time as a theorem proof. Without budget control:

- Simple queries cost 10-50x more than necessary
- No visibility into cost-per-query at the GPU level
- No way to set per-user or per-session compute budgets
- GPU costs scale linearly with thinking tokens

Papers like [Ares](https://arxiv.org/abs/2603.07915), [AutoThink](https://huggingface.co/blog/codelion/autothink), and [Conformal Thinking](https://arxiv.org/abs/2602.03814) describe adaptive reasoning — but no practical, open-source tool exists that you can deploy today.

ThinkBudget fills that gap.

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Application │────▶│   ThinkBudget    │────▶│  vLLM / SGLang  │
│ (any client)│◀────│   Proxy          │◀────│  / Ollama       │
└─────────────┘     │                  │     └────────┬────────┘
                    │  1. Classify     │              │
                    │  2. Set budget   │     ┌────────┴────────┐
                    │  3. Forward      │     │  GPU (RTX 4090/ │
                    │  4. Monitor GPU  │     │  5090/PRO 6000) │
                    │  5. Track cost   │     └─────────────────┘
                    └──────────────────┘
```

1. **Classify** — Heuristic classifier scores query complexity in <1ms (no LLM call)
2. **Budget** — Sets thinking token cap based on tier (32 → 128 → 512 → 2048 → 8192)
3. **Forward** — Proxies to your inference backend with budget-enforced parameters
4. **Monitor** — Samples GPU power/utilization via `pynvml` during inference
5. **Track** — Records per-query cost in dollars, energy in joules, tokens saved

## Quick Start

### Install

```bash
pip install thinkbudget
# With GPU monitoring:
pip install thinkbudget[gpu]
# With benchmarking tools:
pip install thinkbudget[all]
```

### Run the Proxy

```bash
# Point at your vLLM/SGLang/Ollama backend
thinkbudget serve \
  --backend-url http://localhost:8000 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu-cost 0.39 \
  --gpu-name "RTX 4090"
```

Then use `http://localhost:9100` as your OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9100/v1", api_key="unused")

# This gets a 32-token thinking budget (TRIVIAL)
response = client.chat.completions.create(
    model="thinkbudget",
    messages=[{"role": "user", "content": "Hello"}],
)

# This gets a 2048-token thinking budget (COMPLEX)
response = client.chat.completions.create(
    model="thinkbudget",
    messages=[{"role": "user", "content": "Design a distributed task queue with at-least-once delivery"}],
)
```

Response headers include cost metadata:

```
X-ThinkBudget-Tier: complex
X-ThinkBudget-Budget: 2048
X-ThinkBudget-Thinking-Tokens: 1847
X-ThinkBudget-Cost: $0.00018200
X-ThinkBudget-Energy-J: 42.3100
```

### Dashboard

Open `http://localhost:9100/dashboard` for a live view of:

- Query stream with tier, budget, actual tokens, GPU cost
- Cost savings vs. unbounded inference
- Tier distribution histogram
- GPU utilization, power, and memory
- Cumulative cost tracker

### Classify Without Proxying

```bash
thinkbudget classify "What is the capital of France?"
# ┌──────────────────────────────────────┐
# │ Tier:            SIMPLE              │
# │ Confidence:      85.0%               │
# │ Thinking Budget: 128 tokens          │
# └──────────────────────────────────────┘

thinkbudget classify "Prove the halting problem is undecidable"
# ┌──────────────────────────────────────┐
# │ Tier:            COMPLEX             │
# │ Confidence:      70.0%               │
# │ Thinking Budget: 2,048 tokens        │
# └──────────────────────────────────────┘
```

## Complexity Tiers

| Tier | Thinking Budget | Example |
|------|----------------|---------|
| **TRIVIAL** | 32 tokens | "Hello", "What is 2+2?" |
| **SIMPLE** | 128 tokens | "What is the capital of France?" |
| **MODERATE** | 512 tokens | "Compare REST vs GraphQL trade-offs" |
| **COMPLEX** | 2,048 tokens | "Debug this race condition in my Go code" |
| **DEEP** | 8,192 tokens | "Prove the halting problem is undecidable" |

Budgets are configurable per tier, per user, and per session.

## CloudRift GPU Deployment

ThinkBudget is designed to run on [CloudRift](https://cloudrift.ai) GPU instances. With $291 in credits:

| GPU | $/hr | Hours Available | Best For |
|-----|------|-----------------|----------|
| RTX 4090 | $0.39 | ~746 hrs | Training, benchmarks |
| RTX 5090 | $0.65 | ~447 hrs | Latest GPU benchmarks |
| RTX PRO 6000 | $1.29 | ~225 hrs | Large model inference |

### One-Command Deploy

```bash
# On a CloudRift GPU instance:
git clone https://github.com/Siddhant-K-code/ThinkBudget
cd ThinkBudget

# Default: DeepSeek-R1-Distill-Qwen-7B on RTX 4090
./cloudrift/run.sh

# Custom model on RTX 5090:
MODEL=Qwen/QwQ-32B GPU_COST=0.65 GPU_NAME="RTX 5090" ./cloudrift/run.sh
```

### Docker

```bash
docker build -t thinkbudget .
docker run --gpus all -p 9100:9100 -p 8000:8000 thinkbudget
```

## Benchmarks

ThinkBudget ships with 85 benchmark queries across all 5 tiers.

### Generate Datasets

```bash
cd benchmarks/datasets
python generate_datasets.py
```

### Run Comparison Benchmark

```bash
# Compare unbounded inference vs ThinkBudget-controlled inference
python benchmarks/run_benchmark.py --mode compare \
  --backend-url http://localhost:8000 \
  --proxy-url http://localhost:9100 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu-cost 0.39
```

### Generate Report

```bash
python benchmarks/compare.py \
  benchmarks/results/summary_baseline_*.json \
  benchmarks/results/summary_budgeted_*.json \
  benchmarks/results/report.md
```

## API Reference

### Proxy Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy with budget control |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check with GPU status |

### Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Aggregate statistics |
| `/api/history` | GET | Recent query history |
| `/api/gpu` | GET | Current GPU metrics |
| `/api/classify` | POST | Classify a query without forwarding |

### Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `THINKBUDGET_BACKEND_URL` | `http://localhost:8000` | Backend inference server |
| `THINKBUDGET_MODEL` | `default` | Model name |
| `THINKBUDGET_PORT` | `9100` | Proxy port |
| `THINKBUDGET_GPU_COST_PER_HOUR` | `0.39` | GPU hourly rate ($) |
| `THINKBUDGET_GPU_NAME` | `RTX 4090` | GPU name for display |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Project Structure

```
thinkbudget/
├── src/thinkbudget/
│   ├── classifier.py      # Query complexity classifier (<1ms, no LLM)
│   ├── budget.py           # Budget controller with per-user policies
│   ├── gpu_monitor.py      # Real-time GPU cost tracking via pynvml
│   ├── proxy.py            # OpenAI-compatible proxy with budget enforcement
│   ├── dashboard.py        # Live web dashboard
│   ├── models.py           # Data models
│   ├── config.py           # Configuration
│   └── cli.py              # CLI entry point
├── benchmarks/
│   ├── datasets/           # 85 queries across 5 complexity tiers
│   ├── run_benchmark.py    # Benchmark runner (baseline vs budgeted)
│   └── compare.py          # Comparison report generator
├── dashboard/
│   └── index.html          # Dashboard UI
├── cloudrift/
│   └── run.sh              # CloudRift one-command deploy
├── Dockerfile
└── pyproject.toml
```

## How It Connects

ThinkBudget is part of a broader agent reliability stack:

- **[Distill](https://github.com/Siddhant-K-code/Distill)** — Cleans what goes *in* (context deduplication)
- **ThinkBudget** — Controls how much the model *thinks* (inference budget)
- **[LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX)** — Profiles *how* inference runs (GPU kernels)

```
User Request
     ↓
┌─────────┐
│ Distill │ → Clean, deduplicated context (fewer input tokens)
└─────────┘
     ↓
┌─────────────┐
│ ThinkBudget │ → Adaptive thinking budget (fewer thinking tokens)
└─────────────┘
     ↓
┌───────────┐
│LLMTraceFX │ → GPU kernel profiling (optimize inference speed)
└───────────┘
     ↓
  Result + Cost Tracking
```

## License

Apache-2.0

## Author

[Siddhant Khare](https://siddhantkhare.com) · [hi@siddhantkhare.com](mailto:hi@siddhantkhare.com)
