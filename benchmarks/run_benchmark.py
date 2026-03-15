"""Benchmark runner for ThinkBudget.

Runs queries with and without ThinkBudget budget control,
measuring tokens, latency, GPU cost, and quality.

Usage:
    # Against a vLLM/SGLang backend directly (no budget):
    python run_benchmark.py --mode baseline --backend-url http://localhost:8000

    # Against ThinkBudget proxy (with budget):
    python run_benchmark.py --mode budgeted --backend-url http://localhost:9100

    # Both modes for comparison:
    python run_benchmark.py --mode compare --backend-url http://localhost:8000 --proxy-url http://localhost:9100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result for a single query."""

    query_id: str
    tier: str
    query: str
    completion_tokens: int = 0
    prompt_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    ttft_ms: float = 0
    cost_dollars: float = 0
    energy_joules: float = 0
    response_preview: str = ""
    error: str | None = None

    # ThinkBudget-specific
    assigned_tier: str = ""
    thinking_budget: int = 0
    was_truncated: bool = False


@dataclass
class BenchmarkSummary:
    """Aggregate summary of a benchmark run."""

    mode: str
    model: str
    gpu: str
    total_queries: int = 0
    total_tokens: int = 0
    total_thinking_tokens: int = 0
    total_cost: float = 0
    total_energy: float = 0
    avg_latency_ms: float = 0
    results_by_tier: dict = field(default_factory=dict)


def run_query(
    client: httpx.Client,
    url: str,
    messages: list[dict],
    model: str,
    max_tokens: int | None = None,
) -> BenchmarkResult:
    """Run a single query and collect metrics."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    start = time.time()
    try:
        resp = client.post(f"{url}/v1/chat/completions", json=body, timeout=120)
        elapsed = (time.time() - start) * 1000

        if resp.status_code != 200:
            return BenchmarkResult(
                query_id="",
                tier="",
                query="",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                latency_ms=elapsed,
            )

        data = resp.json()
        usage = data.get("usage", {})
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")

        # Parse ThinkBudget headers if present
        assigned_tier = resp.headers.get("X-ThinkBudget-Tier", "")
        thinking_budget = int(resp.headers.get("X-ThinkBudget-Budget", "0"))
        thinking_tokens_header = int(resp.headers.get("X-ThinkBudget-Thinking-Tokens", "0"))
        cost_header = resp.headers.get("X-ThinkBudget-Cost", "")
        energy_header = resp.headers.get("X-ThinkBudget-Energy-J", "")

        # Parse cost
        cost = 0.0
        if cost_header.startswith("$"):
            try:
                cost = float(cost_header[1:])
            except ValueError:
                pass

        energy = 0.0
        if energy_header:
            try:
                energy = float(energy_header)
            except ValueError:
                pass

        return BenchmarkResult(
            query_id="",
            tier="",
            query="",
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            thinking_tokens=thinking_tokens_header,
            latency_ms=elapsed,
            cost_dollars=cost,
            energy_joules=energy,
            response_preview=content[:200],
            assigned_tier=assigned_tier,
            thinking_budget=thinking_budget,
        )

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return BenchmarkResult(
            query_id="",
            tier="",
            query="",
            error=str(e),
            latency_ms=elapsed,
        )


def load_dataset(dataset_dir: str, tier: str) -> list[dict]:
    """Load a tier's dataset from JSONL."""
    filepath = Path(dataset_dir) / f"{tier}.jsonl"
    if not filepath.exists():
        print(f"  Dataset not found: {filepath}")
        return []

    records = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_benchmark(
    url: str,
    model: str,
    dataset_dir: str,
    output_dir: str,
    mode: str,
    max_queries_per_tier: int = 10,
    gpu_cost_per_hour: float = 0.39,
) -> BenchmarkSummary:
    """Run the full benchmark suite."""
    tiers = ["trivial", "simple", "moderate", "complex", "deep"]
    all_results: list[BenchmarkResult] = []

    client = httpx.Client(timeout=120)

    print(f"\n{'='*60}")
    print(f"  ThinkBudget Benchmark — {mode.upper()} mode")
    print(f"  Backend: {url}")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")

    for tier in tiers:
        dataset = load_dataset(dataset_dir, tier)
        if not dataset:
            continue

        queries = dataset[:max_queries_per_tier]
        print(f"  [{tier.upper():>8}] Running {len(queries)} queries...")

        for i, record in enumerate(queries):
            result = run_query(
                client=client,
                url=url,
                messages=record["messages"],
                model=model,
                # In baseline mode, don't set max_tokens (let model think freely)
                max_tokens=None if mode == "baseline" else None,
            )
            result.query_id = record["id"]
            result.tier = tier
            result.query = record["query"][:100]

            all_results.append(result)

            status = "✓" if not result.error else f"✗ {result.error[:50]}"
            tokens_info = f"{result.completion_tokens} tokens"
            if result.assigned_tier:
                tokens_info += f" [tier={result.assigned_tier}, budget={result.thinking_budget}]"

            print(f"    {i+1:>3}/{len(queries)} {status} — {result.latency_ms:.0f}ms, {tokens_info}")

    client.close()

    # Compute summary
    successful = [r for r in all_results if not r.error]
    summary = BenchmarkSummary(
        mode=mode,
        model=model,
        gpu=f"${gpu_cost_per_hour}/hr",
        total_queries=len(successful),
        total_tokens=sum(r.completion_tokens for r in successful),
        total_thinking_tokens=sum(r.thinking_tokens for r in successful),
        total_cost=sum(r.cost_dollars for r in successful),
        total_energy=sum(r.energy_joules for r in successful),
        avg_latency_ms=(
            sum(r.latency_ms for r in successful) / len(successful) if successful else 0
        ),
    )

    # Per-tier breakdown
    for tier in tiers:
        tier_results = [r for r in successful if r.tier == tier]
        if tier_results:
            summary.results_by_tier[tier] = {
                "count": len(tier_results),
                "avg_tokens": sum(r.completion_tokens for r in tier_results) / len(tier_results),
                "avg_latency_ms": sum(r.latency_ms for r in tier_results) / len(tier_results),
                "total_cost": sum(r.cost_dollars for r in tier_results),
            }

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = out_dir / f"benchmark_{mode}_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps({
                "query_id": r.query_id,
                "tier": r.tier,
                "query": r.query,
                "completion_tokens": r.completion_tokens,
                "thinking_tokens": r.thinking_tokens,
                "latency_ms": r.latency_ms,
                "cost_dollars": r.cost_dollars,
                "energy_joules": r.energy_joules,
                "assigned_tier": r.assigned_tier,
                "thinking_budget": r.thinking_budget,
                "error": r.error,
            }) + "\n")

    summary_file = out_dir / f"summary_{mode}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "mode": summary.mode,
            "model": summary.model,
            "gpu": summary.gpu,
            "total_queries": summary.total_queries,
            "total_tokens": summary.total_tokens,
            "total_thinking_tokens": summary.total_thinking_tokens,
            "total_cost": summary.total_cost,
            "total_energy": summary.total_energy,
            "avg_latency_ms": summary.avg_latency_ms,
            "results_by_tier": summary.results_by_tier,
        }, f, indent=2)

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  SUMMARY — {mode.upper()}")
    print(f"{'─'*60}")
    print(f"  Queries:          {summary.total_queries}")
    print(f"  Total tokens:     {summary.total_tokens:,}")
    print(f"  Thinking tokens:  {summary.total_thinking_tokens:,}")
    print(f"  Total cost:       ${summary.total_cost:.6f}")
    print(f"  Total energy:     {summary.total_energy:.2f} J")
    print(f"  Avg latency:      {summary.avg_latency_ms:.0f} ms")
    print(f"\n  Per-tier breakdown:")
    for tier, stats in summary.results_by_tier.items():
        print(f"    {tier:>10}: {stats['count']} queries, "
              f"avg {stats['avg_tokens']:.0f} tokens, "
              f"avg {stats['avg_latency_ms']:.0f}ms, "
              f"${stats['total_cost']:.6f}")
    print(f"\n  Results saved to: {results_file}")
    print(f"  Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ThinkBudget Benchmark Runner")
    parser.add_argument("--mode", choices=["baseline", "budgeted", "compare"], default="budgeted")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--proxy-url", default="http://localhost:9100")
    parser.add_argument("--model", default="default")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).parent / "datasets"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--max-per-tier", type=int, default=10)
    parser.add_argument("--gpu-cost", type=float, default=0.39)
    args = parser.parse_args()

    if args.mode == "compare":
        print("\n" + "="*60)
        print("  Running BASELINE (no budget control)...")
        print("="*60)
        baseline = run_benchmark(
            url=args.backend_url,
            model=args.model,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            mode="baseline",
            max_queries_per_tier=args.max_per_tier,
            gpu_cost_per_hour=args.gpu_cost,
        )

        print("\n" + "="*60)
        print("  Running BUDGETED (with ThinkBudget)...")
        print("="*60)
        budgeted = run_benchmark(
            url=args.proxy_url,
            model=args.model,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            mode="budgeted",
            max_queries_per_tier=args.max_per_tier,
            gpu_cost_per_hour=args.gpu_cost,
        )

        # Comparison
        print("\n" + "="*60)
        print("  COMPARISON: Baseline vs ThinkBudget")
        print("="*60)
        if baseline.total_tokens > 0:
            token_savings = (1 - budgeted.total_tokens / baseline.total_tokens) * 100
            print(f"  Token reduction:  {token_savings:.1f}%")
        if baseline.total_cost > 0:
            cost_savings = (1 - budgeted.total_cost / baseline.total_cost) * 100
            print(f"  Cost reduction:   {cost_savings:.1f}%")
        if baseline.avg_latency_ms > 0:
            latency_change = (1 - budgeted.avg_latency_ms / baseline.avg_latency_ms) * 100
            print(f"  Latency change:   {latency_change:+.1f}%")
        print()

    else:
        url = args.proxy_url if args.mode == "budgeted" else args.backend_url
        run_benchmark(
            url=url,
            model=args.model,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            max_queries_per_tier=args.max_per_tier,
            gpu_cost_per_hour=args.gpu_cost,
        )


if __name__ == "__main__":
    main()
