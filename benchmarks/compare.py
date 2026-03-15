"""Compare benchmark results between baseline and budgeted runs.

Generates a markdown report with tables and analysis.

Usage:
    python compare.py results/summary_baseline_*.json results/summary_budgeted_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(baseline_path: str, budgeted_path: str, output_path: str | None = None):
    baseline = load_summary(baseline_path)
    budgeted = load_summary(budgeted_path)

    lines = []
    lines.append("# ThinkBudget Benchmark Report\n")
    lines.append(f"**Model:** {baseline.get('model', 'unknown')}")
    lines.append(f"**GPU:** {baseline.get('gpu', 'unknown')}\n")

    # Overall comparison
    lines.append("## Overall Comparison\n")
    lines.append("| Metric | Baseline | ThinkBudget | Change |")
    lines.append("|--------|----------|-------------|--------|")

    metrics = [
        ("Total Queries", "total_queries", "", False),
        ("Total Tokens", "total_tokens", "", False),
        ("Thinking Tokens", "total_thinking_tokens", "", False),
        ("Total Cost", "total_cost", "$", True),
        ("Total Energy (J)", "total_energy", "", True),
        ("Avg Latency (ms)", "avg_latency_ms", "", True),
    ]

    for label, key, prefix, is_float in metrics:
        bv = baseline.get(key, 0)
        tv = budgeted.get(key, 0)

        if is_float:
            bv_str = f"{prefix}{bv:.6f}" if "cost" in key.lower() else f"{bv:.2f}"
            tv_str = f"{prefix}{tv:.6f}" if "cost" in key.lower() else f"{tv:.2f}"
        else:
            bv_str = f"{bv:,}"
            tv_str = f"{tv:,}"

        if bv > 0:
            change = ((tv - bv) / bv) * 100
            change_str = f"{change:+.1f}%"
            if change < 0:
                change_str = f"**{change_str}**"  # Bold for savings
        else:
            change_str = "—"

        lines.append(f"| {label} | {bv_str} | {tv_str} | {change_str} |")

    # Per-tier comparison
    lines.append("\n## Per-Tier Breakdown\n")
    lines.append("| Tier | Baseline Tokens | Budgeted Tokens | Token Savings | Baseline Cost | Budgeted Cost | Cost Savings |")
    lines.append("|------|----------------|-----------------|---------------|---------------|---------------|--------------|")

    tiers = ["trivial", "simple", "moderate", "complex", "deep"]
    for tier in tiers:
        bt = baseline.get("results_by_tier", {}).get(tier, {})
        tt = budgeted.get("results_by_tier", {}).get(tier, {})

        if not bt and not tt:
            continue

        b_tokens = bt.get("avg_tokens", 0)
        t_tokens = tt.get("avg_tokens", 0)
        b_cost = bt.get("total_cost", 0)
        t_cost = tt.get("total_cost", 0)

        token_savings = f"{((1 - t_tokens/b_tokens) * 100):.1f}%" if b_tokens > 0 else "—"
        cost_savings = f"{((1 - t_cost/b_cost) * 100):.1f}%" if b_cost > 0 else "—"

        lines.append(
            f"| {tier} | {b_tokens:.0f} | {t_tokens:.0f} | {token_savings} | "
            f"${b_cost:.6f} | ${t_cost:.6f} | {cost_savings} |"
        )

    # Key findings
    total_token_savings = 0
    total_cost_savings = 0
    if baseline.get("total_tokens", 0) > 0:
        total_token_savings = (1 - budgeted["total_tokens"] / baseline["total_tokens"]) * 100
    if baseline.get("total_cost", 0) > 0:
        total_cost_savings = (1 - budgeted["total_cost"] / baseline["total_cost"]) * 100

    lines.append("\n## Key Findings\n")
    lines.append(f"- **Token reduction:** {total_token_savings:.1f}% fewer tokens generated")
    lines.append(f"- **Cost reduction:** {total_cost_savings:.1f}% lower GPU cost")
    lines.append(f"- **Queries processed:** {budgeted.get('total_queries', 0)}")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        print(f"Report saved to {output_path}")
    else:
        print(report)


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare.py <baseline_summary.json> <budgeted_summary.json> [output.md]")
        sys.exit(1)

    output = sys.argv[3] if len(sys.argv) > 3 else None
    compare(sys.argv[1], sys.argv[2], output)


if __name__ == "__main__":
    main()
