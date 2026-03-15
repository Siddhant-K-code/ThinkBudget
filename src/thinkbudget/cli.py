"""CLI entry point for ThinkBudget."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="thinkbudget",
        description="Adaptive inference budget controller for self-hosted LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── serve ──
    serve_parser = subparsers.add_parser("serve", help="Start the ThinkBudget proxy server")
    serve_parser.add_argument(
        "--backend-url", default="http://localhost:8000",
        help="Backend inference server URL (default: http://localhost:8000)",
    )
    serve_parser.add_argument(
        "--model", default="default",
        help="Model name to use on the backend",
    )
    serve_parser.add_argument(
        "--port", type=int, default=9100,
        help="Proxy server port (default: 9100)",
    )
    serve_parser.add_argument(
        "--dashboard-port", type=int, default=9101,
        help="Dashboard port (default: 9101)",
    )
    serve_parser.add_argument(
        "--gpu-cost", type=float, default=0.39,
        help="GPU cost per hour in dollars (default: 0.39 for RTX 4090)",
    )
    serve_parser.add_argument(
        "--gpu-name", default="RTX 4090",
        help="GPU name for display (default: RTX 4090)",
    )
    serve_parser.add_argument(
        "--config", default=None,
        help="Path to JSON config file",
    )

    # ── classify ──
    classify_parser = subparsers.add_parser("classify", help="Classify a query's complexity")
    classify_parser.add_argument("query", help="Query text to classify")

    # ── demo ──
    demo_parser = subparsers.add_parser(
        "demo", help="Run in demo mode (simulated backend, no GPU needed)"
    )
    demo_parser.add_argument(
        "--port", type=int, default=9100, help="Server port (default: 9100)",
    )
    demo_parser.add_argument(
        "--gpu-name", default="RTX 4090", help="Simulated GPU name",
    )
    demo_parser.add_argument(
        "--gpu-cost", type=float, default=0.39, help="Simulated GPU cost/hr",
    )
    demo_parser.add_argument(
        "--auto-traffic", action="store_true",
        help="Auto-send demo queries every few seconds",
    )
    demo_parser.add_argument(
        "--traffic-interval", type=float, default=3.0,
        help="Seconds between auto-traffic queries (default: 3)",
    )

    # ── benchmark ──
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument(
        "--backend-url", default="http://localhost:8000",
        help="Backend inference server URL",
    )
    bench_parser.add_argument(
        "--model", default="default",
        help="Model name",
    )
    bench_parser.add_argument(
        "--dataset-dir", default="benchmarks/datasets",
        help="Directory containing benchmark datasets",
    )
    bench_parser.add_argument(
        "--output-dir", default="benchmarks/results",
        help="Directory for benchmark results",
    )
    bench_parser.add_argument(
        "--gpu-cost", type=float, default=0.39,
        help="GPU cost per hour in dollars",
    )

    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "demo":
        _run_demo(args)
    elif args.command == "classify":
        _run_classify(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_demo(args):
    """Start ThinkBudget in demo mode with simulated backend."""
    import asyncio
    import threading
    import uvicorn
    from thinkbudget.demo import create_demo_app, run_demo_traffic
    from thinkbudget.dashboard import create_dashboard_app
    from thinkbudget.models import BudgetConfig, ProxyConfig

    config = ProxyConfig(
        port=args.port,
        backend_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        budget=BudgetConfig(
            gpu_cost_per_hour=args.gpu_cost,
            gpu_name=args.gpu_name,
        ),
    )

    app = create_demo_app(config)
    dashboard = create_dashboard_app()
    app.mount("/dashboard", dashboard)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║              ThinkBudget v0.1.0 — DEMO MODE              ║
╠══════════════════════════════════════════════════════════╣
║  Dashboard:  http://localhost:{config.port}/dashboard              ║
║  API:        http://localhost:{config.port}/v1/chat/completions    ║
║  GPU:        {config.budget.gpu_name:<20} (simulated)            ║
║  Auto-traffic: {"ON — query every ~" + str(args.traffic_interval) + "s" if args.auto_traffic else "OFF (send queries manually)":<39}║
╚══════════════════════════════════════════════════════════╝

  Try:
    curl -s http://localhost:{config.port}/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{{"model":"demo","messages":[{{"role":"user","content":"Hello"}}]}}' | python3 -m json.tool
    """)

    if args.auto_traffic:
        def _traffic():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                run_demo_traffic(
                    f"http://localhost:{config.port}",
                    interval=args.traffic_interval,
                )
            )

        t = threading.Thread(target=_traffic, daemon=True)
        t.start()

    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")


def _run_serve(args):
    """Start the proxy server."""
    import uvicorn
    from thinkbudget.config import load_config
    from thinkbudget.models import BudgetConfig, ProxyConfig
    from thinkbudget.proxy import create_app
    from thinkbudget.dashboard import create_dashboard_app

    if args.config:
        config = load_config(args.config)
    else:
        config = ProxyConfig(
            backend_url=args.backend_url,
            backend_model=args.model,
            port=args.port,
            dashboard_port=args.dashboard_port,
            budget=BudgetConfig(
                gpu_cost_per_hour=args.gpu_cost,
                gpu_name=args.gpu_name,
            ),
        )

    app = create_app(config)

    # Mount dashboard
    dashboard = create_dashboard_app()
    app.mount("/dashboard", dashboard)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    ThinkBudget v0.1.0                    ║
╠══════════════════════════════════════════════════════════╣
║  Proxy:      http://0.0.0.0:{config.port:<5}                      ║
║  Dashboard:  http://0.0.0.0:{config.port}/dashboard              ║
║  Backend:    {config.backend_url:<43}║
║  GPU:        {config.budget.gpu_name:<20} (${config.budget.gpu_cost_per_hour:.2f}/hr)          ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")


def _run_classify(args):
    """Classify a single query."""
    from rich.console import Console
    from rich.table import Table
    from thinkbudget.classifier import QueryClassifier
    from thinkbudget.models import DEFAULT_BUDGETS

    console = Console()
    classifier = QueryClassifier()
    result = classifier.classify(args.query)

    budget = DEFAULT_BUDGETS[result.tier]

    table = Table(title="Query Classification")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Query", args.query[:80] + "..." if len(args.query) > 80 else args.query)
    table.add_row("Tier", f"[bold]{result.tier.value.upper()}[/bold]")
    table.add_row("Confidence", f"{result.confidence:.1%}")
    table.add_row("Thinking Budget", f"{budget:,} tokens")
    table.add_row("Reasoning", result.reasoning)

    signals = result.signals
    table.add_row("─── Signals ───", "")
    table.add_row("  Token Count", str(signals.token_count))
    table.add_row("  Code Block", "✓" if signals.has_code_block else "✗")
    table.add_row("  Math Operators", "✓" if signals.has_math_operators else "✗")
    table.add_row("  Questions", str(signals.question_count))
    table.add_row("  Reasoning Keywords", str(signals.reasoning_keywords))
    table.add_row("  Multi-step", "✓" if signals.has_multi_step else "✗")
    table.add_row("  Instruction Complexity", str(signals.instruction_complexity))

    console.print(table)


def _run_benchmark(args):
    """Run the benchmark suite."""
    print("Benchmark runner — use benchmarks/run_benchmark.py for full benchmarks")
    print(f"Backend: {args.backend_url}")
    print(f"Model: {args.model}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
