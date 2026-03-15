"""Demo mode for ThinkBudget.

Simulates a backend inference server with realistic token counts,
latency, and GPU metrics. No real GPU or model needed — runs anywhere.

Usage:
    thinkbudget demo
    # Then open http://localhost:9100/dashboard and watch it fill up
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from thinkbudget.budget import BudgetController
from thinkbudget.classifier import QueryClassifier
from thinkbudget.dashboard import create_dashboard_app
from thinkbudget.models import (
    BudgetConfig,
    ComplexityTier,
    DEFAULT_BUDGETS,
    ProxyConfig,
    QueryCost,
    QueryRecord,
)

# Simulated responses per tier
DEMO_RESPONSES = {
    ComplexityTier.TRIVIAL: [
        "Hello! How can I help you today?",
        "4",
        "The sky is blue.",
        "Sure thing!",
        "Python is a programming language.",
    ],
    ComplexityTier.SIMPLE: [
        "The capital of France is Paris.",
        "100°F is approximately 37.8°C.",
        "A REST API uses HTTP methods to interact with resources. It follows a stateless client-server architecture.",
        "Polymorphism allows objects of different types to be treated through a common interface.",
        "HTTPS adds TLS encryption on top of HTTP, protecting data in transit.",
    ],
    ComplexityTier.MODERATE: [
        "REST vs GraphQL trade-offs:\n\nREST is simpler, cacheable, and well-understood. GraphQL reduces over-fetching and lets clients request exactly the data they need. For mobile apps with variable data needs, GraphQL often wins. For simple CRUD APIs with stable clients, REST is easier to maintain.\n\nKey factors: team familiarity, caching requirements, and how varied your client data needs are.",
        "The CAP theorem states that a distributed system can provide at most two of three guarantees: Consistency, Availability, and Partition tolerance.\n\n- CP: MongoDB (strong consistency, may reject writes during partitions)\n- AP: Cassandra (always available, eventually consistent)\n- CA: Traditional RDBMS (no partition tolerance, single node)",
        "B-tree indexes in PostgreSQL maintain sorted data for range queries and equality lookups. They're the default and work well for most cases.\n\nHash indexes only support equality comparisons but use less space. Use hash indexes when you only need exact-match lookups on large tables and never need range scans or sorting.",
    ],
    ComplexityTier.COMPLEX: [
        "<think>\nThe user has a race condition in Go. Looking at the code:\n- `update()` correctly locks the mutex before writing\n- `read()` accesses the map WITHOUT locking\n- Go maps are not safe for concurrent read/write\n- Even though writes are protected, concurrent read + write = panic\n- The fix: also lock the mutex in `read()`, or use `sync.RWMutex`\n- RWMutex is better here: allows concurrent reads, exclusive writes\n</think>\n\nThe bug: `read()` accesses the map without holding the lock. In Go, concurrent map read + write panics even if writes are serialized.\n\n```go\nfunc read(key string) int {\n    mu.RLock()\n    defer mu.RUnlock()\n    return cache[key]\n}\n```\n\nBetter: switch to `sync.RWMutex` so reads don't block each other.",
        "<think>\nDistributed task queue design:\n1. Need at-least-once delivery → acknowledgment + retry\n2. Priorities → priority queue or separate queues per priority\n3. Delayed execution → scheduled queue with timestamp\n4. Dead letter queue → failed messages after N retries\n\nArchitecture options:\n- Redis Streams + consumer groups\n- PostgreSQL with SKIP LOCKED\n- Custom on top of Kafka\n\nFor simplicity and reliability, PostgreSQL with SKIP LOCKED is underrated.\n</think>\n\nArchitecture: PostgreSQL-backed task queue using `SELECT ... FOR UPDATE SKIP LOCKED`.\n\nTables: `tasks` (id, payload, priority, state, scheduled_at, attempts, max_attempts), `dead_letters` (task_id, last_error, failed_at).\n\nWorkers poll with priority ordering. Failed tasks retry with exponential backoff. After max_attempts, moved to dead_letters. Delayed tasks use `scheduled_at > NOW()` filter.",
    ],
    ComplexityTier.DEEP: [
        "<think>\nProving the halting problem is undecidable:\n\n1. Assume for contradiction that H(P, I) exists — a program that decides whether program P halts on input I.\n\n2. Construct D(P):\n   - Run H(P, P)\n   - If H says P halts on P → loop forever\n   - If H says P doesn't halt on P → halt\n\n3. Now ask: does D(D) halt?\n   - If D(D) halts → H(D,D) = halts → D loops forever. Contradiction.\n   - If D(D) doesn't halt → H(D,D) = doesn't halt → D halts. Contradiction.\n\n4. Therefore H cannot exist. The halting problem is undecidable.\n\nImplications for AI agent static analysis:\n- We cannot build a general tool that determines if an agent will terminate\n- Any static analysis of agent behavior is necessarily incomplete\n- This is why runtime monitoring (like AgentTrace) is essential\n- We can still prove termination for restricted agent classes\n- Bounded loops, finite tool sets, and timeout policies provide practical guarantees\n</think>\n\nProof by diagonalization:\n\nAssume a decider H(P, I) exists that returns true iff program P halts on input I. Construct D(P): run H(P,P); if it says \"halts\", loop forever; otherwise halt. Asking whether D(D) halts produces a contradiction in both cases. Therefore H cannot exist.\n\nFor AI agents: no general static analyzer can guarantee termination of arbitrary agent loops. This is why practical systems use runtime bounds — timeouts, step limits, and budget caps (which is exactly what ThinkBudget does for the inference layer).",
    ],
}

# Simulated thinking token counts per tier (realistic ranges)
THINKING_TOKENS_RANGE = {
    ComplexityTier.TRIVIAL: (0, 5),
    ComplexityTier.SIMPLE: (0, 20),
    ComplexityTier.MODERATE: (30, 150),
    ComplexityTier.COMPLEX: (200, 1800),
    ComplexityTier.DEEP: (1500, 7000),
}

# Simulated latency per tier in ms
LATENCY_RANGE = {
    ComplexityTier.TRIVIAL: (80, 200),
    ComplexityTier.SIMPLE: (150, 400),
    ComplexityTier.MODERATE: (400, 1200),
    ComplexityTier.COMPLEX: (1200, 4000),
    ComplexityTier.DEEP: (3000, 12000),
}

# Demo queries that cycle through automatically
DEMO_QUERIES = [
    "Hello",
    "What is the capital of France?",
    "What is Docker?",
    "Compare REST vs GraphQL for a mobile app",
    "Hi there",
    "What is TCP/IP?",
    "Explain the CAP theorem with examples",
    "Debug this race condition in my Go code with goroutines writing to a shared map",
    "Thanks",
    "What is a hash table?",
    "Design a rate limiter using token bucket algorithm",
    "What are the trade-offs between monorepo and polyrepo?",
    "Prove the halting problem is undecidable and explain implications for AI agents",
    "What is CORS?",
    "Design a distributed task queue with at-least-once delivery",
    "OK",
    "Explain how B-tree indexes work in PostgreSQL",
    "What is 2+2?",
    "Implement a circuit breaker pattern with half-open state",
    "Define polymorphism",
]


def create_demo_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create a ThinkBudget app in demo mode with simulated backend."""
    config = config or ProxyConfig()

    app = FastAPI(title="ThinkBudget Demo", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    classifier = QueryClassifier()
    budget_controller = BudgetController(config.budget)
    history: deque[QueryRecord] = deque(maxlen=config.max_history)

    # Simulated GPU info
    sim_gpu = {
        "name": config.budget.gpu_name,
        "available": True,
        "memory_total_mb": 24576 if "4090" in config.budget.gpu_name else 32768,
    }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": "demo",
            "gpu_available": True,
            "gpu_name": sim_gpu["name"],
            "backend_url": "simulated",
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": config.backend_model, "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        classification = classifier.classify_messages(messages)

        user_id = body.pop("user", "default") if "user" in body else "default"
        decision = budget_controller.decide(tier=classification.tier, user_id=user_id)

        # Simulate inference
        tier = classification.tier
        thinking_lo, thinking_hi = THINKING_TOKENS_RANGE[tier]
        thinking_tokens = random.randint(thinking_lo, min(thinking_hi, decision.thinking_budget))
        was_truncated = thinking_tokens >= decision.thinking_budget * 0.95

        response_text = random.choice(DEMO_RESPONSES[tier])
        output_tokens = int(len(response_text.split()) * 1.3)

        latency_lo, latency_hi = LATENCY_RANGE[tier]
        latency_ms = random.uniform(latency_lo, latency_hi)

        # Simulate GPU cost
        duration_s = latency_ms / 1000
        power_watts = random.uniform(180, 320)
        energy_j = power_watts * duration_s
        dollar_cost = (duration_s / 3600) * config.budget.gpu_cost_per_hour

        # Simulate delay
        await asyncio.sleep(min(latency_ms / 1000, 2.0))

        # Extract prompt
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                if isinstance(last_user_msg, list):
                    last_user_msg = str(last_user_msg)
                break

        cost = QueryCost(
            energy_joules=round(energy_j, 4),
            power_watts_avg=round(power_watts, 2),
            duration_seconds=round(duration_s, 4),
            dollar_cost=round(dollar_cost, 8),
            gpu_utilization_avg=round(random.uniform(60, 95), 2),
            memory_used_mb=round(random.uniform(8000, 20000), 2),
        )

        record = QueryRecord(
            prompt=last_user_msg,
            prompt_tokens=int(len(last_user_msg.split()) * 1.3),
            tier=tier,
            signals=classification.signals,
            confidence=classification.confidence,
            thinking_budget=decision.thinking_budget,
            thinking_tokens_used=thinking_tokens,
            output_tokens=output_tokens,
            budget_utilization=(
                thinking_tokens / decision.thinking_budget
                if decision.thinking_budget > 0
                else 0
            ),
            was_truncated=was_truncated,
            cost=cost,
            total_latency_ms=latency_ms,
        )
        history.append(record)
        budget_controller.record_completion(user_id, thinking_tokens, dollar_cost)

        # Build OpenAI-format response
        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.backend_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": thinking_tokens + output_tokens,
                "total_tokens": record.prompt_tokens + thinking_tokens + output_tokens,
            },
        }

        headers = {
            "X-ThinkBudget-Tier": tier.value,
            "X-ThinkBudget-Budget": str(decision.thinking_budget),
            "X-ThinkBudget-Thinking-Tokens": str(thinking_tokens),
            "X-ThinkBudget-Cost": f"${dollar_cost:.8f}",
            "X-ThinkBudget-Energy-J": f"{energy_j:.4f}",
        }

        return JSONResponse(content=result, headers=headers)

    # -- Dashboard API (same as proxy.py) --

    @app.get("/api/stats")
    async def get_stats():
        records = list(history)
        if not records:
            return {
                "total_queries": 0,
                "total_cost": 0,
                "total_tokens_saved": 0,
                "total_cost_saved": 0,
                "tier_distribution": {},
                "avg_latency_ms": 0,
                "gpu_info": {"name": sim_gpu["name"], "available": True},
                "budget_stats": budget_controller.stats.summary(),
            }

        total_cost = sum(r.cost.dollar_cost for r in records)
        total_tokens_saved = sum(r.tokens_saved for r in records)
        total_cost_saved = sum(r.cost_saved for r in records)
        avg_latency = sum(r.total_latency_ms for r in records) / len(records)

        tier_dist: dict[str, int] = {}
        for r in records:
            tier_dist[r.tier.value] = tier_dist.get(r.tier.value, 0) + 1

        return {
            "total_queries": len(records),
            "total_cost": round(total_cost, 6),
            "total_tokens_saved": total_tokens_saved,
            "total_cost_saved": round(total_cost_saved, 6),
            "tier_distribution": tier_dist,
            "avg_latency_ms": round(avg_latency, 2),
            "gpu_info": {"name": sim_gpu["name"], "available": True},
            "budget_stats": budget_controller.stats.summary(),
        }

    @app.get("/api/history")
    async def get_history(limit: int = 50):
        records = list(history)[-limit:]
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "tier": r.tier.value,
                "thinking_budget": r.thinking_budget,
                "thinking_tokens_used": r.thinking_tokens_used,
                "output_tokens": r.output_tokens,
                "budget_utilization": round(r.budget_utilization, 2),
                "was_truncated": r.was_truncated,
                "cost_dollars": round(r.cost.dollar_cost, 8),
                "energy_joules": round(r.cost.energy_joules, 4),
                "latency_ms": round(r.total_latency_ms, 2),
                "tokens_saved": r.tokens_saved,
                "prompt_preview": (
                    r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt
                ),
            }
            for r in reversed(records)
        ]

    @app.get("/api/gpu")
    async def get_gpu_status():
        return {
            "name": sim_gpu["name"],
            "available": True,
            "power_watts": round(random.uniform(180, 320), 1),
            "utilization": round(random.uniform(40, 90), 1),
            "memory_used_mb": round(random.uniform(8000, 20000), 1),
            "memory_total_mb": sim_gpu["memory_total_mb"],
            "temperature_c": round(random.uniform(55, 78), 1),
            "cost_per_hour": config.budget.gpu_cost_per_hour,
        }

    @app.post("/api/classify")
    async def classify_query(request: Request):
        body = await request.json()
        query = body.get("query", "")
        messages = body.get("messages", [])

        if messages:
            result = classifier.classify_messages(messages)
        else:
            result = classifier.classify(query)

        budget = config.budget.get_budget(result.tier)

        return {
            "tier": result.tier.value,
            "confidence": result.confidence,
            "thinking_budget": budget,
            "signals": result.signals.model_dump(),
            "reasoning": result.reasoning,
        }

    return app


async def run_demo_traffic(base_url: str, interval: float = 3.0):
    """Send simulated queries to the demo server at regular intervals."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        idx = 0
        while True:
            query = DEMO_QUERIES[idx % len(DEMO_QUERIES)]
            try:
                await client.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": "demo",
                        "messages": [{"role": "user", "content": query}],
                    },
                )
            except Exception:
                pass
            idx += 1
            await asyncio.sleep(interval + random.uniform(-1, 1))
