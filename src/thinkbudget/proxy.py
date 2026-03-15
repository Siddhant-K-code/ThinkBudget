"""OpenAI-compatible proxy server for ThinkBudget.

Drop-in replacement that intercepts /v1/chat/completions, classifies queries,
enforces thinking budgets, monitors GPU cost, and streams responses back.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from thinkbudget.budget import BudgetController, BudgetDecision
from thinkbudget.classifier import QueryClassifier
from thinkbudget.gpu_monitor import GPUMonitor
from thinkbudget.models import BudgetConfig, ComplexityTier, ProxyConfig, QueryRecord

# Regex to detect thinking tokens in reasoning model output
# DeepSeek-R1 uses <think>...</think>, Qwen-QwQ uses similar patterns
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create the ThinkBudget proxy FastAPI application."""
    config = config or ProxyConfig()

    app = FastAPI(
        title="ThinkBudget",
        description="Adaptive inference budget controller",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core components
    classifier = QueryClassifier()
    budget_controller = BudgetController(config.budget)
    gpu_monitor = GPUMonitor(
        gpu_cost_per_hour=config.budget.gpu_cost_per_hour,
    )

    # Query history for dashboard
    history: deque[QueryRecord] = deque(maxlen=config.max_history)

    # HTTP client for backend
    client = httpx.AsyncClient(timeout=120.0)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "gpu_available": gpu_monitor.is_available,
            "gpu_name": gpu_monitor.gpu_info.name,
            "backend_url": config.backend_url,
        }

    @app.get("/v1/models")
    async def list_models():
        """Proxy model listing to backend."""
        try:
            resp = await client.get(f"{config.backend_url}/v1/models")
            return resp.json()
        except Exception as e:
            return {"object": "list", "data": [{"id": config.backend_model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Main proxy endpoint — classifies, budgets, forwards, monitors."""
        body = await request.json()
        start_time = time.time()

        # 1. Classify query complexity
        messages = body.get("messages", [])
        classification = classifier.classify_messages(messages)

        # 2. Get budget decision
        user_id = body.pop("user", "default") if "user" in body else "default"
        requested_max = body.get("max_tokens")
        decision = budget_controller.decide(
            tier=classification.tier,
            user_id=user_id,
            requested_max_tokens=requested_max,
        )

        # Check if budget exhausted
        if decision.thinking_budget == 0 and decision.was_capped:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": decision.cap_reason,
                        "type": "budget_exhausted",
                        "code": "budget_exceeded",
                    }
                },
                headers={
                    "X-ThinkBudget-Tier": classification.tier.value,
                    "X-ThinkBudget-Budget": "0",
                },
            )

        # 3. Apply budget to request
        modified_body = budget_controller.apply_to_request(body, decision)

        # Remove internal metadata before forwarding
        modified_body.pop("_thinkbudget", None)

        # Set model if not specified
        if "model" not in modified_body or modified_body["model"] == "thinkbudget":
            modified_body["model"] = config.backend_model

        # 4. Start GPU monitoring
        gpu_monitor.start_sampling(interval_ms=100)

        # 5. Forward to backend
        is_streaming = modified_body.get("stream", False)

        if is_streaming:
            return await _handle_streaming(
                client, config, modified_body, classification, decision,
                gpu_monitor, budget_controller, history, start_time, user_id,
            )
        else:
            return await _handle_non_streaming(
                client, config, modified_body, classification, decision,
                gpu_monitor, budget_controller, history, start_time, user_id,
            )

    # ── Dashboard API endpoints ──

    @app.get("/api/stats")
    async def get_stats():
        """Get aggregate statistics."""
        records = list(history)
        if not records:
            return {
                "total_queries": 0,
                "total_cost": 0,
                "total_tokens_saved": 0,
                "total_cost_saved": 0,
                "tier_distribution": {},
                "avg_latency_ms": 0,
                "gpu_info": {
                    "name": gpu_monitor.gpu_info.name,
                    "available": gpu_monitor.is_available,
                },
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
            "gpu_info": {
                "name": gpu_monitor.gpu_info.name,
                "available": gpu_monitor.is_available,
            },
            "budget_stats": budget_controller.stats.summary(),
        }

    @app.get("/api/history")
    async def get_history(limit: int = 50):
        """Get recent query history."""
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
                "prompt_preview": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
            }
            for r in reversed(records)
        ]

    @app.get("/api/gpu")
    async def get_gpu_status():
        """Get current GPU status."""
        snap = gpu_monitor.snapshot()
        return {
            "name": gpu_monitor.gpu_info.name,
            "available": gpu_monitor.is_available,
            "power_watts": snap.power_watts,
            "utilization": snap.gpu_utilization,
            "memory_used_mb": snap.memory_used_mb,
            "memory_total_mb": snap.memory_total_mb,
            "temperature_c": snap.temperature_c,
            "cost_per_hour": gpu_monitor.gpu_cost_per_hour,
        }

    @app.post("/api/classify")
    async def classify_query(request: Request):
        """Classify a query without forwarding it."""
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

    @app.on_event("shutdown")
    async def shutdown():
        await client.aclose()
        gpu_monitor.shutdown()

    return app


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    config: ProxyConfig,
    body: dict,
    classification: Any,
    decision: BudgetDecision,
    gpu_monitor: GPUMonitor,
    budget_controller: BudgetController,
    history: deque,
    start_time: float,
    user_id: str,
) -> JSONResponse:
    """Handle non-streaming completion request."""
    try:
        resp = await client.post(
            f"{config.backend_url}/v1/chat/completions",
            json=body,
        )
        result = resp.json()
    except Exception as e:
        gpu_monitor.stop_sampling()
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Backend error: {str(e)}", "type": "backend_error"}},
        )

    # Stop GPU monitoring
    samples = gpu_monitor.stop_sampling()
    end_time = time.time()
    duration = end_time - start_time

    # Extract token counts from response
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    # Estimate thinking vs output tokens
    content = ""
    if result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content", "")

    thinking_tokens, output_tokens = _count_thinking_tokens(content, completion_tokens)

    # Compute cost
    cost = gpu_monitor.compute_cost(
        samples=samples,
        duration_seconds=duration,
        thinking_tokens=thinking_tokens,
    )

    # Record usage
    budget_controller.record_completion(user_id, thinking_tokens, cost.dollar_cost)

    # Build query record
    last_user_msg = ""
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            if isinstance(last_user_msg, list):
                last_user_msg = str(last_user_msg)
            break

    record = QueryRecord(
        prompt=last_user_msg,
        prompt_tokens=prompt_tokens,
        tier=classification.tier,
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
        was_truncated=thinking_tokens >= decision.thinking_budget,
        cost=cost,
        total_latency_ms=duration * 1000,
    )
    history.append(record)

    # Add ThinkBudget headers to response
    headers = {
        "X-ThinkBudget-Tier": classification.tier.value,
        "X-ThinkBudget-Budget": str(decision.thinking_budget),
        "X-ThinkBudget-Thinking-Tokens": str(thinking_tokens),
        "X-ThinkBudget-Cost": f"${cost.dollar_cost:.8f}",
        "X-ThinkBudget-Energy-J": f"{cost.energy_joules:.4f}",
    }

    return JSONResponse(content=result, headers=headers)


async def _handle_streaming(
    client: httpx.AsyncClient,
    config: ProxyConfig,
    body: dict,
    classification: Any,
    decision: BudgetDecision,
    gpu_monitor: GPUMonitor,
    budget_controller: BudgetController,
    history: deque,
    start_time: float,
    user_id: str,
) -> StreamingResponse:
    """Handle streaming completion request with budget enforcement."""

    async def stream_generator():
        thinking_tokens = 0
        output_tokens = 0
        in_thinking = False
        budget_exceeded = False
        content_buffer = ""

        try:
            async with client.stream(
                "POST",
                f"{config.backend_url}/v1/chat/completions",
                json=body,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        yield f"{line}\n"
                        continue

                    data = line[6:]
                    if data.strip() == "[DONE]":
                        # Final accounting
                        samples = gpu_monitor.stop_sampling()
                        end_time = time.time()
                        duration = end_time - start_time

                        cost = gpu_monitor.compute_cost(
                            samples=samples,
                            duration_seconds=duration,
                            thinking_tokens=thinking_tokens,
                        )

                        budget_controller.record_completion(
                            user_id, thinking_tokens, cost.dollar_cost
                        )

                        last_user_msg = ""
                        for m in reversed(body.get("messages", [])):
                            if m.get("role") == "user":
                                last_user_msg = m.get("content", "")
                                if isinstance(last_user_msg, list):
                                    last_user_msg = str(last_user_msg)
                                break

                        record = QueryRecord(
                            prompt=last_user_msg,
                            prompt_tokens=0,
                            tier=classification.tier,
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
                            was_truncated=budget_exceeded,
                            cost=cost,
                            total_latency_ms=duration * 1000,
                        )
                        history.append(record)

                        yield "data: [DONE]\n\n"
                        return

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        yield f"data: {data}\n\n"
                        continue

                    # Extract content delta
                    delta = ""
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {}).get("content", "") or ""

                    content_buffer += delta

                    # Track thinking vs output tokens
                    if THINK_OPEN_RE.search(delta):
                        in_thinking = True
                    if THINK_CLOSE_RE.search(delta):
                        in_thinking = False

                    if in_thinking:
                        # Rough token estimate for streaming
                        thinking_tokens += max(1, len(delta.split()))

                        # Budget enforcement: if thinking exceeds budget, inject close tag
                        if (
                            thinking_tokens >= decision.thinking_budget
                            and not budget_exceeded
                        ):
                            budget_exceeded = True
                            # Inject thinking close tag to force end of reasoning
                            inject_chunk = dict(chunk)
                            if inject_chunk.get("choices"):
                                inject_chunk["choices"][0]["delta"] = {
                                    "content": "</think>\n"
                                }
                            yield f"data: {json.dumps(inject_chunk)}\n\n"
                            continue
                    else:
                        output_tokens += max(1, len(delta.split()))

                    # If budget was exceeded, skip further thinking tokens
                    if budget_exceeded and in_thinking:
                        continue

                    yield f"data: {data}\n\n"

        except Exception as e:
            gpu_monitor.stop_sampling()
            error_data = {
                "error": {"message": f"Backend stream error: {str(e)}", "type": "backend_error"}
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    headers = {
        "X-ThinkBudget-Tier": classification.tier.value,
        "X-ThinkBudget-Budget": str(decision.thinking_budget),
    }

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


def _count_thinking_tokens(content: str, total_tokens: int) -> tuple[int, int]:
    """Estimate thinking vs output tokens from response content.

    Reasoning models wrap thinking in <think>...</think> tags.
    """
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking_text = think_match.group(1)
        # Rough token estimate: ~1.3 tokens per word
        thinking_tokens = int(len(thinking_text.split()) * 1.3)
        output_tokens = max(0, total_tokens - thinking_tokens)
        return thinking_tokens, output_tokens

    # No thinking tags found — assume all tokens are output
    return 0, total_tokens
