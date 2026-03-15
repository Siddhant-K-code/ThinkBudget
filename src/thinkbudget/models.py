"""Data models for ThinkBudget."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComplexityTier(str, Enum):
    """Query complexity tiers with associated default thinking budgets."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    DEEP = "deep"


# Default thinking token budgets per tier
DEFAULT_BUDGETS: dict[ComplexityTier, int] = {
    ComplexityTier.TRIVIAL: 32,
    ComplexityTier.SIMPLE: 128,
    ComplexityTier.MODERATE: 512,
    ComplexityTier.COMPLEX: 2048,
    ComplexityTier.DEEP: 8192,
}


class BudgetConfig(BaseModel):
    """Configuration for budget allocation per tier."""

    budgets: dict[ComplexityTier, int] = Field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    global_max_thinking_tokens: int = Field(default=16384)
    enable_budget_forcing: bool = Field(default=True)
    # GPU cost rate in $/hour (e.g., 0.39 for RTX 4090 on CloudRift)
    gpu_cost_per_hour: float = Field(default=0.39)
    gpu_name: str = Field(default="RTX 4090")

    def get_budget(self, tier: ComplexityTier) -> int:
        return min(self.budgets.get(tier, 512), self.global_max_thinking_tokens)


class ClassifierSignals(BaseModel):
    """Signals extracted from a query for classification."""

    token_count: int = 0
    has_code_block: bool = False
    has_math_operators: bool = False
    question_count: int = 0
    reasoning_keywords: int = 0
    instruction_complexity: int = 0
    avg_word_length: float = 0.0
    has_multi_step: bool = False


class QueryCost(BaseModel):
    """GPU cost for a single query."""

    energy_joules: float = 0.0
    power_watts_avg: float = 0.0
    duration_seconds: float = 0.0
    dollar_cost: float = 0.0
    gpu_utilization_avg: float = 0.0
    memory_used_mb: float = 0.0


class QueryRecord(BaseModel):
    """Complete record of a processed query."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)

    # Input
    prompt: str = ""
    prompt_tokens: int = 0

    # Classification
    tier: ComplexityTier = ComplexityTier.MODERATE
    signals: ClassifierSignals = Field(default_factory=ClassifierSignals)
    confidence: float = 0.0

    # Budget
    thinking_budget: int = 0
    thinking_tokens_used: int = 0
    output_tokens: int = 0
    budget_utilization: float = 0.0
    was_truncated: bool = False

    # Cost
    cost: QueryCost = Field(default_factory=QueryCost)

    # Quality (filled during benchmarks)
    quality_score: float | None = None

    # Timing
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def tokens_saved(self) -> int:
        """Tokens saved vs. unbounded inference (estimated)."""
        # Estimate: unbounded would use ~4x the budget for non-deep queries
        unbounded_estimate = DEFAULT_BUDGETS[ComplexityTier.DEEP]
        return max(0, unbounded_estimate - self.thinking_tokens_used)

    @property
    def cost_saved(self) -> float:
        """Estimated dollar cost saved vs. unbounded."""
        if self.cost.duration_seconds <= 0:
            return 0.0
        tokens_per_sec = self.thinking_tokens_used / self.cost.duration_seconds if self.cost.duration_seconds > 0 else 1
        time_saved = self.tokens_saved / tokens_per_sec if tokens_per_sec > 0 else 0
        hourly_rate = self.cost.dollar_cost / (self.cost.duration_seconds / 3600) if self.cost.duration_seconds > 0 else 0
        return time_saved * hourly_rate / 3600


class ProxyConfig(BaseModel):
    """Configuration for the ThinkBudget proxy server."""

    # Backend inference server
    backend_url: str = "http://localhost:8000"
    backend_model: str = "default"

    # Proxy server
    host: str = "0.0.0.0"
    port: int = 9100

    # Budget
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    # Dashboard
    enable_dashboard: bool = True
    dashboard_port: int = 9101

    # History
    max_history: int = 10000


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""

    backend_url: str = "http://localhost:8000"
    model: str = "default"
    gpu_cost_per_hour: float = 0.39
    gpu_name: str = "RTX 4090"
    dataset_dir: str = "benchmarks/datasets"
    output_dir: str = "benchmarks/results"
    runs_per_query: int = 1
    judge_model: str | None = None  # For quality scoring
