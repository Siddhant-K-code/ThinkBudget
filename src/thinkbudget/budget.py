"""Budget controller for ThinkBudget.

Enforces thinking token budgets on inference requests. Modifies the request
parameters sent to the backend inference engine to cap reasoning effort.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from thinkbudget.models import BudgetConfig, ComplexityTier, DEFAULT_BUDGETS


@dataclass
class BudgetPolicy:
    """Policy for a specific user or session."""

    user_id: str = "default"
    max_tokens_per_hour: int = 100_000
    max_cost_per_hour: float = 1.0  # dollars
    tier_overrides: dict[ComplexityTier, int] = field(default_factory=dict)
    tokens_used_this_hour: int = 0
    cost_this_hour: float = 0.0
    hour_start: float = field(default_factory=time.time)

    def reset_if_needed(self):
        """Reset hourly counters if the hour has passed."""
        now = time.time()
        if now - self.hour_start >= 3600:
            self.tokens_used_this_hour = 0
            self.cost_this_hour = 0.0
            self.hour_start = now

    def can_proceed(self) -> tuple[bool, str]:
        """Check if the user can make another request."""
        self.reset_if_needed()
        if self.tokens_used_this_hour >= self.max_tokens_per_hour:
            return False, f"Token budget exhausted ({self.tokens_used_this_hour}/{self.max_tokens_per_hour} tokens/hr)"
        if self.cost_this_hour >= self.max_cost_per_hour:
            return False, f"Cost budget exhausted (${self.cost_this_hour:.4f}/${self.max_cost_per_hour:.2f}/hr)"
        return True, "OK"

    def record_usage(self, tokens: int, cost: float):
        """Record token and cost usage."""
        self.reset_if_needed()
        self.tokens_used_this_hour += tokens
        self.cost_this_hour += cost


@dataclass
class BudgetDecision:
    """The budget controller's decision for a query."""

    thinking_budget: int
    max_tokens: int  # Total max_tokens to send to backend
    tier: ComplexityTier
    policy_applied: str
    was_capped: bool = False
    cap_reason: str = ""


class BudgetController:
    """Controls thinking token budgets per query.

    Takes a classified tier and returns the appropriate budget,
    applying user policies and global limits.
    """

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.policies: dict[str, BudgetPolicy] = {}
        self.stats = BudgetStats()

    def get_policy(self, user_id: str = "default") -> BudgetPolicy:
        """Get or create a budget policy for a user."""
        if user_id not in self.policies:
            self.policies[user_id] = BudgetPolicy(user_id=user_id)
        return self.policies[user_id]

    def set_policy(self, policy: BudgetPolicy):
        """Set a custom policy for a user."""
        self.policies[policy.user_id] = policy

    def decide(
        self,
        tier: ComplexityTier,
        user_id: str = "default",
        requested_max_tokens: int | None = None,
    ) -> BudgetDecision:
        """Decide the thinking budget for a query.

        Args:
            tier: Classified complexity tier
            user_id: User identifier for policy lookup
            requested_max_tokens: Client's requested max_tokens (if any)

        Returns:
            BudgetDecision with the thinking budget to enforce
        """
        policy = self.get_policy(user_id)

        # Check if user can proceed
        can_proceed, reason = policy.can_proceed()
        if not can_proceed:
            return BudgetDecision(
                thinking_budget=0,
                max_tokens=0,
                tier=tier,
                policy_applied=f"user:{user_id}",
                was_capped=True,
                cap_reason=reason,
            )

        # Get base budget from tier
        base_budget = self.config.get_budget(tier)

        # Apply user tier overrides
        if tier in policy.tier_overrides:
            base_budget = policy.tier_overrides[tier]

        # Apply global cap
        thinking_budget = min(base_budget, self.config.global_max_thinking_tokens)

        # If client requested a specific max_tokens, respect it as an upper bound
        if requested_max_tokens is not None:
            thinking_budget = min(thinking_budget, requested_max_tokens)

        # Estimate total max_tokens: thinking + output
        # Output tokens are typically 1-3x the thinking tokens for reasoning models
        # We add a generous buffer for the actual response
        output_buffer = max(512, thinking_budget)
        max_tokens = thinking_budget + output_buffer

        was_capped = thinking_budget < DEFAULT_BUDGETS.get(ComplexityTier.DEEP, 8192)
        cap_reason = ""
        if was_capped:
            cap_reason = f"Tier {tier.value} budget: {thinking_budget} tokens"

        self.stats.record(tier, thinking_budget)

        return BudgetDecision(
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            tier=tier,
            policy_applied=f"user:{user_id}",
            was_capped=was_capped,
            cap_reason=cap_reason,
        )

    def apply_to_request(
        self,
        request_body: dict[str, Any],
        decision: BudgetDecision,
    ) -> dict[str, Any]:
        """Apply budget decision to an OpenAI-format request body.

        Modifies the request to enforce the thinking budget via:
        1. Setting max_tokens / max_completion_tokens
        2. Adding budget forcing stop sequences (for reasoning models)
        3. Setting temperature hints for simpler queries
        """
        modified = dict(request_body)

        # Set max completion tokens
        modified["max_tokens"] = decision.max_tokens

        # For reasoning models that support it, set thinking budget directly
        # (DeepSeek-R1, Qwen-QwQ use <think> tags)
        if self.config.enable_budget_forcing:
            # Add metadata for the proxy to enforce
            modified["_thinkbudget"] = {
                "thinking_budget": decision.thinking_budget,
                "tier": decision.tier.value,
                "enforce": True,
            }

        # For trivial/simple queries, lower temperature to reduce rambling
        if decision.tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE):
            if "temperature" not in modified:
                modified["temperature"] = 0.3

        return modified

    def record_completion(
        self,
        user_id: str,
        thinking_tokens: int,
        cost: float,
    ):
        """Record actual usage after completion."""
        policy = self.get_policy(user_id)
        policy.record_usage(thinking_tokens, cost)


class BudgetStats:
    """Tracks budget allocation statistics."""

    def __init__(self):
        self.tier_counts: dict[ComplexityTier, int] = defaultdict(int)
        self.tier_budgets: dict[ComplexityTier, list[int]] = defaultdict(list)
        self.total_queries: int = 0
        self.total_budget_allocated: int = 0

    def record(self, tier: ComplexityTier, budget: int):
        self.tier_counts[tier] += 1
        self.tier_budgets[tier].append(budget)
        self.total_queries += 1
        self.total_budget_allocated += budget

    @property
    def avg_budget(self) -> float:
        if self.total_queries == 0:
            return 0
        return self.total_budget_allocated / self.total_queries

    @property
    def tier_distribution(self) -> dict[str, float]:
        if self.total_queries == 0:
            return {}
        return {
            tier.value: count / self.total_queries
            for tier, count in self.tier_counts.items()
        }

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_budget_allocated": self.total_budget_allocated,
            "avg_budget_per_query": round(self.avg_budget, 1),
            "tier_distribution": self.tier_distribution,
        }
