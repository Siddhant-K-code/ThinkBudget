"""Tests for the budget controller."""

import unittest

from thinkbudget.budget import BudgetController, BudgetPolicy
from thinkbudget.models import BudgetConfig, ComplexityTier


class TestBudgetController(unittest.TestCase):
    def setUp(self):
        self.config = BudgetConfig(gpu_cost_per_hour=0.39)
        self.controller = BudgetController(self.config)

    def test_trivial_gets_small_budget(self):
        decision = self.controller.decide(ComplexityTier.TRIVIAL)
        self.assertEqual(decision.thinking_budget, 32)

    def test_simple_budget(self):
        decision = self.controller.decide(ComplexityTier.SIMPLE)
        self.assertEqual(decision.thinking_budget, 128)

    def test_moderate_budget(self):
        decision = self.controller.decide(ComplexityTier.MODERATE)
        self.assertEqual(decision.thinking_budget, 512)

    def test_complex_budget(self):
        decision = self.controller.decide(ComplexityTier.COMPLEX)
        self.assertEqual(decision.thinking_budget, 2048)

    def test_deep_budget(self):
        decision = self.controller.decide(ComplexityTier.DEEP)
        self.assertEqual(decision.thinking_budget, 8192)

    def test_max_tokens_includes_output_buffer(self):
        decision = self.controller.decide(ComplexityTier.TRIVIAL)
        self.assertGreater(decision.max_tokens, decision.thinking_budget)

    def test_respects_requested_max_tokens(self):
        decision = self.controller.decide(ComplexityTier.DEEP, requested_max_tokens=100)
        self.assertEqual(decision.thinking_budget, 100)

    def test_global_cap(self):
        config = BudgetConfig(global_max_thinking_tokens=64)
        controller = BudgetController(config)
        decision = controller.decide(ComplexityTier.DEEP)
        self.assertEqual(decision.thinking_budget, 64)

    def test_stats_tracked(self):
        self.controller.decide(ComplexityTier.TRIVIAL)
        self.controller.decide(ComplexityTier.COMPLEX)
        self.assertEqual(self.controller.stats.total_queries, 2)
        self.assertEqual(self.controller.stats.total_budget_allocated, 32 + 2048)


class TestBudgetPolicy(unittest.TestCase):
    def setUp(self):
        self.controller = BudgetController()

    def test_default_policy_allows(self):
        policy = self.controller.get_policy("user1")
        can, reason = policy.can_proceed()
        self.assertTrue(can)

    def test_token_limit_enforced(self):
        policy = BudgetPolicy(user_id="user1", max_tokens_per_hour=100)
        self.controller.set_policy(policy)
        policy.record_usage(100, 0.01)
        can, reason = policy.can_proceed()
        self.assertFalse(can)
        self.assertIn("Token budget", reason)

    def test_cost_limit_enforced(self):
        policy = BudgetPolicy(user_id="user2", max_cost_per_hour=0.01)
        self.controller.set_policy(policy)
        policy.record_usage(10, 0.01)
        can, reason = policy.can_proceed()
        self.assertFalse(can)
        self.assertIn("Cost budget", reason)

    def test_exhausted_budget_returns_zero(self):
        policy = BudgetPolicy(user_id="user3", max_tokens_per_hour=10)
        self.controller.set_policy(policy)
        policy.record_usage(10, 0.0)
        decision = self.controller.decide(ComplexityTier.SIMPLE, user_id="user3")
        self.assertEqual(decision.thinking_budget, 0)
        self.assertTrue(decision.was_capped)

    def test_tier_override(self):
        policy = BudgetPolicy(
            user_id="user4",
            tier_overrides={ComplexityTier.TRIVIAL: 256},
        )
        self.controller.set_policy(policy)
        decision = self.controller.decide(ComplexityTier.TRIVIAL, user_id="user4")
        self.assertEqual(decision.thinking_budget, 256)


class TestApplyToRequest(unittest.TestCase):
    def setUp(self):
        self.controller = BudgetController()

    def test_sets_max_tokens(self):
        decision = self.controller.decide(ComplexityTier.SIMPLE)
        body = {"messages": [{"role": "user", "content": "test"}]}
        modified = self.controller.apply_to_request(body, decision)
        self.assertEqual(modified["max_tokens"], decision.max_tokens)

    def test_trivial_gets_low_temperature(self):
        decision = self.controller.decide(ComplexityTier.TRIVIAL)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        modified = self.controller.apply_to_request(body, decision)
        self.assertEqual(modified["temperature"], 0.3)

    def test_complex_keeps_default_temperature(self):
        decision = self.controller.decide(ComplexityTier.COMPLEX)
        body = {"messages": [{"role": "user", "content": "debug this"}]}
        modified = self.controller.apply_to_request(body, decision)
        self.assertNotIn("temperature", modified)

    def test_does_not_override_explicit_temperature(self):
        decision = self.controller.decide(ComplexityTier.TRIVIAL)
        body = {"messages": [], "temperature": 0.9}
        modified = self.controller.apply_to_request(body, decision)
        self.assertEqual(modified["temperature"], 0.9)


if __name__ == "__main__":
    unittest.main()
