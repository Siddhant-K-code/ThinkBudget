"""Tests for data models."""

import unittest

from thinkbudget.models import (
    BudgetConfig,
    ComplexityTier,
    DEFAULT_BUDGETS,
    QueryCost,
    QueryRecord,
)


class TestBudgetConfig(unittest.TestCase):
    def test_default_budgets(self):
        config = BudgetConfig()
        self.assertEqual(config.get_budget(ComplexityTier.TRIVIAL), 32)
        self.assertEqual(config.get_budget(ComplexityTier.DEEP), 8192)

    def test_global_cap(self):
        config = BudgetConfig(global_max_thinking_tokens=100)
        self.assertEqual(config.get_budget(ComplexityTier.DEEP), 100)

    def test_custom_budgets(self):
        config = BudgetConfig(budgets={ComplexityTier.TRIVIAL: 64})
        self.assertEqual(config.get_budget(ComplexityTier.TRIVIAL), 64)


class TestQueryRecord(unittest.TestCase):
    def test_tokens_saved(self):
        record = QueryRecord(
            tier=ComplexityTier.TRIVIAL,
            thinking_tokens_used=10,
        )
        self.assertGreater(record.tokens_saved, 0)

    def test_deep_tier_no_savings(self):
        record = QueryRecord(
            tier=ComplexityTier.DEEP,
            thinking_tokens_used=8192,
        )
        self.assertEqual(record.tokens_saved, 0)

    def test_id_generated(self):
        r1 = QueryRecord()
        r2 = QueryRecord()
        self.assertNotEqual(r1.id, r2.id)

    def test_timestamp_set(self):
        record = QueryRecord()
        self.assertGreater(record.timestamp, 0)


class TestDefaultBudgets(unittest.TestCase):
    def test_all_tiers_have_budgets(self):
        for tier in ComplexityTier:
            self.assertIn(tier, DEFAULT_BUDGETS)

    def test_budgets_increase_with_tier(self):
        tiers = [
            ComplexityTier.TRIVIAL,
            ComplexityTier.SIMPLE,
            ComplexityTier.MODERATE,
            ComplexityTier.COMPLEX,
            ComplexityTier.DEEP,
        ]
        budgets = [DEFAULT_BUDGETS[t] for t in tiers]
        self.assertEqual(budgets, sorted(budgets))


if __name__ == "__main__":
    unittest.main()
