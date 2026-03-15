"""Tests for the query complexity classifier."""

import unittest

from thinkbudget.classifier import QueryClassifier
from thinkbudget.models import ComplexityTier


class TestQueryClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = QueryClassifier()

    # -- Trivial tier --

    def test_greeting_is_trivial(self):
        result = self.classifier.classify("Hello")
        self.assertEqual(result.tier, ComplexityTier.TRIVIAL)

    def test_short_affirmation_is_trivial(self):
        result = self.classifier.classify("Yes")
        self.assertEqual(result.tier, ComplexityTier.TRIVIAL)

    def test_thanks_is_trivial(self):
        result = self.classifier.classify("Thanks")
        self.assertEqual(result.tier, ComplexityTier.TRIVIAL)

    # -- Simple tier --

    def test_factual_question_is_simple(self):
        result = self.classifier.classify("What is the capital of France?")
        self.assertEqual(result.tier, ComplexityTier.SIMPLE)

    def test_definition_is_simple(self):
        result = self.classifier.classify("Define polymorphism in programming")
        self.assertEqual(result.tier, ComplexityTier.SIMPLE)

    # -- At least moderate --

    def test_comparison_is_at_least_moderate(self):
        result = self.classifier.classify(
            "Compare REST vs GraphQL for a mobile app backend. What are the trade-offs?"
        )
        self.assertIn(result.tier, [ComplexityTier.MODERATE, ComplexityTier.COMPLEX, ComplexityTier.DEEP])

    # -- At least complex --

    def test_system_design_is_at_least_complex(self):
        result = self.classifier.classify(
            "Design a distributed task queue with at-least-once delivery, "
            "priorities, delayed execution, and dead letter queues. "
            "Walk through the architecture step by step."
        )
        self.assertIn(result.tier, [ComplexityTier.COMPLEX, ComplexityTier.DEEP])

    def test_debug_with_code_is_at_least_moderate(self):
        result = self.classifier.classify(
            "Debug this race condition:\n```go\nvar mu sync.Mutex\n"
            "var cache = make(map[string]int)\n```\nWhat's wrong?"
        )
        self.assertIn(result.tier, [ComplexityTier.MODERATE, ComplexityTier.COMPLEX, ComplexityTier.DEEP])

    # -- Signals --

    def test_code_block_detected(self):
        result = self.classifier.classify("Fix this:\n```python\nprint('hello')\n```")
        self.assertTrue(result.signals.has_code_block)

    def test_math_detected(self):
        result = self.classifier.classify("Solve x^2 + 3x = 0")
        self.assertTrue(result.signals.has_math_operators)

    def test_question_count(self):
        result = self.classifier.classify("What is X? How does Y work? Why is Z?")
        self.assertEqual(result.signals.question_count, 3)

    def test_multi_step_detected(self):
        result = self.classifier.classify("First do X, then do Y, finally do Z")
        self.assertTrue(result.signals.has_multi_step)

    # -- Messages format --

    def test_classify_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = self.classifier.classify_messages(messages)
        self.assertEqual(result.tier, ComplexityTier.TRIVIAL)

    def test_classify_messages_uses_last_user_message(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        result = self.classifier.classify_messages(messages)
        self.assertEqual(result.tier, ComplexityTier.SIMPLE)

    def test_classify_messages_no_user(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        result = self.classifier.classify_messages(messages)
        self.assertEqual(result.tier, ComplexityTier.MODERATE)  # default fallback

    # -- Confidence --

    def test_trivial_has_high_confidence(self):
        result = self.classifier.classify("Hi")
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_result_has_reasoning(self):
        result = self.classifier.classify("Hello")
        self.assertTrue(len(result.reasoning) > 0)


class TestClassifierEdgeCases(unittest.TestCase):
    def setUp(self):
        self.classifier = QueryClassifier()

    def test_empty_string(self):
        result = self.classifier.classify("")
        self.assertIsNotNone(result.tier)

    def test_very_long_query(self):
        query = "Explain " + "the implications of " * 100 + "this concept."
        result = self.classifier.classify(query)
        self.assertIn(result.tier, [ComplexityTier.MODERATE, ComplexityTier.COMPLEX, ComplexityTier.DEEP])

    def test_unicode_query(self):
        result = self.classifier.classify("什么是人工智能？")
        self.assertIsNotNone(result.tier)


if __name__ == "__main__":
    unittest.main()
