"""Query complexity classifier for ThinkBudget.

Classifies incoming queries into complexity tiers without calling the LLM.
Uses a combination of heuristic signals: token count, structural patterns,
keyword analysis, and question complexity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from thinkbudget.models import ClassifierSignals, ComplexityTier

# Patterns that signal deeper reasoning requirements
REASONING_KEYWORDS = {
    # Mathematical / logical reasoning
    "prove", "derive", "theorem", "lemma", "induction", "contradiction",
    "integral", "differential", "eigenvalue", "convergence", "formal",
    # Debugging / analysis
    "debug", "diagnose", "root cause", "race condition", "deadlock",
    "memory leak", "stack trace", "segfault", "core dump", "intermittent",
    # Deep analysis
    "analyze", "evaluate", "compare and contrast", "trade-offs", "tradeoffs",
    "implications", "consequences", "architecture", "design pattern",
    # System design
    "distributed", "consensus", "replication", "partitioning", "fault-tolerant",
    "byzantine", "consistency", "availability", "scalability",
    "throughput", "latency", "concurrent", "lock-free",
    # Implementation depth
    "from scratch", "implement", "pseudocode", "state machine",
    "data structure", "algorithm", "complexity", "optimization",
    "edge cases", "failure recovery", "error handling",
    # Multi-step
    "step by step", "step-by-step", "first.*then.*finally",
    "walk me through", "explain in detail", "comprehensive",
    "cover", "show the", "show how",
}

SIMPLE_PATTERNS = {
    # Factual lookups
    "what is", "what's", "who is", "who's", "when did", "when was",
    "where is", "where's", "how many", "how much", "define",
    # Simple conversions / lookups
    "convert", "translate", "capital of", "population of",
    # Yes/no
    "is it true", "can you", "does it", "will it",
}

TRIVIAL_PATTERNS = {
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "yes", "no", "ok", "okay", "sure", "got it",
}

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
MATH_RE = re.compile(r"[\+\-\*/\^=<>≤≥∫∑∏√]|\\frac|\\int|\\sum")
QUESTION_RE = re.compile(r"\?")
MULTI_STEP_RE = re.compile(
    r"\b(first|second|third|then|next|finally|step\s*\d|part\s*\d)\b", re.IGNORECASE
)
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)


@dataclass
class ClassificationResult:
    """Result of query classification."""

    tier: ComplexityTier
    confidence: float
    signals: ClassifierSignals
    reasoning: str


class QueryClassifier:
    """Classifies query complexity using heuristic signals.

    No LLM calls — classification runs in <1ms.
    """

    def __init__(self, custom_weights: dict[str, float] | None = None):
        self.weights = {
            "token_count": 1.0,
            "code_block": 1.5,
            "math": 1.3,
            "questions": 1.0,
            "reasoning_keywords": 2.0,
            "multi_step": 1.5,
            "instruction_complexity": 1.0,
            **(custom_weights or {}),
        }

    def extract_signals(self, query: str) -> ClassifierSignals:
        """Extract classification signals from a query string."""
        words = query.split()
        token_estimate = int(len(words) * 1.3)  # rough token estimate

        code_blocks = CODE_BLOCK_RE.findall(query)
        has_code = len(code_blocks) > 0

        has_math = bool(MATH_RE.search(query))

        questions = QUESTION_RE.findall(query)
        question_count = len(questions)

        query_lower = query.lower()
        reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in query_lower)

        has_multi_step = bool(MULTI_STEP_RE.search(query)) or bool(
            NUMBERED_LIST_RE.search(query)
        )

        # Instruction complexity: count distinct action verbs
        action_verbs = {
            "write", "create", "build", "implement", "design", "refactor",
            "optimize", "fix", "test", "deploy", "configure", "setup",
            "explain", "describe", "list", "generate", "modify", "update",
        }
        instruction_complexity = sum(1 for v in action_verbs if v in query_lower)

        avg_word_length = (
            sum(len(w) for w in words) / len(words) if words else 0.0
        )

        return ClassifierSignals(
            token_count=token_estimate,
            has_code_block=has_code,
            has_math_operators=has_math,
            question_count=question_count,
            reasoning_keywords=reasoning_hits,
            instruction_complexity=instruction_complexity,
            avg_word_length=round(avg_word_length, 2),
            has_multi_step=has_multi_step,
        )

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query into a complexity tier."""
        query_stripped = query.strip()

        # Fast path: trivial detection
        words = query_stripped.split()
        if len(words) <= 3:
            lower = query_stripped.lower().rstrip("!?.,")
            # Check it's not a simple factual question
            is_factual = any(lower.startswith(p) for p in SIMPLE_PATTERNS)
            if not is_factual and (lower in TRIVIAL_PATTERNS or len(query_stripped) < 10):
                return ClassificationResult(
                    tier=ComplexityTier.TRIVIAL,
                    confidence=0.95,
                    signals=ClassifierSignals(token_count=len(words)),
                    reasoning="Very short query / greeting",
                )

        signals = self.extract_signals(query_stripped)

        # Compute weighted complexity score (0-100)
        score = 0.0

        # Token count contribution (0-40 points)
        # Longer queries almost always need more reasoning
        token_score = min(40, signals.token_count / 6)
        score += token_score * self.weights["token_count"]

        # Code block contribution (0-15 points)
        if signals.has_code_block:
            score += 15 * self.weights["code_block"]

        # Math contribution (0-10 points)
        if signals.has_math_operators:
            score += 10 * self.weights["math"]

        # Question count (0-10 points)
        score += min(10, signals.question_count * 3) * self.weights["questions"]

        # Reasoning keywords (0-20 points) — strongest signal
        score += min(20, signals.reasoning_keywords * 5) * self.weights["reasoning_keywords"]

        # Multi-step (0-10 points)
        if signals.has_multi_step:
            score += 10 * self.weights["multi_step"]

        # Instruction complexity (0-10 points)
        score += min(10, signals.instruction_complexity * 3) * self.weights[
            "instruction_complexity"
        ]

        # Check for simple pattern override — factual questions should be at least SIMPLE
        query_lower = query_stripped.lower()
        is_simple_pattern = any(query_lower.startswith(p) for p in SIMPLE_PATTERNS)
        if is_simple_pattern and score < 20:
            score = max(score, 10)  # Ensure factual questions reach at least SIMPLE tier

        # Map score to tier
        tier, confidence, reasoning = self._score_to_tier(score, signals)

        return ClassificationResult(
            tier=tier,
            confidence=round(confidence, 3),
            signals=signals,
            reasoning=reasoning,
        )

    def _score_to_tier(
        self, score: float, signals: ClassifierSignals
    ) -> tuple[ComplexityTier, float, str]:
        """Map a complexity score to a tier with confidence."""
        if score < 8:
            return ComplexityTier.TRIVIAL, 0.9, f"Score {score:.1f} < 8"
        elif score < 20:
            # Confidence drops near boundaries
            conf = 0.85 if score < 15 else 0.7
            return ComplexityTier.SIMPLE, conf, f"Score {score:.1f} in [8, 20)"
        elif score < 40:
            conf = 0.8 if 25 < score < 35 else 0.65
            return ComplexityTier.MODERATE, conf, f"Score {score:.1f} in [20, 40)"
        elif score < 65:
            conf = 0.8 if 45 < score < 55 else 0.7
            return ComplexityTier.COMPLEX, conf, f"Score {score:.1f} in [40, 65)"
        else:
            return ComplexityTier.DEEP, 0.85, f"Score {score:.1f} >= 65"

    def classify_messages(self, messages: list[dict]) -> ClassificationResult:
        """Classify from OpenAI-format messages array.

        Uses the last user message for classification.
        """
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return ClassificationResult(
                tier=ComplexityTier.MODERATE,
                confidence=0.3,
                signals=ClassifierSignals(),
                reasoning="No user message found, defaulting to MODERATE",
            )

        last_message = user_messages[-1].get("content", "")
        if isinstance(last_message, list):
            # Handle multimodal content
            text_parts = [p.get("text", "") for p in last_message if p.get("type") == "text"]
            last_message = " ".join(text_parts)

        return self.classify(last_message)
