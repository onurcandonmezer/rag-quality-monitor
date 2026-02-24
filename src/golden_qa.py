"""Golden Q&A Test Suite Management.

Manages golden question-answer test sets for RAG quality regression testing.
Loads Q&A pairs from YAML, runs evaluation suites, and detects regressions
by comparing current scores against baseline results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evaluator import EvaluationResult, RAGEvaluator


@dataclass
class GoldenQAPair:
    """A single golden Q&A pair for testing."""

    id: str
    question: str
    expected_answer: str
    context: str
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class GoldenQATestResult:
    """Result of running a single golden Q&A test."""

    qa_pair: GoldenQAPair
    evaluation: EvaluationResult
    passed: bool
    generated_answer: str


@dataclass
class GoldenQAResult:
    """Result of running the full golden Q&A test suite."""

    results: list[GoldenQATestResult]
    pass_rate: float
    total: int
    passed: int
    failed: int
    regression_detected: bool
    regression_details: list[dict[str, Any]] = field(default_factory=list)
    avg_scores: dict[str, float] = field(default_factory=dict)


class GoldenQAManager:
    """Manages golden Q&A test sets for RAG quality regression testing.

    Loads curated Q&A pairs from YAML files, evaluates RAG system
    answers against them, and detects quality regressions by comparing
    current results against stored baseline scores.
    """

    def __init__(
        self,
        evaluator: RAGEvaluator | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        """Initialize the manager.

        Args:
            evaluator: RAGEvaluator instance. Creates default if None.
            pass_threshold: Minimum overall score for a test to pass.
        """
        self.evaluator = evaluator or RAGEvaluator()
        self.pass_threshold = pass_threshold
        self._qa_pairs: list[GoldenQAPair] = []
        self._baseline_scores: dict[str, float] = {}

    @property
    def qa_pairs(self) -> list[GoldenQAPair]:
        """Get loaded Q&A pairs."""
        return self._qa_pairs

    def load_from_yaml(self, filepath: str | Path) -> list[GoldenQAPair]:
        """Load golden Q&A pairs from a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            List of loaded GoldenQAPair objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML structure is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Golden Q&A file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "qa_pairs" not in data:
            raise ValueError("YAML file must contain a 'qa_pairs' key")

        pairs = []
        for item in data["qa_pairs"]:
            pair = GoldenQAPair(
                id=item.get("id", f"qa_{len(pairs):03d}"),
                question=item["question"],
                expected_answer=item["expected_answer"],
                context=item["context"],
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "medium"),
            )
            pairs.append(pair)

        self._qa_pairs = pairs
        return pairs

    def load_from_list(self, pairs: list[dict[str, Any]]) -> list[GoldenQAPair]:
        """Load golden Q&A pairs from a list of dicts.

        Args:
            pairs: List of dicts with Q&A data.

        Returns:
            List of loaded GoldenQAPair objects.
        """
        loaded = []
        for item in pairs:
            pair = GoldenQAPair(
                id=item.get("id", f"qa_{len(loaded):03d}"),
                question=item["question"],
                expected_answer=item["expected_answer"],
                context=item["context"],
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "medium"),
            )
            loaded.append(pair)

        self._qa_pairs = loaded
        return loaded

    def set_baseline(self, scores: dict[str, float]) -> None:
        """Set baseline scores for regression detection.

        Args:
            scores: Dict mapping Q&A pair IDs to their baseline overall scores.
        """
        self._baseline_scores = scores.copy()

    def set_baseline_from_results(self, result: GoldenQAResult) -> None:
        """Set baseline from a previous test run result.

        Args:
            result: A GoldenQAResult to use as baseline.
        """
        self._baseline_scores = {
            r.qa_pair.id: r.evaluation.overall_score for r in result.results
        }

    def run_test_suite(
        self,
        answer_fn: callable | None = None,
        answers: dict[str, str] | None = None,
        regression_threshold: float = 0.1,
    ) -> GoldenQAResult:
        """Run the golden Q&A test suite.

        Either provide an answer_fn that generates answers given
        (question, context), or a dict mapping Q&A pair IDs to
        pre-generated answers.

        Args:
            answer_fn: Function(question, context) -> answer string.
            answers: Dict mapping Q&A IDs to answer strings.
            regression_threshold: Score drop to trigger regression alert.

        Returns:
            GoldenQAResult with pass/fail details and regression info.

        Raises:
            ValueError: If neither answer_fn nor answers is provided,
                or if no Q&A pairs are loaded.
        """
        if not self._qa_pairs:
            raise ValueError("No Q&A pairs loaded. Call load_from_yaml or load_from_list first.")

        if answer_fn is None and answers is None:
            raise ValueError("Provide either answer_fn or answers dict")

        results: list[GoldenQATestResult] = []
        regression_details: list[dict[str, Any]] = []

        for pair in self._qa_pairs:
            if answers and pair.id in answers:
                generated_answer = answers[pair.id]
            elif answer_fn:
                generated_answer = answer_fn(pair.question, pair.context)
            else:
                generated_answer = ""

            evaluation = self.evaluator.evaluate(
                question=pair.question,
                answer=generated_answer,
                context=pair.context,
                expected_answer=pair.expected_answer,
            )

            passed = evaluation.overall_score >= self.pass_threshold

            test_result = GoldenQATestResult(
                qa_pair=pair,
                evaluation=evaluation,
                passed=passed,
                generated_answer=generated_answer,
            )
            results.append(test_result)

            if pair.id in self._baseline_scores:
                baseline = self._baseline_scores[pair.id]
                drop = baseline - evaluation.overall_score
                if drop > regression_threshold:
                    regression_details.append({
                        "id": pair.id,
                        "question": pair.question,
                        "baseline_score": baseline,
                        "current_score": evaluation.overall_score,
                        "score_drop": round(drop, 4),
                    })

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = total - passed_count
        pass_rate = passed_count / total if total > 0 else 0.0

        avg_scores = {}
        if results:
            avg_scores = {
                "faithfulness": round(
                    sum(r.evaluation.faithfulness for r in results) / total, 4
                ),
                "relevance": round(
                    sum(r.evaluation.relevance for r in results) / total, 4
                ),
                "recall": round(
                    sum(r.evaluation.recall for r in results) / total, 4
                ),
                "precision": round(
                    sum(r.evaluation.precision for r in results) / total, 4
                ),
                "overall": round(
                    sum(r.evaluation.overall_score for r in results) / total, 4
                ),
            }

        return GoldenQAResult(
            results=results,
            pass_rate=round(pass_rate, 4),
            total=total,
            passed=passed_count,
            failed=failed_count,
            regression_detected=len(regression_details) > 0,
            regression_details=regression_details,
            avg_scores=avg_scores,
        )

    def filter_by_tag(self, tag: str) -> list[GoldenQAPair]:
        """Filter loaded Q&A pairs by tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of matching GoldenQAPair objects.
        """
        return [p for p in self._qa_pairs if tag in p.tags]

    def filter_by_difficulty(self, difficulty: str) -> list[GoldenQAPair]:
        """Filter loaded Q&A pairs by difficulty.

        Args:
            difficulty: Difficulty level (easy, medium, hard).

        Returns:
            List of matching GoldenQAPair objects.
        """
        return [p for p in self._qa_pairs if p.difficulty == difficulty]
