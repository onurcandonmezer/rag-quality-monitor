"""Tests for RAG evaluation suite."""

from __future__ import annotations

from collections import Counter

import pytest

from src.evaluator import (
    BatchEvaluationResult,
    EvaluationResult,
    RAGEvaluator,
    _cosine_similarity,
    _jaccard_similarity,
    _tokenize,
)


class TestTokenize:
    """Tests for text tokenization."""

    def test_basic_tokenization(self):
        tokens = _tokenize("The quick brown fox jumps over the lazy dog")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        # Stop words should be removed
        assert "the" not in tokens
        assert "over" not in tokens

    def test_empty_text(self):
        assert _tokenize("") == []

    def test_punctuation_removal(self):
        tokens = _tokenize("Hello, world! This is a test.")
        assert all("," not in t and "!" not in t and "." not in t for t in tokens)


class TestSimilarityFunctions:
    """Tests for similarity helper functions."""

    def test_cosine_similarity_identical(self):
        vec = Counter({"hello": 2, "world": 1})
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_disjoint(self):
        vec_a = Counter({"hello": 1})
        vec_b = Counter({"world": 1})
        assert _cosine_similarity(vec_a, vec_b) == 0.0

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity(Counter(), Counter({"a": 1})) == 0.0

    def test_jaccard_similarity_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_jaccard_similarity_disjoint(self):
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_similarity_partial(self):
        result = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert result == pytest.approx(2 / 4)

    def test_jaccard_similarity_both_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0


class TestRAGEvaluator:
    """Tests for the RAGEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return RAGEvaluator()

    def test_score_faithfulness_grounded(self, evaluator):
        context = "Python is a programming language created by Guido van Rossum."
        answer = "Python is a programming language created by Guido van Rossum."
        score = evaluator.score_faithfulness(answer, context)
        assert 0.5 <= score <= 1.0

    def test_score_faithfulness_ungrounded(self, evaluator):
        context = "Python is a programming language."
        answer = "Java was invented by James Gosling at Sun Microsystems in 1995."
        score = evaluator.score_faithfulness(answer, context)
        assert score < 0.5

    def test_score_faithfulness_empty(self, evaluator):
        assert evaluator.score_faithfulness("", "some context") == 0.0
        assert evaluator.score_faithfulness("some answer", "") == 0.0

    def test_score_relevance_relevant(self, evaluator):
        question = "What is machine learning?"
        answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        score = evaluator.score_relevance(answer, question)
        assert score > 0.3

    def test_score_relevance_irrelevant(self, evaluator):
        question = "What is machine learning?"
        answer = "The weather today is sunny with a high of 75 degrees Fahrenheit."
        score = evaluator.score_relevance(answer, question)
        assert score < 0.3

    def test_score_recall_full_coverage(self, evaluator):
        expected = "RAG combines retrieval with generation for better answers."
        answer = (
            "RAG combines retrieval with text generation to produce better, more grounded answers."
        )
        score = evaluator.score_recall(answer, expected)
        assert score > 0.5

    def test_score_recall_no_coverage(self, evaluator):
        expected = "RAG combines retrieval with generation."
        answer = "The sun rises in the east and sets in the west."
        score = evaluator.score_recall(answer, expected)
        assert score < 0.3

    def test_score_precision_precise(self, evaluator):
        expected = "Python is a programming language."
        context = "Python is a popular programming language used in web development."
        answer = "Python is a programming language."
        score = evaluator.score_precision(answer, expected, context)
        assert score > 0.5

    def test_evaluate_full(self, evaluator):
        result = evaluator.evaluate(
            question="What is Python?",
            answer="Python is a programming language created by Guido van Rossum.",
            context="Python is a high-level programming language created by Guido van Rossum in 1991.",
            expected_answer="Python is a programming language created by Guido van Rossum.",
        )
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.relevance <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
        assert result.question == "What is Python?"

    def test_evaluate_passed(self, evaluator):
        result = evaluator.evaluate(
            question="What is Python?",
            answer="Python is a programming language.",
            context="Python is a programming language.",
            expected_answer="Python is a programming language.",
        )
        assert result.passed(threshold=0.3)

    def test_evaluate_batch(self, evaluator):
        pairs = [
            {
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "context": "Python is a high-level programming language.",
                "expected_answer": "Python is a programming language.",
            },
            {
                "question": "What is JavaScript?",
                "answer": "JavaScript is a scripting language for the web.",
                "context": "JavaScript is a lightweight scripting language for web browsers.",
                "expected_answer": "JavaScript is a scripting language for web development.",
            },
        ]
        result = evaluator.evaluate_batch(pairs)
        assert isinstance(result, BatchEvaluationResult)
        assert len(result.results) == 2
        assert 0.0 <= result.avg_overall <= 1.0
        assert 0.0 <= result.pass_rate <= 1.0

    def test_evaluate_batch_empty(self, evaluator):
        result = evaluator.evaluate_batch([])
        assert result.avg_overall == 0.0
        assert result.pass_rate == 0.0

    def test_custom_weights(self):
        evaluator = RAGEvaluator(
            faithfulness_weight=1.0,
            relevance_weight=0.0,
            recall_weight=0.0,
            precision_weight=0.0,
        )
        assert evaluator.weights["faithfulness"] == pytest.approx(1.0)
