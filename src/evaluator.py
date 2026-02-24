"""RAG Evaluation Suite.

Provides comprehensive evaluation of RAG system responses across multiple
quality dimensions: faithfulness, relevance, recall, and precision.
Uses keyword overlap and text similarity for scoring (no API needed).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG response."""

    question: str
    answer: str
    context: str
    expected_answer: str
    faithfulness: float
    relevance: float
    recall: float
    precision: float
    overall_score: float
    details: dict[str, Any] = field(default_factory=dict)

    def passed(self, threshold: float = 0.5) -> bool:
        """Check if the evaluation passed a minimum threshold."""
        return self.overall_score >= threshold


@dataclass
class BatchEvaluationResult:
    """Result of evaluating multiple RAG responses."""

    results: list[EvaluationResult]
    avg_faithfulness: float
    avg_relevance: float
    avg_recall: float
    avg_precision: float
    avg_overall: float
    pass_rate: float


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, removing punctuation and stopwords."""
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only", "own",
        "same", "than", "too", "very", "just", "because", "if", "when",
        "where", "how", "what", "which", "who", "whom", "this", "that",
        "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "him", "his", "she", "her", "they", "them", "their",
    }
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def _compute_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Compute n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine similarity between two Counter vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _extract_sentences(text: str) -> list[str]:
    """Extract sentences from text."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


class RAGEvaluator:
    """Evaluates RAG system responses across multiple quality dimensions.

    Scoring is done using keyword overlap, n-gram matching, and text
    similarity techniques. No external API calls are required.

    Metrics:
        - Faithfulness: Does the answer stick to the retrieved context? (0-1)
        - Relevance: Is the answer relevant to the question? (0-1)
        - Recall: How much of the expected answer is covered? (0-1)
        - Precision: How precise is the answer (no unnecessary info)? (0-1)
    """

    def __init__(
        self,
        faithfulness_weight: float = 0.3,
        relevance_weight: float = 0.25,
        recall_weight: float = 0.25,
        precision_weight: float = 0.2,
    ) -> None:
        self.weights = {
            "faithfulness": faithfulness_weight,
            "relevance": relevance_weight,
            "recall": recall_weight,
            "precision": precision_weight,
        }
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score_faithfulness(self, answer: str, context: str) -> float:
        """Score how well the answer sticks to the retrieved context.

        Measures the fraction of answer content that can be traced back
        to the context. High faithfulness means the answer does not
        introduce information absent from the context.

        Args:
            answer: The generated answer.
            context: The retrieved context documents.

        Returns:
            Score between 0 and 1.
        """
        if not answer.strip() or not context.strip():
            return 0.0

        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)

        if not answer_tokens:
            return 0.0

        context_set = set(context_tokens)
        supported_count = sum(1 for t in answer_tokens if t in context_set)
        unigram_score = supported_count / len(answer_tokens)

        answer_bigrams = set(_compute_ngrams(answer_tokens, 2))
        context_bigrams = set(_compute_ngrams(context_tokens, 2))
        bigram_score = (
            _jaccard_similarity(answer_bigrams, context_bigrams)
            if answer_bigrams
            else 0.0
        )

        answer_sentences = _extract_sentences(answer)
        sentence_scores = []
        for sent in answer_sentences:
            sent_tokens = set(_tokenize(sent))
            if sent_tokens:
                overlap = len(sent_tokens & context_set) / len(sent_tokens)
                sentence_scores.append(overlap)
        sentence_score = (
            sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
        )

        return 0.4 * unigram_score + 0.3 * bigram_score + 0.3 * sentence_score

    def score_relevance(self, answer: str, question: str) -> float:
        """Score how relevant the answer is to the question.

        Measures semantic alignment between the question and answer
        using token overlap and vector similarity.

        Args:
            answer: The generated answer.
            question: The original question.

        Returns:
            Score between 0 and 1.
        """
        if not answer.strip() or not question.strip():
            return 0.0

        answer_tokens = _tokenize(answer)
        question_tokens = _tokenize(question)

        if not answer_tokens or not question_tokens:
            return 0.0

        question_set = set(question_tokens)
        answer_set = set(answer_tokens)
        keyword_overlap = len(question_set & answer_set) / len(question_set)

        answer_counter = Counter(answer_tokens)
        question_counter = Counter(question_tokens)
        cosine_sim = _cosine_similarity(answer_counter, question_counter)

        return 0.6 * keyword_overlap + 0.4 * cosine_sim

    def score_recall(self, answer: str, expected_answer: str) -> float:
        """Score how much of the expected answer is covered.

        Measures the fraction of expected answer content present in
        the generated answer.

        Args:
            answer: The generated answer.
            expected_answer: The ground-truth expected answer.

        Returns:
            Score between 0 and 1.
        """
        if not expected_answer.strip():
            return 1.0 if not answer.strip() else 0.0
        if not answer.strip():
            return 0.0

        expected_tokens = _tokenize(expected_answer)
        answer_tokens = _tokenize(answer)

        if not expected_tokens:
            return 1.0

        answer_set = set(answer_tokens)
        covered = sum(1 for t in expected_tokens if t in answer_set)
        unigram_recall = covered / len(expected_tokens)

        expected_bigrams = set(_compute_ngrams(expected_tokens, 2))
        answer_bigrams = set(_compute_ngrams(answer_tokens, 2))
        bigram_recall = (
            len(expected_bigrams & answer_bigrams) / len(expected_bigrams)
            if expected_bigrams
            else 0.0
        )

        return 0.6 * unigram_recall + 0.4 * bigram_recall

    def score_precision(self, answer: str, expected_answer: str, context: str) -> float:
        """Score how precise the answer is (no unnecessary information).

        Measures whether the answer contains mostly relevant information
        without extraneous content.

        Args:
            answer: The generated answer.
            expected_answer: The ground-truth expected answer.
            context: The retrieved context.

        Returns:
            Score between 0 and 1.
        """
        if not answer.strip():
            return 0.0

        answer_tokens = _tokenize(answer)
        expected_tokens = set(_tokenize(expected_answer))
        context_tokens = set(_tokenize(context))

        if not answer_tokens:
            return 0.0

        relevant_tokens = expected_tokens | context_tokens
        relevant_count = sum(1 for t in answer_tokens if t in relevant_tokens)
        token_precision = relevant_count / len(answer_tokens)

        answer_len = len(answer_tokens)
        expected_len = len(expected_tokens) if expected_tokens else 1
        length_ratio = min(answer_len, expected_len * 2) / max(answer_len, expected_len * 2)
        length_penalty = min(1.0, length_ratio * 1.5)

        return 0.7 * token_precision + 0.3 * length_penalty

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        expected_answer: str,
    ) -> EvaluationResult:
        """Evaluate a single RAG response across all metrics.

        Args:
            question: The original question.
            answer: The generated answer.
            context: The retrieved context documents.
            expected_answer: The ground-truth expected answer.

        Returns:
            EvaluationResult with all scores.
        """
        faithfulness = self.score_faithfulness(answer, context)
        relevance = self.score_relevance(answer, question)
        recall = self.score_recall(answer, expected_answer)
        precision = self.score_precision(answer, expected_answer, context)

        overall = (
            self.weights["faithfulness"] * faithfulness
            + self.weights["relevance"] * relevance
            + self.weights["recall"] * recall
            + self.weights["precision"] * precision
        )

        return EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            expected_answer=expected_answer,
            faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4),
            recall=round(recall, 4),
            precision=round(precision, 4),
            overall_score=round(overall, 4),
            details={
                "weights": self.weights,
                "answer_length": len(answer.split()),
                "context_length": len(context.split()),
            },
        )

    def evaluate_batch(
        self,
        qa_pairs: list[dict[str, str]],
        pass_threshold: float = 0.5,
    ) -> BatchEvaluationResult:
        """Evaluate a batch of Q&A pairs.

        Each pair should have keys: question, answer, context, expected_answer.

        Args:
            qa_pairs: List of dicts with Q&A data.
            pass_threshold: Minimum overall score to pass.

        Returns:
            BatchEvaluationResult with aggregated metrics.
        """
        results = []
        for pair in qa_pairs:
            result = self.evaluate(
                question=pair["question"],
                answer=pair["answer"],
                context=pair["context"],
                expected_answer=pair["expected_answer"],
            )
            results.append(result)

        if not results:
            return BatchEvaluationResult(
                results=[],
                avg_faithfulness=0.0,
                avg_relevance=0.0,
                avg_recall=0.0,
                avg_precision=0.0,
                avg_overall=0.0,
                pass_rate=0.0,
            )

        n = len(results)
        avg_faithfulness = sum(r.faithfulness for r in results) / n
        avg_relevance = sum(r.relevance for r in results) / n
        avg_recall = sum(r.recall for r in results) / n
        avg_precision = sum(r.precision for r in results) / n
        avg_overall = sum(r.overall_score for r in results) / n
        pass_rate = sum(1 for r in results if r.passed(pass_threshold)) / n

        return BatchEvaluationResult(
            results=results,
            avg_faithfulness=round(avg_faithfulness, 4),
            avg_relevance=round(avg_relevance, 4),
            avg_recall=round(avg_recall, 4),
            avg_precision=round(avg_precision, 4),
            avg_overall=round(avg_overall, 4),
            pass_rate=round(pass_rate, 4),
        )
