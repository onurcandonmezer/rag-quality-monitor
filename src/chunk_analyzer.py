"""Chunk Quality Analysis.

Analyzes text chunks used in RAG systems for quality metrics including
relevance, coherence, information density, and overlap with neighbors.
Provides recommendations for chunk size optimization.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkQuality:
    """Quality metrics for a single text chunk."""

    chunk_index: int
    text: str
    length: int
    relevance_score: float
    coherence_score: float
    information_density: float
    overlap_with_neighbors: float
    issues: list[str] = field(default_factory=list)


@dataclass
class ChunkAnalysisReport:
    """Report from analyzing a set of chunks."""

    chunks: list[ChunkQuality]
    total_chunks: int
    avg_length: float
    avg_relevance: float
    avg_coherence: float
    avg_information_density: float
    avg_overlap: float
    problematic_chunks: list[int]
    recommendations: list[str]
    overall_quality: float
    stats: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 1]


def _unique_tokens(text: str) -> set[str]:
    """Get unique meaningful tokens from text."""
    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "but",
        "or",
        "not",
        "it",
        "its",
        "this",
        "that",
    }
    tokens = _tokenize(text)
    return {t for t in tokens if t not in stop_words}


class ChunkAnalyzer:
    """Analyzes text chunks for quality in RAG systems.

    Evaluates chunks on coherence, information density, overlap with
    neighbors, and length. Provides actionable recommendations for
    improving chunk quality.
    """

    def __init__(
        self,
        min_chunk_length: int = 50,
        max_chunk_length: int = 2000,
        max_overlap_ratio: float = 0.5,
        min_information_density: float = 0.3,
    ) -> None:
        """Initialize the analyzer.

        Args:
            min_chunk_length: Minimum acceptable chunk length in characters.
            max_chunk_length: Maximum acceptable chunk length in characters.
            max_overlap_ratio: Maximum acceptable overlap ratio between neighbors.
            min_information_density: Minimum acceptable information density.
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.max_overlap_ratio = max_overlap_ratio
        self.min_information_density = min_information_density

    def _score_coherence(self, text: str) -> float:
        """Score the coherence of a text chunk.

        Coherence is estimated by sentence structure quality,
        presence of transition words, and consistent topic focus.

        Args:
            text: The chunk text.

        Returns:
            Coherence score between 0 and 1.
        """
        if not text.strip():
            return 0.0

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        sentence_score = min(1.0, len(sentences) / 3)

        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 1.0
        if avg_sentence_len < 5:
            length_score = avg_sentence_len / 5
        elif avg_sentence_len > 40:
            length_score = max(0.3, 1.0 - (avg_sentence_len - 40) / 60)

        transition_words = {
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "meanwhile",
            "nevertheless",
            "similarly",
            "specifically",
            "particularly",
            "including",
            "example",
            "first",
            "second",
            "third",
            "finally",
            "also",
            "such",
        }
        tokens = set(_tokenize(text))
        transition_count = len(tokens & transition_words)
        transition_score = min(1.0, transition_count / 2)

        if len(sentences) >= 2:
            token_sets = [set(_tokenize(s)) for s in sentences]
            overlaps = []
            for i in range(len(token_sets) - 1):
                if token_sets[i] and token_sets[i + 1]:
                    overlap = len(token_sets[i] & token_sets[i + 1])
                    total = len(token_sets[i] | token_sets[i + 1])
                    overlaps.append(overlap / total if total > 0 else 0)
            topic_consistency = sum(overlaps) / len(overlaps) if overlaps else 0.0
        else:
            topic_consistency = 0.5

        coherence = (
            0.25 * sentence_score
            + 0.25 * length_score
            + 0.2 * transition_score
            + 0.3 * topic_consistency
        )

        return min(1.0, max(0.0, coherence))

    def _score_information_density(self, text: str) -> float:
        """Score the information density of a text chunk.

        Information density measures the ratio of unique, meaningful
        content words to total words.

        Args:
            text: The chunk text.

        Returns:
            Information density score between 0 and 1.
        """
        if not text.strip():
            return 0.0

        all_tokens = _tokenize(text)
        if not all_tokens:
            return 0.0

        unique_meaningful = _unique_tokens(text)
        unique_ratio = len(unique_meaningful) / len(all_tokens)

        word_counts = Counter(all_tokens)
        total_words = len(all_tokens)
        repetition = sum(c - 1 for c in word_counts.values() if c > 1) / total_words
        non_repetition_score = max(0.0, 1.0 - repetition)

        capitalized_words = sum(1 for w in text.split() if w and w[0].isupper() and len(w) > 1)
        proper_noun_density = min(1.0, capitalized_words / max(1, len(all_tokens)) * 5)

        density = 0.5 * unique_ratio + 0.3 * non_repetition_score + 0.2 * proper_noun_density

        return min(1.0, max(0.0, density))

    def _compute_overlap(self, chunk_a: str, chunk_b: str) -> float:
        """Compute token overlap ratio between two chunks.

        Args:
            chunk_a: First chunk text.
            chunk_b: Second chunk text.

        Returns:
            Overlap ratio between 0 and 1.
        """
        tokens_a = _unique_tokens(chunk_a)
        tokens_b = _unique_tokens(chunk_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))

        return len(intersection) / smaller if smaller > 0 else 0.0

    def _score_relevance(self, chunk: str, query: str | None = None) -> float:
        """Score chunk relevance to an optional query.

        If no query is provided, scores based on intrinsic content quality.

        Args:
            chunk: The chunk text.
            query: Optional query to measure relevance against.

        Returns:
            Relevance score between 0 and 1.
        """
        if not chunk.strip():
            return 0.0

        if query:
            chunk_tokens = _unique_tokens(chunk)
            query_tokens = _unique_tokens(query)
            if not query_tokens:
                return 0.5
            overlap = len(chunk_tokens & query_tokens) / len(query_tokens)
            return min(1.0, overlap * 1.5)

        tokens = _tokenize(chunk)
        if not tokens:
            return 0.0

        unique_ratio = len(set(tokens)) / len(tokens)
        has_sentences = len(re.split(r"[.!?]+", chunk)) > 1
        has_substance = len(tokens) >= 10

        score = 0.4 * unique_ratio + 0.3 * float(has_sentences) + 0.3 * float(has_substance)
        return min(1.0, max(0.0, score))

    def analyze_chunk(
        self,
        text: str,
        index: int = 0,
        neighbors: tuple[str | None, str | None] = (None, None),
        query: str | None = None,
    ) -> ChunkQuality:
        """Analyze a single chunk for quality.

        Args:
            text: The chunk text.
            index: Index of the chunk in the set.
            neighbors: (previous_chunk, next_chunk) or None.
            query: Optional query for relevance scoring.

        Returns:
            ChunkQuality with all metrics.
        """
        issues: list[str] = []
        length = len(text)

        if length < self.min_chunk_length:
            issues.append(f"Chunk too short ({length} chars < {self.min_chunk_length})")
        if length > self.max_chunk_length:
            issues.append(f"Chunk too long ({length} chars > {self.max_chunk_length})")

        coherence = self._score_coherence(text)
        info_density = self._score_information_density(text)
        relevance = self._score_relevance(text, query)

        overlap_scores = []
        prev_chunk, next_chunk = neighbors
        if prev_chunk:
            overlap_scores.append(self._compute_overlap(text, prev_chunk))
        if next_chunk:
            overlap_scores.append(self._compute_overlap(text, next_chunk))
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

        if avg_overlap > self.max_overlap_ratio:
            issues.append(
                f"High overlap with neighbors ({avg_overlap:.2f} > {self.max_overlap_ratio})"
            )
        if info_density < self.min_information_density:
            issues.append(
                f"Low information density ({info_density:.2f} < {self.min_information_density})"
            )

        return ChunkQuality(
            chunk_index=index,
            text=text,
            length=length,
            relevance_score=round(relevance, 4),
            coherence_score=round(coherence, 4),
            information_density=round(info_density, 4),
            overlap_with_neighbors=round(avg_overlap, 4),
            issues=issues,
        )

    def analyze(
        self,
        chunks: list[str],
        query: str | None = None,
    ) -> ChunkAnalysisReport:
        """Analyze a set of chunks for quality.

        Args:
            chunks: List of text chunks.
            query: Optional query for relevance scoring.

        Returns:
            ChunkAnalysisReport with overall statistics and recommendations.
        """
        if not chunks:
            return ChunkAnalysisReport(
                chunks=[],
                total_chunks=0,
                avg_length=0.0,
                avg_relevance=0.0,
                avg_coherence=0.0,
                avg_information_density=0.0,
                avg_overlap=0.0,
                problematic_chunks=[],
                recommendations=["No chunks provided for analysis."],
                overall_quality=0.0,
            )

        analyzed: list[ChunkQuality] = []
        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
            result = self.analyze_chunk(
                text=chunk,
                index=i,
                neighbors=(prev_chunk, next_chunk),
                query=query,
            )
            analyzed.append(result)

        n = len(analyzed)
        avg_length = sum(c.length for c in analyzed) / n
        avg_relevance = sum(c.relevance_score for c in analyzed) / n
        avg_coherence = sum(c.coherence_score for c in analyzed) / n
        avg_density = sum(c.information_density for c in analyzed) / n
        avg_overlap = sum(c.overlap_with_neighbors for c in analyzed) / n

        problematic = [c.chunk_index for c in analyzed if c.issues]

        recommendations = self._generate_recommendations(analyzed, avg_length)

        overall = (
            0.3 * avg_relevance
            + 0.25 * avg_coherence
            + 0.25 * avg_density
            + 0.2 * (1.0 - min(1.0, avg_overlap / self.max_overlap_ratio))
        )

        lengths = [c.length for c in analyzed]
        stats = {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "median_length": sorted(lengths)[n // 2],
            "std_length": (
                math.sqrt(sum((ln - avg_length) ** 2 for ln in lengths) / n) if n > 1 else 0.0
            ),
            "num_too_short": sum(1 for c in analyzed if c.length < self.min_chunk_length),
            "num_too_long": sum(1 for c in analyzed if c.length > self.max_chunk_length),
            "num_high_overlap": sum(
                1 for c in analyzed if c.overlap_with_neighbors > self.max_overlap_ratio
            ),
            "num_low_density": sum(
                1 for c in analyzed if c.information_density < self.min_information_density
            ),
        }

        return ChunkAnalysisReport(
            chunks=analyzed,
            total_chunks=n,
            avg_length=round(avg_length, 2),
            avg_relevance=round(avg_relevance, 4),
            avg_coherence=round(avg_coherence, 4),
            avg_information_density=round(avg_density, 4),
            avg_overlap=round(avg_overlap, 4),
            problematic_chunks=problematic,
            recommendations=recommendations,
            overall_quality=round(overall, 4),
            stats=stats,
        )

    def _generate_recommendations(
        self,
        chunks: list[ChunkQuality],
        avg_length: float,
    ) -> list[str]:
        """Generate optimization recommendations based on analysis.

        Args:
            chunks: Analyzed chunk quality results.
            avg_length: Average chunk length.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        too_short = [c for c in chunks if c.length < self.min_chunk_length]
        too_long = [c for c in chunks if c.length > self.max_chunk_length]
        high_overlap = [c for c in chunks if c.overlap_with_neighbors > self.max_overlap_ratio]
        low_density = [c for c in chunks if c.information_density < self.min_information_density]

        if too_short:
            pct = len(too_short) / len(chunks) * 100
            recommendations.append(
                f"{len(too_short)} chunks ({pct:.0f}%) are below minimum length "
                f"({self.min_chunk_length} chars). Consider merging with adjacent chunks."
            )

        if too_long:
            pct = len(too_long) / len(chunks) * 100
            recommendations.append(
                f"{len(too_long)} chunks ({pct:.0f}%) exceed maximum length "
                f"({self.max_chunk_length} chars). Consider splitting into smaller chunks."
            )

        if high_overlap:
            pct = len(high_overlap) / len(chunks) * 100
            recommendations.append(
                f"{len(high_overlap)} chunks ({pct:.0f}%) have high overlap with neighbors. "
                f"Consider reducing chunk overlap or using non-overlapping chunking."
            )

        if low_density:
            pct = len(low_density) / len(chunks) * 100
            recommendations.append(
                f"{len(low_density)} chunks ({pct:.0f}%) have low information density. "
                f"Consider removing boilerplate content or improving text extraction."
            )

        if avg_length < 200:
            recommendations.append(
                "Average chunk length is low. Consider increasing chunk size for better context."
            )
        elif avg_length > 1500:
            recommendations.append(
                "Average chunk length is high. Consider decreasing chunk size for better precision."
            )

        if not recommendations:
            recommendations.append("Chunk quality looks good. No major issues detected.")

        return recommendations
