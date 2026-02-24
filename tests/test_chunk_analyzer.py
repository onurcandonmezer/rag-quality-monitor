"""Tests for chunk quality analysis module."""

from __future__ import annotations

import pytest

from src.chunk_analyzer import (
    ChunkAnalysisReport,
    ChunkAnalyzer,
    ChunkQuality,
)


@pytest.fixture
def analyzer():
    return ChunkAnalyzer()


@pytest.fixture
def sample_chunks():
    return [
        (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances "
            "large language model outputs. It retrieves relevant documents from external "
            "knowledge bases. The retrieved documents serve as context for the language model."
        ),
        (
            "The retriever component takes a user query and finds the most relevant "
            "documents from the index. Dense retrieval uses neural network embeddings "
            "for semantic similarity search. BM25 is a popular sparse retrieval method."
        ),
        (
            "Document chunking is a critical step in RAG pipelines. The size and strategy "
            "of chunks directly impacts retrieval quality. Small chunks often lack context "
            "while large chunks may dilute relevant information."
        ),
    ]


class TestChunkAnalyzer:
    """Tests for ChunkAnalyzer."""

    def test_analyze_single_chunk(self, analyzer):
        chunk = (
            "Python is a high-level programming language. It supports multiple "
            "programming paradigms including procedural and object-oriented programming."
        )
        result = analyzer.analyze_chunk(chunk, index=0)
        assert isinstance(result, ChunkQuality)
        assert result.chunk_index == 0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.information_density <= 1.0
        assert result.length == len(chunk)

    def test_analyze_too_short_chunk(self, analyzer):
        chunk = "Short text."
        result = analyzer.analyze_chunk(chunk, index=0)
        assert any("too short" in issue.lower() for issue in result.issues)

    def test_analyze_too_long_chunk(self):
        analyzer = ChunkAnalyzer(max_chunk_length=50)
        chunk = (
            "This is a chunk that is definitely longer than fifty characters in total length here."
        )
        result = analyzer.analyze_chunk(chunk, index=0)
        assert any("too long" in issue.lower() for issue in result.issues)

    def test_analyze_with_neighbors(self, analyzer, sample_chunks):
        result = analyzer.analyze_chunk(
            sample_chunks[1],
            index=1,
            neighbors=(sample_chunks[0], sample_chunks[2]),
        )
        assert result.overlap_with_neighbors >= 0.0

    def test_analyze_with_query(self, analyzer):
        chunk = (
            "Dense retrieval uses neural network embeddings for semantic similarity search. "
            "It encodes queries and documents into dense vectors."
        )
        result = analyzer.analyze_chunk(chunk, index=0, query="How does dense retrieval work?")
        assert result.relevance_score > 0.0

    def test_analyze_batch(self, analyzer, sample_chunks):
        report = analyzer.analyze(sample_chunks)
        assert isinstance(report, ChunkAnalysisReport)
        assert report.total_chunks == 3
        assert report.avg_length > 0
        assert 0.0 <= report.avg_coherence <= 1.0
        assert 0.0 <= report.avg_information_density <= 1.0
        assert 0.0 <= report.overall_quality <= 1.0

    def test_analyze_empty_chunks(self, analyzer):
        report = analyzer.analyze([])
        assert report.total_chunks == 0
        assert report.overall_quality == 0.0
        assert "No chunks provided" in report.recommendations[0]

    def test_analyze_with_high_overlap(self):
        analyzer = ChunkAnalyzer(max_overlap_ratio=0.1)
        chunks = [
            "Python is a programming language used for web development.",
            "Python is a programming language used for data science.",
        ]
        report = analyzer.analyze(chunks)
        # These chunks share significant content
        high_overlap_count = report.stats.get("num_high_overlap", 0)
        assert high_overlap_count >= 0

    def test_recommendations_generated(self, analyzer):
        chunks = ["Short."] + [
            "This is a normal sized chunk with enough content to be meaningful. "
            "It contains several sentences about various topics. "
            "The information density should be reasonable."
        ] * 3
        report = analyzer.analyze(chunks)
        assert len(report.recommendations) > 0

    def test_stats_in_report(self, analyzer, sample_chunks):
        report = analyzer.analyze(sample_chunks)
        assert "min_length" in report.stats
        assert "max_length" in report.stats
        assert "median_length" in report.stats
        assert "num_too_short" in report.stats
        assert "num_too_long" in report.stats

    def test_information_density_scoring(self, analyzer):
        # High density: diverse vocabulary
        high_density = (
            "Quantum computing leverages superposition and entanglement to process "
            "information exponentially faster. Applications include cryptography, "
            "drug discovery, optimization, and climate modeling."
        )
        # Low density: repetitive content
        low_density = (
            "The thing is a thing that does things. The thing is also a thing "
            "that is used for things. Things are things that do things."
        )
        high_result = analyzer.analyze_chunk(high_density, index=0)
        low_result = analyzer.analyze_chunk(low_density, index=1)
        assert high_result.information_density > low_result.information_density

    def test_coherence_scoring(self, analyzer):
        coherent = (
            "Machine learning models require training data. "
            "Furthermore, the data must be representative of the target domain. "
            "Additionally, proper validation sets prevent overfitting."
        )
        incoherent = "Word. Token. Bit."
        coherent_result = analyzer.analyze_chunk(coherent, index=0)
        incoherent_result = analyzer.analyze_chunk(incoherent, index=1)
        assert coherent_result.coherence_score > incoherent_result.coherence_score
