"""Tests for hallucination detection module."""

from __future__ import annotations

import pytest

from src.hallucination import (
    BatchHallucinationResult,
    Claim,
    ClaimStatus,
    HallucinationDetector,
    HallucinationResult,
)


class TestClaimExtraction:
    """Tests for claim extraction from answers."""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()

    def test_extract_single_sentence(self, detector):
        answer = "Python was created by Guido van Rossum in 1991."
        claims = detector.extract_claims(answer)
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_extract_multiple_sentences(self, detector):
        answer = (
            "Python is a programming language. "
            "It was created by Guido van Rossum. "
            "Python supports multiple paradigms."
        )
        claims = detector.extract_claims(answer)
        assert len(claims) >= 2

    def test_extract_empty_answer(self, detector):
        claims = detector.extract_claims("")
        assert claims == []

    def test_extract_short_fragment_filtered(self, detector):
        answer = "Yes no."
        claims = detector.extract_claims(answer)
        # Very short fragments (< 3 words) should be filtered
        assert len(claims) == 0


class TestClaimVerification:
    """Tests for claim verification against context."""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()

    def test_supported_claim(self, detector):
        claim = Claim(text="Python was created by Guido van Rossum")
        context = "Python is a programming language created by Guido van Rossum in 1991."
        verified = detector.verify_claim(claim, context)
        assert verified.status == ClaimStatus.SUPPORTED
        assert verified.confidence > 0

    def test_unsupported_claim(self, detector):
        claim = Claim(text="Java was invented by Oracle Corporation in 2020")
        context = "Python is a programming language created by Guido van Rossum."
        verified = detector.verify_claim(claim, context)
        assert verified.status == ClaimStatus.UNSUPPORTED

    def test_claim_with_empty_tokens(self, detector):
        claim = Claim(text="a the is")
        context = "Some context here."
        verified = detector.verify_claim(claim, context)
        assert verified.status == ClaimStatus.UNSUPPORTED


class TestHallucinationDetection:
    """Tests for the full hallucination detection pipeline."""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()

    def test_detect_no_hallucination(self, detector):
        context = (
            "Retrieval-Augmented Generation combines information retrieval "
            "with text generation. It retrieves relevant documents from a "
            "knowledge base and uses them as context."
        )
        answer = (
            "RAG combines information retrieval with text generation. "
            "It retrieves relevant documents from a knowledge base."
        )
        result = detector.detect(answer, context)
        assert isinstance(result, HallucinationResult)
        assert result.score < 0.5
        assert result.num_supported > 0

    def test_detect_hallucinated_answer(self, detector):
        context = "Python is a programming language."
        answer = (
            "Python was created in the year 2025 by Elon Musk. "
            "It runs exclusively on quantum computers. "
            "Python requires a special NASA-approved compiler."
        )
        result = detector.detect(answer, context)
        assert result.score > 0.3
        assert result.num_unsupported > 0

    def test_detect_empty_answer(self, detector):
        result = detector.detect("", "some context")
        assert result.score == 0.0
        assert result.total_claims == 0

    def test_hallucination_result_properties(self, detector):
        result = detector.detect(
            "Python is a language created by Guido van Rossum.",
            "Python was created by Guido van Rossum in 1991.",
        )
        assert isinstance(result.total_claims, int)
        assert isinstance(result.is_hallucinated, bool)

    def test_detect_batch(self, detector):
        pairs = [
            {
                "answer": "Python is a programming language.",
                "context": "Python is a high-level programming language used worldwide.",
            },
            {
                "answer": "JavaScript runs in the browser.",
                "context": "JavaScript is a client-side scripting language that runs in web browsers.",
            },
        ]
        result = detector.detect_batch(pairs)
        assert isinstance(result, BatchHallucinationResult)
        assert len(result.results) == 2
        assert 0.0 <= result.avg_score <= 1.0

    def test_detect_batch_empty(self, detector):
        result = detector.detect_batch([])
        assert result.avg_score == 0.0
        assert result.total_claims == 0

    def test_custom_threshold(self):
        detector = HallucinationDetector(support_threshold=0.8)
        result = detector.detect("Python is a language.", "Python is a programming language.")
        assert isinstance(result, HallucinationResult)
