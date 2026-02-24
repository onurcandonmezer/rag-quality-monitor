"""Hallucination Detection and Scoring.

Detects and scores hallucinations in RAG system outputs by extracting
claims from generated answers and verifying them against the retrieved context.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ClaimStatus(StrEnum):
    """Status of a claim after verification."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class Claim:
    """A single factual claim extracted from an answer."""

    text: str
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    confidence: float = 0.0
    supporting_evidence: str = ""


@dataclass
class HallucinationResult:
    """Result of hallucination detection for a single answer."""

    answer: str
    context: str
    score: float  # 0 = no hallucination, 1 = fully hallucinated
    claims: list[Claim]
    num_supported: int
    num_unsupported: int
    num_contradicted: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_claims(self) -> int:
        """Total number of claims."""
        return len(self.claims)

    @property
    def is_hallucinated(self) -> bool:
        """Whether the answer contains significant hallucination."""
        return self.score > 0.5


@dataclass
class BatchHallucinationResult:
    """Result of hallucination detection for multiple answers."""

    results: list[HallucinationResult]
    avg_score: float
    total_claims: int
    total_supported: int
    total_unsupported: int
    total_contradicted: int
    hallucination_rate: float


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize_simple(text: str) -> set[str]:
    """Simple tokenization into unique word set."""
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
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "and",
        "but",
        "or",
        "not",
        "so",
        "if",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
    }
    normalized = _normalize_text(text)
    return {w for w in normalized.split() if w not in stop_words and len(w) > 1}


def _find_contradictions(claim_tokens: set[str], context_text: str) -> bool:
    """Check if a claim contradicts the context using negation patterns."""
    negation_words = {"not", "never", "no", "none", "neither", "nor", "cannot", "cant", "wont"}
    context_normalized = _normalize_text(context_text)

    claim_has_negation = bool(negation_words & claim_tokens)
    context_words = set(context_normalized.split())

    # Check if context has negation near words that overlap with the claim
    context_has_negation_near_overlap = False
    for neg_word in negation_words:
        # Only match whole words by checking word boundaries
        words_list = context_normalized.split()
        if neg_word not in words_list:
            continue
        idx = context_normalized.find(f" {neg_word} ")
        if idx == -1 and context_normalized.startswith(f"{neg_word} "):
            idx = 0
        if idx == -1:
            continue
        window = context_normalized[max(0, idx - 40) : idx + 40]
        window_tokens = set(window.split()) - negation_words
        if len(claim_tokens & window_tokens) >= 3:
            context_has_negation_near_overlap = True
            break

    if claim_has_negation != context_has_negation_near_overlap:
        overlap = claim_tokens & context_words
        if len(overlap) >= 3:
            return True

    return False


class HallucinationDetector:
    """Detects hallucinations in RAG-generated answers.

    Extracts claims from answers and verifies each claim against the
    provided context. Assigns a hallucination score from 0 (fully grounded)
    to 1 (fully hallucinated).
    """

    def __init__(
        self,
        support_threshold: float = 0.4,
        contradiction_penalty: float = 1.5,
    ) -> None:
        """Initialize the detector.

        Args:
            support_threshold: Minimum token overlap ratio to consider a
                claim supported by the context.
            contradiction_penalty: Multiplier for contradicted claims when
                computing the hallucination score.
        """
        self.support_threshold = support_threshold
        self.contradiction_penalty = contradiction_penalty

    def extract_claims(self, answer: str) -> list[Claim]:
        """Extract individual factual claims from an answer.

        Splits the answer into sentence-level claims, filtering out
        very short or non-informative fragments.

        Args:
            answer: The generated answer text.

        Returns:
            List of Claim objects.
        """
        if not answer.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sub_claims = re.split(r"[;,]\s+(?=[A-Z])", sent)
            for sub in sub_claims:
                sub = sub.strip().rstrip(".")
                words = sub.split()
                if len(words) >= 3:
                    claims.append(Claim(text=sub))

        return claims

    def verify_claim(self, claim: Claim, context: str) -> Claim:
        """Verify a single claim against the context.

        Args:
            claim: The claim to verify.
            context: The retrieved context documents.

        Returns:
            Updated Claim with status and confidence.
        """
        claim_tokens = _tokenize_simple(claim.text)
        context_tokens = _tokenize_simple(context)

        if not claim_tokens:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.confidence = 0.0
            return claim

        overlap = claim_tokens & context_tokens
        overlap_ratio = len(overlap) / len(claim_tokens)

        if _find_contradictions(claim_tokens, context):
            claim.status = ClaimStatus.CONTRADICTED
            claim.confidence = 0.3
            claim.supporting_evidence = "Contradicting evidence found in context"
            return claim

        if overlap_ratio >= self.support_threshold:
            claim.status = ClaimStatus.SUPPORTED
            claim.confidence = min(1.0, overlap_ratio * 1.2)

            context_sentences = re.split(r"[.!?]+", context)
            best_match = ""
            best_score = 0.0
            for sent in context_sentences:
                sent_tokens = _tokenize_simple(sent)
                if sent_tokens:
                    sent_overlap = len(claim_tokens & sent_tokens) / len(claim_tokens)
                    if sent_overlap > best_score:
                        best_score = sent_overlap
                        best_match = sent.strip()
            claim.supporting_evidence = best_match
        else:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.confidence = 1.0 - overlap_ratio

        return claim

    def detect(self, answer: str, context: str) -> HallucinationResult:
        """Detect hallucinations in an answer given the context.

        Args:
            answer: The generated answer.
            context: The retrieved context documents.

        Returns:
            HallucinationResult with score and claim details.
        """
        if not answer.strip():
            return HallucinationResult(
                answer=answer,
                context=context,
                score=0.0,
                claims=[],
                num_supported=0,
                num_unsupported=0,
                num_contradicted=0,
            )

        claims = self.extract_claims(answer)

        if not claims:
            return HallucinationResult(
                answer=answer,
                context=context,
                score=0.0,
                claims=[],
                num_supported=0,
                num_unsupported=0,
                num_contradicted=0,
            )

        verified_claims = [self.verify_claim(claim, context) for claim in claims]

        num_supported = sum(1 for c in verified_claims if c.status == ClaimStatus.SUPPORTED)
        num_unsupported = sum(1 for c in verified_claims if c.status == ClaimStatus.UNSUPPORTED)
        num_contradicted = sum(1 for c in verified_claims if c.status == ClaimStatus.CONTRADICTED)

        total = len(verified_claims)
        weighted_unsupported = num_unsupported + num_contradicted * self.contradiction_penalty
        score = min(1.0, weighted_unsupported / total)

        return HallucinationResult(
            answer=answer,
            context=context,
            score=round(score, 4),
            claims=verified_claims,
            num_supported=num_supported,
            num_unsupported=num_unsupported,
            num_contradicted=num_contradicted,
            details={
                "support_threshold": self.support_threshold,
                "total_claims": total,
            },
        )

    def detect_batch(
        self,
        pairs: list[dict[str, str]],
    ) -> BatchHallucinationResult:
        """Detect hallucinations for multiple answer-context pairs.

        Each pair should have keys: answer, context.

        Args:
            pairs: List of dicts with answer and context.

        Returns:
            BatchHallucinationResult with aggregated metrics.
        """
        results = []
        for pair in pairs:
            result = self.detect(
                answer=pair["answer"],
                context=pair["context"],
            )
            results.append(result)

        if not results:
            return BatchHallucinationResult(
                results=[],
                avg_score=0.0,
                total_claims=0,
                total_supported=0,
                total_unsupported=0,
                total_contradicted=0,
                hallucination_rate=0.0,
            )

        n = len(results)
        avg_score = sum(r.score for r in results) / n
        total_claims = sum(r.total_claims for r in results)
        total_supported = sum(r.num_supported for r in results)
        total_unsupported = sum(r.num_unsupported for r in results)
        total_contradicted = sum(r.num_contradicted for r in results)
        hallucination_rate = sum(1 for r in results if r.is_hallucinated) / n

        return BatchHallucinationResult(
            results=results,
            avg_score=round(avg_score, 4),
            total_claims=total_claims,
            total_supported=total_supported,
            total_unsupported=total_unsupported,
            total_contradicted=total_contradicted,
            hallucination_rate=round(hallucination_rate, 4),
        )
