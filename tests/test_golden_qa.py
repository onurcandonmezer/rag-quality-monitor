"""Tests for golden Q&A test suite management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.golden_qa import (
    GoldenQAManager,
    GoldenQAPair,
    GoldenQAResult,
)


@pytest.fixture
def sample_qa_data():
    """Sample Q&A data for testing."""
    return [
        {
            "id": "test_001",
            "question": "What is Python?",
            "expected_answer": "Python is a high-level programming language.",
            "context": "Python is a high-level, interpreted programming language known for readability.",
            "tags": ["basics", "python"],
            "difficulty": "easy",
        },
        {
            "id": "test_002",
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of AI that learns from data.",
            "context": "Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
            "tags": ["ml", "ai"],
            "difficulty": "medium",
        },
        {
            "id": "test_003",
            "question": "What is deep learning?",
            "expected_answer": "Deep learning uses neural networks with many layers.",
            "context": "Deep learning is a subset of machine learning using neural networks with multiple layers for complex pattern recognition.",
            "tags": ["ml", "deep-learning"],
            "difficulty": "hard",
        },
    ]


@pytest.fixture
def sample_yaml_file(sample_qa_data):
    """Create a temporary YAML file with sample data."""
    data = {"version": "1.0", "qa_pairs": sample_qa_data}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        return Path(f.name)


@pytest.fixture
def manager():
    """Create a GoldenQAManager instance."""
    return GoldenQAManager()


class TestGoldenQAManager:
    """Tests for GoldenQAManager."""

    def test_load_from_yaml(self, manager, sample_yaml_file):
        pairs = manager.load_from_yaml(sample_yaml_file)
        assert len(pairs) == 3
        assert all(isinstance(p, GoldenQAPair) for p in pairs)
        assert pairs[0].id == "test_001"

    def test_load_from_yaml_file_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load_from_yaml("/nonexistent/file.yaml")

    def test_load_from_yaml_invalid_structure(self, manager):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid": "data"}, f)
            path = Path(f.name)
        with pytest.raises(ValueError, match="qa_pairs"):
            manager.load_from_yaml(path)

    def test_load_from_list(self, manager, sample_qa_data):
        pairs = manager.load_from_list(sample_qa_data)
        assert len(pairs) == 3
        assert pairs[0].question == "What is Python?"

    def test_filter_by_tag(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        ml_pairs = manager.filter_by_tag("ml")
        assert len(ml_pairs) == 2
        assert all("ml" in p.tags for p in ml_pairs)

    def test_filter_by_difficulty(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        easy = manager.filter_by_difficulty("easy")
        assert len(easy) == 1
        assert easy[0].difficulty == "easy"

    def test_run_test_suite_with_answers(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        answers = {
            "test_001": "Python is a high-level programming language.",
            "test_002": "Machine learning is a subset of AI that learns from data.",
            "test_003": "Deep learning uses neural networks with many layers.",
        }
        result = manager.run_test_suite(answers=answers)
        assert isinstance(result, GoldenQAResult)
        assert result.total == 3
        assert result.passed + result.failed == result.total
        assert 0.0 <= result.pass_rate <= 1.0

    def test_run_test_suite_with_function(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)

        def answer_fn(question, context):
            return f"Based on the context: {context[:50]}"

        result = manager.run_test_suite(answer_fn=answer_fn)
        assert isinstance(result, GoldenQAResult)
        assert result.total == 3

    def test_run_test_suite_no_pairs_loaded(self, manager):
        with pytest.raises(ValueError, match="No Q&A pairs loaded"):
            manager.run_test_suite(answers={})

    def test_run_test_suite_no_answers(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        with pytest.raises(ValueError, match="answer_fn or answers"):
            manager.run_test_suite()

    def test_regression_detection(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)

        # Set high baseline
        manager.set_baseline(
            {
                "test_001": 0.95,
                "test_002": 0.90,
                "test_003": 0.85,
            }
        )

        # Run with intentionally poor answers
        answers = {
            "test_001": "Something completely unrelated about cooking recipes.",
            "test_002": "The weather is nice today with sunshine.",
            "test_003": "Football is a popular sport played worldwide.",
        }
        result = manager.run_test_suite(answers=answers, regression_threshold=0.1)
        assert result.regression_detected is True
        assert len(result.regression_details) > 0

    def test_set_baseline_from_results(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        answers = {p["id"]: p["expected_answer"] for p in sample_qa_data}
        result = manager.run_test_suite(answers=answers)

        manager.set_baseline_from_results(result)
        assert len(manager._baseline_scores) == 3

    def test_avg_scores_in_result(self, manager, sample_qa_data):
        manager.load_from_list(sample_qa_data)
        answers = {p["id"]: p["expected_answer"] for p in sample_qa_data}
        result = manager.run_test_suite(answers=answers)
        assert "faithfulness" in result.avg_scores
        assert "relevance" in result.avg_scores
        assert "recall" in result.avg_scores
        assert "precision" in result.avg_scores
        assert "overall" in result.avg_scores

    def test_load_real_golden_qa_set(self, manager):
        """Test loading the actual golden Q&A set from the project."""
        golden_path = Path(__file__).parent.parent / "data" / "golden_qa_set.yaml"
        if golden_path.exists():
            pairs = manager.load_from_yaml(golden_path)
            assert len(pairs) >= 15
