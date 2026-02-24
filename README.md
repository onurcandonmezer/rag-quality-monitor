# RAG Quality Monitor

[![CI](https://github.com/onurcandnmz/rag-quality-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/onurcandnmz/rag-quality-monitor/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A comprehensive RAG (Retrieval-Augmented Generation) quality monitoring and assurance platform. Evaluate, detect hallucinations, run regression tests, analyze chunk quality, and monitor system health -- all without requiring external API keys.

## Overview

RAG Quality Monitor provides a complete toolkit for ensuring and maintaining the quality of RAG system outputs. It enables teams to catch quality degradation early, detect hallucinations, run automated regression testing against golden Q&A sets, and continuously monitor system health through configurable alerts.

## Features

- **Multi-Metric Evaluation** -- Score RAG responses across faithfulness, relevance, recall, and precision using keyword overlap and text similarity (no API keys required)
- **Hallucination Detection** -- Extract claims from generated answers and verify each one against retrieved context, categorizing them as supported, unsupported, or contradicted
- **Golden Q&A Test Suites** -- Manage curated question-answer pairs, run automated regression tests, and detect score drops against baselines
- **Chunk Quality Analysis** -- Analyze text chunks for coherence, information density, neighbor overlap, and length issues with actionable optimization recommendations
- **Continuous Monitoring** -- Track quality metrics over time with configurable alert thresholds, trend detection, and degradation alerts
- **Interactive Dashboard** -- Streamlit-based monitoring dashboard with quality overview, hallucination monitor, test suite runner, chunk analysis, and alert management

## Architecture

```
                    +------------------+
                    |   Streamlit UI   |
                    |   (app.py)       |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v------+  +--------v-------+  +-------v--------+
|  RAGEvaluator  |  | Hallucination  |  |  GoldenQA      |
|  (evaluator)   |  | Detector       |  |  Manager       |
+--------+-------+  +--------+-------+  +-------+--------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                    +--------v---------+
                    | QualityMonitor   |
                    | (monitor)        |
                    +--------+---------+
                             |
                    +--------v---------+
                    | ChunkAnalyzer    |
                    | (chunk_analyzer) |
                    +------------------+
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/onurcandnmz/rag-quality-monitor.git
cd rag-quality-monitor

# Install with uv
uv venv
uv pip install -e ".[dev]"

# Or install with pip
pip install -e ".[dev]"
```

### Run Tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

### Launch Dashboard

```bash
uv run streamlit run src/app.py
```

## Usage Examples

### Evaluate a RAG Response

```python
from src.evaluator import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG combines retrieval with generation for grounded answers.",
    context="RAG retrieves relevant documents and uses them as context for LLM generation.",
    expected_answer="RAG combines retrieval with text generation.",
)
print(f"Overall: {result.overall_score:.2f}")
print(f"Faithfulness: {result.faithfulness:.2f}")
print(f"Relevance: {result.relevance:.2f}")
```

### Detect Hallucinations

```python
from src.hallucination import HallucinationDetector

detector = HallucinationDetector()
result = detector.detect(
    answer="Python was created by Guido van Rossum in 1991.",
    context="Python is a high-level programming language created by Guido van Rossum.",
)
print(f"Hallucination Score: {result.score:.2f}")
for claim in result.claims:
    print(f"  [{claim.status.value}] {claim.text}")
```

### Run Golden Q&A Tests

```python
from src.golden_qa import GoldenQAManager

manager = GoldenQAManager()
manager.load_from_yaml("data/golden_qa_set.yaml")

answers = {"qa_001": "RAG combines retrieval with generation..."}
result = manager.run_test_suite(answers=answers)
print(f"Pass Rate: {result.pass_rate:.1%}")
print(f"Regression Detected: {result.regression_detected}")
```

### Analyze Chunk Quality

```python
from src.chunk_analyzer import ChunkAnalyzer

analyzer = ChunkAnalyzer()
chunks = [
    "First chunk about RAG systems and their components.",
    "Second chunk about vector databases and embeddings.",
    "Third chunk about evaluation metrics for retrieval.",
]
report = analyzer.analyze(chunks, query="How does RAG work?")
print(f"Overall Quality: {report.overall_quality:.2f}")
for rec in report.recommendations:
    print(f"  - {rec}")
```

### Monitor Quality Over Time

```python
from src.monitor import QualityMonitor

monitor = QualityMonitor(config_path="configs/monitor_config.yaml")

alerts = monitor.record_evaluation({
    "faithfulness": 0.85,
    "relevance": 0.78,
    "recall": 0.72,
    "precision": 0.68,
})
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.message}")

trend = monitor.detect_trend("faithfulness")
print(f"Trend: {trend.direction.value} (slope: {trend.slope:.4f})")
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Data Validation | Pydantic |
| Configuration | PyYAML |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| CLI Output | Rich |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| Package Management | uv |

## Project Structure

```
rag-quality-monitor/
├── README.md
├── pyproject.toml
├── Makefile
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── evaluator.py          # RAG evaluation suite (4 metrics)
│   ├── hallucination.py      # Hallucination detection & scoring
│   ├── golden_qa.py          # Golden Q&A test suite management
│   ├── chunk_analyzer.py     # Chunk quality analysis
│   ├── monitor.py            # Continuous monitoring & alerts
│   └── app.py                # Streamlit monitoring dashboard
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py     # 23 tests
│   ├── test_hallucination.py # 14 tests
│   ├── test_golden_qa.py     # 14 tests
│   ├── test_chunk_analyzer.py # 12 tests
│   └── test_monitor.py       # 22 tests
├── configs/
│   └── monitor_config.yaml   # Alert thresholds & monitoring settings
├── data/
│   └── golden_qa_set.yaml    # 18 curated golden Q&A pairs
└── assets/
```

## Configuration

Alert thresholds and monitoring settings are configured in `configs/monitor_config.yaml`:

```yaml
thresholds:
  faithfulness:
    warning: 0.7
    critical: 0.5
  hallucination:
    warning: 0.3
    critical: 0.5

monitoring:
  history_window: 50
  min_evaluations: 5

trend_detection:
  min_data_points: 10
  degradation_threshold: 0.1
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Onurcan Donmezer
