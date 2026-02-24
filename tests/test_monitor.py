"""Tests for continuous quality monitoring module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from src.monitor import (
    MetricRecord,
    MonitoringReport,
    QualityMonitor,
    Severity,
    TrendDirection,
    TrendInfo,
)


@pytest.fixture
def monitor():
    return QualityMonitor()


@pytest.fixture
def monitor_with_config():
    config_path = Path(__file__).parent.parent / "configs" / "monitor_config.yaml"
    return QualityMonitor(config_path=str(config_path))


class TestQualityMonitor:
    """Tests for QualityMonitor."""

    def test_record_metric(self, monitor):
        alerts = monitor.record_metric("faithfulness", 0.85)
        assert isinstance(alerts, list)

    def test_record_metric_triggers_warning(self, monitor):
        alerts = monitor.record_metric("faithfulness", 0.6)
        assert len(alerts) >= 1
        assert alerts[0].severity == Severity.WARNING

    def test_record_metric_triggers_critical(self, monitor):
        alerts = monitor.record_metric("faithfulness", 0.3)
        assert len(alerts) >= 1
        assert any(a.severity == Severity.CRITICAL for a in alerts)

    def test_record_metric_no_alert(self, monitor):
        alerts = monitor.record_metric("faithfulness", 0.9)
        assert len(alerts) == 0

    def test_hallucination_alert_direction(self, monitor):
        # For hallucination, higher values are worse
        alerts = monitor.record_metric("hallucination", 0.6)
        assert len(alerts) >= 1

    def test_record_evaluation(self, monitor):
        scores = {
            "faithfulness": 0.4,
            "relevance": 0.8,
            "recall": 0.7,
            "precision": 0.6,
        }
        alerts = monitor.record_evaluation(scores)
        assert isinstance(alerts, list)
        # faithfulness 0.4 should trigger critical alert
        faith_alerts = [a for a in alerts if a.metric == "faithfulness"]
        assert len(faith_alerts) >= 1

    def test_get_metric_history(self, monitor):
        base_time = time.time()
        for i in range(5):
            monitor.record_metric("faithfulness", 0.7 + i * 0.05, timestamp=base_time + i)

        history = monitor.get_metric_history("faithfulness")
        assert len(history) == 5
        assert all(isinstance(r, MetricRecord) for r in history)
        # Check sorted by timestamp
        for i in range(len(history) - 1):
            assert history[i].timestamp <= history[i + 1].timestamp

    def test_get_metric_history_with_limit(self, monitor):
        base_time = time.time()
        for i in range(10):
            monitor.record_metric("relevance", 0.8, timestamp=base_time + i)

        history = monitor.get_metric_history("relevance", limit=5)
        assert len(history) == 5

    def test_detect_trend_stable(self, monitor):
        base_time = time.time()
        for i in range(15):
            monitor.record_metric("faithfulness", 0.8, timestamp=base_time + i)

        trend = monitor.detect_trend("faithfulness")
        assert isinstance(trend, TrendInfo)
        assert trend.direction == TrendDirection.STABLE
        assert trend.data_points == 15

    def test_detect_trend_degrading(self, monitor):
        base_time = time.time()
        for i in range(15):
            value = 0.9 - i * 0.04  # Steadily decreasing
            monitor.record_metric("faithfulness", value, timestamp=base_time + i)

        trend = monitor.detect_trend("faithfulness")
        assert trend.direction == TrendDirection.DEGRADING
        assert trend.slope < 0

    def test_detect_trend_improving(self, monitor):
        base_time = time.time()
        for i in range(15):
            value = 0.5 + i * 0.03  # Steadily increasing
            monitor.record_metric("faithfulness", value, timestamp=base_time + i)

        trend = monitor.detect_trend("faithfulness")
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.slope > 0

    def test_detect_trend_insufficient_data(self, monitor):
        monitor.record_metric("faithfulness", 0.8)
        trend = monitor.detect_trend("faithfulness")
        assert trend.direction == TrendDirection.STABLE
        assert trend.data_points == 1

    def test_get_alerts_filter_severity(self, monitor):
        monitor.record_metric("faithfulness", 0.3)  # Critical
        monitor.record_metric("relevance", 0.6)  # Warning

        critical = monitor.get_alerts(severity=Severity.CRITICAL)
        assert all(a.severity == Severity.CRITICAL for a in critical)

    def test_get_alerts_filter_metric(self, monitor):
        monitor.record_metric("faithfulness", 0.3)
        monitor.record_metric("relevance", 0.3)

        faith_alerts = monitor.get_alerts(metric="faithfulness")
        assert all(a.metric == "faithfulness" for a in faith_alerts)

    def test_generate_report(self, monitor):
        base_time = time.time()
        for i in range(5):
            monitor.record_metric("faithfulness", 0.8, timestamp=base_time + i)
            monitor.record_metric("relevance", 0.7, timestamp=base_time + i)

        report = monitor.generate_report()
        assert isinstance(report, MonitoringReport)
        assert "faithfulness" in report.latest_metrics
        assert "relevance" in report.latest_metrics
        assert report.summary["total_metrics_tracked"] == 2
        assert report.generated_at is not None

    def test_clear_alerts(self, monitor):
        monitor.record_metric("faithfulness", 0.3)
        assert len(monitor.get_alerts()) > 0

        monitor.clear_alerts()
        assert len(monitor.get_alerts()) == 0

    def test_clear_history(self, monitor):
        monitor.record_metric("faithfulness", 0.8)
        monitor.clear_history()

        history = monitor.get_metric_history("faithfulness")
        assert len(history) == 0

    def test_sqlite_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        monitor = QualityMonitor(db_path=db_path)
        monitor.record_metric("faithfulness", 0.8)
        history = monitor.get_metric_history("faithfulness")
        assert len(history) == 1

    def test_config_loading(self, monitor_with_config):
        config = monitor_with_config._config
        assert "thresholds" in config
        assert "monitoring" in config
        assert "faithfulness" in config["thresholds"]

    def test_max_alerts_per_hour_limit(self, monitor):
        # Override to low limit
        monitor._config["alerts"]["max_alerts_per_hour"] = 2
        base_time = time.time()

        all_alerts = []
        for i in range(5):
            alerts = monitor.record_metric("faithfulness", 0.3, timestamp=base_time + i)
            all_alerts.extend(alerts)

        # Should be capped
        assert len(all_alerts) <= 2

    def test_health_status_healthy(self, monitor):
        monitor.record_metric("faithfulness", 0.9)
        report = monitor.generate_report()
        assert report.summary["health_status"] == "healthy"

    def test_health_status_critical(self, monitor):
        monitor.record_metric("faithfulness", 0.3)
        report = monitor.generate_report()
        assert report.summary["health_status"] == "critical"
