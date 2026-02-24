"""Continuous Quality Monitoring and Alerts.

Tracks RAG quality metrics over time, detects degradation trends,
and generates alerts when quality drops below configured thresholds.
Supports in-memory and SQLite storage backends.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Severity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TrendDirection(StrEnum):
    """Direction of quality trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class MetricRecord:
    """A single metric measurement."""

    metric: str
    value: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationAlert:
    """Alert when quality drops below a threshold."""

    metric: str
    current_value: float
    threshold: float
    severity: Severity
    timestamp: float
    message: str


@dataclass
class TrendInfo:
    """Trend information for a metric."""

    metric: str
    direction: TrendDirection
    slope: float
    recent_avg: float
    historical_avg: float
    data_points: int


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report."""

    alerts: list[DegradationAlert]
    trends: list[TrendInfo]
    latest_metrics: dict[str, float]
    summary: dict[str, Any]
    generated_at: str


class QualityMonitor:
    """Monitors RAG quality metrics over time and generates alerts.

    Tracks metrics in-memory or in SQLite, detects quality degradation,
    and provides trend analysis. Configurable via YAML config file.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            config_path: Path to monitoring config YAML file.
            db_path: Path to SQLite database. Uses in-memory if None.
        """
        self._config = self._load_config(config_path)
        self._db_path = str(db_path) if db_path else None
        self._records: list[MetricRecord] = []
        self._alerts: list[DegradationAlert] = []
        self._alert_timestamps: list[float] = []

        if self._db_path:
            self._init_db()

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load monitoring configuration from YAML.

        Args:
            config_path: Path to config file.

        Returns:
            Config dict with defaults for missing values.
        """
        defaults: dict[str, Any] = {
            "thresholds": {
                "faithfulness": {"warning": 0.7, "critical": 0.5},
                "relevance": {"warning": 0.7, "critical": 0.5},
                "recall": {"warning": 0.6, "critical": 0.4},
                "precision": {"warning": 0.6, "critical": 0.4},
                "overall": {"warning": 0.65, "critical": 0.45},
                "hallucination": {"warning": 0.3, "critical": 0.5},
            },
            "monitoring": {
                "check_interval": 300,
                "history_window": 50,
                "min_evaluations": 5,
            },
            "trend_detection": {
                "min_data_points": 10,
                "degradation_threshold": 0.1,
                "rolling_window": 5,
            },
            "alerts": {
                "max_alerts_per_hour": 10,
                "enable_console": True,
            },
        }

        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    loaded = yaml.safe_load(f)
                if loaded:
                    for key in defaults:
                        if key in loaded:
                            if isinstance(defaults[key], dict):
                                defaults[key].update(loaded[key])
                            else:
                                defaults[key] = loaded[key]

        return defaults

    def _init_db(self) -> None:
        """Initialize SQLite database tables."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def record_metric(
        self,
        metric: str,
        value: float,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DegradationAlert]:
        """Record a metric measurement and check for alerts.

        Args:
            metric: Name of the metric (e.g., 'faithfulness').
            value: Metric value.
            timestamp: Unix timestamp. Uses current time if None.
            metadata: Optional metadata dict.

        Returns:
            List of any alerts triggered by this measurement.
        """
        if timestamp is None:
            timestamp = time.time()

        record = MetricRecord(
            metric=metric,
            value=value,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        self._records.append(record)

        if self._db_path:
            self._save_record_to_db(record)

        alerts = self._check_thresholds(metric, value, timestamp)
        self._alerts.extend(alerts)
        return alerts

    def record_evaluation(
        self,
        scores: dict[str, float],
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DegradationAlert]:
        """Record a full evaluation result with all metrics.

        Args:
            scores: Dict mapping metric names to values.
            timestamp: Unix timestamp.
            metadata: Optional metadata.

        Returns:
            List of all alerts triggered.
        """
        if timestamp is None:
            timestamp = time.time()

        all_alerts: list[DegradationAlert] = []
        for metric, value in scores.items():
            alerts = self.record_metric(metric, value, timestamp, metadata)
            all_alerts.extend(alerts)

        return all_alerts

    def _save_record_to_db(self, record: MetricRecord) -> None:
        """Save a metric record to SQLite."""
        import json

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO metrics (metric, value, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (record.metric, record.value, record.timestamp, json.dumps(record.metadata)),
        )
        conn.commit()
        conn.close()

    def _check_thresholds(
        self,
        metric: str,
        value: float,
        timestamp: float,
    ) -> list[DegradationAlert]:
        """Check if a metric value violates configured thresholds.

        Args:
            metric: The metric name.
            value: Current value.
            timestamp: When the measurement was taken.

        Returns:
            List of triggered alerts.
        """
        alerts: list[DegradationAlert] = []
        thresholds = self._config["thresholds"].get(metric)

        if not thresholds:
            return alerts

        max_per_hour = self._config["alerts"].get("max_alerts_per_hour", 10)
        cutoff = timestamp - 3600
        recent_alerts = [t for t in self._alert_timestamps if t > cutoff]
        if len(recent_alerts) >= max_per_hour:
            return alerts

        is_hallucination = metric == "hallucination"

        if is_hallucination:
            if value >= thresholds.get("critical", 0.5):
                alert = DegradationAlert(
                    metric=metric,
                    current_value=round(value, 4),
                    threshold=thresholds["critical"],
                    severity=Severity.CRITICAL,
                    timestamp=timestamp,
                    message=(
                        f"CRITICAL: {metric} score {value:.4f} exceeds "
                        f"critical threshold {thresholds['critical']}"
                    ),
                )
                alerts.append(alert)
                self._alert_timestamps.append(timestamp)
            elif value >= thresholds.get("warning", 0.3):
                alert = DegradationAlert(
                    metric=metric,
                    current_value=round(value, 4),
                    threshold=thresholds["warning"],
                    severity=Severity.WARNING,
                    timestamp=timestamp,
                    message=(
                        f"WARNING: {metric} score {value:.4f} exceeds "
                        f"warning threshold {thresholds['warning']}"
                    ),
                )
                alerts.append(alert)
                self._alert_timestamps.append(timestamp)
        else:
            if value <= thresholds.get("critical", 0.5):
                alert = DegradationAlert(
                    metric=metric,
                    current_value=round(value, 4),
                    threshold=thresholds["critical"],
                    severity=Severity.CRITICAL,
                    timestamp=timestamp,
                    message=(
                        f"CRITICAL: {metric} score {value:.4f} below "
                        f"critical threshold {thresholds['critical']}"
                    ),
                )
                alerts.append(alert)
                self._alert_timestamps.append(timestamp)
            elif value <= thresholds.get("warning", 0.7):
                alert = DegradationAlert(
                    metric=metric,
                    current_value=round(value, 4),
                    threshold=thresholds["warning"],
                    severity=Severity.WARNING,
                    timestamp=timestamp,
                    message=(
                        f"WARNING: {metric} score {value:.4f} below "
                        f"warning threshold {thresholds['warning']}"
                    ),
                )
                alerts.append(alert)
                self._alert_timestamps.append(timestamp)

        return alerts

    def get_metric_history(
        self,
        metric: str,
        limit: int | None = None,
    ) -> list[MetricRecord]:
        """Get historical records for a metric.

        Args:
            metric: The metric name.
            limit: Maximum number of records to return (most recent).

        Returns:
            List of MetricRecord objects, sorted by timestamp.
        """
        records = [r for r in self._records if r.metric == metric]
        records.sort(key=lambda r: r.timestamp)
        if limit:
            records = records[-limit:]
        return records

    def detect_trend(self, metric: str) -> TrendInfo:
        """Detect the quality trend for a metric.

        Uses linear regression slope to determine if quality is
        improving, stable, or degrading.

        Args:
            metric: The metric name.

        Returns:
            TrendInfo with direction and statistics.
        """
        config = self._config["trend_detection"]
        min_points = config.get("min_data_points", 10)
        window = config.get("rolling_window", 5)
        degradation_threshold = config.get("degradation_threshold", 0.1)

        records = self.get_metric_history(metric)

        if len(records) < min_points:
            avg_val = statistics.mean(r.value for r in records) if records else 0.0
            return TrendInfo(
                metric=metric,
                direction=TrendDirection.STABLE,
                slope=0.0,
                recent_avg=round(avg_val, 4),
                historical_avg=round(avg_val, 4),
                data_points=len(records),
            )

        values = [r.value for r in records]

        n = len(values)
        x_values = list(range(n))
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values, strict=True))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0.0

        recent = values[-window:]
        historical = values[:-window] if len(values) > window else values
        recent_avg = statistics.mean(recent)
        historical_avg = statistics.mean(historical)

        is_hallucination = metric == "hallucination"

        if is_hallucination:
            if slope > degradation_threshold / n:
                direction = TrendDirection.DEGRADING
            elif slope < -degradation_threshold / n:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.STABLE
        else:
            if slope < -degradation_threshold / n:
                direction = TrendDirection.DEGRADING
            elif slope > degradation_threshold / n:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.STABLE

        return TrendInfo(
            metric=metric,
            direction=direction,
            slope=round(slope, 6),
            recent_avg=round(recent_avg, 4),
            historical_avg=round(historical_avg, 4),
            data_points=n,
        )

    def get_alerts(
        self,
        severity: Severity | None = None,
        metric: str | None = None,
        limit: int | None = None,
    ) -> list[DegradationAlert]:
        """Get recorded alerts with optional filtering.

        Args:
            severity: Filter by severity level.
            metric: Filter by metric name.
            limit: Maximum number of alerts to return.

        Returns:
            List of DegradationAlert objects.
        """
        alerts = self._alerts.copy()

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if metric:
            alerts = [a for a in alerts if a.metric == metric]

        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        if limit:
            alerts = alerts[:limit]

        return alerts

    def generate_report(self) -> MonitoringReport:
        """Generate a comprehensive monitoring report.

        Returns:
            MonitoringReport with alerts, trends, and summary.
        """
        metrics = set(r.metric for r in self._records)

        latest: dict[str, float] = {}
        for metric in metrics:
            history = self.get_metric_history(metric)
            if history:
                latest[metric] = history[-1].value

        trends = [self.detect_trend(metric) for metric in metrics]

        degrading = [t for t in trends if t.direction == TrendDirection.DEGRADING]
        improving = [t for t in trends if t.direction == TrendDirection.IMPROVING]

        total_records = len(self._records)
        total_alerts = len(self._alerts)
        critical_alerts = sum(1 for a in self._alerts if a.severity == Severity.CRITICAL)

        summary: dict[str, Any] = {
            "total_metrics_tracked": len(metrics),
            "total_measurements": total_records,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "degrading_metrics": [t.metric for t in degrading],
            "improving_metrics": [t.metric for t in improving],
            "health_status": (
                "critical" if critical_alerts > 0 else "warning" if total_alerts > 0 else "healthy"
            ),
        }

        return MonitoringReport(
            alerts=self._alerts.copy(),
            trends=trends,
            latest_metrics=latest,
            summary=summary,
            generated_at=datetime.now(UTC).isoformat(),
        )

    def clear_alerts(self) -> None:
        """Clear all recorded alerts."""
        self._alerts.clear()
        self._alert_timestamps.clear()

    def clear_history(self) -> None:
        """Clear all metric history."""
        self._records.clear()
        self._alerts.clear()
        self._alert_timestamps.clear()

        if self._db_path:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metrics")
            cursor.execute("DELETE FROM alerts")
            conn.commit()
            conn.close()
