"""Streamlit Monitoring Dashboard.

Interactive dashboard for monitoring RAG quality metrics, hallucination
detection results, golden Q&A test suite results, and chunk analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .chunk_analyzer import ChunkAnalyzer
from .evaluator import RAGEvaluator
from .golden_qa import GoldenQAManager
from .hallucination import HallucinationDetector
from .monitor import QualityMonitor, Severity, TrendDirection


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def _init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "monitor" not in st.session_state:
        config_path = _get_project_root() / "configs" / "monitor_config.yaml"
        st.session_state.monitor = QualityMonitor(
            config_path=str(config_path) if config_path.exists() else None
        )
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = RAGEvaluator()
    if "detector" not in st.session_state:
        st.session_state.detector = HallucinationDetector()
    if "chunk_analyzer" not in st.session_state:
        st.session_state.chunk_analyzer = ChunkAnalyzer()
    if "qa_manager" not in st.session_state:
        st.session_state.qa_manager = GoldenQAManager()


def render_quality_overview() -> None:
    """Render the Quality Overview panel."""
    st.header("Quality Overview")

    monitor: QualityMonitor = st.session_state.monitor
    report = monitor.generate_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = report.summary.get("health_status", "unknown")
        st.metric("Health Status", status.upper())
    with col2:
        st.metric("Total Measurements", report.summary.get("total_measurements", 0))
    with col3:
        st.metric("Total Alerts", report.summary.get("total_alerts", 0))
    with col4:
        st.metric("Critical Alerts", report.summary.get("critical_alerts", 0))

    if report.latest_metrics:
        st.subheader("Latest Metric Scores")
        metrics_df = pd.DataFrame(
            [{"Metric": k, "Score": v} for k, v in report.latest_metrics.items()]
        )
        st.dataframe(metrics_df, use_container_width=True)

    if report.trends:
        st.subheader("Metric Trends")
        trends_data = []
        for trend in report.trends:
            icon = {
                TrendDirection.IMPROVING: "^",
                TrendDirection.STABLE: "-",
                TrendDirection.DEGRADING: "v",
            }.get(trend.direction, "?")
            trends_data.append(
                {
                    "Metric": trend.metric,
                    "Direction": f"{icon} {trend.direction.value}",
                    "Recent Avg": trend.recent_avg,
                    "Historical Avg": trend.historical_avg,
                    "Data Points": trend.data_points,
                }
            )
        st.dataframe(pd.DataFrame(trends_data), use_container_width=True)

    st.subheader("Record New Evaluation")
    with st.form("eval_form"):
        col1, col2 = st.columns(2)
        with col1:
            question = st.text_area("Question", height=80)
            context = st.text_area("Context", height=120)
        with col2:
            answer = st.text_area("Answer", height=80)
            expected = st.text_area("Expected Answer", height=120)

        submitted = st.form_submit_button("Evaluate")
        if submitted and question and answer and context and expected:
            evaluator: RAGEvaluator = st.session_state.evaluator
            result = evaluator.evaluate(question, answer, context, expected)

            scores = {
                "faithfulness": result.faithfulness,
                "relevance": result.relevance,
                "recall": result.recall,
                "precision": result.precision,
                "overall": result.overall_score,
            }
            alerts = monitor.record_evaluation(scores)

            st.success(f"Overall Score: {result.overall_score:.4f}")
            score_df = pd.DataFrame([{"Metric": k, "Score": v} for k, v in scores.items()])
            st.dataframe(score_df, use_container_width=True)

            if alerts:
                for alert in alerts:
                    if alert.severity == Severity.CRITICAL:
                        st.error(alert.message)
                    else:
                        st.warning(alert.message)


def render_hallucination_monitor() -> None:
    """Render the Hallucination Monitor panel."""
    st.header("Hallucination Monitor")

    detector: HallucinationDetector = st.session_state.detector

    with st.form("hallucination_form"):
        answer = st.text_area("Answer to Check", height=120)
        context = st.text_area("Context", height=120)
        submitted = st.form_submit_button("Detect Hallucinations")

    if submitted and answer and context:
        result = detector.detect(answer, context)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hallucination Score", f"{result.score:.2f}")
        with col2:
            st.metric("Supported Claims", result.num_supported)
        with col3:
            st.metric("Unsupported Claims", result.num_unsupported)
        with col4:
            st.metric("Contradicted Claims", result.num_contradicted)

        if result.claims:
            st.subheader("Claim Analysis")
            claims_data = []
            for claim in result.claims:
                claims_data.append(
                    {
                        "Claim": claim.text[:100] + ("..." if len(claim.text) > 100 else ""),
                        "Status": claim.status.value,
                        "Confidence": f"{claim.confidence:.2f}",
                    }
                )
            st.dataframe(pd.DataFrame(claims_data), use_container_width=True)

        monitor: QualityMonitor = st.session_state.monitor
        monitor.record_metric("hallucination", result.score)


def render_golden_qa() -> None:
    """Render the Golden Q&A Test Suite panel."""
    st.header("Golden Q&A Test Suite")

    manager: GoldenQAManager = st.session_state.qa_manager

    golden_qa_path = _get_project_root() / "data" / "golden_qa_set.yaml"
    if golden_qa_path.exists() and st.button("Load Golden Q&A Set"):
        pairs = manager.load_from_yaml(golden_qa_path)
        st.success(f"Loaded {len(pairs)} Q&A pairs")

    if manager.qa_pairs:
        st.subheader(f"Loaded Q&A Pairs: {len(manager.qa_pairs)}")

        pairs_data = []
        for pair in manager.qa_pairs:
            pairs_data.append(
                {
                    "ID": pair.id,
                    "Question": pair.question[:80] + "...",
                    "Difficulty": pair.difficulty,
                    "Tags": ", ".join(pair.tags),
                }
            )
        st.dataframe(pd.DataFrame(pairs_data), use_container_width=True)

        st.subheader("Run Test Suite")
        st.info(
            "Enter answers for each Q&A pair below. "
            "Leave empty to use the expected answer as a baseline test."
        )

        if st.button("Run with Expected Answers (Baseline)"):
            answers = {pair.id: pair.expected_answer for pair in manager.qa_pairs}
            result = manager.run_test_suite(answers=answers)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pass Rate", f"{result.pass_rate:.1%}")
            with col2:
                st.metric("Passed", result.passed)
            with col3:
                st.metric("Failed", result.failed)

            if result.avg_scores:
                st.subheader("Average Scores")
                scores_df = pd.DataFrame(
                    [{"Metric": k, "Score": v} for k, v in result.avg_scores.items()]
                )
                st.dataframe(scores_df, use_container_width=True)
    else:
        st.info("No Q&A pairs loaded. Click 'Load Golden Q&A Set' to begin.")


def render_chunk_analysis() -> None:
    """Render the Chunk Analysis panel."""
    st.header("Chunk Analysis")

    analyzer: ChunkAnalyzer = st.session_state.chunk_analyzer

    with st.form("chunk_form"):
        chunks_text = st.text_area(
            "Enter chunks (one per paragraph, separated by blank lines)",
            height=200,
        )
        query = st.text_input("Optional query for relevance scoring")
        submitted = st.form_submit_button("Analyze Chunks")

    if submitted and chunks_text:
        chunks = [c.strip() for c in chunks_text.split("\n\n") if c.strip()]

        if chunks:
            report = analyzer.analyze(chunks, query=query or None)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", report.total_chunks)
            with col2:
                st.metric("Avg Length", f"{report.avg_length:.0f}")
            with col3:
                st.metric("Overall Quality", f"{report.overall_quality:.2f}")
            with col4:
                st.metric("Problematic", len(report.problematic_chunks))

            st.subheader("Chunk Quality Heatmap")
            heatmap_data = []
            for chunk in report.chunks:
                heatmap_data.append(
                    {
                        "Index": chunk.chunk_index,
                        "Length": chunk.length,
                        "Relevance": chunk.relevance_score,
                        "Coherence": chunk.coherence_score,
                        "Info Density": chunk.information_density,
                        "Overlap": chunk.overlap_with_neighbors,
                        "Issues": len(chunk.issues),
                    }
                )
            st.dataframe(pd.DataFrame(heatmap_data), use_container_width=True)

            if report.recommendations:
                st.subheader("Recommendations")
                for rec in report.recommendations:
                    st.write(f"- {rec}")


def render_alerts() -> None:
    """Render the Alerts and Notifications panel."""
    st.header("Alerts & Notifications")

    monitor: QualityMonitor = st.session_state.monitor

    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "Critical", "Warning", "Info"],
        )
    with col2:
        if st.button("Clear All Alerts"):
            monitor.clear_alerts()
            st.success("Alerts cleared")

    severity = None
    if severity_filter != "All":
        severity = Severity(severity_filter.lower())

    alerts = monitor.get_alerts(severity=severity)

    if alerts:
        alerts_data = []
        for alert in alerts:
            alerts_data.append(
                {
                    "Severity": alert.severity.value.upper(),
                    "Metric": alert.metric,
                    "Current Value": f"{alert.current_value:.4f}",
                    "Threshold": f"{alert.threshold:.4f}",
                    "Message": alert.message,
                }
            )
        st.dataframe(pd.DataFrame(alerts_data), use_container_width=True)
    else:
        st.success("No alerts. System is healthy.")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="RAG Quality Monitor",
        page_icon="",
        layout="wide",
    )

    st.title("RAG Quality Monitor")
    st.caption("Comprehensive RAG quality monitoring and assurance platform")

    _init_session_state()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Quality Overview",
            "Hallucination Monitor",
            "Golden Q&A Suite",
            "Chunk Analysis",
            "Alerts",
        ]
    )

    with tab1:
        render_quality_overview()
    with tab2:
        render_hallucination_monitor()
    with tab3:
        render_golden_qa()
    with tab4:
        render_chunk_analysis()
    with tab5:
        render_alerts()


if __name__ == "__main__":
    main()
