from typing import Callable

import numpy as np

from mrd.detection.regression import RegressionDetector
from mrd.drift.detector import DriftDetector
from mrd.reports.report import ReportGenerator
from mrd.schemas import (
    ModelEvaluation,
    RegressionReport,
)


class RegressionPipeline:
    """
    End-to-end ML model regression detection pipeline.

    The pipeline combines:
    1. Model performance comparison
    2. Feature drift detection
    3. Segment analysis
    4. Structured reporting
    """

    def __init__(
        self,
        regression_detector: RegressionDetector,
        drift_detector: DriftDetector,
        report_generator: ReportGenerator,
    ) -> None:
        self.regression_detector = regression_detector
        self.drift_detector = drift_detector
        self.report_generator = report_generator

    def run(
        self,
        baseline: ModelEvaluation,
        candidate: ModelEvaluation,
        feature_baseline: dict[str, np.ndarray] | None = None,
        feature_candidate: dict[str, np.ndarray] | None = None,
    ) -> RegressionReport:
        """
        Run the complete regression detection pipeline.
        """

        regression_report = (
            self.regression_detector.compare(
                baseline=baseline,
                candidate=candidate,
            )
        )

        drift_results = []

        if (
            feature_baseline is not None
            and feature_candidate is not None
        ):
            drift_results = (
                self._detect_feature_drift(
                    feature_baseline,
                    feature_candidate,
                )
            )

        regression_detected = (
            regression_report.status == "regression"
            or any(
                result.severity == "high"
                for result in drift_results
            )
        )

        summary = self._build_summary(
            regression_report=regression_report,
            drift_results=drift_results,
        )

        return self.report_generator.create_report(
            model_name=baseline.model_name,
            baseline_version=baseline.model_version,
            candidate_version=candidate.model_version,
            dataset_name=baseline.dataset_version,
            regression_detected=regression_detected,
            performance=regression_report.performance,
            drift=drift_results,
            segments=[],
            summary=summary,
        )

    def _detect_feature_drift(
        self,
        baseline_features: dict[str, np.ndarray],
        candidate_features: dict[str, np.ndarray],
    ) -> list:
        """Detect drift for all shared features (numeric and categorical)."""

        results = []

        shared_features = (
            set(baseline_features)
            & set(candidate_features)
        )

        for feature_name in shared_features:
            result = (
                self.drift_detector.detect_feature(
                    feature_name=feature_name,
                    baseline=baseline_features[
                        feature_name
                    ],
                    current=candidate_features[
                        feature_name
                    ],
                )
            )

            results.append(result)

        return results

    def _build_summary(
        self,
        regression_report: RegressionReport,
        drift_results: list,
    ) -> str:
        """Build a deterministic human-readable summary."""

        high_drift = sum(
            result.severity == "high"
            for result in drift_results
        )

        if regression_report.status == "regression":
            if high_drift > 0:
                return (
                    "Model performance regression "
                    "detected with high feature drift."
                )

            return (
                "Model performance regression detected."
            )

        if high_drift > 0:
            return (
                "No performance regression detected, "
                "but high feature drift was detected."
            )

        return (
            "No significant model regression detected."
        )