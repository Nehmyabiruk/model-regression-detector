from mrd.config import load_thresholds

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from mrd.statistics.bootstrap import (
    bootstrap_metric_difference,
)
from mrd.schemas import (
    MetricComparison,
    ModelEvaluation,
    RegressionReport,
)


class RegressionDetector:
    """Detect performance regressions between model versions."""

    def __init__(
        self,
        thresholds: dict | None = None,
    ) -> None:
        self.thresholds = (
            thresholds
            if thresholds is not None
            else load_thresholds()
        )

    def compare(
        self,
        baseline: ModelEvaluation,
        candidate: ModelEvaluation,
        task_type: str = "classification",
    ) -> RegressionReport:
        """Compare a candidate model against a baseline model."""

        self._validate_evaluations(
            baseline=baseline,
            candidate=candidate,
        )

        baseline_metrics = {
            metric.name: metric
            for metric in baseline.metrics
        }

        candidate_metrics = {
            metric.name: metric
            for metric in candidate.metrics
        }

        comparisons: list[MetricComparison] = []

        for metric_name, baseline_metric in baseline_metrics.items():
            candidate_metric = candidate_metrics.get(metric_name)

            if candidate_metric is None:
                continue

            difference = (
                candidate_metric.value
                - baseline_metric.value
            )

            relative_change = self._calculate_relative_change(
                baseline_value=baseline_metric.value,
                candidate_value=candidate_metric.value,
            )

            regression = self._is_regression(
                metric_name=metric_name,
                difference=difference,
                higher_is_better=baseline_metric.higher_is_better,
                task_type=task_type,
            )

            comparisons.append(
                MetricComparison(
                    metric_name=metric_name,
                    baseline_value=baseline_metric.value,
                    candidate_value=candidate_metric.value,
                    difference=difference,
                    relative_change=relative_change,
                    regression=regression,
                )
            )

        regression_detected = any(
            comparison.regression
            for comparison in comparisons
        )

        status = "regression" if regression_detected else "pass"

        return RegressionReport(
            report_id=str(uuid4()),
            model_name=candidate.model_name,
            baseline_version=baseline.model_version,
            candidate_version=candidate.model_version,
            dataset_name=candidate.dataset_version,
            created_at=datetime.now(timezone.utc),
            task_type=task_type,
            status=status,
            performance={
                comparison.metric_name: comparison.model_dump()
                for comparison in comparisons
            },
            drift=[],
            segments=[],
            summary=(
                "Regression detected."
                if regression_detected
                else "No regression detected."
            ),
        )

    @staticmethod
    def _calculate_relative_change(
        baseline_value: float,
        candidate_value: float,
    ) -> float:
        if baseline_value == 0:
            return 0.0

        return (
            candidate_value - baseline_value
        ) / abs(baseline_value)

    def _is_regression(
        self,
        metric_name: str,
        difference: float,
        higher_is_better: bool,
        task_type: str | None = None,
    ) -> bool:
        metric_config = self._get_metric_config(
            metric_name,
            task_type=task_type,
        )

        max_regression = float(
            metric_config["max_regression"]
        )

        if higher_is_better:
            return difference < -max_regression

        return difference > max_regression

    def _get_metric_config(
        self,
        metric_name: str,
        task_type: str | None = None,
    ) -> dict:
        if task_type and task_type in self.thresholds:
            task_cfg = self.thresholds[task_type]
            if isinstance(task_cfg, dict):
                if metric_name in task_cfg and isinstance(task_cfg[metric_name], dict) and "max_regression" in task_cfg[metric_name]:
                    return task_cfg[metric_name]
                for sub_cfg in task_cfg.values():
                    if isinstance(sub_cfg, dict) and metric_name in sub_cfg and isinstance(sub_cfg[metric_name], dict) and "max_regression" in sub_cfg[metric_name]:
                        return sub_cfg[metric_name]

        for section in ("classification", "regression", "timeseries", "forecasting"):
            section_cfg = self.thresholds.get(section, {})
            if isinstance(section_cfg, dict):
                if metric_name in section_cfg and isinstance(section_cfg[metric_name], dict) and "max_regression" in section_cfg[metric_name]:
                    return section_cfg[metric_name]
                for sub_cfg in section_cfg.values():
                    if isinstance(sub_cfg, dict) and metric_name in sub_cfg and isinstance(sub_cfg[metric_name], dict) and "max_regression" in sub_cfg[metric_name]:
                        return sub_cfg[metric_name]

        def _find_metric(cfg: dict) -> dict | None:
            if metric_name in cfg and isinstance(cfg[metric_name], dict) and "max_regression" in cfg[metric_name]:
                return cfg[metric_name]
            for val in cfg.values():
                if isinstance(val, dict):
                    res = _find_metric(val)
                    if res is not None:
                        return res
            return None

        found = _find_metric(self.thresholds)
        if found is not None:
            return found

        raise ValueError(
            f"No regression threshold configured "
            f"for metric: {metric_name}"
        )

    @staticmethod
    def _validate_evaluations(
        baseline: ModelEvaluation,
        candidate: ModelEvaluation,
    ) -> None:
        if baseline.model_name != candidate.model_name:
            raise ValueError(
                "Baseline and candidate must belong "
                "to the same model."
            )

        if baseline.dataset_version != candidate.dataset_version:
            raise ValueError(
                "Baseline and candidate must use "
                "the same evaluation dataset."
            )

        if baseline.sample_count != candidate.sample_count:
            raise ValueError(
                "Baseline and candidate must use "
                "the same number of evaluation samples."
            )


def validate_statistical_significance(
    y_true: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    metric_function: Callable,
    confidence_level: float = 0.95,
    n_iterations: int = 1000,
) -> tuple[float, float, float, bool]:
    """
    Determine whether the observed metric difference
    has statistical evidence of being different from zero.
    """

    (
        observed_difference,
        lower_bound,
        upper_bound,
    ) = bootstrap_metric_difference(
        y_true=y_true,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        metric_function=metric_function,
        n_iterations=n_iterations,
        confidence_level=confidence_level,
    )

    statistically_significant = (
        lower_bound > 0
        or upper_bound < 0
    )

    return (
        observed_difference,
        lower_bound,
        upper_bound,
        statistically_significant,
    )
