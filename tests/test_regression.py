from datetime import datetime, timezone

import pytest

from mrd.detection.regression import RegressionDetector
from mrd.schemas import MetricResult, ModelEvaluation


def create_evaluation(
    version: str,
    metrics: list[MetricResult],
    dataset_version: str = "eval-v1",
) -> ModelEvaluation:
    return ModelEvaluation(
        model_name="credit-risk",
        model_version=version,
        dataset_version=dataset_version,
        evaluated_at=datetime.now(timezone.utc),
        sample_count=1000,
        metrics=metrics,
    )


def test_detects_roc_auc_regression() -> None:
    baseline = create_evaluation(
        version="2.2.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.98,
                higher_is_better=True,
            )
        ],
    )

    candidate = create_evaluation(
        version="2.3.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.96,
                higher_is_better=True,
            )
        ],
    )

    detector = RegressionDetector(
        thresholds={
            "classification": {
                "roc_auc": {
                    "max_regression": 0.01,
                }
            }
        }
    )

    report = detector.compare(
        baseline=baseline,
        candidate=candidate,
    )

    assert report.status == "regression"
    assert report.performance["roc_auc"]["regression"] is True


def test_passes_when_regression_is_below_threshold() -> None:
    baseline = create_evaluation(
        version="2.2.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.98,
                higher_is_better=True,
            )
        ],
    )

    candidate = create_evaluation(
        version="2.3.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.975,
                higher_is_better=True,
            )
        ],
    )

    detector = RegressionDetector(
        thresholds={
            "classification": {
                "roc_auc": {
                    "max_regression": 0.01,
                }
            }
        }
    )

    report = detector.compare(
        baseline=baseline,
        candidate=candidate,
    )

    assert report.status == "pass"
    assert report.performance["roc_auc"]["regression"] is False


def test_handles_lower_is_better_metric() -> None:
    baseline = create_evaluation(
        version="2.2.0",
        metrics=[
            MetricResult(
                name="log_loss",
                value=0.30,
                higher_is_better=False,
            )
        ],
    )

    candidate = create_evaluation(
        version="2.3.0",
        metrics=[
            MetricResult(
                name="log_loss",
                value=0.40,
                higher_is_better=False,
            )
        ],
    )

    detector = RegressionDetector(
        thresholds={
            "classification": {
                "log_loss": {
                    "max_regression": 0.05,
                }
            }
        }
    )

    report = detector.compare(
        baseline=baseline,
        candidate=candidate,
    )

    assert report.status == "regression"


def test_rejects_different_datasets() -> None:
    baseline = create_evaluation(
        version="2.2.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.98,
                higher_is_better=True,
            )
        ],
        dataset_version="eval-v1",
    )

    candidate = create_evaluation(
        version="2.3.0",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=0.96,
                higher_is_better=True,
            )
        ],
        dataset_version="eval-v2",
    )

    detector = RegressionDetector(
        thresholds={
            "classification": {
                "roc_auc": {
                    "max_regression": 0.01,
                }
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="same evaluation dataset",
    ):
        detector.compare(
            baseline=baseline,
            candidate=candidate,
        )
