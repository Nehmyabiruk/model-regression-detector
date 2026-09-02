import numpy as np

from mrd.detection.regression import RegressionDetector
from mrd.drift.detector import DriftDetector
from mrd.pipeline import RegressionPipeline
from mrd.reports.report import ReportGenerator
from mrd.schemas import (
    MetricResult,
    ModelEvaluation,
)


def create_evaluation(
    version: str,
    roc_auc: float,
) -> ModelEvaluation:

    return ModelEvaluation(
        model_name="credit-risk",
        model_version=version,
        dataset_version="eval-v1",
        metrics=[
            MetricResult(
                name="roc_auc",
                value=roc_auc,
                higher_is_better=True,
            )
        ],
        sample_count=1000,
    )


def test_pipeline_detects_regression() -> None:

    baseline = create_evaluation(
        version="2.2.0",
        roc_auc=0.98,
    )

    candidate = create_evaluation(
        version="2.3.0",
        roc_auc=0.90,
    )

    regression_detector = RegressionDetector(
        thresholds={
            "classification": {
                "roc_auc": {
                    "max_regression": 0.05,
                }
            }
        }
    )

    drift_detector = DriftDetector(
        thresholds={
            "drift": {
                "psi": {
                    "low": 0.10,
                    "moderate": 0.25,
                }
            }
        }
    )

    pipeline = RegressionPipeline(
        regression_detector=regression_detector,
        drift_detector=drift_detector,
        report_generator=ReportGenerator(),
    )

    report = pipeline.run(
        baseline=baseline,
        candidate=candidate,
    )

    assert report.status == "regression"
    assert report.model_name == "credit-risk"
    assert report.baseline_version == "2.2.0"
    assert report.candidate_version == "2.3.0"


def test_pipeline_detects_categorical_feature_drift() -> None:
    baseline = create_evaluation(
        version="2.2.0",
        roc_auc=0.98,
    )

    candidate = create_evaluation(
        version="2.3.0",
        roc_auc=0.98,
    )

    pipeline = RegressionPipeline(
        regression_detector=RegressionDetector(
            thresholds={
                "classification": {
                    "roc_auc": {
                        "max_regression": 0.05,
                    }
                }
            }
        ),
        drift_detector=DriftDetector(
            thresholds={
                "drift": {
                    "psi": {
                        "low": 0.10,
                        "moderate": 0.25,
                    }
                }
            }
        ),
        report_generator=ReportGenerator(),
    )

    report = pipeline.run(
        baseline=baseline,
        candidate=candidate,
        feature_baseline={
            "gender": np.array(["female"] * 90 + ["male"] * 10),
        },
        feature_candidate={
            "gender": np.array(["female"] * 10 + ["male"] * 90),
        },
    )

    assert len(report.drift) == 1
    assert report.drift[0].feature_name == "gender"
    assert report.drift[0].method == "psi_categorical"
    assert report.drift[0].severity == "high"