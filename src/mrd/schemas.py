from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result of calculating a single model metric."""

    name: str
    value: float
    higher_is_better: bool


class ModelEvaluation(BaseModel):
    """Evaluation results for one model version."""

    model_name: str
    model_version: str
    dataset_version: str
    metrics: list[MetricResult]
    sample_count: int
    evaluated_at: datetime | None = Field(default=None)


class MetricComparison(BaseModel):
    """Comparison between a baseline and candidate metric."""

    metric_name: str
    baseline_value: float
    candidate_value: float
    difference: float
    relative_change: float
    regression: bool


class DriftResult(BaseModel):
    """Result of feature distribution drift analysis."""

    feature_name: str
    method: str
    score: float
    severity: Literal[
        "low",
        "moderate",
        "high",
    ]


class PredictionDriftResult(BaseModel):
    """Result of prediction distribution drift."""

    score: float
    severity: Literal[
        "low",
        "moderate",
        "high",
    ]


class SegmentResult(BaseModel):
    """Result of comparing model performance within a segment."""

    segment_name: str
    metric_name: str
    baseline_value: float
    candidate_value: float
    difference: float
    relative_change: float
    regression: bool
    sample_count: int


class ModelVersion(BaseModel):
    """Identity and location of a model version."""

    model_name: str
    version: str
    framework: str
    artifact_uri: str | None = None


class RegressionReport(BaseModel):
    """Complete structured model regression report."""

    report_id: str
    model_name: str
    baseline_version: str
    candidate_version: str
    dataset_name: str
    created_at: datetime
    task_type: str = "classification"

    status: Literal[
        "pass",
        "regression",
    ]

    performance: dict
    drift: list[DriftResult]
    segments: list[SegmentResult]

    summary: str