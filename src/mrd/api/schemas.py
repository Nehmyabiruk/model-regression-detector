from typing import Literal
from pydantic import BaseModel
from mrd.schemas import RegressionReport


class InvestigationRequest(BaseModel):

    report: dict


class InvestigationResponse(BaseModel):

    investigation: dict

    root_cause: dict

    recommendations: dict

    agent_analysis: str


class EvaluationRequest(BaseModel):
    """Request to run a model evaluation."""

    model_name: str
    baseline_version: str
    candidate_version: str
    dataset_version: str
    target_column: str
    model_type: Literal["classification", "regression", "timeseries"] = "classification"
    time_column: str | None = None


class EvaluationResponse(BaseModel):
    """Response containing the regression report."""

    report: RegressionReport