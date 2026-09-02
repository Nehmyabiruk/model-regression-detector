from typing import Literal

from pydantic import BaseModel, Field


class RootCauseHypothesis(BaseModel):
    """A possible explanation for a model regression."""

    cause: str

    category: Literal[
        "data_drift",
        "feature",
        "model",
        "segment",
        "training",
        "evaluation",
        "unknown",
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: list[str]

    reasoning: str

    recommended_checks: list[str]


class RootCauseAnalysis(BaseModel):
    """Ranked root-cause hypotheses."""

    hypotheses: list[RootCauseHypothesis]