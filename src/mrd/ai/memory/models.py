from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RegressionIncident(BaseModel):
    """Historical record of a model regression."""

    incident_id: str

    model_name: str

    baseline_version: str

    candidate_version: str

    dataset_version: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    status: str

    evidence: dict

    root_causes: list[dict] = []

    resolution: str | None = None