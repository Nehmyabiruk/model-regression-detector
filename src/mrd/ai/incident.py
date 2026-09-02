from datetime import datetime, timezone
from uuid import uuid4


def create_ai_incident(
    report: dict,
    investigation: dict,
    root_cause: dict,
    recommendations: dict,
) -> dict:

    return {
        "incident_id": str(uuid4()),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": report.get(
            "model_name"
        ),
        "baseline_version": report.get(
            "baseline_version"
        ),
        "candidate_version": report.get(
            "candidate_version"
        ),
        "investigation": investigation,
        "root_cause": root_cause,
        "recommendations": recommendations,
    }