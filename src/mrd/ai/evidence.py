from typing import Any

from mrd.schemas import RegressionReport


def build_evidence(
    report: RegressionReport,
) -> dict[str, Any]:
    """
    Convert a regression report into a compact
    evidence package for AI investigation.
    """

    return {
        "model": {
            "name": report.model_name,
            "baseline_version": report.baseline_version,
            "candidate_version": report.candidate_version,
            "dataset": report.dataset_name,
            "task_type": getattr(report, "task_type", "classification"),
        },

        "status": report.status,

        "performance": report.performance,

        "drift": [
            result.model_dump()
            for result in report.drift
        ],

        "segments": [
            result.model_dump()
            for result in report.segments
        ],

        "summary": report.summary,
    }