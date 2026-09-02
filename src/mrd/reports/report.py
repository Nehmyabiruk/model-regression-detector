
from datetime import datetime, timezone
from uuid import uuid4

from mrd.schemas import RegressionReport


class ReportGenerator:
    """Generate reproducible model regression reports."""

    def create_report(
        self,
        model_name: str,
        baseline_version: str,
        candidate_version: str,
        dataset_name: str,
        regression_detected: bool,
        summary: str,
        performance: dict | None = None,
        drift: list | None = None,
        segments: list | None = None,
        task_type: str = "classification",
    ) -> RegressionReport:
        """
        Create a structured regression report.

        performance, drift, and segments are optional so that
        the report generator can also create a basic report
        when detailed evidence is not available yet.
        """

        status = (
            "regression"
            if regression_detected
            else "pass"
        )

        return RegressionReport(
            report_id=str(uuid4()),
            model_name=model_name,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            dataset_name=dataset_name,
            created_at=datetime.now(
                timezone.utc
            ),
            task_type=task_type,
            status=status,
            performance=(
                performance
                if performance is not None
                else {}
            ),
            drift=(
                drift
                if drift is not None
                else []
            ),
            segments=(
                segments
                if segments is not None
                else []
            ),
            summary=summary,
        )

    def to_json(
        self,
        report: RegressionReport,
    ) -> str:
        """Convert a report to a JSON string."""

        return report.model_dump_json(
            indent=2
        )
