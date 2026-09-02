from mrd.reports.report import ReportGenerator


def test_regression_report_is_created() -> None:
    generator = ReportGenerator()

    report = generator.create_report(
    model_name="credit-risk",
    baseline_version="1.4.0",
    candidate_version="1.5.0",
    dataset_name="validation_2026_08",
    regression_detected=True,
    performance={
        "roc_auc": {
            "baseline": 0.981,
            "candidate": 0.964,
            "difference": -0.017,
        }
    },
    drift=[],
    segments=[],
    summary="ROC-AUC decreased significantly.",
)

    assert report.model_name == "credit-risk"
    assert report.baseline_version == "1.4.0"
    assert report.candidate_version == "1.5.0"
    assert report.status == "regression"
    assert report.report_id


def test_pass_report_is_created() -> None:
    generator = ReportGenerator()

    report = generator.create_report(
        model_name="credit-risk",
        baseline_version="1.4.0",
        candidate_version="1.5.0",
        dataset_name="validation_2026_08",
        regression_detected=False,
        summary="No meaningful regression detected.",
    )

    assert report.status == "pass"


def test_report_can_be_serialized_to_json() -> None:
    generator = ReportGenerator()

    report = generator.create_report(
        model_name="credit-risk",
        baseline_version="1.4.0",
        candidate_version="1.5.0",
        dataset_name="validation_2026_08",
        regression_detected=True,
        summary="Regression detected.",
    )

    json_data = generator.to_json(
        report
    )

    assert '"model_name": "credit-risk"' in json_data
    assert '"status": "regression"' in json_data