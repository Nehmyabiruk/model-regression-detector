import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mrd.detection.regression import RegressionDetector
from mrd.drift.temporal import TemporalDriftDetector
from mrd.evaluation.dataset_loader import DatasetLoader
from mrd.evaluation.timeseries import TimeSeriesModelEvaluator
from mrd.metrics.forecasting import calculate_forecasting_metrics
from mrd.schemas import MetricResult, ModelEvaluation


def test_forecasting_metrics_calculation() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([9.5, 21.0, 28.5, 42.0])

    metrics = calculate_forecasting_metrics(y_true, y_pred)
    metric_dict = {m.name: m for m in metrics}

    assert 'mae' in metric_dict
    assert 'rmse' in metric_dict
    assert 'r2' in metric_dict

    assert metric_dict['mae'].higher_is_better is False
    assert metric_dict["rmse"].higher_is_better is False
    assert metric_dict["r2"].higher_is_better is True

    assert metric_dict['mae'].value == pytest.approx(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    assert metric_dict['rmse'].value == pytest.approx(np.sqrt(mse))
    assert metric_dict["r2"].value == pytest.approx(r2_score(y_true, y_pred))


def test_timeseries_evaluator() -> None:
    X = pd.DataFrame({
        "time_step": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    model = LinearRegression()
    model.fit(X, y)

    evaluator = TimeSeriesModelEvaluator()
    result = evaluator.evaluate(
        model=model,
        X=X,
        y=y,
        model_name="demand-forecaster",
        model_version="1.0.0",
        dataset_version="2026-Q3",
    )

    assert result.model_name == "demand-forecaster"
    assert result.sample_count == 5
    assert len(result.metrics) == 3
    metric_dict = {m.name: m.value for m in result.metrics}
    assert metric_dict['r2'] == pytest.approx(1.0)


def test_dataset_loader_sorts_chronologically() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        # We write out of order timestamps
        csv_path.write_text(
            "date,feature1,target\n"
            "2026-01-03,10,300\n"
            "2026-01-01,10,100\n"
            "2026-01-02,10,200\n"
        )

        X, y = DatasetLoader.load_csv(
            csv_path,
            target_column="target",
            time_column="date",
        )

        assert np.array_equal(y, np.array([100, 200, 300]))
        assert list(X['date'].astype(str)) == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_dataset_loader_missing_time_column() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        csv_path.write_text("feature1,target\n10,100\n")

        with pytest.raises(ValueError, match="Time column 'date' not found"):
            DatasetLoader.load_csv(csv_path, target_column="target", time_column="date")


def test_dataset_loader_invalid_timestamps() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        csv_path.write_text(
            "date,feature1,target\n"
            "2026-01-01,10,100\n"
            "not-a-date,10,200\n"
        )

        with pytest.raises(ValueError, match="invalid or unparseable timestamps"):
            DatasetLoader.load_csv(csv_path, target_column="target", time_column="date")


def test_temporal_drift_detector_validation() -> None:
    dates = pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"])
    result = TemporalDriftDetector.validate_temporal_series(dates)

    assert result.is_monotonic is True
    assert result.sample_count == 3
    assert result.temporal_drift_supported is False


def test_timeseries_regression_detection() -> None:
    baseline = ModelEvaluation(
        model_name="forecaster",
        model_version="1.0.0",
        dataset_version="2026-Q3_validation",
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name="mae", value=0.5, higher_is_better=False),
            MetricResult(name="r2", value=0.95, higher_is_better=True),
        ],
    )

    candidate = ModelEvaluation(
        model_name="forecaster",
        model_version="1.1.0",
        dataset_version="2026-Q3_validation",
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name="mae", value=1.0, higher_is_better=False),
            MetricResult(name="r2", value=0.75, higher_is_better=True),
        ],
    )

    detector = RegressionDetector()
    report = detector.compare(baseline=baseline, candidate=candidate, task_type="timeseries")

    assert report.status == "regression"
    assert report.task_type == "timeseries"
    assert report.performance["mae"]["regression"] is True
    assert report.performance["r2"]["regression"] is True
