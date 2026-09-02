import tempfile
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mrd.detection.regression import RegressionDetector
from mrd.evaluation.loader import ModelLoader
from mrd.evaluation.regression import RegressionModelEvaluator
from mrd.metrics.regression import calculate_regression_metrics
from mrd.schemas import MetricResult, ModelEvaluation


def test_regression_metrics_calculation() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    metrics = calculate_regression_metrics(y_true, y_pred)
    metric_dict = {m.name: m for m in metrics}

    assert 'mae' in metric_dict
    assert 'mse' in metric_dict
    assert 'rmse' in metric_dict
    assert 'r2' in metric_dict

    assert metric_dict['mae'].higher_is_better is False
    assert metric_dict['mse'].higher_is_better is False
    assert metric_dict["rmse"].higher_is_better is False
    assert metric_dict["r2"].higher_is_better is True

    assert metric_dict['mae'].value == pytest.approx(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    assert metric_dict['mse'].value == pytest.approx(mse)
    assert metric_dict['rmse'].value == pytest.approx(np.sqrt(mse))
    assert metric_dict['r2'].value == pytest.approx(r2_score(y_true, y_pred))


def test_regression_evaluator() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    model = LinearRegression()
    model.fit(X, y)

    evaluator = RegressionModelEvaluator()
    result = evaluator.evaluate(
        model=model,
        X=X,
        y=y,
        model_name="housing-predictor",
        model_version="1.0.0",
        dataset_version="eval-v1",
    )

    assert result.model_name == "housing-predictor"
    assert result.sample_count == 5
    assert len(result.metrics) == 4
    metric_dict = {m.name: m.value for m in result.metrics}
    assert metric_dict["r2"] == pytest.approx(1.0)
    assert metric_dict["mae"] == pytest.approx(0.0, abs=1e-6)


def test_model_loader_for_regression() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression()
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "regressor.pkl"
        joblib.dump(model, model_path)

        loaded = ModelLoader.load(model_path, model_type="regression")
        assert hasattr(loaded, "predict")
        assert not hasattr(loaded, "predict_proba")


def test_regression_detection_directions() -> None:
    baseline = ModelEvaluation(
        model_name="regression-project",
        model_version="1.0.0",
        dataset_version="eval-v1",
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name="mae", value=1.0, higher_is_better=False),
            MetricResult(name="r2", value=0.90, higher_is_better=True),
        ],
    )

    # Candidate has worse MAE (1.15 > 1.0 + 0.05) and worse R2 (0.80 < 0.90 - 0.02)
    candidate = ModelEvaluation(
        model_name="regression-project",
        model_version="1.1.0",
        dataset_version="eval-v1",
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name="mae", value=1.15, higher_is_better=False),
            MetricResult(name="r2", value=0.80, higher_is_better=True),
        ],
    )

    detector = RegressionDetector()
    report = detector.compare(baseline=baseline, candidate=candidate, task_type="regression")

    assert report.status == "regression"
    assert report.task_type == "regression"
    assert report.performance["mae"]["regression"] is True
    assert report.performance["r2"]["regression"] is True
