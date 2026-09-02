import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss

from mrd.metrics.classification import calculate_classification_metrics
from mrd.evaluation.evaluator import ModelEvaluator
from mrd.detection.regression import RegressionDetector
from mrd.schemas import MetricResult, ModelEvaluation
from datetime import datetime, timezone


def test_multiclass_metrics_calculation() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    # 3-class probabilities
    y_score = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.1, 0.7],
        [0.1, 0.7, 0.2],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    metric_dict = {m.name: m.value for m in metrics}

    assert 'accuracy' in metric_dict
    assert 'precision' in metric_dict
    assert 'recall' in metric_dict
    assert 'f1' in metric_dict
    assert 'roc_auc' in metric_dict
    assert 'log_loss' in metric_dict
    assert 'average_precision' not in metric_dict

    assert metric_dict['accuracy'] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metric_dict['precision'] == pytest.approx(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    assert metric_dict['recall'] == pytest.approx(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    assert metric_dict['f1'] == pytest.approx(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    assert metric_dict['roc_auc'] == pytest.approx(roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted'))
    assert metric_dict['log_loss'] == pytest.approx(log_loss(y_true, y_score))


def test_multiclass_evaluator() -> None:
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [1.1, 1.2],
        [1.2, 1.1],
        [2.1, 2.2],
        [2.2, 2.1],
        [0.15, 0.25],
        [1.15, 1.25],
        [2.15, 2.25],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])

    model = LogisticRegression()
    model.fit(X, y)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate(
        model=model,
        X=X,
        y=y,
        model_name='multiclass-classifier',
        model_version='1.0.0',
        dataset_version='eval-v1',
    )

    assert result.sample_count == 9
    assert len(result.metrics) == 6
    metric_names = [m.name for m in result.metrics]
    assert 'accuracy' in metric_names
    assert 'roc_auc' in metric_names
    assert 'average_precision' not in metric_names


def test_multiclass_regression_detection() -> None:
    baseline = ModelEvaluation(
        model_name='multiclass-model',
        model_version='1.0.0',
        dataset_version='eval-v1',
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name='accuracy', value=0.95, higher_is_better=True),
            MetricResult(name='f1', value=0.94, higher_is_better=True),
            MetricResult(name='log_loss', value=0.15, higher_is_better=False),
        ],
    )

    candidate = ModelEvaluation(
        model_name='multiclass-model',
        model_version='1.1.0',
        dataset_version='eval-v1',
        evaluated_at=datetime.now(timezone.utc),
        sample_count=100,
        metrics=[
            MetricResult(name='accuracy', value=0.88, higher_is_better=True),
            MetricResult(name='f1', value=0.85, higher_is_better=True),
            MetricResult(name='log_loss', value=0.35, higher_is_better=False),
        ],
    )

    detector = RegressionDetector()
    report = detector.compare(baseline=baseline, candidate=candidate, task_type='classification')

    assert report.status == 'regression'
    assert report.performance['accuracy']['regression'] is True
    assert report.performance['f1']['regression'] is True
    assert report.performance['log_loss']['regression'] is True
