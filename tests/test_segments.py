import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mrd.segments.analyzer import (
    SegmentAnalyzer,
    create_categorical_segments,
    create_numeric_segments,
)


def test_numeric_segments_create_correct_masks() -> None:
    values = np.array(
        [18, 20, 24, 25, 30, 35, 40, 50, 60]
    )

    segments = create_numeric_segments(
        values=values,
        boundaries=[18, 25, 40, 60],
    )

    assert segments["18_25"].sum() == 3
    assert segments["25_40"].sum() == 3
    assert segments["40_60"].sum() == 3


def test_segment_regression_is_detected() -> None:
    y_true = np.array(
        [0, 1, 0, 1, 0, 1] * 20
    )

    baseline_predictions = np.array(
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7] * 20
    )

    candidate_predictions = np.array(
        [0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 20
    )

    analyzer = SegmentAnalyzer(
        minimum_sample_count=100
    )

    result = analyzer.compare_segment(
        segment_name="age_18_25",
        y_true=y_true,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        metric_name="roc_auc",
        metric_function=roc_auc_score,
        max_regression=0.05,
        higher_is_better=True,
    )

    assert result.regression is True
    assert result.difference < -0.05


def test_small_segment_is_rejected() -> None:
    y_true = np.array([0, 1, 0])
    baseline_predictions = np.array(
        [0.1, 0.9, 0.2]
    )
    candidate_predictions = np.array(
        [0.2, 0.8, 0.3]
    )

    analyzer = SegmentAnalyzer(
        minimum_sample_count=100
    )

    with pytest.raises(ValueError):
        analyzer.compare_segment(
            segment_name="tiny",
            y_true=y_true,
            baseline_predictions=baseline_predictions,
            candidate_predictions=candidate_predictions,
            metric_name="roc_auc",
            metric_function=roc_auc_score,
            max_regression=0.05,
        )


def test_categorical_segments_create_correct_masks() -> None:
    values = np.array(
        ["female", "male", "female", "other"]
    )

    segments = create_categorical_segments(values)

    assert segments["female"].sum() == 2
    assert segments["male"].sum() == 1
    assert segments["other"].sum() == 1


def test_mismatched_lengths_are_rejected() -> None:
    y_true = np.array([0, 1, 0])

    baseline_predictions = np.array(
        [0.1, 0.9]
    )

    candidate_predictions = np.array(
        [0.2, 0.8, 0.3]
    )

    analyzer = SegmentAnalyzer()

    with pytest.raises(ValueError):
        analyzer.compare_segment(
            segment_name="invalid",
            y_true=y_true,
            baseline_predictions=baseline_predictions,
            candidate_predictions=candidate_predictions,
            metric_name="roc_auc",
            metric_function=roc_auc_score,
            max_regression=0.05,
        )