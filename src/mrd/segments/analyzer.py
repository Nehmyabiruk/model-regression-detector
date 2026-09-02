from collections.abc import Callable

import numpy as np
import pandas as pd

from mrd.schemas import SegmentResult


class SegmentAnalyzer:
    """Analyze model performance across dataset segments."""

    def __init__(
        self,
        minimum_sample_count: int = 100,
    ) -> None:
        if minimum_sample_count <= 0:
            raise ValueError(
                "minimum_sample_count must be greater than zero."
            )

        self.minimum_sample_count = (
            minimum_sample_count
        )

    def compare_segment(
        self,
        segment_name: str,
        y_true: np.ndarray,
        baseline_predictions: np.ndarray,
        candidate_predictions: np.ndarray,
        metric_name: str,
        metric_function: Callable,
        max_regression: float,
        higher_is_better: bool = True,
    ) -> SegmentResult:
        """
        Compare baseline and candidate performance
        for a single segment.
        """

        self._validate_inputs(
            y_true=y_true,
            baseline_predictions=baseline_predictions,
            candidate_predictions=candidate_predictions,
        )

        sample_count = len(y_true)

        if sample_count < self.minimum_sample_count:
            raise ValueError(
                f"Segment '{segment_name}' contains "
                f"{sample_count} samples, but at least "
                f"{self.minimum_sample_count} are required."
            )

        baseline_value = metric_function(
            y_true,
            baseline_predictions,
        )

        candidate_value = metric_function(
            y_true,
            candidate_predictions,
        )

        difference = (
            candidate_value
            - baseline_value
        )

        relative_change = (
            difference / abs(baseline_value)
            if baseline_value != 0
            else 0.0
        )

        regression = self._is_regression(
            difference=difference,
            max_regression=max_regression,
            higher_is_better=higher_is_better,
        )

        return SegmentResult(
            segment_name=segment_name,
            metric_name=metric_name,
            baseline_value=float(baseline_value),
            candidate_value=float(candidate_value),
            difference=float(difference),
            relative_change=float(relative_change),
            regression=regression,
            sample_count=sample_count,
        )

    @staticmethod
    def _is_regression(
        difference: float,
        max_regression: float,
        higher_is_better: bool,
    ) -> bool:
        if higher_is_better:
            return difference < -max_regression

        return difference > max_regression

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        baseline_predictions: np.ndarray,
        candidate_predictions: np.ndarray,
    ) -> None:
        if not (
            len(y_true)
            == len(baseline_predictions)
            == len(candidate_predictions)
        ):
            raise ValueError(
                "y_true and prediction arrays must "
                "have the same length."
            )


def create_numeric_segments(
    values: np.ndarray,
    boundaries: list[float],
) -> dict[str, np.ndarray]:
    """
    Create mutually exclusive boolean masks
    for numerical segments.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(boundaries) < 2:
        raise ValueError(
            "At least two boundaries are required."
        )

    segments: dict[str, np.ndarray] = {}

    for index in range(
        len(boundaries) - 1
    ):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        if index == len(boundaries) - 2:
            mask = (
                (values >= lower)
                & (values <= upper)
            )
        else:
            mask = (
                (values >= lower)
                & (values < upper)
            )

        segment_name = (
            f"{lower:g}_{upper:g}"
        )

        segments[segment_name] = mask

    return segments


def create_categorical_segments(
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create boolean masks for each observed category.

    Numeric-only helpers such as create_numeric_segments must not be
    used on string columns; call this instead.
    """

    series = pd.Series(values)
    segments: dict[str, np.ndarray] = {}

    for category in series.dropna().unique():
        segment_name = str(category)
        segments[segment_name] = (
            series == category
        ).to_numpy()

    return segments