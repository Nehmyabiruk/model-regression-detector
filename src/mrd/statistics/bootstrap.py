from collections.abc import Callable

import numpy as np


def bootstrap_metric_difference(
    y_true: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    metric_function: Callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Estimate the confidence interval of the difference
    between candidate and baseline metric values.
    """

    if not (
        len(y_true)
        == len(baseline_predictions)
        == len(candidate_predictions)
    ):
        raise ValueError(
            "y_true and prediction arrays must have "
            "the same length."
        )

    if n_iterations <= 0:
        raise ValueError(
            "n_iterations must be greater than zero."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    rng = np.random.default_rng(random_state)

    sample_count = len(y_true)

    observed_difference = (
        metric_function(
            y_true,
            candidate_predictions,
        )
        - metric_function(
            y_true,
            baseline_predictions,
        )
    )

    bootstrap_differences = np.empty(
        n_iterations,
        dtype=float,
    )

    for iteration in range(n_iterations):
        indices = rng.integers(
            low=0,
            high=sample_count,
            size=sample_count,
        )

        bootstrap_y = y_true[indices]

        bootstrap_baseline = (
            baseline_predictions[indices]
        )

        bootstrap_candidate = (
            candidate_predictions[indices]
        )

        baseline_score = metric_function(
            bootstrap_y,
            bootstrap_baseline,
        )

        candidate_score = metric_function(
            bootstrap_y,
            bootstrap_candidate,
        )

        bootstrap_differences[iteration] = (
            candidate_score - baseline_score
        )

    alpha = 1 - confidence_level

    lower_percentile = (
        alpha / 2
    ) * 100

    upper_percentile = (
        1 - alpha / 2
    ) * 100

    lower_bound, upper_bound = np.percentile(
        bootstrap_differences,
        [
            lower_percentile,
            upper_percentile,
        ],
    )

    return (
        float(observed_difference),
        float(lower_bound),
        float(upper_bound),
    )