import numpy as np
import pandas as pd


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index (PSI)
    between a baseline and current numerical distribution.
    """

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    if baseline.size == 0:
        raise ValueError(
            "Baseline data cannot be empty."
        )

    if current.size == 0:
        raise ValueError(
            "Current data cannot be empty."
        )

    if bins < 2:
        raise ValueError(
            "bins must be at least 2."
        )

    breakpoints = np.percentile(
        baseline,
        np.linspace(
            0,
            100,
            bins + 1,
        ),
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:
        return 0.0

    baseline_counts, _ = np.histogram(
        baseline,
        bins=breakpoints,
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints,
    )

    baseline_percentages = (
        baseline_counts / baseline.size
    )

    current_percentages = (
        current_counts / current.size
    )

    baseline_percentages = np.clip(
        baseline_percentages,
        epsilon,
        None,
    )

    current_percentages = np.clip(
        current_percentages,
        epsilon,
        None,
    )

    psi_values = (
        current_percentages
        - baseline_percentages
    ) * np.log(
        current_percentages
        / baseline_percentages
    )

    return float(
        np.sum(psi_values)
    )


def calculate_categorical_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate PSI between two categorical distributions.

    Categories are compared by relative frequency. Missing values are
    treated as an explicit category so they are not silently dropped.
    """

    baseline_series = pd.Series(baseline, dtype="object")
    current_series = pd.Series(current, dtype="object")

    if baseline_series.empty:
        raise ValueError(
            "Baseline data cannot be empty."
        )

    if current_series.empty:
        raise ValueError(
            "Current data cannot be empty."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    baseline_series = (
        baseline_series.fillna("__missing__").astype(str)
    )
    current_series = (
        current_series.fillna("__missing__").astype(str)
    )

    categories = sorted(
        set(baseline_series.unique())
        | set(current_series.unique())
    )

    if len(categories) < 2:
        return 0.0

    baseline_counts = baseline_series.value_counts()
    current_counts = current_series.value_counts()

    psi = 0.0

    for category in categories:
        baseline_share = max(
            float(baseline_counts.get(category, 0))
            / len(baseline_series),
            epsilon,
        )
        current_share = max(
            float(current_counts.get(category, 0))
            / len(current_series),
            epsilon,
        )
        psi += (current_share - baseline_share) * np.log(
            current_share / baseline_share
        )

    return float(psi)