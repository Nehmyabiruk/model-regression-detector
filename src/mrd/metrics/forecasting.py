import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


from mrd.schemas import MetricResult


def calculate_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[MetricResult]:
    """Calculate the time-series forecasting metrics used by the detector."""

    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse_value = float(mean_squared_error(y_true, y_pred))
    rmse_value = float(np.sqrt(mse_value))

    metrics = [
        MetricResult(
            name="mae",
            value=float(mean_absolute_error(y_true, y_pred)),
            higher_is_better=False,
        ),
        MetricResult(
            name="rmse",
            value=rmse_value,
            higher_is_better=False,
        ),
        MetricResult(
            name="r2",
            value=float(r2_score(y_true, y_pred)),
            higher_is_better=True,
        ),
    ]

    return metrics
