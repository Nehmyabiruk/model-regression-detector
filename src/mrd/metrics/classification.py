from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mrd.schemas import MetricResult


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> list[MetricResult]:
    """Calculate the classification metrics used by the detector."""

    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError(
            "y_true, y_pred, and y_score must contain the same number of samples."
        )

    unique_classes = np.unique(y_true)
    is_multiclass = len(unique_classes) > 2 or (y_score.ndim == 2 and y_score.shape[1] > 2)

    if is_multiclass:
        metrics = [
            MetricResult(
                name="accuracy",
                value=float(accuracy_score(y_true, y_pred)),
                higher_is_better=True,
            ),
            MetricResult(
                name="precision",
                value=float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="recall",
                value=float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="f1",
                value=float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="roc_auc",
                value=float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class="ovr",
                        average="weighted",
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="log_loss",
                value=float(
                    log_loss(
                        y_true,
                        y_score,
                    )
                ),
                higher_is_better=False,
            ),
        ]
    else:
        metrics = [
            MetricResult(
                name="accuracy",
                value=float(accuracy_score(y_true, y_pred)),
                higher_is_better=True,
            ),
            MetricResult(
                name="precision",
                value=float(
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="recall",
                value=float(
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="f1",
                value=float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="roc_auc",
                value=float(
                    roc_auc_score(
                        y_true,
                        y_score,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="average_precision",
                value=float(
                    average_precision_score(
                        y_true,
                        y_score,
                    )
                ),
                higher_is_better=True,
            ),
            MetricResult(
                name="log_loss",
                value=float(
                    log_loss(
                        y_true,
                        y_score,
                    )
                ),
                higher_is_better=False,
            ),
        ]

    return metrics