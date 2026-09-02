from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from mrd.evaluation.model_input import (
    format_prediction_failure,
    prepare_model_input,
)
from mrd.metrics.regression import calculate_regression_metrics
from mrd.schemas import ModelEvaluation


class RegressionModelEvaluator:
    """Evaluate regression models against a target dataset."""

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        model_name: str,
        model_version: str,
        dataset_version: str,
    ) -> ModelEvaluation:
        """Run the model and produce a structured regression evaluation result."""

        self._validate_inputs(X, y)

        X_input = prepare_model_input(model, X)

        try:
            y_pred = model.predict(X_input)
        except Exception as exc:
            raise RuntimeError(
                format_prediction_failure(model, X, exc)
            ) from exc

        y_pred = np.asarray(y_pred)

        metrics = calculate_regression_metrics(
            y_true=y,
            y_pred=y_pred,
        )

        return ModelEvaluation(
            model_name=model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            evaluated_at=datetime.now(timezone.utc),
            sample_count=len(y),
            metrics=metrics,
        )

    @staticmethod
    def _validate_inputs(
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Validate the basic evaluation inputs."""

        if len(X) == 0:
            raise ValueError("X cannot be empty.")

        if len(y) == 0:
            raise ValueError("y cannot be empty.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )
