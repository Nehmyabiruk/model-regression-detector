from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from mrd.evaluation.model_input import (
    format_prediction_failure,
    prepare_model_input,
)
from mrd.metrics.classification import calculate_classification_metrics
from mrd.schemas import ModelEvaluation


class ModelEvaluator:
    """Evaluate classification models against a labeled dataset."""

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        model_name: str,
        model_version: str,
        dataset_version: str,
    ) -> ModelEvaluation:
        """Run the model and produce a structured evaluation result.

        X may be a DataFrame of mixed dtypes (preferred for uploaded CSVs)
        or a numeric ndarray. Preprocessing is never applied here; the
        uploaded model is expected to consume the features as trained.
        """

        self._validate_inputs(X, y)

        X_input = prepare_model_input(model, X)

        try:
            y_pred = model.predict(X_input)
        except Exception as exc:
            raise RuntimeError(
                format_prediction_failure(model, X, exc)
            ) from exc

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "The model must implement predict_proba() "
                "for classification evaluation."
            )

        try:
            probabilities = model.predict_proba(X_input)
        except Exception as exc:
            raise RuntimeError(
                format_prediction_failure(model, X, exc)
            ) from exc

        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "Classification requires predict_proba() "
                "to return an array with at least two columns."
            )

        unique_targets = np.unique(y)
        if probabilities.shape[1] == 2 and len(unique_targets) <= 2:
            y_score = probabilities[:, 1]
        else:
            y_score = probabilities

        metrics = calculate_classification_metrics(
            y_true=y,
            y_pred=y_pred,
            y_score=y_score,
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
