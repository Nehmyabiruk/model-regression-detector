"""Adapt evaluation features to an uploaded model's input contract.

The evaluation dataset is never re-encoded here. Preprocessing belongs to the
uploaded artifact (typically a sklearn Pipeline fitted at training time).

This module only:
- preserves a DataFrame when the model can consume raw mixed-type features
- converts to a numeric array only when every column is already numeric and
  the model is a bare estimator trained without feature names
- rejects mixed-type data against a bare estimator with a clear error
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline


def non_numeric_columns(X: pd.DataFrame) -> list[str]:
    """Return feature names that are not numeric dtypes."""

    return [
        str(column)
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]


def unwrap_fitted_estimator(model: Any) -> Any:
    """Unwrap search wrappers (GridSearchCV, RandomizedSearchCV, etc.)."""

    seen: set[int] = set()
    current = model

    while id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "best_estimator_", None)
        if nested is None:
            break
        current = nested

    return current


def contains_preprocessing(model: Any) -> bool:
    """
    Return True when the artifact looks like a preprocessing+estimator pipeline.

    Bare estimators (LogisticRegression, RandomForest, ...) return False even
    if they were fitted on a named DataFrame.
    """

    return _contains_preprocessing(model, seen=set())


def prepare_model_input(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """
    Return the feature object that should be passed to predict/predict_proba.

    Raises:
        ValueError: If the dataset has non-numeric columns and the model does
            not include preprocessing that can consume them.
        TypeError: If X is not a DataFrame or ndarray.
    """

    if isinstance(X, np.ndarray):
        return _prepare_from_array(model, X)

    if isinstance(X, pd.DataFrame):
        return _prepare_from_frame(model, X)

    raise TypeError(
        f"Features must be a pandas DataFrame or numpy ndarray, "
        f"got {type(X).__name__}."
    )


def format_prediction_failure(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    exc: Exception,
) -> str:
    """Build an actionable error when model.predict() rejects the features."""

    model_type = type(model).__name__
    message = (
        f"The uploaded {model_type} failed during prediction: {exc}."
    )

    if isinstance(X, pd.DataFrame):
        categorical = non_numeric_columns(X)
        if categorical:
            message += (
                " The evaluation dataset contains non-numeric columns: "
                f"{', '.join(categorical)}."
            )
            if not contains_preprocessing(model):
                message += " " + _missing_preprocessing_message(categorical)
            else:
                message += (
                    " The uploaded pipeline could not consume these columns. "
                    "Confirm that the serialized artifact is the same "
                    "preprocessing+model pipeline used at training time, "
                    "and that the dataset columns match the training schema."
                )

    return message


def _prepare_from_array(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(
            f"Feature array must be 2-dimensional, got {X.ndim}."
        )

    if _array_is_non_numeric(X) and not contains_preprocessing(model):
        raise ValueError(
            _missing_preprocessing_message(
                ["<unnamed array columns>"]
            )
        )

    _validate_feature_count(model, X.shape[1])
    return X


def _prepare_from_frame(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame | np.ndarray:
    X = _align_feature_frame(X, model)
    categorical = non_numeric_columns(X)

    if contains_preprocessing(model):
        _validate_feature_count(model, X.shape[1])
        return X

    if categorical:
        raise ValueError(
            _missing_preprocessing_message(categorical)
        )

    _validate_feature_count(model, X.shape[1])

    # Bare estimator trained on a named numeric DataFrame can consume one.
    if hasattr(model, "feature_names_in_"):
        return X

    try:
        return X.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Failed to convert numeric features to a float array: "
            f"{exc}"
        ) from exc


def _align_feature_frame(
    X: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        return X

    expected = list(feature_names)
    missing = [
        name for name in expected
        if name not in X.columns
    ]

    if missing:
        available = ", ".join(map(str, X.columns))
        missing_text = ", ".join(map(str, missing))
        raise ValueError(
            "Dataset is missing columns required by the model: "
            f"{missing_text}. Dataset columns: {available}."
        )

    return X.loc[:, expected]


def _validate_feature_count(model: Any, n_features: int) -> None:
    expected = getattr(model, "n_features_in_", None)

    if expected is None:
        return

    expected_count = int(expected)

    if n_features != expected_count:
        raise ValueError(
            f"The model expects {expected_count} features "
            f"but the dataset provides {n_features}."
        )


def _array_is_non_numeric(X: np.ndarray) -> bool:
    return X.dtype.kind in ("O", "U", "S")


def _is_pipeline(model: Any) -> bool:
    if isinstance(model, SklearnPipeline):
        return True

    steps = getattr(model, "steps", None)
    named = getattr(model, "named_steps", None)
    return isinstance(steps, (list, tuple)) and named is not None


def _is_column_transformer(model: Any) -> bool:
    if isinstance(model, ColumnTransformer):
        return True

    return (
        hasattr(model, "transformers")
        and hasattr(model, "transform")
        and hasattr(model, "remainder")
    )


def _contains_preprocessing(
    model: Any,
    seen: set[int],
) -> bool:
    if model is None:
        return False

    marker = id(model)
    if marker in seen:
        return False
    seen.add(marker)

    model = unwrap_fitted_estimator(model)

    if _is_column_transformer(model):
        return True

    if hasattr(model, "transformer_list"):
        return True

    if _is_pipeline(model):
        steps = list(model.steps)
        if len(steps) >= 2:
            return True
        if steps:
            return _contains_preprocessing(steps[0][1], seen)
        return False

    for attr in ("estimator", "base_estimator", "estimator_"):
        nested = getattr(model, attr, None)
        if nested is not None and nested is not model:
            if _contains_preprocessing(nested, seen):
                return True

    return False


def _missing_preprocessing_message(non_numeric: list[str]) -> str:
    shown = ", ".join(str(column) for column in non_numeric[:12])
    extra = ""
    if len(non_numeric) > 12:
        extra = f" (and {len(non_numeric) - 12} more)"

    return (
        f"The evaluation dataset contains non-numeric feature columns "
        f"({shown}{extra}) but the uploaded model does not include a "
        "preprocessing pipeline that can consume them. Upload the complete "
        "training artifact (for example a sklearn.pipeline.Pipeline with "
        "encoding/scaling plus the estimator), or provide a dataset whose "
        "features already match the model's numeric input. This system will "
        "not invent categorical encodings at evaluation time, because they "
        "would not match the encodings used when the model was trained."
    )
