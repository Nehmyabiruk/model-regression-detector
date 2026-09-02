import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mrd.evaluation.model_input import (
    contains_preprocessing,
    non_numeric_columns,
    prepare_model_input,
)


def test_non_numeric_columns_detects_object_dtype() -> None:
    X = pd.DataFrame(
        {
            "gender": ["female", "male"],
            "age": [21, 30],
        }
    )

    assert non_numeric_columns(X) == ["gender"]


def test_contains_preprocessing_for_pipeline() -> None:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )

    assert contains_preprocessing(pipeline) is True
    assert contains_preprocessing(LogisticRegression()) is False


def test_prepare_passes_dataframe_to_pipeline() -> None:
    X = pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "age": [21, 30, 40, 50],
        }
    )
    y = np.array([0, 1, 0, 1])

    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["gender"],
                        ),
                        ("num", "passthrough", ["age"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    prepared = prepare_model_input(pipeline, X)

    assert isinstance(prepared, pd.DataFrame)
    assert list(prepared.columns) == ["gender", "age"]
    assert prepared["gender"].tolist() == X["gender"].tolist()


def test_prepare_rejects_bare_estimator_with_categoricals() -> None:
    X = pd.DataFrame({"gender": ["female", "male"]})
    model = LogisticRegression()
    model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

    with pytest.raises(ValueError, match="preprocessing pipeline"):
        prepare_model_input(model, X)


def test_prepare_converts_numeric_frame_for_array_trained_estimator() -> None:
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    model = LogisticRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 0]))

    prepared = prepare_model_input(model, X)

    assert isinstance(prepared, np.ndarray)
    assert prepared.dtype == np.float64
    assert prepared.shape == (3, 1)


def test_prepare_aligns_named_numeric_dataframe() -> None:
    X = pd.DataFrame(
        {
            "b": [1.0, 2.0, 3.0],
            "a": [4.0, 5.0, 6.0],
        }
    )
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X[["a", "b"]], y)

    prepared = prepare_model_input(model, X)

    assert isinstance(prepared, pd.DataFrame)
    assert list(prepared.columns) == ["a", "b"]


def test_prepare_reports_missing_expected_columns() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    y = np.array([0, 1])
    model = LogisticRegression()
    model.fit(train, y)

    with pytest.raises(ValueError, match="missing columns"):
        prepare_model_input(
            model,
            pd.DataFrame({"a": [0.0, 1.0]}),
        )
