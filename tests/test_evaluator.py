import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mrd.evaluation.evaluator import ModelEvaluator


@pytest.fixture
def trained_model() -> LogisticRegression:
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    return model


def test_evaluator_returns_model_evaluation(
    trained_model: LogisticRegression,
) -> None:
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    y = np.array([0, 0, 0, 1, 1, 1])

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=trained_model,
        X=X,
        y=y,
        model_name="credit-risk",
        model_version="2.2.0",
        dataset_version="eval-v1",
    )

    assert result.model_name == "credit-risk"
    assert result.model_version == "2.2.0"
    assert result.dataset_version == "eval-v1"
    assert result.sample_count == 6
    assert len(result.metrics) == 7


def test_evaluator_rejects_empty_data(
    trained_model: LogisticRegression,
) -> None:
    X = np.empty((0, 1))
    y = np.array([])

    evaluator = ModelEvaluator()

    with pytest.raises(ValueError, match="X cannot be empty"):
        evaluator.evaluate(
            model=trained_model,
            X=X,
            y=y,
            model_name="credit-risk",
            model_version="2.2.0",
            dataset_version="eval-v1",
        )


def test_evaluator_rejects_mismatched_lengths(
    trained_model: LogisticRegression,
) -> None:
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([0, 1])

    evaluator = ModelEvaluator()

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        evaluator.evaluate(
            model=trained_model,
            X=X,
            y=y,
            model_name="credit-risk",
            model_version="2.2.0",
            dataset_version="eval-v1",
        )


def _mixed_frame(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "gender": rng.choice(["female", "male"], size=n),
            "age": rng.integers(18, 70, size=n),
        }
    )
    y = (
        (X["gender"] == "male").astype(int)
        + (X["age"] > 40).astype(int)
        > 0
    ).to_numpy()
    return X, y


def test_evaluator_accepts_pipeline_with_categorical_features() -> None:
    X, y = _mixed_frame()

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
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X, y)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate(
        model=pipeline,
        X=X,
        y=y,
        model_name="generic-classifier",
        model_version="1.0.0",
        dataset_version="eval-v1",
    )

    assert result.sample_count == len(y)
    assert len(result.metrics) == 7


def test_evaluator_rejects_bare_estimator_with_categorical_features(
    trained_model: LogisticRegression,
) -> None:
    X = pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male", "female", "male"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    evaluator = ModelEvaluator()

    with pytest.raises(ValueError, match="preprocessing pipeline"):
        evaluator.evaluate(
            model=trained_model,
            X=X,
            y=y,
            model_name="generic-classifier",
            model_version="1.0.0",
            dataset_version="eval-v1",
        )


def test_evaluator_accepts_numeric_dataframe_with_bare_estimator() -> None:
    X = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X.to_numpy(), y)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate(
        model=model,
        X=X,
        y=y,
        model_name="generic-classifier",
        model_version="1.0.0",
        dataset_version="eval-v1",
    )

    assert result.sample_count == 6
    assert len(result.metrics) == 7