import numpy as np
import pandas as pd
import pytest

from mrd.drift.detector import DriftDetector
from mrd.drift.psi import calculate_categorical_psi, calculate_psi


def test_identical_distributions_have_low_psi() -> None:
    data = np.arange(1000)

    score = calculate_psi(
        baseline=data,
        current=data.copy(),
    )

    assert score < 0.01


def test_different_distributions_have_higher_psi() -> None:
    baseline = np.random.default_rng(
        42
    ).normal(
        loc=0,
        scale=1,
        size=5000,
    )

    current = np.random.default_rng(
        43
    ).normal(
        loc=3,
        scale=1,
        size=5000,
    )

    score = calculate_psi(
        baseline=baseline,
        current=current,
    )

    assert score > 0.25


def test_empty_baseline_fails() -> None:
    with pytest.raises(ValueError):
        calculate_psi(
            baseline=np.array([]),
            current=np.array([1, 2, 3]),
        )


def test_empty_current_fails() -> None:
    with pytest.raises(ValueError):
        calculate_psi(
            baseline=np.array([1, 2, 3]),
            current=np.array([]),
        )


def test_drift_detector_classifies_high_drift() -> None:
    baseline = np.random.default_rng(
        42
    ).normal(
        loc=0,
        scale=1,
        size=5000,
    )

    current = np.random.default_rng(
        43
    ).normal(
        loc=3,
        scale=1,
        size=5000,
    )

    detector = DriftDetector(
        thresholds={
            "drift": {
                "psi": {
                    "low": 0.10,
                    "moderate": 0.25,
                }
            }
        }
    )

    result = detector.detect_numeric_feature(
        feature_name="income",
        baseline=baseline,
        current=current,
    )

    assert result.severity == "high"
    assert result.score > 0.25


def test_identical_categorical_distributions_have_low_psi() -> None:
    data = np.array(["female", "male", "female", "male"] * 50)

    score = calculate_categorical_psi(
        baseline=data,
        current=data.copy(),
    )

    assert score < 0.01


def test_shifted_categorical_distributions_have_higher_psi() -> None:
    baseline = np.array(["female"] * 90 + ["male"] * 10)
    current = np.array(["female"] * 10 + ["male"] * 90)

    score = calculate_categorical_psi(
        baseline=baseline,
        current=current,
    )

    assert score > 0.25


def test_detect_feature_routes_categorical_columns() -> None:
    detector = DriftDetector(
        thresholds={
            "drift": {
                "psi": {
                    "low": 0.10,
                    "moderate": 0.25,
                }
            }
        }
    )

    result = detector.detect_feature(
        feature_name="gender",
        baseline=np.array(["female"] * 90 + ["male"] * 10),
        current=np.array(["female"] * 10 + ["male"] * 90),
    )

    assert result.method == "psi_categorical"
    assert result.feature_name == "gender"
    assert result.severity == "high"


def test_detect_dataframe_uses_column_names() -> None:
    detector = DriftDetector()
    frame = pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "age": [21, 30, 40, 50],
        }
    )

    results = detector.detect_dataframe(frame, frame)
    names = {result.feature_name for result in results}

    assert names == {"gender", "age"}
    methods = {
        result.feature_name: result.method
        for result in results
    }
    assert methods["gender"] == "psi_categorical"
    assert methods["age"] == "psi"