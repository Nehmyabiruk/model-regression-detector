import numpy as np
import pandas as pd

from mrd.config import load_thresholds
from mrd.schemas import DriftResult

from mrd.drift.psi import (
    calculate_categorical_psi,
    calculate_psi,
)


class DriftDetector:
    """Detect distribution drift in model inputs."""

    def __init__(
        self,
        thresholds: dict | None = None,
    ) -> None:
        self.thresholds = (
            thresholds
            if thresholds is not None
            else load_thresholds()
        )

    def detect_feature(
        self,
        feature_name: str,
        baseline: np.ndarray | pd.Series,
        current: np.ndarray | pd.Series,
    ) -> DriftResult:
        """Dispatch numeric vs categorical drift by dtype."""

        baseline_series = pd.Series(baseline)
        current_series = pd.Series(current)

        if (
            pd.api.types.is_numeric_dtype(baseline_series)
            and pd.api.types.is_numeric_dtype(current_series)
        ):
            return self.detect_numeric_feature(
                feature_name=feature_name,
                baseline=baseline_series.to_numpy(),
                current=current_series.to_numpy(),
            )

        return self.detect_categorical_feature(
            feature_name=feature_name,
            baseline=baseline_series.to_numpy(),
            current=current_series.to_numpy(),
        )

    def detect_dataframe(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
    ) -> list[DriftResult]:
        """Detect drift for every shared column in two frames."""

        results: list[DriftResult] = []
        shared_columns = [
            column
            for column in baseline.columns
            if column in current.columns
        ]

        for column in shared_columns:
            results.append(
                self.detect_feature(
                    feature_name=str(column),
                    baseline=baseline[column],
                    current=current[column],
                )
            )

        return results

    def detect_numeric_feature(
        self,
        feature_name: str,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Calculate PSI and classify its severity."""

        score = calculate_psi(
            baseline=baseline,
            current=current,
        )

        severity = self._classify_severity(
            score
        )

        return DriftResult(
            feature_name=feature_name,
            method="psi",
            score=score,
            severity=severity,
        )

    def detect_categorical_feature(
        self,
        feature_name: str,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Calculate categorical PSI and classify its severity."""

        score = calculate_categorical_psi(
            baseline=baseline,
            current=current,
        )

        severity = self._classify_severity(
            score
        )

        return DriftResult(
            feature_name=feature_name,
            method="psi_categorical",
            score=score,
            severity=severity,
        )

    def _classify_severity(
        self,
        score: float,
    ) -> str:
        psi_config = self.thresholds[
            "drift"
        ]["psi"]

        if score < psi_config["low"]:
            return "low"

        if score < psi_config["moderate"]:
            return "moderate"

        return "high"
