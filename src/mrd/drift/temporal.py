from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


class TemporalValidationResult(BaseModel):
    """Result of temporal data validation and consistency checks."""

    is_monotonic: bool
    start_time: str | None = None
    end_time: str | None = None
    sample_count: int
    temporal_drift_supported: bool = False
    summary: str


class TemporalDriftDetector:
    """Validate temporal characteristics without inventing fake drift scores."""

    @staticmethod
    def validate_temporal_series(
        time_series: pd.Series | np.ndarray,
    ) -> TemporalValidationResult:
        """
        Validate chronological ordering and return temporal metadata.
        """

        series = pd.Series(time_series)

        if series.empty:
            raise ValueError("time_series cannot be empty.")

        try:
            dt_series = pd.to_datetime(series)
        except Exception as exc:
            raise ValueError(
                f"Time series contains invalid timestamps: {exc}"
            ) from exc

        is_mono = bool(dt_series.is_monotonic_increasing)
        start = str(dt_series.iloc[0])
        end = str(dt_series.iloc[-1])

        return TemporalValidationResult(
            is_monotonic=is_mono,
            start_time=start,
            end_time=end,
            sample_count=len(dt_series),
            temporal_drift_supported=False,
            summary=(
                "Chronological ordering validated. Advanced temporal drift diagnostics are extensible."
                if is_mono
                else "Timestamps are not strictly monotonically increasing."
            ),
        )
