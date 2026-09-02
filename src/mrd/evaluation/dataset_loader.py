"""Dataset loading and validation for CSV and other formats."""

from pathlib import Path

import numpy as np
import pandas as pd


class DatasetLoader:
    """Load evaluation datasets while preserving raw feature types.

    Features are returned as a pandas DataFrame so categorical columns
    remain strings (or other original dtypes). Encoding and scaling are
    the uploaded model's responsibility, not the dataset loader's.
    """

    SUPPORTED_FORMATS = {
        ".csv",
        ".parquet",
    }

    @staticmethod
    def load_csv(
        path: Path | str,
        target_column: str,
        time_column: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Load a CSV file and separate features from target.

        Args:
            path: Path to the CSV file
            target_column: Name of the target column
            time_column: Optional name of the time/date column for chronological sorting

        Returns:
            Tuple of (X, y) where:
            - X: Feature frame with original dtypes preserved
            - y: Target values as a numpy array

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If target column doesn't exist or data validation fails
            RuntimeError: If file cannot be read
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read CSV file {path}: {str(exc)}"
            ) from exc

        if time_column is not None:
            if time_column not in df.columns:
                raise ValueError(
                    f"Time column '{time_column}' not found in dataset. "
                    f"Available columns: {', '.join(map(str, df.columns))}"
                )

            if time_column == target_column:
                raise ValueError(
                    "Time column cannot be the same as target column."
                )

            try:
                parsed_time = pd.to_datetime(df[time_column], errors="coerce")
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse time column '{time_column}' as datetime: {exc}"
                ) from exc

            if parsed_time.isna().any():
                raise ValueError(
                    f"Time column '{time_column}' contains invalid or unparseable timestamps."
                )

            df[time_column] = parsed_time
            df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)

        return DatasetLoader._split_features_and_target(
            df,
            target_column=target_column,
        )

    @staticmethod
    def load(
        path: Path | str,
        target_column: str,
        format: str = "csv",
        time_column: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Load a dataset in the specified format.

        Args:
            path: Path to the dataset file
            target_column: Name of the target column
            format: Format of the file ("csv", "parquet")
            time_column: Optional name of the time/date column

        Returns:
            Tuple of (X, y) with X as a DataFrame of original dtypes

        Raises:
            ValueError: If format is not supported
        """

        if format == "csv":
            return DatasetLoader.load_csv(
                path,
                target_column=target_column,
                time_column=time_column,
            )

        if format == "parquet":
            raise NotImplementedError(
                "Parquet support is not yet implemented."
            )

        raise ValueError(
            f"Unsupported dataset format: {format}. "
            f"Supported: csv, parquet"
        )

    @staticmethod
    def _split_features_and_target(
        df: pd.DataFrame,
        target_column: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )

        if df[target_column].isna().any():
            missing_count = int(df[target_column].isna().sum())
            raise ValueError(
                f"Target column has {missing_count} missing values. "
                f"Please handle missing values before evaluation."
            )

        X_df = df.drop(columns=[target_column])
        y_series = df[target_column]

        missing_feature_cols = X_df.columns[X_df.isna().any()].tolist()
        if missing_feature_cols:
            raise ValueError(
                f"Features have missing values: "
                f"{', '.join(map(str, missing_feature_cols))}. "
                f"Please handle missing values before evaluation."
            )

        if len(X_df) == 0:
            raise ValueError(
                "Dataset is empty after removing missing values."
            )

        if X_df.shape[1] == 0:
            raise ValueError(
                "Dataset has no feature columns after removing the target."
            )

        y = y_series.to_numpy()

        if len(X_df) != len(y):
            raise ValueError(
                "Mismatch between feature and target dimensions."
            )

        return X_df, y
