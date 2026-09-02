"""Model loading and validation for scikit-learn compatible models.

Security
--------
``joblib.load`` / pickle deserialization executes arbitrary Python bytecode
from the file being loaded. An uploaded ``.pkl`` / ``.joblib`` is therefore
equivalent to uploaded executable code.

This process-local loader is acceptable only for trusted artifacts (for
example models produced by the same organization). For untrusted uploads,
production deployments MUST:

1. Deserialize in an isolated worker (container or VM, no network egress,
   non-root, read-only mounts, memory/CPU/time limits).
2. Treat the evaluation API as an untrusted-code host: never load pickles
   in the same process that holds secrets, the database, or other tenants.
3. Prefer safer model formats when available (ONNX, skops) and reject
   pickle from unauthenticated users.
4. Cap artifact size and scan/allow-list file types before load.

Until that isolation exists, only load models you would otherwise execute.
"""

from pathlib import Path
from typing import Any
import joblib


class ModelLoader:
    """Load and validate serialized ML models safely."""

    SUPPORTED_FORMATS = {
        ".pkl",
        ".pickle",
        ".joblib",
    }

    @staticmethod
    def load(path: Path | str, model_type: str = "classification") -> Any:
        """
        Load a serialized model from disk.

        Supports joblib and pickle formats.

        Args:
            path: Path to the model file
            model_type: The ML task type ("classification", "regression", "timeseries")

        Returns:
            The loaded model object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            RuntimeError: If model cannot be loaded
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() not in ModelLoader.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported model format: {path.suffix}. "
                f"Supported: {', '.join(ModelLoader.SUPPORTED_FORMATS)}"
            )

        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {path}: {str(exc)}"
            ) from exc

        ModelLoader._validate_model(model, model_type=model_type)

        return model

    @staticmethod
    def _validate_model(model: Any, model_type: str = "classification") -> None:
        """
        Validate that the model supports required prediction methods.

        For classification:
        - Must have predict() method
        - Must have predict_proba() method for probability estimates

        For regression and timeseries:
        - Must have predict() method
        """

        if not hasattr(model, "predict"):
            raise ValueError(
                "Model must implement predict() method. "
                "The supplied object does not have this method."
            )

        if model_type == "classification" and not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model must implement predict_proba() method. "
                "The supplied object does not have this method. "
                "This is required for classification evaluation."
            )
