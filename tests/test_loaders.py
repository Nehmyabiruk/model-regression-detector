"""Tests for model and dataset loading."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
import joblib

from mrd.evaluation.loader import ModelLoader
from mrd.evaluation.dataset_loader import DatasetLoader


class TestModelLoader:
    """Test model loading and validation."""

    @pytest.fixture
    def trained_model(self) -> LogisticRegression:
        """Create a simple trained model."""
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression()
        model.fit(X, y)
        return model

    def test_load_joblib_model(self, trained_model):
        """Test loading a joblib-serialized model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.joblib"
            joblib.dump(trained_model, model_path)

            loaded_model = ModelLoader.load(model_path)

            assert loaded_model is not None
            assert hasattr(loaded_model, "predict")
            assert hasattr(loaded_model, "predict_proba")

    def test_load_pickle_model(self, trained_model):
        """Test loading a pickle-serialized model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            joblib.dump(trained_model, model_path)

            loaded_model = ModelLoader.load(model_path)

            assert loaded_model is not None
            assert hasattr(loaded_model, "predict")

    def test_file_not_found(self):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ModelLoader.load(Path("/nonexistent/path/model.joblib"))

    def test_unsupported_format(self):
        """Test error for unsupported file format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.h5"
            model_path.touch()

            with pytest.raises(ValueError, match="Unsupported model format"):
                ModelLoader.load(model_path)

    def test_invalid_model_file(self):
        """Test error for corrupted model file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.joblib"
            model_path.write_text("invalid data")

            with pytest.raises(RuntimeError, match="Failed to load model"):
                ModelLoader.load(model_path)

    def test_model_missing_predict(self):
        """Test validation when model lacks predict method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.joblib"
            # Create an object without predict method
            invalid_obj = {"not": "a model"}
            joblib.dump(invalid_obj, model_path)

            with pytest.raises(ValueError, match="must implement predict"):
                ModelLoader.load(model_path)


class TestDatasetLoader:
    """Test dataset loading and validation."""

    def test_load_csv_with_target(self):
        """Test loading a valid CSV file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            
            # Create a simple CSV
            csv_path.write_text("feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0\n")

            X, y = DatasetLoader.load_csv(csv_path, target_column="target")

            assert isinstance(X, pd.DataFrame)
            assert X.shape == (3, 2)
            assert list(X.columns) == ["feature1", "feature2"]
            assert len(y) == 3
            assert np.array_equal(y, np.array([0, 1, 0]))

    def test_missing_target_column(self):
        """Test error when target column doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text("feature1,feature2\n1,2\n3,4\n")

            with pytest.raises(ValueError, match="Target column 'target' not found"):
                DatasetLoader.load_csv(csv_path, target_column="target")

    def test_missing_values_in_target(self):
        """Test error when target has missing values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text("feature1,target\n1,0\n3,\n5,1\n")

            with pytest.raises(ValueError, match="missing values"):
                DatasetLoader.load_csv(csv_path, target_column="target")

    def test_missing_values_in_features(self):
        """Test error when features have missing values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text("feature1,target\n,0\n3,1\n5,0\n")

            with pytest.raises(ValueError, match="missing values"):
                DatasetLoader.load_csv(csv_path, target_column="target")

    def test_empty_dataset(self):
        """Test error with empty dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text("feature1,target\n")

            with pytest.raises(ValueError, match="Dataset is empty"):
                DatasetLoader.load_csv(csv_path, target_column="target")

    def test_file_not_found(self):
        """Test error when CSV file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_csv(Path("/nonexistent/data.csv"), target_column="target")

    def test_preserves_categorical_feature_types(self):
        """Categorical strings must not be coerced to float at load time."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text(
                "gender,age,target\n"
                "female,21,0\n"
                "male,30,1\n"
            )

            X, y = DatasetLoader.load_csv(csv_path, target_column="target")

            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["gender", "age"]
            assert X["gender"].tolist() == ["female", "male"]
            assert pd.api.types.is_numeric_dtype(X["age"])
            assert not pd.api.types.is_numeric_dtype(X["gender"])
            assert np.array_equal(y, np.array([0, 1]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
