"""Persistence for versioned model-evaluation baselines."""

from __future__ import annotations

import json
from pathlib import Path

from mrd.schemas import ModelEvaluation


class BaselineManager:
    """Save and load model evaluations used as regression baselines."""

    def __init__(self, base_dir: str | Path = "baselines") -> None:
        self.base_dir = Path(base_dir)

    def save(self, evaluation: ModelEvaluation) -> Path:
        """Save an evaluation by version and update that model's latest baseline."""

        model_dir = self.base_dir / evaluation.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        version_path = model_dir / f"{evaluation.model_version}.json"
        latest_path = model_dir / "latest.json"
        payload = evaluation.model_dump(mode="json")

        self._write_json(version_path, payload)
        self._write_json(latest_path, payload)
        return version_path

    def load(
        self,
        model_name: str,
        version: str | None = None,
    ) -> ModelEvaluation:
        """Load a model baseline, using the latest version by default."""

        filename = "latest.json" if version is None else f"{version}.json"
        path = self.base_dir / model_name / filename
        if not path.is_file():
            raise FileNotFoundError(f"Baseline not found: {path}")

        return ModelEvaluation.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

