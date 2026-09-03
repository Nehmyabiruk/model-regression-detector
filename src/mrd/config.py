from pathlib import Path
import yaml
import os


def load_thresholds(path: Path | None = None) -> dict:
    """Load regression thresholds from a YAML configuration file."""

    if path is None:
        candidates = []

        # 1. Explicit environment variable (highest priority)
        env_path = os.getenv("THRESHOLDS_PATH")
        if env_path:
            candidates.append(Path(env_path))

        # 2. Common locations
        candidates.extend([
            Path.cwd() / "config" / "thresholds.yaml",
            Path.cwd() / "thresholds.yaml",
            Path(__file__).resolve().parent / "thresholds.yaml",
            Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml",
            Path(__file__).resolve().parents[2] / "thresholds.yaml",
        ])

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "Threshold configuration not found. Tried:\n"
                + "\n".join(f"  - {c}" for c in candidates)
            )

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Threshold configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("Threshold configuration must contain a YAML mapping.")

    return configuration