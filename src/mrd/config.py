from pathlib import Path
import yaml
import os


def load_thresholds(path: Path | None = None) -> dict:
    """Load regression thresholds from a YAML configuration file."""

    if path is None:
        # Try several possible locations
        candidates = [
            # 1. Explicit environment variable
            Path(os.getenv("THRESHOLDS_PATH", "")),

            # 2. Project root / config/thresholds.yaml  (local development)
            Path.cwd() / "config" / "thresholds.yaml",

            # 3. Project root / thresholds.yaml  (if you put it in root)
            Path.cwd() / "thresholds.yaml",

            # 4. Relative to this file (works after pip install if you move the file later)
            Path(__file__).resolve().parent / "thresholds.yaml",
            Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml",
            Path(__file__).resolve().parents[2] / "thresholds.yaml",
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "Threshold configuration not found. Tried:\n"
                + "\n".join(f"  - {c}" for c in candidates if c)
            )

    if not path.exists():
        raise FileNotFoundError(f"Threshold configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("Threshold configuration must contain a YAML mapping.")

    return configuration