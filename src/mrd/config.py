from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"


def load_thresholds(
    path: Path = DEFAULT_THRESHOLDS_PATH,
) -> dict:
    """Load regression thresholds from a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError(
            "Threshold configuration must contain a YAML mapping."
        )

    return configuration