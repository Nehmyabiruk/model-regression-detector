from pathlib import Path

from mrd.evaluation.dataset_loader import DatasetLoader
from mrd.evaluation.evaluator import ModelEvaluator
from mrd.evaluation.loader import ModelLoader
from mrd.detection.regression import RegressionDetector


BASE_DIR = Path(__file__).resolve().parent

BASELINE_MODEL_PATH = (
    BASE_DIR / "models" / "baseline.pkl"
)

CANDIDATE_MODEL_PATH = (
    BASE_DIR / "models" / "candidate.pkl"
)

DATA_PATH = (
    BASE_DIR / "data" / "evaluation.csv"
)


def load_evaluation_data() -> tuple:
    return DatasetLoader.load(
        DATA_PATH,
        target_column="target",
    )


def evaluate_model(
    model_path: Path,
    model_version: str,
    X,
    y,
):
    model = ModelLoader.load(model_path)

    evaluator = ModelEvaluator()

    return evaluator.evaluate(
        model=model,
        X=X,
        y=y,
        model_name="credit-risk",
        model_version=model_version,
        dataset_version="credit-risk-eval-v1",
    )


def main() -> None:
    X, y = load_evaluation_data()

    baseline = evaluate_model(
        model_path=BASELINE_MODEL_PATH,
        model_version="2.2.0",
        X=X,
        y=y,
    )

    candidate = evaluate_model(
        model_path=CANDIDATE_MODEL_PATH,
        model_version="2.3.0",
        X=X,
        y=y,
    )

    detector = RegressionDetector()

    report = detector.compare(
        baseline=baseline,
        candidate=candidate,
    )

    print(
        report.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()