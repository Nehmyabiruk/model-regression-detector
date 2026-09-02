import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

logger = logging.getLogger(__name__)

from mrd.ai.agent.agent import RegressionAgent
from mrd.ai.agent.tools import InvestigationTools
from mrd.ai.investigator import AIInvestigator
from mrd.ai.memory.store import IncidentMemory
from mrd.ai.rag.embeddings import EmbeddingService
from mrd.ai.rag.retriever import RAGRetriever
from mrd.ai.rag.store import VectorStore
from mrd.ai.recommendations import RecommendationEngine
from mrd.api.schemas import (
    InvestigationRequest,
    InvestigationResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from mrd.evaluation.evaluator import ModelEvaluator
from mrd.evaluation.regression import RegressionModelEvaluator
from mrd.evaluation.timeseries import TimeSeriesModelEvaluator
from mrd.evaluation.loader import ModelLoader
from mrd.evaluation.dataset_loader import DatasetLoader
from mrd.evaluation.model_input import non_numeric_columns
from mrd.detection.regression import RegressionDetector
from mrd.drift.detector import DriftDetector
from mrd.reports.report import ReportGenerator


app = FastAPI(
    title="Model Regression Detector AI",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # Alternative dev port
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = os.getenv(
    "AI_MODEL",
    "meta-llama/llama-3-8b-instruct:free",
)


DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured."
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


memory = IncidentMemory()


embedding_service = EmbeddingService()


vector_store = VectorStore(
    DATABASE_URL
)


rag = RAGRetriever(
    store=vector_store,
    embeddings=embedding_service,
)


tools = InvestigationTools(
    memory=memory,
    rag=rag,
)


investigator = AIInvestigator(
    model=MODEL
)


agent = RegressionAgent(
    client=client,
    model=MODEL,
    tools=tools,
)


recommendation_engine = RecommendationEngine(
    client=client,
    model=MODEL,
)

# ML Evaluation Components
model_evaluator = ModelEvaluator()
regression_evaluator = RegressionModelEvaluator()
timeseries_evaluator = TimeSeriesModelEvaluator()
regression_detector = RegressionDetector()
drift_detector = DriftDetector()
report_generator = ReportGenerator()

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _safe_upload_name(upload: UploadFile, default: str) -> str:
    """Keep only the basename so uploaded names cannot escape tmp_dir."""

    name = Path(upload.filename or default).name
    if not name or name in {".", ".."}:
        return default
    return name


def _enforce_upload_size(content: bytes, label: str) -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{label} exceeds the maximum upload size of "
            f"{MAX_UPLOAD_BYTES} bytes."
        )


@app.get("/health")
def health() -> dict:

    return {
        "status": "ok",
        "service": "mrd-ai",
    }


@app.post(
    "/evaluations/run",
    response_model=EvaluationResponse,
)
async def run_evaluation(
    model_name: str = Form(...),
    baseline_version: str = Form(...),
    candidate_version: str = Form(...),
    dataset_version: str = Form(...),
    target_column: str = Form(...),
    model_type: str = Form("classification"),
    time_column: str | None = Form(None),
    baseline_model: UploadFile = File(...),
    candidate_model: UploadFile = File(...),
    evaluation_dataset: UploadFile = File(...),
):
    """
    Run a complete model evaluation workflow.

    This endpoint:
    1. Accepts uploaded model files (baseline and candidate)
    2. Accepts an evaluation dataset
    3. Loads both models and validates them according to task type
    4. Loads the dataset and validates the target (and time column if applicable)
    5. Evaluates both models on the dataset
    6. Compares performance and detects regression
    7. Detects feature drift
    8. Generates a comprehensive RegressionReport

    Args:
        model_name: Name of the model (e.g., "credit-risk")
        baseline_version: Baseline model version (e.g., "2.2.0")
        candidate_version: Candidate model version (e.g., "2.3.0")
        dataset_version: Dataset version identifier
        target_column: Name of the target column in the dataset
        model_type: Type of model ("classification", "regression", "timeseries")
        time_column: Optional name of the time column (required for timeseries)
        baseline_model: Uploaded baseline model file (joblib/pickle)
        candidate_model: Uploaded candidate model file (joblib/pickle)
        evaluation_dataset: Uploaded evaluation dataset (CSV)

    Returns:
        EvaluationResponse with the RegressionReport

    Raises:
        HTTPException: If validation fails or evaluation cannot be completed
    """

    try:
        valid_types = {"classification", "regression", "timeseries"}
        if model_type not in valid_types:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Supported types: {', '.join(sorted(valid_types))}."
            )

        if model_type == "timeseries" and not time_column:
            raise ValueError(
                "time_column is required for time-series evaluation."
            )

        logger.info(
            "Starting evaluation: %s (baseline: %s, candidate: %s, type: %s)",
            model_name,
            baseline_version,
            candidate_version,
            model_type,
        )

        # Use temporary directory for uploaded files
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save uploaded files
            baseline_path = tmp_path / _safe_upload_name(
                baseline_model,
                "baseline.pkl",
            )
            candidate_path = tmp_path / _safe_upload_name(
                candidate_model,
                "candidate.pkl",
            )
            dataset_path = tmp_path / _safe_upload_name(
                evaluation_dataset,
                "evaluation.csv",
            )

            logger.info("Saving uploaded files to temporary directory")
            baseline_content = await baseline_model.read()
            _enforce_upload_size(baseline_content, "Baseline model")
            baseline_path.write_bytes(baseline_content)

            candidate_content = await candidate_model.read()
            _enforce_upload_size(candidate_content, "Candidate model")
            candidate_path.write_bytes(candidate_content)

            dataset_content = await evaluation_dataset.read()
            _enforce_upload_size(dataset_content, "Evaluation dataset")
            dataset_path.write_bytes(dataset_content)

            # Pickle/joblib deserialization executes arbitrary code.
            # Production must load these artifacts in an isolated worker.
            logger.info("Loading baseline model from %s", baseline_path)
            baseline_model_obj = ModelLoader.load(
                baseline_path,
                model_type=model_type,
            )

            logger.info("Loading candidate model from %s", candidate_path)
            candidate_model_obj = ModelLoader.load(
                candidate_path,
                model_type=model_type,
            )

            logger.info(
                "Loading dataset from %s (target: %s, time_column: %s)",
                dataset_path,
                target_column,
                time_column,
            )
            X, y = DatasetLoader.load(
                path=dataset_path,
                target_column=target_column,
                time_column=time_column if model_type == "timeseries" else None,
            )

            logger.info(
                "Dataset loaded: %d samples, %d features (%d non-numeric)",
                len(X),
                X.shape[1],
                len(non_numeric_columns(X)),
            )

            # Dispatch evaluator based on model_type
            if model_type == "classification":
                active_evaluator = model_evaluator
            elif model_type == "regression":
                active_evaluator = regression_evaluator
            elif model_type == "timeseries":
                active_evaluator = timeseries_evaluator

            logger.info("Evaluating baseline model with %s", type(active_evaluator).__name__)
            baseline_eval = active_evaluator.evaluate(
                model=baseline_model_obj,
                X=X,
                y=y,
                model_name=model_name,
                model_version=baseline_version,
                dataset_version=dataset_version,
            )

            logger.info("Baseline evaluation: %d metrics", len(baseline_eval.metrics))

            logger.info("Evaluating candidate model with %s", type(active_evaluator).__name__)
            candidate_eval = active_evaluator.evaluate(
                model=candidate_model_obj,
                X=X,
                y=y,
                model_name=model_name,
                model_version=candidate_version,
                dataset_version=dataset_version,
            )

            logger.info("Candidate evaluation: %d metrics", len(candidate_eval.metrics))

            logger.info("Comparing models and detecting regression")
            report = regression_detector.compare(
                baseline=baseline_eval,
                candidate=candidate_eval,
                task_type=model_type,
            )

            logger.info(
                "Regression detection complete: status=%s",
                report.status,
            )

            # Single evaluation dataset: drift is intra-dataset (near zero)
            # until a dedicated baseline reference sample is supplied.
            logger.info("Detecting feature drift")
            drift_results = drift_detector.detect_dataframe(
                baseline=X,
                current=X,
            )

            logger.info("Drift detection complete: %d features analyzed", len(drift_results))

            # Update report with drift results
            report.drift = drift_results

            logger.info(
                "Evaluation complete: %s (regression: %s, drift features: %d)",
                model_name,
                report.status,
                len(drift_results),
            )

            return EvaluationResponse(report=report)

    except ValueError as exc:
        logger.error("Validation error during evaluation: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logger.error("File error during evaluation: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.error("Evaluation runtime error: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "Unexpected error during evaluation: %s",
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}",
        ) from exc


@app.post(
    "/ai/investigate",
    response_model=InvestigationResponse,
)
def investigate(
    request: InvestigationRequest,
):

    try:
        logger.info("Starting investigation with evidence: %s", request.report)

        evidence = request.report

        logger.info("Calling investigator.investigate()")
        investigation = (
            investigator.investigate(
                evidence
            )
        )
        logger.info("Investigation completed: %s", investigation)

        logger.info("Calling investigator.analyze_root_cause()")
        root_cause = (
            investigator.analyze_root_cause(
                evidence
            )
        )
        logger.info("Root cause analysis completed: %s", root_cause)

        logger.info("Calling recommendation_engine.generate()")
        recommendations = (
            recommendation_engine.generate(
                evidence=evidence,
                root_cause=root_cause.model_dump(),
            )
        )
        logger.info("Recommendations generated: %s", recommendations)

        logger.info("Calling agent.investigate()")
        agent_analysis = agent.investigate(
            evidence
        )
        logger.info("Agent analysis completed: %s", agent_analysis)

        response = InvestigationResponse(
            investigation=(
                investigation.model_dump()
            ),
            root_cause=(
                root_cause.model_dump()
            ),
            recommendations=(
                recommendations.model_dump()
            ),
            agent_analysis=agent_analysis,
        )
        
        logger.info("Investigation successful")
        return response

    except Exception as exc:
        logger.error("Investigation failed with error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc