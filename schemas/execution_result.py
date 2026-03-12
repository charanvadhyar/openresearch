"""
ExecutionResult â€” what came back from running a method on Kaggle Kernels.
EvaluationReport â€” the Evaluator agent's verdict after comparing all methods.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# â”€â”€ ExecutionResult â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ExecutionStatus(str, Enum):
    SUCCESS  = "success"
    FAILED   = "failed"
    TIMEOUT  = "timeout"
    RETRYING = "retrying"


class ModelArtifact(BaseModel):
    """The saved model and everything needed to use it."""
    kaggle_output_path: str    # e.g. "/kaggle/working/model.pkl"
    format: str                # "pickle", "pytorch", "keras", "joblib"
    size_mb: float
    inference_script: str      # Relative path to inference script
    requirements: list[str]    # pip packages needed to run inference


class ExecutionResult(BaseModel):
    """
    What happened when we ran a method's notebook on Kaggle Kernels.

    Both successes and failures are captured here.
    Failures include the traceback and what the retry attempt changed.
    """
    method_id: str
    method_name: str

    status: ExecutionStatus
    kaggle_kernel_url: Optional[str] = None  # Link the researcher can visit

    # â”€â”€ Timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    runtime_minutes: Optional[float] = None
    gpu_used: bool = False

    # â”€â”€ Metrics (populated on success) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    primary_metric_value: Optional[float]  = None
    primary_metric_name: Optional[str]     = None
    all_metrics: dict[str, float]           = Field(default_factory=dict)

    # Training vs validation gap â€” used to detect overfitting
    train_metric: Optional[float] = None
    val_metric:   Optional[float] = None

    # â”€â”€ Model artifact â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model_artifact: Optional[ModelArtifact] = None

    # â”€â”€ Error handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    error_message: Optional[str]  = None
    error_plain_english: Optional[str] = Field(
        None,
        description="What went wrong, explained in plain English for non-experts.",
    )
    retry_count: int = 0
    retry_changes: list[str] = Field(
        default_factory=list,
        description="What the Code Generator changed on each retry attempt.",
    )

    # â”€â”€ Plain English results summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    results_plain_english: Optional[str] = Field(
        None,
        description=(
            "What do these results mean? Written for a researcher who may not "
            "immediately know if 0.847 AUC is good or bad for their problem. "
            "Includes context: 'This is considered strong for medical data â€” "
            "most published papers in this domain report 0.80â€“0.88.'"
        ),
    )


# â”€â”€ EvaluationReport â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RiskFlag(BaseModel):
    """Something the researcher should investigate before trusting the results."""
    method_id: str
    flag_type: str   # "overfitting", "data_leakage", "unstable_cv", "suspicious_score"
    severity: str    # "warning", "critical"
    explanation: str # Plain English. What is the risk and what should they do?


class MethodScore(BaseModel):
    """Multi-criteria score for a single method."""
    method_id: str
    method_name: str

    # Raw metric (from ExecutionResult)
    primary_metric: float

    # Normalized scores (0â€“1) for each criterion
    score_performance:      float  # Primary metric, normalized across all methods
    score_speed:            float  # Inverse of training time, normalized
    score_interpretability: float  # Hand-scored by evaluator agent
    score_robustness:       float  # Based on CV std â€” lower variance = higher score

    # Weighted total (weights from config.yaml)
    total_score: float

    summary: str = Field(
        ...,
        description="2â€“3 sentence plain English summary of this method's performance.",
    )


class FailedMethodSummary(BaseModel):
    """Structured diagnostics for a method that failed execution."""
    method_id: str
    method_name: str
    error_type: str
    error_plain_english: str
    retry_count: int = 0


class EvaluationReport(BaseModel):
    """
    The Evaluator agent's full comparison and recommendation.

    This is NOT a declaration of truth â€” it's a structured starting point
    for the researcher's own judgment.
    """
    method_scores: list[MethodScore]

    # The winner
    winner_method_id: str
    winner_explanation: str = Field(
        ...,
        description=(
            "Why did this method win? References specific numbers and characteristics. "
            "Acknowledges any caveats or uncertainties. "
            "e.g. 'XGBoost scored highest overall (0.847 AUC, 2.3 min training). "
            "The neural network scored higher on raw performance (0.851) but took "
            "18 minutes to train and showed signs of overfitting (train: 0.94 vs val: 0.851). "
            "Given your constraint of interpretability, XGBoost is the clear recommendation.'"
        ),
    )

    # Runner up (for the researcher to explore further)
    runner_up_method_id: Optional[str] = None
    runner_up_explanation: Optional[str] = None

    # Risk flags the researcher must read
    risk_flags: list[RiskFlag] = Field(default_factory=list)

    # Structured summary of failed methods and diagnostics
    failed_methods_summary: list[FailedMethodSummary] = Field(default_factory=list)

    # What should the researcher do next?
    next_steps: list[str] = Field(
        ...,
        description=(
            "3â€“5 concrete actions ranked by impact. "
            "Each one explains WHY it's worth doing. "
            "e.g. '1. Collect more data for the minority class â€” you have only 45 positive "
            "examples which limits model learning. Even 100 more would likely improve AUC by 3â€“5%.'"
        ),
    )

    # Suggested novel research directions (for the paper)
    research_directions: list[str] = Field(
        default_factory=list,
        description=(
            "Directions the researcher could pursue as their novel contribution. "
            "AutoResearch ran the baselines â€” what could they do that no one has done yet?"
        ),
    )

