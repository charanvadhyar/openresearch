"""
ProblemSpec — the structured output of the Problem Analyst agent.

Every downstream agent reads from this. If this is wrong, everything is wrong.
That's why the Problem Analyst has a confidence score and asks for confirmation
when it's not sure.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .dataset_diagnostics import DatasetDiagnosticsReport
from .data_prep import DataPrepReport


class TaskType(str, Enum):
    """The 6 ML problem types AutoResearch supports in v1."""
    CLASSIFICATION  = "classification"
    REGRESSION      = "regression"
    NLP             = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES     = "time_series"
    CLUSTERING      = "clustering"


class EvalMetric(str, Enum):
    """Primary evaluation metrics per task type."""
    # Classification
    ACCURACY         = "accuracy"
    F1_MACRO         = "f1_macro"
    F1_WEIGHTED      = "f1_weighted"
    AUC_ROC          = "auc_roc"
    PRECISION_RECALL = "precision_recall_auc"
    LOG_LOSS         = "log_loss"
    # Regression
    RMSE  = "rmse"
    MAE   = "mae"
    R2    = "r2"
    MAPE  = "mape"
    # NLP
    BLEU     = "bleu"
    ROUGE_L  = "rouge_l"
    # CV
    MAP      = "map"
    IOU      = "iou"
    # Time series
    SMAPE    = "smape"
    # Clustering
    SILHOUETTE        = "silhouette"
    DAVIES_BOULDIN    = "davies_bouldin"
    CALINSKI_HARABASZ = "calinski_harabasz"


class Constraint(BaseModel):
    """
    A constraint the researcher cares about beyond raw performance.

    Examples:
      - "model must be explainable to non-technical doctors"
      - "inference must run in < 100ms on CPU"
      - "model size must be < 50MB"
    """
    name: str
    description: str
    # How hard is this constraint? "must" vs "nice to have"
    is_hard: bool = False

    class Config:
        # Plain English examples shown to the user
        json_schema_extra = {
            "example": {
                "name": "interpretability",
                "description": "Model decisions must be explainable to cardiologists",
                "is_hard": True,
            }
        }


class ProblemSpec(BaseModel):
    """
    The structured understanding of the research problem.

    This is the single most important schema in AutoResearch. Every agent
    downstream reads from this. The Problem Analyst produces it. The user
    can override any field if the agent got something wrong.

    Plain English note (shown to user):
        Think of this as AutoResearch's understanding of your problem.
        If anything looks wrong, you can correct it before the run continues.
    """

    # ── Core classification ────────────────────────────────────────────────
    task_type: TaskType = Field(
        ...,
        description="What kind of ML problem is this? Classification, regression, etc.",
    )
    domain: str = Field(
        ...,
        description="The subject area. e.g. 'medical imaging', 'financial forecasting', 'NLP sentiment'",
    )

    # ── What does success look like? ───────────────────────────────────────
    primary_metric: EvalMetric = Field(
        ...,
        description="The single number we optimize for. Everything else is secondary.",
    )
    secondary_metrics: list[EvalMetric] = Field(
        default_factory=list,
        description="Additional metrics to track but not optimize for.",
    )

    # ── What are we working with? ──────────────────────────────────────────
    target_column: Optional[str] = Field(
        None,
        description="The column we're trying to predict. None for unsupervised tasks.",
    )
    input_description: str = Field(
        ...,
        description="Plain English description of the input data. e.g. 'Tabular data with 50 features about hospital patients'",
    )
    estimated_row_count: Optional[int] = Field(
        None,
        description="Approximate number of rows. Affects method selection — deep learning needs 100k+.",
    )
    has_class_imbalance: Optional[bool] = Field(
        None,
        description="Are the classes very unequal? e.g. 99% negative, 1% positive. Affects metric and method choices.",
    )
    dataset_diagnostics: Optional[DatasetDiagnosticsReport] = Field(
        None,
        description="Pre-planning diagnostics for dataset quality and risks.",
    )
    data_prep: Optional[DataPrepReport] = Field(
        None,
        description="Lightweight data preparation guidance derived from EDA.",
    )

    # ── Constraints ────────────────────────────────────────────────────────
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Non-performance requirements: interpretability, speed, model size, etc.",
    )

    # ── Confidence ─────────────────────────────────────────────────────────
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident is AutoResearch in this classification? "
            "Below 0.80, it will ask you to confirm before continuing."
        ),
    )
    confidence_explanation: str = Field(
        ...,
        description="Plain English explanation of why the confidence is what it is.",
    )

    # ── Use GPU? ───────────────────────────────────────────────────────────
    requires_gpu: bool = Field(
        False,
        description=(
            "Does this problem benefit from GPU? "
            "True for deep learning (CV, NLP transformers, LSTM). "
            "False for classical ML — saves your Kaggle GPU quota."
        ),
    )

    # ── Plain English summary ──────────────────────────────────────────────
    plain_english_summary: str = Field(
        ...,
        description=(
            "A 2–3 sentence explanation of the problem in plain English. "
            "No jargon. Written as if explaining to a smart friend who doesn't know ML."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "classification",
                "domain": "medical diagnosis",
                "primary_metric": "auc_roc",
                "secondary_metrics": ["f1_weighted", "precision_recall_auc"],
                "target_column": "diagnosis",
                "input_description": "Tabular data with 14 features from cardiac tests — age, cholesterol, blood pressure, etc.",
                "estimated_row_count": 303,
                "has_class_imbalance": False,
                "constraints": [
                    {
                        "name": "interpretability",
                        "description": "Doctors need to understand why the model made a decision",
                        "is_hard": True,
                    }
                ],
                "confidence": 0.95,
                "confidence_explanation": "The problem statement clearly describes a binary classification task with a named target column and tabular medical data.",
                "requires_gpu": False,
                "plain_english_summary": (
                    "You want to predict whether a patient has heart disease based on 14 medical measurements. "
                    "This is a classification problem — the model outputs yes or no. "
                    "Because doctors will act on these predictions, we'll prioritize methods that explain their reasoning."
                ),
            }
        }
