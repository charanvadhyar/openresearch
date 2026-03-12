"""
DataHealthReport — output of the EDA Agent.

This is what you get after AutoResearch has looked at your data.
Written to be understood by someone who didn't write the EDA code.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DataQualityFlag(str, Enum):
    """Things the EDA agent noticed that the researcher should know about."""
    MISSING_VALUES       = "missing_values"
    HIGH_CARDINALITY     = "high_cardinality"       # Too many unique categories
    CLASS_IMBALANCE      = "class_imbalance"         # Skewed target distribution
    HIGH_CORRELATION     = "high_correlation"         # Features that are nearly duplicates
    OUTLIERS_DETECTED    = "outliers_detected"
    LEAKAGE_RISK         = "leakage_risk"             # Feature that might "cheat" by using future info
    LOW_VARIANCE         = "low_variance"             # Feature that barely changes — probably useless
    DATE_NOT_PARSED      = "date_not_parsed"          # Looks like a date but stored as string
    SMALL_DATASET        = "small_dataset"            # Under 1000 rows — model choices change
    DUPLICATE_ROWS       = "duplicate_rows"


class Severity(str, Enum):
    INFO    = "info"     # Good to know, not urgent
    WARNING = "warning"  # Should look at this
    CRITICAL = "critical" # This might break your results


class FeatureInsight(BaseModel):
    """
    What AutoResearch learned about a single feature (column) in your dataset.

    Written in plain English so a domain expert (not necessarily an ML expert)
    can understand and validate the findings.
    """
    name: str
    dtype: str  # int, float, categorical, text, datetime, etc.
    missing_pct: float = Field(..., ge=0.0, le=100.0)
    unique_count: int

    # Statistics (None for non-numeric features)
    mean: Optional[float]   = None
    std: Optional[float]    = None
    min: Optional[float]    = None
    max: Optional[float]    = None
    skewness: Optional[float] = None

    # Flags for this specific feature
    flags: list[DataQualityFlag] = Field(default_factory=list)

    # Plain English: what should the researcher know about this feature?
    insight: str = Field(
        ...,
        description=(
            "1–2 sentences explaining what's notable about this feature. "
            "Written for the researcher, not for another ML engineer."
        ),
    )

    # Is this feature likely useful for prediction?
    usefulness_estimate: str = Field(
        ...,
        description="One of: 'likely useful', 'possibly useful', 'low signal', 'investigate further'",
    )


class CorrelationPair(BaseModel):
    """Two features that are strongly correlated with each other."""
    feature_a: str
    feature_b: str
    correlation: float  # Pearson or Cramér's V depending on feature types
    plain_english: str  # e.g. "Age and systolic blood pressure move together strongly (r=0.71)"


class DataHealthReport(BaseModel):
    """
    The full EDA output. What AutoResearch found when it looked at your data.

    Every finding is accompanied by a plain English explanation of what it means
    and what (if anything) the researcher should do about it.

    Plain English note (shown to user):
        This is AutoResearch's first look at your data. Read through the flags
        and insights — you know your data better than any algorithm does.
        If something looks wrong, trust your judgment.
    """

    # ── Overview ───────────────────────────────────────────────────────────
    row_count: int
    column_count: int
    duplicate_row_count: int
    memory_usage_mb: float

    # ── Target variable ────────────────────────────────────────────────────
    target_distribution: dict[str, float] = Field(
        ...,
        description="For classification: % of each class. For regression: key stats.",
    )
    target_plain_english: str = Field(
        ...,
        description="What does the target distribution look like? Written in plain English.",
    )

    # ── Features ───────────────────────────────────────────────────────────
    features: list[FeatureInsight]
    high_importance_features: list[str] = Field(
        default_factory=list,
        description="Features with the strongest relationship to the target (mutual information).",
    )
    low_importance_features: list[str] = Field(
        default_factory=list,
        description="Features with very weak or no relationship to the target.",
    )

    # ── Correlations ───────────────────────────────────────────────────────
    strong_correlations: list[CorrelationPair] = Field(
        default_factory=list,
        description="Pairs of features that carry similar information.",
    )

    # ── Flags ──────────────────────────────────────────────────────────────
    flags: list[tuple[DataQualityFlag, Severity, str]] = Field(
        default_factory=list,
        description="(flag, severity, plain_english_explanation) tuples.",
    )

    # ── Overall health ─────────────────────────────────────────────────────
    health_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "An overall data quality score from 0–100. "
            "Above 70: good to go. 50–70: some cleaning needed. Below 50: serious issues."
        ),
    )

    health_score_explanation: str = Field(
        ...,
        description="Why did the data get this score? What are the main issues?",
    )

    # ── Recommendations ────────────────────────────────────────────────────
    recommendations: list[str] = Field(
        ...,
        description=(
            "Concrete, ranked actions the researcher can take to improve data quality. "
            "Each one explains WHY it matters, not just what to do."
        ),
    )

    # ── Method hints ───────────────────────────────────────────────────────
    method_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Based on the data characteristics, what should the Method Formulator know? "
            "e.g. 'Dataset is small (303 rows) — avoid deep learning', "
            "'Strong outliers in 3 features — tree methods will handle this better than linear ones'"
        ),
    )

    # ── Rich EDA context (passed directly to CodeGen) ──────────────────────
    # Populated from eda_output.json — codegen uses these to generate correct
    # file paths, column names, and data loading code without guessing.
    train_path:          Optional[str] = None
    test_path:           Optional[str] = None
    label_column:        Optional[str] = None
    all_columns:         list[str]     = Field(default_factory=list)
    feature_columns:     list[str]     = Field(default_factory=list)
    numeric_columns:     list[str]     = Field(default_factory=list)
    categorical_columns: list[str]     = Field(default_factory=list)
    column_analysis:     dict          = Field(default_factory=dict)
    sample_rows:         list[dict]    = Field(default_factory=list)
