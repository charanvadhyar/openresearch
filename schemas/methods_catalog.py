"""
MethodSpec & MethodsCatalog — output of the Method Formulator agent.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    LOW    = "low"    # < 5 min on CPU
    MEDIUM = "medium" # 5–30 min on CPU or < 10 min GPU
    HIGH   = "high"   # 30+ min, GPU recommended


class MethodSpec(BaseModel):
    """
    A single candidate ML method to try.

    Includes not just WHAT to run but WHY it was chosen for this specific
    problem and data — so the researcher understands the reasoning.
    """
    id: str               # e.g. "xgboost_baseline"
    name: str             # e.g. "XGBoost with feature engineering"
    algorithm_family: str # e.g. "Gradient Boosting", "Neural Network", "Linear"

    # The most important field — why THIS method for THIS problem
    why_chosen: str = Field(
        ...,
        description=(
            "Plain English explanation of why this method is appropriate. "
            "Must reference specific characteristics from the ProblemSpec and DataHealthReport. "
            "e.g. 'Chosen because your dataset has 303 rows and mixed numeric/categorical features. "
            "XGBoost handles both without needing feature scaling, and works well on small datasets "
            "unlike neural networks which typically need 100k+ rows.'"
        ),
    )

    # Honest tradeoffs
    strengths: list[str]
    weaknesses: list[str]
    complexity: ComplexityLevel

    requires_gpu: bool = False

    # Hyperparameter search space (fed to Optuna)
    hyperparam_space: dict[str, Any] = Field(
        default_factory=dict,
        description="The ranges Optuna will search. Keys are param names, values are [low, high] or lists.",
    )

    # Feature engineering to apply before this method
    feature_engineering: list[str] = Field(
        default_factory=list,
        description="Steps to apply. e.g. ['StandardScaler on numeric', 'OneHotEncoder on categorical']",
    )

    # What the researcher should watch for
    watch_out_for: str = Field(
        ...,
        description=(
            "One concrete thing to watch out for with this method on this data. "
            "e.g. 'XGBoost can overfit on small datasets — watch the gap between train and val scores.'"
        ),
    )

    # Estimated training time
    estimated_runtime_minutes: Optional[float] = None


class MethodsCatalog(BaseModel):
    """All proposed methods for this run, ranked by expected performance."""
    methods: list[MethodSpec]
    ranking_explanation: str = Field(
        ...,
        description="Why are the methods ranked in this order? Plain English.",
    )
    methods_not_tried: list[str] = Field(
        default_factory=list,
        description="Methods that were considered but excluded, and why.",
    )
