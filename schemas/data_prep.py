"""
DataPrepReport - lightweight, rule-based data preparation guidance.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DataPrepReport(BaseModel):
    """
    Minimal, actionable data prep plan derived from EDA + diagnostics.
    This is guidance for CodeGen and researchers, not an automated transform.
    """

    target_column: str = Field(
        default="",
        description="Target column to predict. Empty for unsupervised tasks.",
    )
    normalize_column_names: bool = Field(
        default=False,
        description="Whether to strip whitespace in column names.",
    )
    missing_value_columns: list[str] = Field(
        default_factory=list,
        description="Columns with missing values that should be imputed.",
    )
    high_cardinality_columns: list[str] = Field(
        default_factory=list,
        description="Categorical columns with many unique values.",
    )
    leakage_candidates: list[str] = Field(
        default_factory=list,
        description="Columns that may leak the target.",
    )
    duplicate_ratio: float = Field(
        default=0.0,
        description="Fraction of duplicate rows.",
    )
    drop_duplicate_rows: bool = Field(
        default=False,
        description="Whether to drop duplicates before training.",
    )
    impute_numeric: str = Field(
        default="median",
        description="Imputation strategy for numeric columns.",
    )
    impute_categorical: str = Field(
        default="most_frequent",
        description="Imputation strategy for categorical columns.",
    )
    stratify_split: bool = Field(
        default=False,
        description="Use stratified train/val split for classification tasks.",
    )
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to drop if they are confirmed leakage or IDs.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Short, plain-English notes about prep risks.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable prep steps for code generation.",
    )
