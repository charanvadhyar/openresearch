"""
Dataset diagnostics schema used across planner/codegen/evaluator stages.
"""

from pydantic import BaseModel, Field


class DatasetDiagnosticsReport(BaseModel):
    rows: int
    columns: int
    target_column: str
    class_distribution: dict[str, float] = Field(default_factory=dict)
    missing_value_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    duplicate_ratio: float = 0.0
    leakage_candidates: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
