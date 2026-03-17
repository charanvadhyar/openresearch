"""
Data Prep Agent

Generates a minimal, rule-based preparation plan from EDA and diagnostics.
This is used to steer CodeGen and prevent common data interpretation errors.
"""

import logging
from typing import Optional

from autoresearch.schemas import ProblemSpec, DataHealthReport, DataPrepReport, TaskType

logger = logging.getLogger(__name__)


class DataPrepAgent:
    """
    Minimal data preparation planner based on EDA + diagnostics.
    No heavy ML libs. No data loading. Just guidance.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def prepare(
        self,
        spec: ProblemSpec,
        report: DataHealthReport,
    ) -> DataPrepReport:
        target_col = spec.target_column or report.label_column or ""
        all_cols = report.all_columns or []

        missing_cols = [
            f.name for f in report.features
            if f.missing_pct and f.missing_pct > 0.0
        ]

        high_cardinality_cols = []
        if report.row_count > 0:
            for f in report.features:
                if f.unique_count and f.unique_count / report.row_count > 0.30:
                    high_cardinality_cols.append(f.name)

        diagnostics = spec.dataset_diagnostics
        leakage_candidates = []
        if diagnostics is not None:
            leakage_candidates = list(diagnostics.leakage_candidates or [])
            # Prefer diagnostics-based high cardinality if present
            if diagnostics.high_cardinality_columns:
                high_cardinality_cols = list(diagnostics.high_cardinality_columns)
            if diagnostics.missing_value_columns:
                missing_cols = list(diagnostics.missing_value_columns)

        duplicate_ratio = 0.0
        if report.row_count > 0:
            duplicate_ratio = report.duplicate_row_count / report.row_count

        normalize_column_names = any(
            (c != c.strip()) for c in all_cols if isinstance(c, str)
        )

        stratify_split = spec.task_type == TaskType.CLASSIFICATION

        recommendations = []
        notes = []

        if normalize_column_names:
            recommendations.append(
                "Strip whitespace from column names and remap target_col if needed."
            )
        if missing_cols:
            recommendations.append(
                "Impute missing values (numeric: median, categorical: most_frequent)."
            )
        if high_cardinality_cols:
            recommendations.append(
                "High-cardinality categoricals detected; avoid full one-hot if it explodes features."
            )
        if leakage_candidates:
            recommendations.append(
                "Investigate leakage candidates and drop if they encode the target."
            )
        if duplicate_ratio > 0.05:
            recommendations.append("Drop duplicate rows before split/training.")
        if not target_col and spec.task_type != TaskType.CLUSTERING:
            notes.append("Target column is missing or unclear; confirm before training.")
        if stratify_split:
            recommendations.append("Use stratified train/val split for classification.")

        prep = DataPrepReport(
            target_column=target_col,
            normalize_column_names=normalize_column_names,
            missing_value_columns=missing_cols,
            high_cardinality_columns=high_cardinality_cols,
            leakage_candidates=leakage_candidates,
            duplicate_ratio=duplicate_ratio,
            drop_duplicate_rows=duplicate_ratio > 0.05,
            impute_numeric="median",
            impute_categorical="most_frequent",
            stratify_split=stratify_split,
            drop_columns=[],
            notes=notes,
            recommendations=recommendations,
        )

        if self.verbose:
            self._print_summary(prep)

        return prep

    def _print_summary(self, prep: DataPrepReport) -> None:
        print("\n" + "-" * 60)
        print("DATA PREP SUMMARY")
        print("-" * 60)
        print(f"Target: {prep.target_column or 'none'}")
        print(f"Missing columns: {len(prep.missing_value_columns)}")
        if prep.high_cardinality_columns:
            print(f"High-cardinality: {', '.join(prep.high_cardinality_columns[:5])}")
        if prep.leakage_candidates:
            print(f"Leakage candidates: {', '.join(prep.leakage_candidates[:5])}")
        if prep.duplicate_ratio > 0:
            print(f"Duplicate ratio: {prep.duplicate_ratio:.2%}")
        if prep.recommendations:
            print("Recommendations:")
            for r in prep.recommendations[:5]:
                print(f"  - {r}")
        print("-" * 60 + "\n")
