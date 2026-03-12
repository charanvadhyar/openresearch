"""
Dataset diagnostics used before method planning.

Lightweight, pandas+numpy only, intended to run quickly on a DataFrame already in memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoresearch.schemas import DataHealthReport, DatasetDiagnosticsReport


class DatasetDiagnostics:
    """Detect common data issues and emit actionable recommendations."""

    @staticmethod
    def analyze_dataset(df: pd.DataFrame, target_column: str) -> DatasetDiagnosticsReport:
        rows = int(len(df))
        columns = int(df.shape[1]) if rows > 0 else int(len(df.columns))
        risk_flags: list[str] = []
        recommendations: list[str] = []

        class_distribution: dict[str, float] = {}
        missing_value_columns: list[str] = []
        high_cardinality_columns: list[str] = []
        leakage_candidates: list[str] = []

        duplicate_ratio = float(df.duplicated().mean()) if rows > 0 else 0.0

        if target_column in df.columns and rows > 0:
            target = df[target_column]
            dist = target.value_counts(dropna=False, normalize=True)
            class_distribution = {str(k): float(v * 100.0) for k, v in dist.items()}

            if pd.api.types.is_numeric_dtype(target):
                for col in df.columns:
                    if col == target_column:
                        continue
                    series = df[col]
                    if not pd.api.types.is_numeric_dtype(series):
                        continue
                    pair = pd.concat([series, target], axis=1).dropna()
                    if len(pair) < 5:
                        continue
                    corr = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
                    if np.isfinite(corr) and corr > 0.95:
                        leakage_candidates.append(col)
                    # Suspicious perfect separability for numeric features
                    if col not in leakage_candidates and corr >= 0.9999:
                        leakage_candidates.append(col)
            else:
                # Categorical perfect mapping can also indicate leakage.
                for col in df.columns:
                    if col == target_column:
                        continue
                    series = df[col]
                    if not (series.dtype == "object" or str(series.dtype).startswith("category")):
                        continue
                    tmp = pd.DataFrame({"x": series, "y": target}).dropna()
                    if tmp.empty:
                        continue
                    map_counts = tmp.groupby("x")["y"].nunique(dropna=False)
                    if not map_counts.empty and int(map_counts.max()) == 1:
                        # If mapping is near 1-to-1 and high cardinality, treat as suspicious.
                        if tmp["x"].nunique(dropna=False) >= max(2, int(0.3 * rows)):
                            leakage_candidates.append(col)

            if len(class_distribution) > 1:
                max_class_pct = max(class_distribution.values())
                if max_class_pct > 70.0:
                    risk_flags.append("class_imbalance")
                    recommendations.append(
                        f"Dataset is highly imbalanced ({max_class_pct:.0f}% in one class). "
                        "Consider stratified splits."
                    )

        if rows > 0:
            missing_pct = (df.isna().mean() * 100.0).to_dict()
            missing_value_columns = [
                col for col, pct in missing_pct.items()
                if float(pct) > 20.0
            ]
            if missing_value_columns:
                risk_flags.append("missing_values")
                recommendations.append(
                    "Columns with high missingness detected. Use imputation and/or drop low-value columns."
                )

            high_cardinality_columns = [
                col for col in df.columns
                if col != target_column and (df[col].nunique(dropna=False) / max(1, rows)) > 0.30
            ]
            if high_cardinality_columns:
                risk_flags.append("high_cardinality")
                recommendations.append(
                    "High-cardinality categorical columns detected. Consider target encoding, hashing, or dropping IDs."
                )

            if duplicate_ratio > 0.05:
                risk_flags.append("duplicate_rows")
                recommendations.append(
                    f"Duplicate rows are high ({duplicate_ratio * 100:.1f}%). Consider deduplication."
                )

            if rows < 500:
                risk_flags.append("small_dataset")
                recommendations.append(
                    f"Dataset is small ({rows} rows). Prefer simpler classical models and stronger cross-validation."
                )
            elif rows > 100_000:
                risk_flags.append("large_dataset")
                recommendations.append(
                    f"Dataset is large ({rows:,} rows). Consider sampling and memory-aware training."
                )

        if leakage_candidates:
            risk_flags.append("label_leakage")
            for col in leakage_candidates[:3]:
                recommendations.append(
                    f"Column '{col}' strongly correlates with target - possible leakage."
                )

        # Perfect separability check: any single feature uniquely determines target.
        if target_column in df.columns and rows > 0:
            target = df[target_column]
            for col in df.columns:
                if col == target_column:
                    continue
                tmp = pd.DataFrame({"x": df[col], "y": target}).dropna()
                if tmp.empty:
                    continue
                grouped = tmp.groupby("x")["y"].nunique(dropna=False)
                if not grouped.empty and int(grouped.max()) == 1 and tmp["x"].nunique(dropna=False) > 1:
                    if col not in leakage_candidates:
                        leakage_candidates.append(col)
                    if "suspicious_separability" not in risk_flags:
                        risk_flags.append("suspicious_separability")
                    recommendations.append(
                        f"Feature '{col}' may perfectly separate the target. Verify for leakage."
                    )
                    break

        return DatasetDiagnosticsReport(
            rows=rows,
            columns=columns,
            target_column=target_column,
            class_distribution=class_distribution,
            missing_value_columns=sorted(set(missing_value_columns)),
            high_cardinality_columns=sorted(set(high_cardinality_columns)),
            duplicate_ratio=duplicate_ratio,
            leakage_candidates=sorted(set(leakage_candidates)),
            risk_flags=sorted(set(risk_flags)),
            recommendations=recommendations,
        )

    @staticmethod
    def enrich_from_data_health(
        diagnostics: DatasetDiagnosticsReport,
        report: DataHealthReport,
        target_column: str,
    ) -> DatasetDiagnosticsReport:
        """
        Override preview-derived diagnostics using full EDA report statistics.
        Keeps run-time light while improving accuracy.
        """
        rows = int(report.row_count)
        diagnostics.rows = rows
        diagnostics.columns = int(report.column_count)
        diagnostics.target_column = target_column
        diagnostics.class_distribution = dict(report.target_distribution or {})
        diagnostics.duplicate_ratio = (
            float(report.duplicate_row_count) / max(1, int(report.row_count))
        )

        missing_cols = [
            f.name for f in report.features
            if float(f.missing_pct) > 20.0
        ]
        high_card_cols = [
            f.name for f in report.features
            if (f.unique_count / max(1, rows)) > 0.30
        ]
        diagnostics.missing_value_columns = sorted(set(missing_cols))
        diagnostics.high_cardinality_columns = sorted(set(high_card_cols))

        leakage_cols = set(diagnostics.leakage_candidates)
        for corr in report.strong_correlations:
            if abs(float(corr.correlation)) > 0.95 and target_column in {corr.feature_a, corr.feature_b}:
                leakage_cols.add(corr.feature_a if corr.feature_b == target_column else corr.feature_b)
        diagnostics.leakage_candidates = sorted(leakage_cols)

        risk_flags = set(diagnostics.risk_flags)
        recs = list(diagnostics.recommendations)

        if len(diagnostics.class_distribution) > 1:
            max_class = max(float(v) for v in diagnostics.class_distribution.values())
            if max_class > 70.0:
                risk_flags.add("class_imbalance")
                recs.append(
                    f"Dataset is highly imbalanced ({max_class:.0f}% in one class). Consider stratified splits."
                )

        if diagnostics.missing_value_columns:
            risk_flags.add("missing_values")
            for col in diagnostics.missing_value_columns[:3]:
                recs.append(f"Use imputation for {col}.")
        if diagnostics.high_cardinality_columns:
            risk_flags.add("high_cardinality")
        if diagnostics.duplicate_ratio > 0.05:
            risk_flags.add("duplicate_rows")
        if diagnostics.leakage_candidates:
            risk_flags.add("label_leakage")
        if rows < 500:
            risk_flags.add("small_dataset")
        elif rows > 100_000:
            risk_flags.add("large_dataset")

        diagnostics.risk_flags = sorted(risk_flags)
        # Preserve order while removing duplicates.
        seen = set()
        dedup_recs = []
        for rec in recs:
            if rec not in seen:
                dedup_recs.append(rec)
                seen.add(rec)
        diagnostics.recommendations = dedup_recs
        return diagnostics

    @staticmethod
    def print_summary(diagnostics: DatasetDiagnosticsReport) -> None:
        print("\nDATASET DIAGNOSTICS")
        print(f"Rows: {diagnostics.rows}")
        print(f"Columns: {diagnostics.columns}")
        if diagnostics.class_distribution:
            rounded = {
                k: f"{float(v):.0f}%"
                for k, v in diagnostics.class_distribution.items()
            }
            print(f"Class distribution: {rounded}")
        if diagnostics.risk_flags:
            print("\nWarnings:")
            for warning in diagnostics.risk_flags:
                if warning == "missing_values" and diagnostics.missing_value_columns:
                    for col in diagnostics.missing_value_columns[:3]:
                        print(f"- Column '{col}' has >20% missing values")
                elif warning == "label_leakage" and diagnostics.leakage_candidates:
                    for col in diagnostics.leakage_candidates[:3]:
                        print(f"- Possible leakage: '{col}'")
                elif warning == "duplicate_rows":
                    print(f"- Duplicate rows ratio is {diagnostics.duplicate_ratio * 100:.1f}%")
                elif warning == "class_imbalance":
                    print("- Target appears class-imbalanced")
                elif warning == "high_cardinality" and diagnostics.high_cardinality_columns:
                    for col in diagnostics.high_cardinality_columns[:3]:
                        print(f"- High-cardinality feature: '{col}'")
                elif warning == "small_dataset":
                    print("- Very small dataset detected")
                elif warning == "large_dataset":
                    print("- Large dataset detected")
                elif warning == "suspicious_separability":
                    print("- Suspiciously perfect separability detected")
        if diagnostics.recommendations:
            print("\nRecommendations:")
            for rec in diagnostics.recommendations[:5]:
                print(f"- {rec}")
