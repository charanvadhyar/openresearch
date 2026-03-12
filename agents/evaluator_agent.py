"""
Evaluator Agent

Compares all ExecutionResults across multiple dimensions, applies a
weighted scoring matrix, and declares a winner â€” with honest caveats.

Key design principle:
    This is NOT a black-box verdict. Every score is shown, every
    weight is explained, and every risk flag is surfaced. The researcher
    makes the final call. AutoResearch makes a recommendation.

Mentor voice:
    "XGBoost scored highest overall. But the neural network had a higher
     raw AUC (0.851 vs 0.847). Given your interpretability constraint,
     XGBoost is the clearer choice â€” doctors can understand SHAP plots,
     not neural network weights."

Scoring dimensions (weights from config.yaml):
    - Performance:      How good is the primary metric?         (default 40%)
    - Speed:            How fast did it train?                  (default 20%)
    - Interpretability: Can you explain the predictions?        (default 20%)
    - Robustness:       Is the CV score stable (low std)?       (default 20%)
"""

import json
import logging
import re
from typing import Any, Optional

from autoresearch.agents.api_utils import LLMClient
from autoresearch.memory.experiment_memory import ExperimentMemory

from autoresearch.schemas import (
    ProblemSpec,
    ExecutionResult,
    ExecutionStatus,
    EvaluationReport,
    FailedMethodSummary,
    MethodScore,
    RiskFlag,
)

logger = logging.getLogger(__name__)


# â”€â”€ Interpretability scores per algorithm family â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# These are reasonable defaults. Researcher can override in config.
# Scale: 0.0 (black box) â†’ 1.0 (fully interpretable)

INTERPRETABILITY_SCORES = {
    "Linear":                        1.0,   # Coefficients directly readable
    "Gradient Boosting":             0.6,   # SHAP available, not inherent
    "Ensemble (Bagging)":            0.65,  # Feature importances, SHAP
    "CNN (Transfer Learning)":       0.2,   # Grad-CAM possible but hard
    "Transformer":                   0.15,  # Attention maps, not reliable
    "Recurrent Neural Network":      0.1,   # Very hard to explain
    "Neural Network (Attention)":    0.35,  # TabNet has built-in attention
    "Additive Model":                0.9,   # Prophet components are explicit
    "Classical NLP":                 0.85,  # TF-IDF weights are readable
    "Centroid-based":                0.7,   # Cluster centers are interpretable
    "Density-based":                 0.5,   # Less interpretable than k-means
    "Gradient Boosting + Feature Engineering": 0.55,
}


def get_interpretability_score(algorithm_family: str) -> float:
    """Look up interpretability score, default to 0.5 if unknown."""
    return INTERPRETABILITY_SCORES.get(algorithm_family, 0.5)


def normalize_scores(values: list[float], higher_is_better: bool = True) -> list[float]:
    """
    Normalize a list of values to [0, 1] range.
    handles edge case where all values are equal.
    """
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0] * len(values)  # All equal â€” full score to everyone
    normalized = [(v - min_v) / (max_v - min_v) for v in values]
    if not higher_is_better:
        normalized = [1.0 - n for n in normalized]
    return normalized


def _extract_error_type(result: ExecutionResult) -> str:
    """Infer canonical error type from traceback/message/plain-English fields."""
    text_parts = [
        result.error_message or "",
        result.error_plain_english or "",
        " ".join(result.retry_changes) if result.retry_changes else "",
    ]
    text = "\n".join(text_parts)
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*Error)\b", text)
    if match:
        return match.group(1)

    lower = text.lower()
    if "shape" in lower and "valueerror" in lower:
        return "ValueError"
    if "shape mismatch" in lower:
        return "ValueError"
    if "missing column" in lower or "column name wasn't found" in lower:
        return "KeyError"
    if "module" in lower and "not found" in lower:
        return "ImportError"
    if "memory" in lower and ("out of" in lower or "ran out" in lower):
        return "MemoryError"
    return "UnknownError"


def _error_plain_english_for_type(error_type: str, fallback: Optional[str]) -> str:
    """Map technical exceptions to plain-English diagnostics."""
    mapping = {
        "KeyError": "A required column is missing or misspelled in the dataset.",
        "ValueError": "Input shapes are incompatible, likely from preprocessing mismatch.",
        "ImportError": "A required Python package is missing in the runtime environment.",
        "ModuleNotFoundError": "A required Python package is missing in the runtime environment.",
        "MemoryError": "The model or data is too large for available memory.",
    }
    if error_type in mapping:
        return mapping[error_type]
    return fallback or "The method failed due to an execution error."


def analyze_failures(results: list[ExecutionResult]) -> dict[str, Any]:
    """
    Analyze failed execution results for repeated error patterns and systemic issues.
    """
    failure_results = [r for r in results if r.status == ExecutionStatus.FAILED]
    error_type_counts: dict[str, int] = {}
    failed_methods_summary: list[dict[str, Any]] = []

    for result in failure_results:
        error_type = _extract_error_type(result)
        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        failed_methods_summary.append({
            "method_id": result.method_id,
            "method_name": result.method_name,
            "error_type": error_type,
            "error_plain_english": _error_plain_english_for_type(
                error_type,
                result.error_plain_english,
            ),
            "retry_count": result.retry_count,
        })

    total_failures = len(failure_results)
    repeated_error_types = {
        err: count for err, count in error_type_counts.items() if count > 1
    }
    systemic_issues: list[str] = []

    if total_failures > 0:
        if error_type_counts.get("KeyError", 0) == total_failures:
            systemic_issues.append(
                "All failed methods hit KeyError, likely a shared missing/renamed dataset column."
            )
        if error_type_counts.get("ImportError", 0) + error_type_counts.get("ModuleNotFoundError", 0) == total_failures:
            systemic_issues.append(
                "All failed methods are missing dependencies in the runtime environment."
            )
        if error_type_counts.get("MemoryError", 0) >= max(2, total_failures // 2):
            systemic_issues.append(
                "Many methods failed with memory pressure, suggesting resource limits or oversized models."
            )
        if error_type_counts.get("ValueError", 0) >= max(2, total_failures // 2):
            systemic_issues.append(
                "Many methods failed with shape mismatch, suggesting a shared preprocessing bug."
            )

    return {
        "failed_methods_summary": failed_methods_summary,
        "error_type_counts": error_type_counts,
        "repeated_error_types": repeated_error_types,
        "systemic_issues": systemic_issues,
        "total_failures": total_failures,
    }


# â”€â”€ System prompt for winner explanation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

EVALUATOR_SYSTEM_PROMPT = """\
You are the Evaluator for AutoResearch â€” a research tool for people who may
not have deep ML expertise.

You receive a scoring table comparing ML methods and must:
1. Write a clear, honest explanation of why the top method won
2. Identify any risks or caveats the researcher must know
3. Suggest concrete next steps
4. Suggest novel research directions they could pursue

## Rules

- Be specific: reference actual numbers, not vague statements
- Be honest: if the winner barely beat the runner-up, say so
- Surface risks prominently â€” overfitting, data leakage, small sample
- Write for a domain expert, not an ML engineer
- next_steps must be actionable with reasons, not vague advice

## Tone examples

BAD winner_explanation:
  "XGBoost performed best across all metrics."

GOOD winner_explanation:
  "XGBoost scored highest overall (0.847 AUC, ~3 min training time).
   The neural network had a slightly higher raw AUC (0.851) but showed
   signs of overfitting â€” its training score (0.94) was much higher than
   its validation score (0.851), suggesting it memorized the training data
   rather than learning the underlying patterns. Given your interpretability
   constraint (doctors need to understand the decisions), XGBoost with SHAP
   explanations is the clearer recommendation."

BAD next_steps:
  "Collect more data and tune hyperparameters."

GOOD next_steps:
  "1. Add SHAP explanations to the XGBoost model â€” this shows doctors exactly
      which features drove each prediction. The code is in the notebook.
   2. Your dataset has only 303 rows. Even 200 more examples of the minority
      class (positive diagnosis) would likely improve AUC by 3â€“5%.
   3. The 'thal' feature had the highest importance â€” interview cardiologists
      about whether this aligns with clinical intuition. If not, investigate."

## Output format

Respond with valid JSON only. No extra text.
"""


class EvaluatorAgent:
    """
    The Evaluator Agent.

    Takes all ExecutionResults, scores them on multiple dimensions,
    and produces an EvaluationReport with an honest, plain English
    recommendation that the researcher owns.

    Usage:
        agent = EvaluatorAgent(api_key="...", weights={...})
        report = agent.evaluate(spec, results)
    """

    DEFAULT_WEIGHTS = {
        "performance":      0.40,
        "speed":            0.20,
        "interpretability": 0.20,
        "robustness":       0.20,
    }

    def __init__(self, llm: LLMClient, weights: Optional[dict] = None, verbose: bool = True):
        self.llm     = llm
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.verbose = verbose
        self.memory  = ExperimentMemory()
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Evaluation weights sum to {total:.2f}, not 1.0. "
                f"Normalizing automatically."
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate(
        self,
        spec: ProblemSpec,
        results: list[ExecutionResult],
    ) -> EvaluationReport:
        """
        Compare all methods and produce a recommendation.

        Args:
            spec:    ProblemSpec (used for context in explanations)
            results: All ExecutionResults (success and failure)

        Returns:
            EvaluationReport with winner, scores, risks, next steps
        """
        if self.verbose:
            print("\n" + "â”" * 60)
            print("âš–ï¸  EVALUATING RESULTS")
            print("â”" * 60)

        # Analyze failures before scoring so diagnostics are carried into the report.
        failure_analysis = analyze_failures(results)

        # Only score successful runs
        successful = [r for r in results if r.status == ExecutionStatus.SUCCESS
                      and r.primary_metric_value is not None]
        failed     = [r for r in results if r.status == ExecutionStatus.FAILED]

        if not successful:
            raise RuntimeError(
                "None of the experiments completed successfully. "
                "Check the error messages above for each method. "
                "The most common cause is a data format issue â€” "
                "make sure your CSV has the right column names."
            )

        if self.verbose:
            print(f"\n  Comparing {len(successful)} successful experiments")
            if failed:
                print(f"  ({len(failed)} failed â€” excluded from comparison)")
            print()

        # 1. Compute raw scores
        method_scores = self._compute_scores(spec, successful)

        # 2. Detect risk flags
        risk_flags = self._detect_risks(successful)
        if results and (len(failed) / len(results)) > 0.5:
            risk_flags.append(RiskFlag(
                method_id="all_methods",
                flag_type="systemic_failure",
                severity="critical",
                explanation=(
                    f"{len(failed)} of {len(results)} methods failed (>50%). "
                    "This suggests a shared root cause (dataset schema, environment, or preprocessing). "
                    "Fix systemic issues before trusting rankings."
                ),
            ))

        # 3. Ask Claude to write the explanation, next steps, research directions
        winner     = max(method_scores, key=lambda s: s.total_score)
        runner_up  = sorted(method_scores, key=lambda s: s.total_score, reverse=True)[1] \
                     if len(method_scores) > 1 else None

        explanations = self._generate_explanations(
            spec,
            method_scores,
            winner,
            runner_up,
            risk_flags,
            failure_analysis,
        )

        failed_methods_summary = [
            FailedMethodSummary(**item)
            for item in failure_analysis["failed_methods_summary"]
        ]

        report = EvaluationReport(
            method_scores=method_scores,
            winner_method_id=winner.method_id,
            winner_explanation=explanations.get("winner_explanation", winner.summary),
            runner_up_method_id=runner_up.method_id if runner_up else None,
            runner_up_explanation=explanations.get("runner_up_explanation"),
            risk_flags=risk_flags,
            failed_methods_summary=failed_methods_summary,
            next_steps=explanations.get("next_steps", []),
            research_directions=explanations.get("research_directions", []),
        )

        if self.verbose:
            self._print_report(report, method_scores)

        self._save_experiments_to_memory(spec, successful, method_scores)
        return report

    def _save_experiments_to_memory(
        self,
        spec: ProblemSpec,
        successful_results: list[ExecutionResult],
        method_scores: list[MethodScore],
    ) -> None:
        """Persist successful method outcomes for future recommendations."""
        if not successful_results:
            return
        runtime_by_method = {r.method_id: (r.runtime_minutes or 0.0) for r in successful_results}
        score_by_method = {s.method_id: s.total_score for s in method_scores}
        rows = 0
        columns = 0
        if spec.dataset_diagnostics is not None:
            rows = spec.dataset_diagnostics.rows
            columns = spec.dataset_diagnostics.columns
        elif spec.estimated_row_count is not None:
            rows = spec.estimated_row_count

        dataset_type = self._dataset_type_from_task(spec.task_type.value)
        dataset_profile = {
            "dataset_type": dataset_type,
            "rows": rows,
            "columns": columns,
        }
        signature = self.memory.compute_dataset_signature(dataset_profile, spec.task_type.value)

        for r in successful_results:
            record = {
                "dataset_signature": signature,
                "dataset_type": dataset_type,
                "rows": rows,
                "columns": columns,
                "task_type": spec.task_type.value,
                "method_id": r.method_id,
                "primary_metric": spec.primary_metric.value,
                "runtime_minutes": float(runtime_by_method.get(r.method_id, 0.0)),
                "score": float(score_by_method.get(r.method_id, 0.0)),
            }
            try:
                self.memory.save_experiment(record)
            except Exception as e:
                logger.warning(f"Failed to persist experiment memory for {r.method_id}: {e}")

    def _dataset_type_from_task(self, task_type: str) -> str:
        if task_type == "nlp":
            return "text"
        if task_type == "computer_vision":
            return "image"
        return "tabular"

    def _compute_scores(
        self,
        spec: ProblemSpec,
        results: list[ExecutionResult],
    ) -> list[MethodScore]:
        """Compute multi-criteria scores for each successful method."""

        # Raw values
        perf_values     = [r.primary_metric_value for r in results]
        speed_values    = [r.runtime_minutes or 60.0 for r in results]  # minutes
        robust_values   = self._compute_robustness(results)

        # Normalize: performance higher=better, speed lower=better
        perf_norm   = normalize_scores(perf_values,   higher_is_better=True)
        speed_norm  = normalize_scores(speed_values,  higher_is_better=False)
        robust_norm = normalize_scores(robust_values, higher_is_better=True)

        method_scores = []
        for i, result in enumerate(results):
            interp_score = get_interpretability_score(
                self._get_algorithm_family(result.method_id)
            )

            # Weighted total
            total = (
                self.weights["performance"]      * perf_norm[i] +
                self.weights["speed"]            * speed_norm[i] +
                self.weights["interpretability"] * interp_score +
                self.weights["robustness"]       * robust_norm[i]
            )

            summary = (
                f"{result.primary_metric_name} = {result.primary_metric_value:.4f}, "
                f"trained in {result.runtime_minutes:.1f} min"
                if result.runtime_minutes else
                f"{result.primary_metric_name} = {result.primary_metric_value:.4f}"
            )

            method_scores.append(MethodScore(
                method_id=result.method_id,
                method_name=result.method_name,
                primary_metric=result.primary_metric_value,
                score_performance=perf_norm[i],
                score_speed=speed_norm[i],
                score_interpretability=interp_score,
                score_robustness=robust_norm[i],
                total_score=total,
                summary=summary,
            ))

        return sorted(method_scores, key=lambda s: s.total_score, reverse=True)

    def _compute_robustness(self, results: list[ExecutionResult]) -> list[float]:
        """
        Robustness = how stable is the model? Measured by train/val gap.
        Smaller gap = more robust. Returns a score (higher = more robust).
        """
        robustness = []
        for r in results:
            if r.train_metric is not None and r.val_metric is not None:
                gap = abs(r.train_metric - r.val_metric)
                # Convert gap to robustness score: 0 gap = 1.0, 0.5+ gap = 0.0
                score = max(0.0, 1.0 - (gap / 0.5))
            else:
                score = 0.5  # Unknown â€” neutral score
            # Slightly penalize methods that required retries to succeed.
            retry_penalty = min(0.15, 0.05 * max(0, r.retry_count))
            score = max(0.0, score - retry_penalty)
            robustness.append(score)
        return robustness

    def _detect_risks(self, results: list[ExecutionResult]) -> list[RiskFlag]:
        """Flag real risks the researcher needs to know about."""
        flags = []

        for r in results:
            # Overfitting check
            if r.train_metric is not None and r.val_metric is not None:
                gap = abs(r.train_metric - r.val_metric)
                if gap > 0.15:
                    flags.append(RiskFlag(
                        method_id=r.method_id,
                        flag_type="overfitting",
                        severity="critical",
                        explanation=(
                            f"'{r.method_name}' has a large gap between training "
                            f"({r.train_metric:.3f}) and validation ({r.val_metric:.3f}) scores "
                            f"(gap = {gap:.3f}). This suggests the model memorized the training "
                            f"data rather than learning general patterns. "
                            f"Don't trust this validation score â€” it's likely optimistic. "
                            f"Try adding more regularization or getting more data."
                        ),
                    ))
                elif gap > 0.08:
                    flags.append(RiskFlag(
                        method_id=r.method_id,
                        flag_type="overfitting",
                        severity="warning",
                        explanation=(
                            f"'{r.method_name}' shows a moderate train/val gap ({gap:.3f}). "
                            f"Worth monitoring â€” if your test data comes from a different "
                            f"distribution, real-world performance may be lower."
                        ),
                    ))

            # Suspiciously perfect score
            if r.primary_metric_value is not None and r.primary_metric_value > 0.99:
                flags.append(RiskFlag(
                    method_id=r.method_id,
                    flag_type="suspicious_score",
                    severity="critical",
                    explanation=(
                        f"'{r.method_name}' achieved a near-perfect score "
                        f"({r.primary_metric_value:.4f}). Real-world ML problems rarely "
                        f"achieve this. Most likely cause: data leakage â€” a feature in your "
                        f"dataset directly encodes the answer. Check your features carefully "
                        f"for any that couldn't be known at prediction time."
                    ),
                ))

        return flags

    def _get_algorithm_family(self, method_id: str) -> str:
        """Look up algorithm family from method_id."""
        family_map = {
            "logistic_regression": "Linear",
            "ridge_regression":    "Linear",
            "random_forest":       "Ensemble (Bagging)",
            "xgboost":             "Gradient Boosting",
            "xgboost_regression":  "Gradient Boosting",
            "xgboost_timeseries":  "Gradient Boosting + Feature Engineering",
            "lightgbm":            "Gradient Boosting",
            "lightgbm_regression": "Gradient Boosting",
            "tabnet":              "Neural Network (Attention)",
            "distilbert":          "Transformer",
            "tfidf_logreg":        "Classical NLP",
            "efficientnet_transfer": "CNN (Transfer Learning)",
            "resnet50_transfer":   "CNN (Transfer Learning)",
            "prophet":             "Additive Model",
            "lstm_timeseries":     "Recurrent Neural Network",
            "kmeans":              "Centroid-based",
            "hdbscan":             "Density-based",
        }
        return family_map.get(method_id, "Unknown")

    def _generate_explanations(
        self,
        spec: ProblemSpec,
        scores: list[MethodScore],
        winner: MethodScore,
        runner_up: Optional[MethodScore],
        risks: list[RiskFlag],
        failure_analysis: dict[str, Any],
    ) -> dict:
        """Use Claude to write the plain English evaluation narrative."""

        scores_table = "\n".join([
            f"  {s.method_name}: total={s.total_score:.3f} | "
            f"perf={s.score_performance:.3f} | speed={s.score_speed:.3f} | "
            f"interp={s.score_interpretability:.3f} | robust={s.score_robustness:.3f} | "
            f"primary_metric={s.primary_metric:.4f}"
            for s in scores
        ])

        risk_summary = "\n".join([
            f"  [{r.severity.upper()}] {r.method_id}: {r.explanation}"
            for r in risks
        ]) or "  None detected"

        failed_rows = failure_analysis.get("failed_methods_summary", [])
        if failed_rows:
            failed_methods_table = "\n".join([
                f"  - {row['method_name']} ({row['method_id']}): "
                f"{row['error_type']} | retries={row['retry_count']} | "
                f"{row['error_plain_english']}"
                for row in failed_rows
            ])
        else:
            failed_methods_table = "  None"

        failure_pattern_summary = (
            f"  Error counts: {failure_analysis.get('error_type_counts', {})}\n"
            f"  Repeated errors: {failure_analysis.get('repeated_error_types', {})}\n"
            f"  Systemic issues: {failure_analysis.get('systemic_issues', [])}"
        )

        constraints = "\n".join([
            f"  - [{('MUST' if c.is_hard else 'nice-to-have')}] {c.name}: {c.description}"
            for c in spec.constraints
        ]) or "  None"

        runner_up_name  = runner_up.method_name if runner_up else "none"
        runner_up_score = f"{runner_up.total_score:.3f}" if runner_up else "n/a"
        diagnostics = spec.dataset_diagnostics
        diagnostics_summary = "  None"
        if diagnostics is not None:
            diagnostics_summary = (
                f"  Risk flags: {diagnostics.risk_flags}\n"
                f"  Missing columns: {diagnostics.missing_value_columns}\n"
                f"  Leakage candidates: {diagnostics.leakage_candidates}\n"
                f"  High-cardinality: {diagnostics.high_cardinality_columns}\n"
                f"  Duplicate ratio: {diagnostics.duplicate_ratio:.3f}\n"
                f"  Class distribution: {diagnostics.class_distribution}"
            )

        prompt = f"""## Problem Context
Domain: {spec.domain}
Task type: {spec.task_type.value}
Primary metric: {spec.primary_metric.value}
Researcher constraints:
{constraints}

## Scoring Results (sorted by total score, highest first)

{scores_table}

## Weight configuration used
Performance: {self.weights['performance']:.0%}
Speed: {self.weights['speed']:.0%}
Interpretability: {self.weights['interpretability']:.0%}
Robustness: {self.weights['robustness']:.0%}

## Winner: {winner.method_name} (score: {winner.total_score:.3f})
## Runner-up: {runner_up_name} (score: {runner_up_score})

## Risk Flags
{risk_summary}

## Failed Methods
{failed_methods_table}

## Failure Patterns
{failure_pattern_summary}

## Dataset Diagnostics
{diagnostics_summary}

## Your task

Write:
1. winner_explanation: Why did {winner.method_name} win? Be specific. Acknowledge any caveats.
2. runner_up_explanation: What would make the runner-up worth trying instead?
3. next_steps: 3-5 concrete, ranked actions with reasons, including failure diagnostics
4. research_directions: 2-3 novel directions the researcher could pursue beyond these baselines

Respond with valid JSON:
{{
  "winner_explanation": "...",
  "runner_up_explanation": "...",
  "next_steps": ["...", "...", "..."],
  "research_directions": ["...", "..."]
}}
"""

        raw = self.llm.create(
            system=EVALUATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            verbose=self.verbose,
        )
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception:
            logger.warning("Could not parse evaluator explanations â€” using defaults")
            return {
                "winner_explanation": winner.summary,
                "runner_up_explanation": runner_up.summary if runner_up else None,
                "next_steps": ["Review the experiment notebooks on Kaggle for full details."],
                "research_directions": ["Explore ensemble combinations of the top methods."],
            }

    def _print_report(self, report: EvaluationReport, scores: list[MethodScore]) -> None:
        """Print evaluation results in mentor voice."""
        print(f"\n  ðŸ† Best method: {report.winner_method_id}")
        print(f"\n  {report.winner_explanation}\n")

        print("  Score breakdown:")
        print(f"  {'Method':<30} {'Perf':>6} {'Speed':>6} {'Interp':>7} {'Robust':>7} {'TOTAL':>7}")
        print("  " + "â”€" * 70)
        for s in scores:
            print(
                f"  {s.method_name:<30} "
                f"{s.score_performance:>6.3f} "
                f"{s.score_speed:>6.3f} "
                f"{s.score_interpretability:>7.3f} "
                f"{s.score_robustness:>7.3f} "
                f"{s.total_score:>7.3f}"
            )

        if report.risk_flags:
            print("\n  âš ï¸  Risks to review:")
            for flag in report.risk_flags:
                emoji = "ðŸš¨" if flag.severity == "critical" else "âš ï¸"
                print(f"  {emoji} {flag.explanation[:120]}")


        if report.failed_methods_summary:
            print("\n  \u274c Failed methods:")
            for failed in report.failed_methods_summary:
                print(f"  {failed.method_name} — {failed.error_plain_english}")

        print("\n  ðŸ“‹ Recommended next steps:")
        for i, step in enumerate(report.next_steps, 1):
            print(f"  {i}. {step[:120]}")

        if report.research_directions:
            print("\n  ðŸ”­ Novel research directions you could pursue:")
            for d in report.research_directions:
                print(f"  â†’ {d[:120]}")

        print("\n" + "â”" * 60 + "\n")


