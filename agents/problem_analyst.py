"""
Problem Analyst Agent

The first agent in the pipeline. Reads the researcher's problem statement
and produces a structured ProblemSpec that every downstream agent uses.

Design principle:
    If this agent gets the task_type wrong, every other agent is wrong.
    So we ask for confirmation when confidence < 0.80, and we always
    explain our reasoning in plain English.

Mentor voice:
    We write like a knowledgeable friend helping someone understand their
    own problem — not like a classifier outputting a label.
"""

import json
import logging
import re
from typing import Optional

from autoresearch.agents.api_utils import LLMClient

from autoresearch.schemas import ProblemSpec, TaskType, EvalMetric, Constraint

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Problem Analyst for AutoResearch — a tool that helps
researchers (including those without deep ML expertise) run rigorous ML experiments.

Your job: read a research problem statement and produce a structured analysis.

## Your core responsibilities

1. Classify the ML task type (classification, regression, nlp, computer_vision,
   time_series, or clustering)
2. Identify the right evaluation metric for THIS specific problem and domain
3. Extract any constraints the researcher cares about (interpretability, speed, etc.)
4. Assign a confidence score — and be honest when you're not sure
5. Write a plain English summary a domain expert (non-ML) can validate

## The most important rule: explain your reasoning

Never just output a classification. Always explain WHY in plain English.
The researcher needs to understand and validate your analysis.

Bad:  "task_type: classification"
Good: "This is a classification problem because you're trying to predict a yes/no
       outcome (heart disease present or absent). The target column 'diagnosis'
       has two values: 0 and 1."

## Metric selection guidance

- Medical / high-stakes binary classification → AUC-ROC (handles class imbalance)
- When false positives are costly → precision_recall_auc
- Balanced multi-class → f1_macro
- Regression where outliers matter → RMSE
- Regression where proportional errors matter → MAPE
- Small datasets with imbalance → f1_weighted

## GPU guidance

Only set requires_gpu=True for:
- Computer vision (always)
- NLP with transformer models (BERT, etc.)
- Time series with LSTM / Temporal Fusion Transformer
- Dataset > 500k rows with neural networks

Classical ML (XGBoost, Random Forest, Linear models) should be requires_gpu=False
to save the researcher's free Kaggle GPU quota.

## Confidence scoring

- 0.95+: Problem statement is unambiguous, task type is obvious
- 0.80–0.94: Some ambiguity but reasonable inference possible
- 0.60–0.79: Multiple interpretations possible — explain them
- Below 0.60: Too ambiguous — ask a clarifying question instead of guessing

## Output format

Respond with a valid JSON object matching the ProblemSpec schema exactly.
No extra text before or after the JSON.

CRITICAL: Always include these exact fields in your JSON: domain, input_description,
confidence, confidence_explanation, requires_gpu, constraints (as array of objects
with name/description/is_hard keys, NOT plain strings).
"""


def build_analysis_prompt(
    problem_statement: str,
    dataset_info: Optional[dict] = None,
) -> str:
    """Build the user message for the Problem Analyst."""

    prompt = f"""## Research Problem Statement

{problem_statement}
"""

    if dataset_info:
        prompt += f"""
## Dataset Information

{json.dumps(dataset_info, indent=2)}
"""

    prompt += """
## Your task

Analyze this problem statement and produce a ProblemSpec JSON.

Remember:
- Explain your reasoning in plain_english_summary — write for the researcher, not another ML engineer
- Be honest about confidence — if you're not sure, say so in confidence_explanation
- Reference specific details from the problem statement in your explanations
- If this is a medical, legal, or high-stakes domain, note that in constraints

Respond with valid JSON only.
"""
    return prompt


def parse_problem_spec(raw_json: str) -> ProblemSpec:
    """
    Parse the LLM response into a ProblemSpec.
    Raises ValueError with a plain English message if parsing fails.
    """
    if not raw_json or not raw_json.strip():
        raise ValueError(
            "AutoResearch received an empty response from the LLM. "
            "Check your API key and that the model name in config.yaml is correct."
        )

    # Strip markdown code fences and extract just the JSON object
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_json).replace("```", "").strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1:
        cleaned = cleaned[brace_start:brace_end + 1]

    try:
        data = json.loads(cleaned)

        # Normalize/infer task_type first, because it's required by ProblemSpec.
        task_aliases = {
            "classification": "classification",
            "classify": "classification",
            "classifier": "classification",
            "regression": "regression",
            "regress": "regression",
            "nlp": "nlp",
            "text": "nlp",
            "computer_vision": "computer_vision",
            "vision": "computer_vision",
            "image": "computer_vision",
            "time_series": "time_series",
            "timeseries": "time_series",
            "forecasting": "time_series",
            "clustering": "clustering",
            "cluster": "clustering",
        }
        task_val = str(data.get("task_type", "")).strip().lower()
        if task_val:
            data["task_type"] = task_aliases.get(task_val, task_val)

        # Normalize constraints: Claude sometimes returns plain strings instead of dicts
        if "constraints" in data and isinstance(data["constraints"], list):
            fixed = []
            for c in data["constraints"]:
                if isinstance(c, str):
                    fixed.append({"name": "constraint", "description": c, "is_hard": False})
                elif isinstance(c, dict):
                    fixed.append(c)
            data["constraints"] = fixed

        # Normalize primary_metric: Claude sometimes uses alternate field names
        if "primary_metric" not in data:
            for alt in ("metric", "eval_metric", "evaluation_metric", "primary_eval_metric"):
                if alt in data:
                    data["primary_metric"] = data[alt]
                    break
            else:
                task = data.get("task_type", "")
                if task == "computer_vision":
                    data["primary_metric"] = "accuracy"
                elif task == "regression":
                    data["primary_metric"] = "rmse"
                elif task in ("nlp", "classification"):
                    data["primary_metric"] = "f1_weighted"
                else:
                    data["primary_metric"] = "accuracy"

        # Coerce unknown metric names to the closest valid EvalMetric value
        _metric_aliases = {
            "log_loss": "log_loss",
            "logloss": "log_loss",
            "cross_entropy": "log_loss",
            "roc_auc": "auc_roc",
            "roc_auc_score": "auc_roc",
            "f1": "f1_weighted",
            "f1_score": "f1_weighted",
            "mean_squared_error": "rmse",
            "mse": "rmse",
            "mean_absolute_error": "mae",
            "r2_score": "r2",
        }
        for field in ("primary_metric", ):
            val = data.get(field)
            if val and val not in {m.value for m in EvalMetric}:
                data[field] = _metric_aliases.get(val.lower(), "accuracy")
        data["secondary_metrics"] = [
            _metric_aliases.get(m.lower(), None) or m
            for m in data.get("secondary_metrics", [])
            if m in {x.value for x in EvalMetric} or m.lower() in _metric_aliases
        ]

        # Fill missing required fields with sensible defaults
        if "task_type" not in data or not data.get("task_type"):
            metric_hint = str(data.get("primary_metric", "")).lower()
            if metric_hint in {"rmse", "mae", "r2", "mape"}:
                data["task_type"] = "regression"
            elif metric_hint in {"bleu", "rouge_l"}:
                data["task_type"] = "nlp"
            elif metric_hint in {"silhouette", "davies_bouldin", "calinski_harabasz"}:
                data["task_type"] = "clustering"
            else:
                # Safe default for most tabular prediction statements.
                data["task_type"] = "classification"

        data.setdefault("domain", "machine learning")
        data.setdefault("input_description", data.get("plain_english_summary", "dataset provided"))
        data.setdefault("confidence", 0.85)
        data.setdefault("confidence_explanation", "Analysis based on problem statement.")
        data.setdefault("requires_gpu", False)
        data.setdefault("secondary_metrics", [])
        data.setdefault("constraints", [])
        data.setdefault("estimated_row_count", None)
        data.setdefault("has_class_imbalance", False)
        data.setdefault("target_column", None)
        data.setdefault(
            "plain_english_summary",
            f"This appears to be a {data['task_type']} problem in the {data['domain']} domain.",
        )

        return ProblemSpec(**data)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AutoResearch couldn't parse its own analysis (this is a bug, not your fault). "
            f"Technical detail: {e}"
        )
    except Exception as e:
        raise ValueError(
            f"The problem analysis produced an unexpected structure. "
            f"Try rephrasing your problem statement to be more specific. "
            f"Technical detail: {e}"
        )


class ProblemAnalystAgent:
    """
    The Problem Analyst agent.

    Usage:
        agent = ProblemAnalystAgent(api_key="your-anthropic-key")
        spec = agent.analyze("I want to predict hospital readmission...")

        if spec.confidence < 0.80:
            # Show spec to user and ask them to confirm or correct
            confirmed_spec = ask_user_to_confirm(spec)
    """

    CONFIDENCE_THRESHOLD = 0.80  # Below this, ask researcher to confirm

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        problem_statement: str,
        dataset_info: Optional[dict] = None,
        verbose: bool = True,
    ) -> ProblemSpec:
        """
        Analyze a problem statement and return a ProblemSpec.

        Args:
            problem_statement: The researcher's description of their problem.
            dataset_info: Optional dict with column names, dtypes, sample rows.
            verbose: If True, print mentor-style progress updates.

        Returns:
            ProblemSpec — the structured understanding of the problem.

        Raises:
            ValueError: If the problem statement is too ambiguous to analyze.
        """
        if verbose:
            print("\n🔍 Reading your problem statement...")
            print("   AutoResearch is figuring out what kind of ML problem this is.\n")

        raw = self.llm.create(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_analysis_prompt(problem_statement, dataset_info)}],
            max_tokens=2000,
            verbose=verbose,
        )
        spec = parse_problem_spec(raw)

        if verbose:
            self._print_analysis(spec)

        if spec.confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Low confidence ({spec.confidence:.0%}): {spec.confidence_explanation}"
            )

        return spec

    def _print_analysis(self, spec: ProblemSpec) -> None:
        """Print the analysis in mentor voice."""

        confidence_emoji = (
            "✅" if spec.confidence >= 0.80
            else "⚠️"
        )

        print("━" * 60)
        print("📋 PROBLEM ANALYSIS")
        print("━" * 60)
        print(f"\n{spec.plain_english_summary}\n")
        print(f"  Task type:        {spec.task_type.value}")
        print(f"  Domain:           {spec.domain}")
        print(f"  Primary metric:   {spec.primary_metric.value}")
        print(f"  GPU needed:       {'Yes' if spec.requires_gpu else 'No (saves your Kaggle quota)'}")
        print(f"  Confidence:       {confidence_emoji} {spec.confidence:.0%}")

        if spec.confidence < self.CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠️  {spec.confidence_explanation}")
            print("  → AutoResearch will ask you to confirm before continuing.\n")
        else:
            print(f"\n  {spec.confidence_explanation}\n")

        if spec.constraints:
            print("  Constraints identified:")
            for c in spec.constraints:
                marker = "MUST" if c.is_hard else "nice to have"
                print(f"    [{marker}] {c.name}: {c.description}")

        print("━" * 60)
        print("  If anything above looks wrong, you can edit config.yaml")
        print("  before the run continues.")
        print("━" * 60 + "\n")
