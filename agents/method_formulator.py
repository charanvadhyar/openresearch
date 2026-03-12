"""
Method Formulator Agent

Reads the ProblemSpec and DataHealthReport, queries the local ML knowledge
base, and proposes 3â€“5 candidate methods â€” each with a plain English
explanation of WHY it was chosen for this specific problem and data.

This is where AutoResearch earns its keep for non-expert researchers.
Instead of a blank list of algorithms, they get:
  "We chose XGBoost because your dataset has 303 rows with mixed features.
   XGBoost handles both without scaling, and works well on small datasets
   unlike neural networks which typically need 100k+ rows."

Design principle:
    The `why_chosen` field is NOT optional and NOT generic.
    It must reference specific numbers from the DataHealthReport.
    A generic "XGBoost is good for tabular data" fails this test.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from autoresearch.agents.api_utils import LLMClient
from autoresearch.memory.experiment_memory import ExperimentMemory
from autoresearch.schemas import (
    ProblemSpec,
    DataHealthReport,
    MethodSpec,
    MethodsCatalog,
    ComplexityLevel,
    TaskType,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base" / "ml_methods.json"


# â”€â”€ System prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SYSTEM_PROMPT = """You are the Method Formulator for AutoResearch â€” a tool that helps
researchers (including those without deep ML expertise) run rigorous experiments.

Your job: given a problem specification and EDA findings, select 3â€“5 candidate
ML methods and explain why each one is appropriate for THIS specific problem.

## The most important rule: be specific about WHY

Every method needs a `why_chosen` explanation that references specific numbers
and characteristics from the problem spec and EDA report.

BAD (generic):
  "XGBoost is a strong gradient boosting method that works well on tabular data."

GOOD (specific):
  "Chosen because your dataset has 303 rows with 13 numeric features and no
   missing values. XGBoost handles mixed feature types without scaling, and at
   this dataset size (300 rows) it typically outperforms neural networks, which
   need 10k+ rows to learn effectively. Your class balance (54%/46%) is nearly
   perfect, so no special handling needed."

## Method selection rules

1. Never propose more than 5 methods (respects Kaggle's 30hr/week GPU quota)
2. Always include at least one simple interpretable baseline (logistic regression,
   ridge, TF-IDF+LR) â€” it's honest research practice
3. Only propose GPU methods if requires_gpu=True in the ProblemSpec
4. If the dataset is small (under 1000 rows), warn against deep learning
5. If interpretability is a hard constraint, rank interpretable methods higher
6. If class imbalance was flagged, mention how each method handles it

## watch_out_for rule

Every method needs a `watch_out_for` that is specific to this dataset.
Not "XGBoost can overfit" â€” that's generic.
Instead: "XGBoost can overfit on your 303-row dataset. Watch for a gap
larger than 5% between your training score and validation score."

## Output format

Respond with valid JSON only. No extra text.
Schema: { "methods": [...], "ranking_explanation": "...", "methods_not_tried": [...] }
"""


def load_knowledge_base(task_type: TaskType) -> list[dict]:
    """Load candidate algorithms for this task type from the knowledge base."""
    with open(KNOWLEDGE_BASE_PATH) as f:
        kb = json.load(f)
    return kb.get(task_type.value, [])


def filter_candidates(
    candidates: list[dict],
    spec: ProblemSpec,
    report: DataHealthReport,
) -> list[dict]:
    """
    Hard-filter candidates before sending to Claude.
    Removes obviously wrong choices based on data characteristics.
    This saves tokens and gives Claude better candidates to reason about.
    """
    filtered = []
    row_count = report.row_count
    diagnostics = spec.dataset_diagnostics
    is_small_dataset = bool(diagnostics and diagnostics.rows < 1000)

    for c in candidates:
        # Skip GPU methods if GPU not available/needed
        if c["requires_gpu"] and not spec.requires_gpu:
            logger.debug(f"Skipping {c['name']}: requires GPU but GPU not enabled")
            continue

        # For very small datasets, prefer classical models by filtering heavy GPU/deep methods.
        if is_small_dataset and c["requires_gpu"]:
            logger.debug(f"Skipping {c['name']}: small dataset ({diagnostics.rows} rows), prefer classical models")
            continue

        # Skip methods that need more data than we have
        if row_count < c.get("min_rows", 0):
            logger.debug(f"Skipping {c['name']}: needs {c['min_rows']} rows, have {row_count}")
            continue

        filtered.append(c)

    # Always keep at least 3 candidates (relax GPU filter if needed)
    if len(filtered) < 3:
        filtered = candidates[:5]

    return filtered


def _eda_summary(spec: ProblemSpec, report: DataHealthReport) -> str:
    """Compact EDA summary for use in prompts - keeps token count low."""
    diagnostics = spec.dataset_diagnostics
    diagnostics_summary = "none"
    if diagnostics is not None:
        diagnostics_summary = (
            f"risk_flags={diagnostics.risk_flags}; "
            f"missing_cols={diagnostics.missing_value_columns[:5]}; "
            f"high_cardinality={diagnostics.high_cardinality_columns[:5]}; "
            f"leakage={diagnostics.leakage_candidates[:5]}; "
            f"duplicate_ratio={diagnostics.duplicate_ratio:.3f}"
        )

    flags_summary = [
        f"  [{str(severity).upper()}] {explanation}"
        for _, severity, explanation in report.flags
    ]
    return f"""Task type: {spec.task_type.value}
Domain: {spec.domain}
Primary metric: {spec.primary_metric.value}
Target column: {spec.target_column or "none"}
GPU available: {spec.requires_gpu}
Dataset: {report.row_count:,} rows x {report.column_count} columns
Health score: {report.health_score:.0f}/100
Data issues: {chr(10).join(flags_summary) if flags_summary else "none"}
Dataset diagnostics: {diagnostics_summary}
Method hints: {"; ".join(report.method_hints[:5]) if report.method_hints else "none"}
High-importance features: {", ".join(report.high_importance_features[:5]) or "not determined"}
Summary: {spec.plain_english_summary}"""


def build_selection_prompt(
    spec: ProblemSpec,
    report: DataHealthReport,
    candidates: list[dict],
) -> str:
    """
    Call 1 â€” small prompt: just pick which method IDs to use.
    Returns JSON: {"selected_ids": [...], "excluded": [{"id": ..., "reason": ...}]}
    """
    menu = "\n".join(
        f"  {c['id']}: {c['name']} (min_rows={c.get('min_rows',0)}, gpu={c['requires_gpu']})"
        for c in candidates
    )
    return f"""{_eda_summary(spec, report)}

Available methods:
{menu}

Pick 3â€“5 methods best suited for this problem. Respond with JSON only:
{{"selected_ids": ["id1", "id2"], "excluded": [{{"id": "id3", "reason": "why skipped"}}]}}"""


def build_explanation_prompt(
    spec: ProblemSpec,
    report: DataHealthReport,
    selected: list[dict],
) -> str:
    """
    Call 2 â€” for each selected method, generate why_chosen + watch_out_for.
    selected is a list of KB entries (dicts) for the chosen methods only.
    """
    methods_block = "\n\n".join(
        f"ID: {m['id']}\nName: {m['name']}\nStrengths: {', '.join(m.get('strengths', []))}"
        for m in selected
    )
    return f"""{_eda_summary(spec, report)}

Selected methods to explain:
{methods_block}

For each method write:
- why_chosen: 2â€“3 sentences referencing specific numbers from the dataset above
- watch_out_for: 1 sentence specific to this dataset (not generic)
- complexity: low / medium / high
- estimated_runtime_minutes: integer

Respond with JSON only:
{{"methods": [{{"id": "...", "why_chosen": "...", "watch_out_for": "...", "complexity": "...", "estimated_runtime_minutes": 0}}]}}"""


class MethodFormulatorAgent:
    """
    The Method Formulator agent.

    Queries the local knowledge base and uses Claude to reason about
    which methods are most appropriate for this specific problem and data.

    Usage:
        agent = MethodFormulatorAgent(api_key="...")
        catalog = agent.formulate(spec, report)
        # catalog.methods[0].why_chosen â†’ specific explanation
    """

    MAX_METHODS = 5

    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm     = llm
        self.verbose = verbose
        self.memory  = ExperimentMemory()

    def formulate(
        self,
        spec: ProblemSpec,
        report: DataHealthReport,
    ) -> MethodsCatalog:
        """
        Propose 3â€“5 methods with plain English explanations.

        Args:
            spec: ProblemSpec from Problem Analyst
            report: DataHealthReport from EDA Agent

        Returns:
            MethodsCatalog with ranked methods and explanations
        """
        if self.verbose:
            print("\n" + "â”" * 60)
            print("ðŸ§ª METHOD SELECTION")
            print("â”" * 60)
            print(f"\n  Looking at the knowledge base for {spec.task_type.value} methods...")

        # 1. Load and filter candidates from knowledge base
        all_candidates = load_knowledge_base(spec.task_type)
        candidates     = filter_candidates(all_candidates, spec, report)

        dataset_profile = {
            "dataset_type": self._dataset_type_from_task(spec.task_type),
            "rows": (spec.dataset_diagnostics.rows if spec.dataset_diagnostics else report.row_count),
            "columns": (spec.dataset_diagnostics.columns if spec.dataset_diagnostics else report.column_count),
        }
        recommended_ids = self.memory.recommend_methods(dataset_profile, spec.task_type.value)
        rec_stats = self.memory.recommendation_stats(dataset_profile, spec.task_type.value)
        if recommended_ids:
            rank = {mid: i for i, mid in enumerate(recommended_ids)}
            candidates = sorted(
                candidates,
                key=lambda c: rank.get(c["id"], len(recommended_ids) + 1),
            )

        if self.verbose:
            print(f"  Found {len(all_candidates)} methods in knowledge base")
            print(f"  {len(candidates)} are appropriate for your data characteristics")
            print("\n  PAST EXPERIMENT MEMORY")
            print(f"  Similar datasets found: {rec_stats['similar_datasets_found']}")
            if rec_stats["top_methods"]:
                print("\n  Top performing methods:")
                for i, item in enumerate(rec_stats["top_methods"], 1):
                    print(
                        f"  {i}. {item['method_id']} "
                        f"(avg {spec.primary_metric.value.upper()} {item['avg_score']:.3f})"
                    )
                print("\n  These methods will be prioritized.")
            print(f"\n  Reasoning about which ones fit your specific problem...\n")

        # 2a. Call 1 â€” pick which methods to use (small prompt)
        sel_text = self.llm.create(
            system="You are a method selection assistant. Respond with valid JSON only.",
            messages=[{"role": "user", "content": build_selection_prompt(spec, report, candidates)}],
            max_tokens=500,
            verbose=self.verbose,
        )
        selected_ids, excluded_raw = self._parse_selection(sel_text, candidates)

        candidate_map = {c["id"]: c for c in candidates}
        selected_entries = [candidate_map[sid] for sid in selected_ids if sid in candidate_map]

        if self.verbose:
            print(f"  Chose {len(selected_entries)} methods: {', '.join(selected_ids)}")
            print(f"\n  Generating detailed explanations...\n")

        # 2b. Call 2 â€” explain only the selected methods (small prompt)
        exp_text = self.llm.create(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_explanation_prompt(spec, report, selected_entries)}],
            max_tokens=2000,
            verbose=self.verbose,
        )

        # 3. Parse into MethodsCatalog
        catalog = self._parse_catalog(exp_text, selected_entries, excluded_raw)

        # 3b. Planner adjustment for class imbalance:
        # add class_weight handling to classification methods that support it.
        diagnostics = spec.dataset_diagnostics
        severe_imbalance = (
            diagnostics is not None and "class_imbalance" in diagnostics.risk_flags
        )
        if severe_imbalance and spec.task_type == TaskType.CLASSIFICATION:
            for method in catalog.methods:
                if method.hyperparam_space is None:
                    method.hyperparam_space = {}
                method.hyperparam_space.setdefault("class_weight", ["balanced", None])
                if "class weight" not in method.watch_out_for.lower():
                    method.watch_out_for += " Also evaluate class_weight='balanced'."

        # 4. Enforce max methods limit (quota protection)
        if len(catalog.methods) > self.MAX_METHODS:
            catalog.methods = catalog.methods[:self.MAX_METHODS]
            logger.info(f"Trimmed to {self.MAX_METHODS} methods to protect Kaggle GPU quota")

        if self.verbose:
            self._print_catalog(catalog, spec)

        return catalog

    def _dataset_type_from_task(self, task_type: TaskType) -> str:
        if task_type == TaskType.NLP:
            return "text"
        if task_type == TaskType.COMPUTER_VISION:
            return "image"
        return "tabular"

    def _parse_selection(self, raw: str, candidates: list[dict]) -> tuple[list[str], list]:
        """Parse Call 1 response â†’ (selected_ids, excluded_list)."""
        import re
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            m = re.search(r"\{[\s\S]*\}", cleaned)
            data = json.loads(m.group() if m else cleaned)
            selected = data.get("selected_ids", [])
            excluded = data.get("excluded", [])
        except Exception:
            # Fallback: use first 4 candidates
            selected = [c["id"] for c in candidates[:4]]
            excluded = []
        # Clamp to MAX_METHODS
        return selected[:self.MAX_METHODS], excluded

    def _parse_catalog(self, raw_json: str, candidates: list[dict], excluded_raw: list = None) -> MethodsCatalog:
        """Parse LLM response into a validated MethodsCatalog."""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try to extract JSON block if there's surrounding text
            import re
            match = re.search(r'\{[\s\S]*\}', raw_json)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("Could not parse method selection response as JSON")

        # Build candidate lookup for filling in fields LLM might have omitted
        candidate_map = {c["id"]: c for c in candidates}

        methods = []
        for m in data.get("methods", []):
            # Fill in fields from knowledge base if LLM omitted them
            kb_entry = candidate_map.get(m.get("id"), {})

            method = MethodSpec(
                id=m.get("id", f"method_{len(methods)}"),
                name=m.get("name", kb_entry.get("name", "Unknown Method")),
                algorithm_family=m.get("algorithm_family", kb_entry.get("family", "")),
                why_chosen=m.get("why_chosen", "Selected as appropriate for this problem type."),
                strengths=m.get("strengths", kb_entry.get("strengths", [])),
                weaknesses=m.get("weaknesses", kb_entry.get("weaknesses", [])),
                complexity=ComplexityLevel(m.get("complexity", kb_entry.get("complexity", "medium"))),
                requires_gpu=m.get("requires_gpu", kb_entry.get("requires_gpu", False)),
                hyperparam_space=m.get("hyperparam_space", kb_entry.get("hyperparam_space", {})),
                feature_engineering=m.get("feature_engineering", kb_entry.get("feature_engineering", [])),
                watch_out_for=m.get("watch_out_for", "Monitor train vs validation score gap."),
                estimated_runtime_minutes=m.get("estimated_runtime_minutes",
                                                 kb_entry.get("estimated_runtime_minutes")),
            )
            methods.append(method)

        # Normalize methods_not_tried â€” merge Call 1 exclusions + any in Call 2 response
        methods_not_tried = data.get("methods_not_tried", []) + (excluded_raw or [])
        if isinstance(methods_not_tried, list):
            fixed = []
            for m in methods_not_tried:
                if isinstance(m, dict):
                    text = (m.get("id") or m.get("name") or "") + (f" â€” {m['reason']}" if m.get("reason") else "")
                    text = text or str(m)
                    fixed.append(text)
                elif isinstance(m, str):
                    fixed.append(m)
            methods_not_tried = fixed

        return MethodsCatalog(
            methods=methods,
            ranking_explanation=data.get("ranking_explanation", "Methods ranked by expected performance."),
            methods_not_tried=methods_not_tried,
        )

    def _print_catalog(self, catalog: MethodsCatalog, spec: ProblemSpec) -> None:
        """Print the method selection in mentor voice."""

        gpu_warning = spec.requires_gpu
        total_gpu_methods = sum(1 for m in catalog.methods if m.requires_gpu)

        print(f"  Selected {len(catalog.methods)} methods to try:\n")

        for i, method in enumerate(catalog.methods, 1):
            complexity_emoji = {"low": "ðŸŸ¢", "medium": "ðŸŸ¡", "high": "ðŸ”´"}.get(
                method.complexity.value, "âšª"
            )
            gpu_tag = " [GPU]" if method.requires_gpu else " [CPU]"
            runtime = f"~{method.estimated_runtime_minutes:.0f} min" if method.estimated_runtime_minutes else "?"

            print(f"  {i}. {method.name}{gpu_tag} {complexity_emoji} {runtime}")
            print(f"     Why: {method.why_chosen[:120]}{'...' if len(method.why_chosen) > 120 else ''}")
            print(f"     Watch out for: {method.watch_out_for[:100]}{'...' if len(method.watch_out_for) > 100 else ''}")
            print()

        print(f"  Ranking: {catalog.ranking_explanation[:150]}")

        if catalog.methods_not_tried:
            print(f"\n  Methods considered but skipped:")
            for m in catalog.methods_not_tried[:3]:
                print(f"    âœ— {m}")

        if total_gpu_methods > 0:
            print(f"\n  âš¡ {total_gpu_methods} method(s) will use your Kaggle GPU quota")
            print(f"     Estimated GPU time: ~{sum(m.estimated_runtime_minutes or 30 for m in catalog.methods if m.requires_gpu):.0f} min")

        print("â”" * 60 + "\n")

