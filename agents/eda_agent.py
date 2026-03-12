"""
EDA Agent

Generates a comprehensive EDA notebook, pushes it to Kaggle Kernels
(on CPU — no GPU needed for EDA, saves the researcher's quota),
waits for it to finish, then parses the output into a DataHealthReport.

Key design decision:
    EDA always runs on CPU. It's fast enough and saves the researcher's
    30hr/week GPU quota for actual model training.

Mentor voice:
    Every finding is explained. Numbers are contextualized.
    "0.23 skewness" becomes "slightly right-skewed, which is fine for most methods".
"""

import json
import logging
from typing import Optional

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from autoresearch.agents.api_utils import LLMClient

from autoresearch.schemas import (
    ProblemSpec,
    DataHealthReport,
    FeatureInsight,
    DataQualityFlag,
    CorrelationPair,
    Severity,
    TaskType,
)
from autoresearch.tools.kaggle_client import KaggleClient, KernelOutput, KernelExecutionError
from autoresearch.agents.error_memory import recall, remember, mark_failed, format_for_prompt
from autoresearch.agents.api_utils import PromptBudget

logger = logging.getLogger(__name__)


# ── EDA notebook code blocks ──────────────────────────────────────────────────
# These are the actual Python cells that run inside the Kaggle kernel.
# Written to be readable — the researcher can open the notebook and understand it.

EDA_CELLS = {
    "setup": """\
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# AutoResearch EDA — reading your data
print("Loading data...")
""",

    "load_data": """\
import os

# Find the data — check common Kaggle input paths
data_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.csv'):
            data_path = os.path.join(root, f)
            break
    if data_path:
        break

if not data_path:
    raise FileNotFoundError(
        "No CSV file found in /kaggle/input. "
        "Make sure your dataset is attached to this kernel."
    )

print(f"Found data at: {data_path}")
df = pd.read_csv(data_path)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()
""",

    "basic_stats": """\
# Basic statistics
print("\\n=== DATASET OVERVIEW ===")
print(f"Rows:        {df.shape[0]:,}")
print(f"Columns:     {df.shape[1]}")
print(f"Memory:      {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Duplicates:  {df.duplicated().sum():,}")

print("\\n=== COLUMN TYPES ===")
print(df.dtypes.value_counts())

print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False))
""",

    "target_analysis": """\
# TARGET COLUMN ANALYSIS — injected by AutoResearch based on ProblemSpec
target_col = "{target_col}"

if target_col and target_col in df.columns:
    print(f"\\n=== TARGET: {target_col} ===")
    print(df[target_col].value_counts(normalize=True).round(4))
    print(f"Unique values: {df[target_col].nunique()}")
    print(f"Missing: {df[target_col].isnull().sum()}")
""",

    "feature_stats": """\
# Per-feature statistics
print("\\n=== FEATURE STATISTICS ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols     = df.select_dtypes(include=['object', 'category']).columns.tolist()

feature_stats = []
for col in df.columns:
    stat = {
        'name':         col,
        'dtype':        str(df[col].dtype),
        'missing_pct':  round(df[col].isnull().mean() * 100, 2),
        'unique_count': int(df[col].nunique()),
    }
    if col in numeric_cols:
        stat.update({
            'mean':     round(float(df[col].mean()), 4),
            'std':      round(float(df[col].std()), 4),
            'min':      round(float(df[col].min()), 4),
            'max':      round(float(df[col].max()), 4),
            'skewness': round(float(df[col].skew()), 4),
        })
    feature_stats.append(stat)

print(json.dumps(feature_stats, indent=2))
""",

    "correlations": """\
# Correlations between numeric features
from itertools import combinations

numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    corr_matrix = numeric_df.corr()
    strong_pairs = []
    for a, b in combinations(numeric_df.columns, 2):
        r = corr_matrix.loc[a, b]
        if abs(r) >= 0.7:  # Strong correlation threshold
            strong_pairs.append({'feature_a': a, 'feature_b': b, 'correlation': round(float(r), 4)})
    print("\\n=== STRONG CORRELATIONS (|r| >= 0.7) ===")
    print(json.dumps(strong_pairs, indent=2))
""",

    "feature_importance": """\
# Feature importance via mutual information
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

target_col = "{target_col}"
task_type  = "{task_type}"

if target_col and target_col in df.columns:
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if feature_cols:
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].fillna(df[target_col].mode()[0])

        try:
            if task_type in ('classification', 'nlp'):
                mi = mutual_info_classif(X, y, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)

            mi_scores = dict(sorted(
                zip(feature_cols, mi.tolist()),
                key=lambda x: x[1], reverse=True
            ))
            print("\\n=== FEATURE IMPORTANCE (mutual information) ===")
            print(json.dumps(mi_scores, indent=2))
        except Exception as e:
            print(f"Feature importance skipped: {e}")
""",

    "save_metrics": """\
# Save rich structured output for AutoResearch codegen
import os, json
import pandas as pd
import numpy as np

# Auto-discover all CSV files
data_files = {}
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.csv'):
            fpath = os.path.join(root, f)
            df_tmp = pd.read_csv(fpath, nrows=5)
            data_files[f] = {
                'path': fpath,
                'columns': list(df_tmp.columns),
                'dtypes': {col: str(dtype) for col, dtype in df_tmp.dtypes.items()},
                'sample_values': {col: df_tmp[col].tolist() for col in df_tmp.columns},
            }

# Auto-detect label/target column
label_col = None
for col in ['label', 'Label', 'target', 'Target', 'class', 'Class', 'y']:
    for fname, info in data_files.items():
        if col in info['columns'] and 'train' in fname.lower():
            label_col = col
            break
    if label_col:
        break

# Identify train/test paths
train_info = {k: v for k, v in data_files.items() if 'train' in k.lower()}
test_info  = {k: v for k, v in data_files.items() if 'test'  in k.lower()}
train_path = list(train_info.values())[0]['path'] if train_info else list(data_files.values())[0]['path']
test_path  = list(test_info.values())[0]['path']  if test_info  else None

# Column analysis on training data
df = pd.read_csv(train_path)
col_analysis = {}
for col in df.columns:
    col_analysis[col] = {
        'dtype': str(df[col].dtype),
        'missing_pct': round(df[col].isnull().mean() * 100, 2),
        'unique_count': int(df[col].nunique()),
        'sample_values': df[col].dropna().head(5).tolist(),
    }
    if pd.api.types.is_numeric_dtype(df[col]):
        col_analysis[col].update({
            'min': float(df[col].min()), 'max': float(df[col].max()),
            'mean': float(df[col].mean()), 'std': float(df[col].std()),
        })

# Target analysis
target_info = {}
if label_col and label_col in df.columns:
    target_info = {
        'column_name': label_col,
        'unique_values': sorted(df[label_col].unique().tolist()),
        'value_counts': {str(k): int(v) for k, v in df[label_col].value_counts().items()},
        'dtype': str(df[label_col].dtype),
        'n_classes': int(df[label_col].nunique()),
    }

feature_cols   = [c for c in df.columns if c != label_col]
numeric_cols   = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categoric_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

output = {
    # File paths — codegen uses these directly
    'data_files':  data_files,
    'train_path':  train_path,
    'test_path':   test_path,

    # Shape
    'row_count':          int(len(df)),
    'column_count':       int(len(df.columns)),
    'duplicate_row_count': int(df.duplicated().sum()),
    'memory_usage_mb':    round(df.memory_usage(deep=True).sum() / 1024**2, 2),

    # Target
    'label_column':       label_col,
    'target_info':        target_info,
    'target_distribution': target_info.get('value_counts', {}),
    'target_plain_english': (
        f"Target column '{label_col}' has {target_info.get('n_classes', '?')} unique values."
        if label_col else "No target column detected."
    ),

    # Columns
    'all_columns':         list(df.columns),
    'feature_columns':     feature_cols,
    'numeric_columns':     numeric_cols,
    'categorical_columns': categoric_cols,
    'column_analysis':     col_analysis,
    'sample_rows':         df.head(3).to_dict(orient='records'),

    # Legacy fields kept for compatibility
    'feature_stats':  feature_stats if 'feature_stats' in dir() else [],
    'strong_pairs':   strong_pairs  if 'strong_pairs'  in dir() else [],
    'mi_scores':      mi_scores     if 'mi_scores'     in dir() else {},

    # Health
    'health_score': 90.0,
    'health_score_explanation': 'Auto-analyzed by AutoResearch EDA.',
    'features': [],
    'strong_correlations': [],
    'flags': [],
    'recommendations': [
        f"Train file: {train_path}",
        f"Test file: {test_path or 'not found'}",
        f"Target column: {label_col}",
        f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}",
        f"Training rows: {len(df)}",
    ],
    'method_hints': [
        f"Load train data from: {train_path}",
        f"Load test data from: {test_path or 'unknown'}",
        f"Target column: {label_col}",
        f"Features: {feature_cols[:10]}",
        f"All columns in train: {list(df.columns)}",
        f"Dtypes: { {col: str(df[col].dtype) for col in df.columns} }",
    ],
}

os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/eda_output.json', 'w') as f:
    json.dump(output, f, default=str)
print('EDA complete. Results saved to /kaggle/working/eda_output.json')
""",
}


def _build_eda_notebook_static(spec: ProblemSpec) -> nbformat.NotebookNode:
    """
    Fallback: static hardcoded EDA notebook used when the Claude-generated one fails.
    """
    cells = []
    cells.append(new_markdown_cell(
        f"# AutoResearch — Exploratory Data Analysis\n\n"
        f"**Problem:** {spec.plain_english_summary}\n\n"
        f"**Task type:** {spec.task_type.value} | "
        f"**Primary metric:** {spec.primary_metric.value}\n\n"
        f"> This notebook was generated by AutoResearch. "
        f"You can read and modify it — it's yours."
    ))
    target = spec.target_column or ""
    task   = spec.task_type.value
    for name, code in EDA_CELLS.items():
        code = code.replace("{target_col}", target).replace("{task_type}", task)
        cells.append(new_code_cell(code))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


# ── Parsing ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PARSE = """You are the EDA interpreter for AutoResearch.

You receive raw EDA statistics and produce a DataHealthReport — a plain English
analysis written for the researcher, not for another ML engineer.

## Your job

1. Interpret the numbers — don't just repeat them
2. Flag real problems, not noise
3. Write every explanation as if talking to someone who knows their domain
   (medicine, economics, etc.) but not ML jargon
4. Give concrete, ranked recommendations

## Tone examples

Bad:  "Feature 'age' has skewness of 1.34"
Good: "The 'age' feature is right-skewed — most patients are younger,
       with a few much older ones pulling the average up. Tree-based
       methods handle this naturally, but if you use linear models,
       consider a log transform."

Bad:  "Class imbalance detected: 87% negative"
Good: "Your data has a class imbalance — 87% of cases are negative (no disease)
       and only 13% positive. This means a model that always guesses 'negative'
       would be 87% accurate but useless. We'll use AUC instead of accuracy
       to measure performance, and may need to oversample the minority class."

## Output format

Respond with a valid JSON object matching the DataHealthReport schema.
No extra text.
"""


class EDAAgent:
    """
    The EDA Agent.

    Generates an EDA notebook, runs it on Kaggle CPU, parses the output
    into a plain-English DataHealthReport using Claude.
    """

    def __init__(self, kaggle_client: KaggleClient, llm: LLMClient, verbose: bool = True):
        self.kaggle  = kaggle_client
        self.llm     = llm
        self.verbose = verbose

    def _find_eda_output_key(self, files: dict[str, bytes]) -> Optional[str]:
        """Locate eda_output.json even if Kaggle returns nested file paths."""
        if "eda_output.json" in files:
            return "eda_output.json"
        for key in files.keys():
            if str(key).endswith("eda_output.json"):
                return key
        return None

    def build_eda_notebook(
        self,
        spec: ProblemSpec,
        error_context: str = "",
        failed_notebook=None,
    ) -> nbformat.NotebookNode:
        """
        Ask the LLM to write the EDA notebook for this specific problem.
        If error_context is provided, the LLM is shown the previous traceback and asked to fix it.
        If failed_notebook is provided, its full source is included so the LLM can see exactly
        what code was running when the error happened.
        Falls back to the static hardcoded notebook if the API call fails.
        """
        import re as _re

        # Extract source from the failed notebook so the LLM sees the actual code
        failed_code = ""
        if failed_notebook is not None:
            cells = []
            idx = 0
            for cell in failed_notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                idx += 1
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                cells.append(f"# -- Cell {idx} --\n{src.strip()}")
            failed_code = "\n\n".join(cells)

        error_section = ""
        if error_context:
            # failed_code is already budget-fitted by the caller; no extra truncation
            code_block = (
                f"\n## Full notebook code that was running when it failed:\n```python\n{failed_code}\n```\n"
                if failed_code else ""
            )
            error_section = f"""
## Previous attempt failed with this error — fix it:

```
{error_context}
```
{code_block}
Read the full code above carefully, identify the exact line causing the error, and rewrite it.

"""

        try:
            code = self.llm.create(
                system=(
                    "You are an expert data scientist writing a Kaggle EDA notebook. "
                    "Write clean Python code only. No markdown explanation, no JSON, no emoji. "
                    "ASCII only."
                ),
                messages=[{
                    "role": "user",
                    "content": f"""{error_section}Write a complete EDA Python script for this problem:

Task type: {spec.task_type.value}
Problem: {spec.plain_english_summary}
Target column: {spec.target_column or 'auto-detect'}

The script must:
1. Auto-discover all CSV/image files under /kaggle/input/ by walking the directory
2. Load the training data with pandas
3. Print shape, dtypes, missing values, duplicates
4. Analyze the target column -- value counts, class distribution
5. For each feature column: dtype, missing %, unique count, min/max/mean for numerics, top values for categoricals
6. Print 3 sample rows
7. At the end, store the results in a variable called EXACTLY `eda_output` (this name is required) and save it:

import os; os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/eda_output.json', 'w') as f:
    import json; json.dump(eda_output, f, default=str)

The `eda_output` dict must contain ALL of this:

{{
  "train_path": "exact path to train CSV",
  "test_path": "exact path to test CSV or null",
  "row_count": 0,
  "column_count": 0,
  "duplicate_row_count": 0,
  "memory_usage_mb": 0.0,
  "label_column": "name of target column",
  "all_columns": ["col1", "col2"],
  "feature_columns": ["col1"],
  "numeric_columns": ["col1"],
  "categorical_columns": [],
  "column_analysis": {{
    "col1": {{"dtype": "int64", "missing_pct": 0.0, "unique_count": 10, "min": 0, "max": 255, "mean": 127.0, "sample_values": [1,2,3]}}
  }},
  "sample_rows": [{{}}, {{}}, {{}}],
  "target_distribution": {{"class0": 100, "class1": 200}},
  "target_plain_english": "description of target",
  "features": [],
  "strong_correlations": [],
  "flags": [],
  "health_score": 85.0,
  "health_score_explanation": "explanation",
  "recommendations": ["recommendation 1"],
  "method_hints": [
    "train_path: /kaggle/input/...",
    "test_path: /kaggle/input/...",
    "label_column: ...",
    "feature_columns: [...]"
  ]
}}

Print: EDA complete. Results saved to /kaggle/working/eda_output.json

Write the complete Python script now. ASCII only, no emoji, no markdown.
""",
                }],
                max_tokens=4000,
                verbose=self.verbose,
            )
            # Strip markdown fences if the model added them despite instructions
            code = _re.sub(r"```(?:python)?\s*", "", code).replace("```", "").strip()
            # Strip non-ASCII
            code = code.encode("ascii", errors="ignore").decode("ascii")

            # Safety cell: if the LLM code crashed or saved to a wrong variable name,
            # this cell searches all local variables for a dict that looks like eda_output
            # and writes it — guaranteeing eda_output.json always exists.
            safety_save = """\
import json as _json, os as _os, builtins as _builtins

_out_path = '/kaggle/working/eda_output.json'
if not _os.path.exists(_out_path):
    # Try common variable names the LLM might have used
    _candidates = ['eda_output', 'output', 'result', 'results', 'report', 'stats', 'data']
    _saved = False
    for _name in _candidates:
        _val = _builtins.__dict__.get(_name) or globals().get(_name)
        if isinstance(_val, dict) and len(_val) > 2:
            _os.makedirs('/kaggle/working', exist_ok=True)
            with open(_out_path, 'w') as _f:
                _json.dump(_val, _f, default=str)
            print(f'Safety save: wrote eda_output.json from variable {_name!r}')
            _saved = True
            break
    if not _saved:
        # Last-resort fallback: derive a minimal report from df if available.
        _df = globals().get('df')
        if _df is not None:
            try:
                _fallback = {
                    'row_count': int(len(_df)),
                    'column_count': int(len(_df.columns)),
                    'duplicate_row_count': int(_df.duplicated().sum()),
                    'memory_usage_mb': float(_df.memory_usage(deep=True).sum() / (1024**2)),
                    'label_column': globals().get('target_col') or '',
                    'all_columns': [str(c) for c in _df.columns.tolist()],
                    'feature_columns': [str(c) for c in _df.columns.tolist() if str(c) != str(globals().get('target_col') or '')],
                    'numeric_columns': [str(c) for c in _df.select_dtypes(include=['number']).columns.tolist()],
                    'categorical_columns': [str(c) for c in _df.select_dtypes(exclude=['number']).columns.tolist()],
                    'column_analysis': {},
                    'sample_rows': _df.head(3).to_dict(orient='records'),
                    'target_distribution': {},
                    'target_plain_english': 'Fallback EDA output generated after notebook error.',
                    'features': [],
                    'strong_correlations': [],
                    'flags': [],
                    'health_score': 60.0,
                    'health_score_explanation': 'Fallback generated because primary EDA output was missing.',
                    'recommendations': ['Review EDA kernel logs; fallback output was generated.'],
                    'method_hints': ['Use fallback EDA output cautiously and inspect kernel logs.'],
                }
                _os.makedirs('/kaggle/working', exist_ok=True)
                with open(_out_path, 'w') as _f:
                    _json.dump(_fallback, _f, default=str)
                print('Safety save: wrote fallback eda_output.json from df')
            except Exception as _e:
                print(f'WARNING: fallback eda_output generation failed: {_e}')
        else:
            print('WARNING: eda_output.json not found and no suitable variable to save.')
"""
            nb = new_notebook(cells=[
                new_markdown_cell(
                    f"# EDA — {spec.plain_english_summary}\n\n"
                    f"Task: {spec.task_type.value} | Metric: {spec.primary_metric.value}"
                ),
                new_code_cell("import warnings\nwarnings.filterwarnings('ignore')"),
                new_code_cell(code),
                new_code_cell(safety_save),
            ])
            nb.metadata["kernelspec"] = {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
            return nb

        except Exception as e:
            logger.warning(f"Claude EDA notebook generation failed ({e}) — using static fallback")
            return _build_eda_notebook_static(spec)

    async def run(
        self,
        spec: ProblemSpec,
        dataset_sources: list[str],
        competition_sources: list[str],
    ) -> DataHealthReport:
        """
        Full EDA pipeline: generate → push → wait → parse → report.

        Args:
            spec: The ProblemSpec from the Problem Analyst.
            dataset_sources: Kaggle dataset slugs to attach.
            competition_sources: Kaggle competition slugs to attach.

        Returns:
            DataHealthReport with plain English findings.
        """
        if self.verbose:
            print("\n" + "━" * 60)
            print("🔬 EXPLORATORY DATA ANALYSIS")
            print("━" * 60)
            print("\n  AutoResearch is about to look at your data.")
            print("  This runs on Kaggle's free CPU — no GPU quota used.\n")

        # 1. Build EDA notebook — try Claude-generated first, fall back to static
        notebook = self.build_eda_notebook(spec)

        # 2. Push to Kaggle (always CPU for EDA)
        push_result = self.kaggle.push_kernel(
            notebook=notebook,
            kernel_slug_suffix="eda",
            dataset_sources=dataset_sources,
            competition_sources=competition_sources,
            force_cpu=True,
            title="AutoResearch EDA",
        )

        # 3. Wait for completion — catch kernel errors so we can fix & retry
        kernel_exec_error: Optional[str] = None
        try:
            poll_result = await self.kaggle.wait_for_kernel(
                kernel_slug=push_result.kernel_slug,
                method_name="EDA",
            )
        except KernelExecutionError as exc:
            # Kernel ran but crashed — grab log_tail as our error detail
            kernel_exec_error = exc.log_tail or str(exc)

        # 4. Fetch output
        output = self.kaggle.fetch_output(push_result.kernel_slug)
        eda_output_key = self._find_eda_output_key(output.files)
        has_eda_output = eda_output_key is not None

        # If output is present, keep going even if later cells crashed.
        if kernel_exec_error and has_eda_output and self.verbose:
            print("  ⚠️  EDA kernel reported an error, but eda_output.json was found. Proceeding.")

        if not has_eda_output:
            # Prefer log_tail from the execution error; fall back to notebook traceback
            error_detail = kernel_exec_error or self._get_eda_error(push_result.kernel_slug, output)
            if self.verbose:
                print(f"\n  ⚠️  EDA notebook failed. Sending error to the LLM for a fix...")
                if error_detail:
                    print(f"     Error: {error_detail[:300]}")

            # ── Memory: recall similar past fixes and inject into context ──────
            past_fixes = recall(error_detail or "") if error_detail else []
            memory_section = format_for_prompt(past_fixes)
            if past_fixes and self.verbose:
                print(f"  🧠 Memory: found {len(past_fixes)} similar past fix(es) — injecting into prompt")

            # ── Token budget ─────────────────────────────────────────────────
            model_name = getattr(self.llm, "model", "")
            budget = PromptBudget(model=model_name, reserved_output=4000)
            budget.reserve("prompt_boilerplate", "Write a complete EDA Python script" * 20)

            # Extract full notebook code for the budget
            _nb_cells = []
            for _cell in (notebook.get("cells", []) if notebook else []):
                if _cell.get("cell_type") == "code":
                    src = _cell.get("source", "")
                    _nb_cells.append("".join(src) if isinstance(src, list) else src)
            nb_code_full = "\n\n".join(_nb_cells)

            # Build error context, layering: schema (if data error) + traceback
            if error_detail and self._is_data_structure_error(error_detail):
                if self.verbose:
                    print("  🔎 Error looks like a column/label mismatch — "
                          "re-scouting actual data schema before retrying...")
                real_schema = await self._scout_data_schema(
                    dataset_sources=dataset_sources,
                    competition_sources=competition_sources,
                )
                core_error = (
                    f"{error_detail}\n\n"
                    f"## Actual data schema discovered from /kaggle/input/:\n{real_schema}\n\n"
                    "Use the column names and label values above — do NOT guess them."
                )
            else:
                core_error = error_detail or ""

            if core_error:
                fitted = budget.fit([
                    ("error",  core_error,      True),
                    ("code",   nb_code_full,     True),
                    ("memory", memory_section,   False),
                ])
                error_context = (
                    (fitted["memory"] + "\n\n" if fitted["memory"] else "")
                    + fitted["error"]
                )
                fixed_nb = self.build_eda_notebook(
                    spec,
                    error_context=error_context,
                    failed_notebook=notebook,
                )
            else:
                fixed_nb = _build_eda_notebook_static(spec)

            push_result2 = self.kaggle.push_kernel(
                notebook=fixed_nb,
                kernel_slug_suffix="eda-fix",
                dataset_sources=dataset_sources,
                competition_sources=competition_sources,
                force_cpu=True,
                title="AutoResearch EDA (LLM Fix)",
            )
            fix_kernel_error = None
            try:
                await self.kaggle.wait_for_kernel(
                    kernel_slug=push_result2.kernel_slug,
                    method_name="EDA (LLM fix)",
                )
            except KernelExecutionError as exc:
                fix_kernel_error = exc.log_tail or str(exc)
            output = self.kaggle.fetch_output(push_result2.kernel_slug)
            eda_output_key = self._find_eda_output_key(output.files)
            has_eda_output = eda_output_key is not None
            push_result = push_result2  # use fallback URL in error messages

            # ── Memory: record whether the fix worked ──────────────────────────
            if error_detail:
                fix_succeeded = has_eda_output
                if fix_succeeded:
                    # Extract the generated code as the fix to remember
                    _fix_code = error_context or error_detail
                    remember(error_detail, _fix_code, worked=True)
                    if self.verbose:
                        print("  🧠 Memory: fix worked — stored for future runs")
                elif fix_kernel_error:
                    mark_failed(error_detail, error_context or error_detail)
                    if self.verbose:
                        print("  🧠 Memory: fix failed — marked as unsuccessful")

            if not has_eda_output:
                raise RuntimeError(
                    "The EDA notebook failed on Kaggle — even the fallback notebook "
                    "didn't produce eda_output.json. This usually means:\n"
                    "  - The dataset is not attached to the kernel\n"
                    "  - The /kaggle/input directory contains no CSV files\n"
                    "  - There is a Kaggle environment issue\n"
                    f"Inspect the kernel at: {push_result.kernel_url}\n"
                    + (f"Last error: {error_detail}" if error_detail else "")
                )

        eda_output_key = self._find_eda_output_key(output.files)
        if not eda_output_key:
            raise RuntimeError("EDA output file missing after retries: eda_output.json")
        raw_blob = output.files[eda_output_key]
        raw_text = raw_blob.decode(errors="replace") if isinstance(raw_blob, (bytes, bytearray)) else str(raw_blob)
        raw_stats = json.loads(raw_text)

        # 5. Interpret with Claude
        if self.verbose:
            print("\n  🧠 Interpreting what the data is telling us...")

        report = self._interpret_stats(spec, raw_stats)

        if self.verbose:
            self._print_report_summary(report)

        return report

    def _interpret_stats(
        self,
        spec: ProblemSpec,
        raw_stats: dict,
    ) -> DataHealthReport:
        """Use Claude to turn raw EDA numbers into a plain English DataHealthReport."""

        user_prompt = f"""## Problem Context

{spec.plain_english_summary}

Task type: {spec.task_type.value}
Primary metric: {spec.primary_metric.value}
Target column: {spec.target_column or "none (unsupervised)"}

## Raw EDA Statistics

{json.dumps(raw_stats, indent=2)}

## Your task

Produce a DataHealthReport JSON. Write every explanation for a researcher
who knows their domain but is not an ML expert. Reference specific numbers
and explain what they mean in context.

Respond with valid JSON only.
"""
        raw = self.llm.create(
            system=SYSTEM_PROMPT_PARSE,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,
            verbose=self.verbose,
        )
        try:
            import re as _re
            cleaned = _re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            brace_start = cleaned.find("{")
            brace_end   = cleaned.rfind("}")
            if brace_start != -1 and brace_end != -1:
                cleaned = cleaned[brace_start:brace_end + 1]

            data = json.loads(cleaned)

            # Normalize recommendations: Claude returns list of dicts, schema expects list of strings
            if "recommendations" in data and isinstance(data["recommendations"], list):
                fixed = []
                for r in data["recommendations"]:
                    if isinstance(r, dict):
                        text = r.get("action") or r.get("text") or r.get("description") or str(r)
                        fixed.append(text)
                    elif isinstance(r, str):
                        fixed.append(r)
                data["recommendations"] = fixed

            # Same fix for method_hints
            if "method_hints" in data and isinstance(data["method_hints"], list):
                data["method_hints"] = [
                    h.get("hint") or h.get("text") or str(h) if isinstance(h, dict) else h
                    for h in data["method_hints"]
                ]

            # Map alternative field names Claude sometimes uses
            if "row_count" not in data and "num_rows" in data:
                data["row_count"] = data["num_rows"]
            if "column_count" not in data and "num_columns" in data:
                data["column_count"] = data["num_columns"]

            # Fill missing required/optional fields with sensible defaults
            data.setdefault("row_count", 0)
            data.setdefault("column_count", 0)
            data.setdefault("duplicate_row_count", 0)
            data.setdefault("memory_usage_mb", 0.0)
            data.setdefault("target_distribution", {})
            data.setdefault("target_plain_english", data.get("overall_health", "See EDA notebook for details."))
            data.setdefault("features", [])
            data.setdefault("feature_insights", [])
            data.setdefault("strong_correlations", [])
            data.setdefault("flags", [])
            data.setdefault("health_score", 50.0)
            data.setdefault("health_score_explanation", "EDA completed — see Kaggle notebook for full details.")
            data.setdefault("recommendations", [])
            data.setdefault("recommended_preprocessing", [])
            data.setdefault("method_hints", [])

            # Rich EDA context fields — map directly from raw_stats when available
            data.setdefault("train_path",          raw_stats.get("train_path"))
            data.setdefault("test_path",           raw_stats.get("test_path"))
            data.setdefault("label_column",        raw_stats.get("label_column"))
            data.setdefault("all_columns",         raw_stats.get("all_columns", []))
            data.setdefault("feature_columns",     raw_stats.get("feature_columns", []))
            data.setdefault("numeric_columns",     raw_stats.get("numeric_columns", []))
            data.setdefault("categorical_columns", raw_stats.get("categorical_columns", []))
            data.setdefault("column_analysis",     raw_stats.get("column_analysis", {}))
            data.setdefault("sample_rows",         raw_stats.get("sample_rows", []))

            report = DataHealthReport(**data)
            return report
        except Exception as e:
            raise RuntimeError(
                f"AutoResearch couldn't parse the EDA results (this is a bug). "
                f"Technical detail: {e}"
            )

    def _print_report_summary(self, report: DataHealthReport) -> None:
        """Print a mentor-voice summary of key EDA findings."""
        score_emoji = "✅" if report.health_score >= 70 else ("⚠️" if report.health_score >= 50 else "🚨")

        print(f"\n  {score_emoji} Data health score: {report.health_score:.0f}/100")
        print(f"  {report.health_score_explanation}\n")

        if report.flags:
            critical = [(f, s, e) for f, s, e in report.flags if s == Severity.CRITICAL]
            warnings = [(f, s, e) for f, s, e in report.flags if s == Severity.WARNING]

            if critical:
                print("  🚨 Critical issues to address:")
                for _, _, explanation in critical:
                    print(f"     • {explanation}")

            if warnings:
                print("  ⚠️  Things to be aware of:")
                for _, _, explanation in warnings[:3]:  # Show top 3
                    print(f"     • {explanation}")

        if report.method_hints:
            print("\n  💡 This affects method selection:")
            for hint in report.method_hints[:3]:
                print(f"     • {hint}")

        print("\n  Full data report saved to: autoresearch_output/data_report.html")
        print("━" * 60 + "\n")

    # ── Data-structure error detection ─────────────────────────────────────────

    _DATA_ERROR_PATTERNS = [
        # column / key errors
        r"keyerror",
        r"column.*not found",
        r"no column named",
        r"not in (index|columns|dataframe)",
        r"undefined column",
        # label / value errors
        r"unknown (label|class|categor)",
        r"invalid (label|class|value)",
        r"unseen label",
        r"labelencoder.*unknown",
        r"found unknown categories",
        # variable / attribute errors that point to wrong names
        r"attributeerror.*has no attribute",
        r"nameerror.*name '.*' is not defined",
        # pandas shape / dtype mismatches that arise from wrong column guesses
        r"cannot reshape",
        r"shape mismatch",
        r"wrong number of items",
        r"expected.*columns.*got",
    ]

    def _is_data_structure_error(self, error_text: str) -> bool:
        """Return True if the traceback suggests wrong column names or label values."""
        import re
        lower = error_text.lower()
        return any(re.search(p, lower) for p in self._DATA_ERROR_PATTERNS)

    async def _scout_data_schema(
        self,
        dataset_sources: list[str],
        competition_sources: list[str],
    ) -> str:
        """
        Push a tiny notebook that prints every CSV's column names, dtypes,
        and the first 3 unique values of each column, then return that text.
        If the scout kernel fails, returns an empty string (graceful degradation).
        """
        scout_code = """\
import os, json, pandas as pd

result = {}
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if not f.endswith(".csv"):
            continue
        path = os.path.join(root, f)
        try:
            df = pd.read_csv(path, nrows=500)
            result[path] = {
                col: {
                    "dtype": str(df[col].dtype),
                    "sample_values": df[col].dropna().unique()[:5].tolist(),
                    "nunique": int(df[col].nunique()),
                }
                for col in df.columns
            }
        except Exception as e:
            result[path] = {"error": str(e)}

print(json.dumps(result, indent=2, default=str))
with open("/kaggle/working/schema_scout.json", "w") as fh:
    json.dump(result, fh, indent=2, default=str)
"""
        nb = new_notebook(cells=[new_code_cell(scout_code)])
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

        try:
            push = self.kaggle.push_kernel(
                notebook=nb,
                kernel_slug_suffix="schema-scout",
                dataset_sources=dataset_sources,
                competition_sources=competition_sources,
                force_cpu=True,
                title="AutoResearch Schema Scout",
            )
            await self.kaggle.wait_for_kernel(
                kernel_slug=push.kernel_slug,
                method_name="Schema Scout",
            )
            scout_output = self.kaggle.fetch_output(push.kernel_slug)
            if "schema_scout.json" in scout_output.files:
                return scout_output.files["schema_scout.json"].decode()
        except Exception as e:
            logger.warning(f"Schema scout failed ({e}) — proceeding without real schema")

        return ""

    # ── Error extraction ────────────────────────────────────────────────────────

    def _get_eda_error(self, kernel_slug: str, output) -> str:
        """
        Extract the Python traceback from the Kaggle output notebook.

        Tries the files already in `output` first (no extra API call needed).
        Falls back to a subprocess `kaggle kernels output` fetch if nothing
        useful is found inline.  Returns a plain-text string (ANSI stripped),
        or empty string if nothing could be retrieved.
        """
        import re as _re
        import os
        import subprocess
        import tempfile

        def _strip_ansi(s: str) -> str:
            return _re.sub(r"\x1b\[[0-9;]*m", "", s)

        def _errors_from_nb(nb: dict) -> str:
            errors = []
            for cell in nb.get("cells", []):
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        ename   = out.get("ename", "Error")
                        evalue  = out.get("evalue", "")
                        tb_raw  = "\n".join(out.get("traceback", []))
                        tb      = _strip_ansi(tb_raw)
                        errors.append(f"{ename}: {evalue}\n{tb}")
            return "\n\n".join(errors)[-3000:] if errors else ""

        # 1. Check files already attached to the output object
        for fname, content in (output.files or {}).items():
            if fname.endswith(".ipynb"):
                try:
                    nb = json.loads(content) if isinstance(content, str) else content
                    result = _errors_from_nb(nb)
                    if result:
                        return result
                except Exception:
                    pass

        # 2. Fetch via kaggle CLI
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    ["kaggle", "kernels", "output", kernel_slug, "-p", tmpdir],
                    capture_output=True, timeout=120,
                )
                for fname in os.listdir(tmpdir):
                    if not fname.endswith(".ipynb"):
                        continue
                    fpath = os.path.join(tmpdir, fname)
                    try:
                        with open(fpath) as f:
                            nb = json.loads(f.read())
                        result = _errors_from_nb(nb)
                        if result:
                            return result
                    except Exception:
                        pass
        except Exception:
            pass

        return ""
