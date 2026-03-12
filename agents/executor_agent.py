"""
Execution Agent

Takes generated notebooks, pushes them all to Kaggle in parallel,
waits for results, and handles failures by sending tracebacks back
to the Code Generator for automatic fixing (max 2 retries per method).

The retry loop is the key safety net:
    1. Push kernel
    2. Wait for result
    3. If error â†’ read traceback â†’ CodeGen fixes it â†’ push again
    4. Max 2 retries, then flag for researcher

Plain English errors are shown at every step so the researcher
always knows what's happening and why.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import nbformat

from autoresearch.schemas import (
    ProblemSpec,
    DataHealthReport,
    MethodSpec,
    ExecutionResult,
    ExecutionStatus,
    ModelArtifact,
)
from autoresearch.tools.kaggle_client import (
    KaggleClient,
    KernelExecutionError,
    KernelTimeoutError,
)
from autoresearch.agents.codegen_agent import CodeGenAgent
from autoresearch.agents.error_memory import recall, remember, mark_failed, format_for_prompt
from autoresearch.agents.api_utils import PromptBudget

logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # Per method. After this, mark as failed and move on.
MAX_PARALLEL_KERNELS = 4
MAX_GPU_KERNELS = 2
KAGGLE_API_MAX_RETRIES = 3
KAGGLE_API_BACKOFF_SECONDS = 5
KERNEL_TIMEOUT_SECONDS = 7200


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _notebook_source(notebook) -> str:
    """
    Extract all code cell source from a notebook object as a single string,
    with cell numbers for easy reference.
    """
    if notebook is None:
        return ""
    lines = []
    code_idx = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_idx += 1
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines.append(f"# -- Cell {code_idx} ------------------------------")
        lines.append(src.strip())
        lines.append("")
    return "\n".join(lines)


# â”€â”€ Retry prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

FIX_PROMPT_TEMPLATE = """\
The notebook for '{method_name}' failed on Kaggle with this error:

```
{error_message}
```

Last lines of output:
```
{log_tail}
```

What went wrong (plain English): {plain_english_diagnosis}

## Full notebook code that was running when it failed:
```python
{notebook_code}
```

Read the full code above carefully, find the exact line(s) causing the error,
and rewrite them to fix the problem. Common causes:
- Import not available in Kaggle environment (use pip install in a cell)
- Wrong column name (check df.columns)
- Shape mismatch between X and y
- Memory error (reduce batch size or n_estimators)

Return the fixed notebook code sections as JSON (same format as before).
"""


FIX_PROMPT_EXTRA_RULES = """\
Additional strict requirements for the fix:
- Keep all required output variables: primary_metric_value, primary_metric_name, all_metrics, train_metric, val_metric, runtime_seconds, method_name
- Ensure metrics are saved to /kaggle/working/metrics.json
- Ensure a model artifact is saved to /kaggle/working/model.pkl (or equivalent model file)
- If target column may be wrong, infer from dataframe columns and validate before training
- If split logic fails, regenerate train/val split with shape checks before fitting
- Add defensive checks with clear error messages for missing columns and empty features
"""


def diagnose_error(error_message: str, log_tail: str) -> str:
    """
    Produce a plain English diagnosis of what went wrong.
    Used both for the retry prompt and for showing the researcher.
    """
    error_lower = (error_message + log_tail).lower()

    if "modulenotfounderror" in error_lower or "importerror" in error_lower:
        pkg = "a required package"
        for line in log_tail.splitlines():
            if "no module named" in line.lower():
                pkg = line.split("'")[-2] if "'" in line else "a package"
                break
        return f"A Python package ({pkg}) wasn't available in the Kaggle environment. The fix: add a !pip install cell at the top."

    if "memoryerror" in error_lower or "out of memory" in error_lower:
        return "The model ran out of memory. The fix: reduce batch size, number of trees, or dataset size."

    if "keyerror" in error_lower:
        return "A column name wasn't found in the dataset. The fix: check the exact column names with df.columns."

    if "valueerror" in error_lower and "shape" in error_lower:
        return "The input data shape doesn't match what the model expects. Usually a preprocessing issue."

    if "cuda" in error_lower or "gpu" in error_lower:
        return "A GPU/CUDA error occurred. The fix: add error handling or fall back to CPU."

    if "timeout" in error_lower:
        return "The kernel exceeded the time limit. The fix: reduce n_trials, n_estimators, or epochs."

    return "An unexpected error occurred. Check the full traceback above."


def _actionable_log_tail(log_tail: str, limit_lines: int = 120) -> str:
    """
    Keep the most actionable traceback chunks and trim noisy output.
    """
    if not log_tail:
        return ""
    lines = [ln.rstrip() for ln in log_tail.splitlines() if ln.strip()]
    if not lines:
        return ""
    error_lines = []
    capture = False
    for ln in lines:
        low = ln.lower()
        if "traceback (most recent call last)" in low:
            capture = True
        if capture:
            error_lines.append(ln)
        if re.search(r"\b([A-Za-z_][A-Za-z0-9_]*Error|Exception)\b", ln):
            error_lines.append(ln)
    selected = error_lines if error_lines else lines[-limit_lines:]
    return "\n".join(selected[-limit_lines:])


def _notebook_minimum_contract_issues(notebook) -> list[str]:
    """
    Lightweight static checks before pushing a retry notebook.
    """
    code = _notebook_source(notebook)
    issues = []
    required_tokens = [
        "primary_metric_value",
        "primary_metric_name",
        "train_metric",
        "val_metric",
        "metrics.json",
    ]
    for token in required_tokens:
        if token not in code:
            issues.append(f"missing token: {token}")
    if "/kaggle/working" not in code and "OUTPUT_DIR" not in code:
        issues.append("missing output path handling")
    return issues


class ExecutionAgent:
    """
    The Execution Agent.

    Pushes all method notebooks to Kaggle in parallel, polls for completion,
    handles failures with automatic retry, and returns structured results.

    Usage:
        agent = ExecutionAgent(kaggle_client, codegen_agent)
        results = await agent.run_all(spec, report, method_notebook_pairs)
    """

    def __init__(
        self,
        kaggle_client: KaggleClient,
        codegen_agent: CodeGenAgent,
        verbose: bool = True,
    ):
        self.kaggle   = kaggle_client
        self.codegen  = codegen_agent
        self.verbose  = verbose

    def _extract_metrics(self, output, method: MethodSpec) -> dict:
        """
        Robust metrics extraction from KernelOutput, tolerant to schema drift.
        """
        metrics = output.metrics_json or {}
        if metrics:
            return metrics

        # Fallback: inspect any JSON file for likely metric keys.
        for name, blob in (output.files or {}).items():
            if not str(name).lower().endswith(".json"):
                continue
            try:
                text = blob.decode(errors="replace") if isinstance(blob, (bytes, bytearray)) else str(blob)
                obj = json.loads(text)
                if isinstance(obj, dict) and (
                    "primary_value" in obj
                    or "primary_metric" in obj
                    or "all_metrics" in obj
                    or "metrics" in obj
                ):
                    return obj
            except Exception:
                continue
        return {}

    def _normalize_metrics_fields(self, raw: dict, method: MethodSpec) -> tuple[Optional[float], str, dict, Optional[float], Optional[float]]:
        """
        Normalize common metric key variants from different generated notebooks.
        """
        all_metrics = raw.get("all_metrics")
        if all_metrics is None and isinstance(raw.get("metrics"), dict):
            all_metrics = raw.get("metrics")
        if all_metrics is None:
            all_metrics = {}

        primary_name = (
            raw.get("primary_metric")
            or raw.get("primary_metric_name")
            or raw.get("metric_name")
            or method.id
        )
        primary_value = (
            raw.get("primary_value")
            if raw.get("primary_value") is not None else
            raw.get("primary_metric_value")
        )
        if primary_value is None and isinstance(all_metrics, dict) and primary_name in all_metrics:
            primary_value = all_metrics.get(primary_name)
        if primary_value is None and isinstance(all_metrics, dict) and len(all_metrics) == 1:
            primary_name = next(iter(all_metrics.keys()))
            primary_value = all_metrics.get(primary_name)

        train_metric = raw.get("train_metric")
        if train_metric is None:
            train_metric = raw.get("train_score")
        val_metric = raw.get("val_metric")
        if val_metric is None:
            val_metric = raw.get("validation_metric")
        if val_metric is None:
            val_metric = raw.get("val_score")

        try:
            primary_value = float(primary_value) if primary_value is not None else None
        except Exception:
            primary_value = None
        try:
            train_metric = float(train_metric) if train_metric is not None else None
        except Exception:
            train_metric = None
        try:
            val_metric = float(val_metric) if val_metric is not None else None
        except Exception:
            val_metric = None

        # Ensure all_metrics is numeric-friendly dict.
        if isinstance(all_metrics, dict):
            cleaned = {}
            for k, v in all_metrics.items():
                try:
                    cleaned[str(k)] = float(v)
                except Exception:
                    continue
            all_metrics = cleaned
        else:
            all_metrics = {}

        return primary_value, str(primary_name), all_metrics, train_metric, val_metric

    async def run_all(
        self,
        spec: ProblemSpec,
        report: DataHealthReport,
        method_notebooks: list[tuple[MethodSpec, nbformat.NotebookNode]],
        dataset_sources: list[str],
        competition_sources: list[str],
    ) -> list[ExecutionResult]:
        """
        Run all method notebooks in parallel on Kaggle.

        Returns one ExecutionResult per method, win or lose.
        Failed methods are included with status=FAILED - the Evaluator
        will skip them and the researcher is told what happened.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("RUNNING EXPERIMENTS")
            print("=" * 60)
            print(f"\n  Pushing {len(method_notebooks)} experiments to Kaggle...")
            print("  They will run in parallel with controlled concurrency.\n")

        cpu_pool = asyncio.Semaphore(MAX_PARALLEL_KERNELS)
        gpu_pool = asyncio.Semaphore(MAX_GPU_KERNELS)
        total = len(method_notebooks)

        async def run_kernel(
            index: int,
            method: MethodSpec,
            notebook: nbformat.NotebookNode,
        ) -> ExecutionResult:
            pool = gpu_pool if method.requires_gpu else cpu_pool
            async with pool:
                if self.verbose:
                    print(f"  [{index}/{total}] Running {method.name}")

                try:
                    push = await self._push_kernel_with_backoff(
                        notebook=notebook,
                        kernel_slug_suffix=method.id,
                        dataset_sources=dataset_sources,
                        competition_sources=competition_sources,
                        force_cpu=not method.requires_gpu,
                    )
                except Exception as e:
                    logger.error(f"Failed to push {method.name}: {e}")
                    return self._make_failed_result(
                        method,
                        f"Failed to push notebook to Kaggle: {e}",
                        "AutoResearch could not upload the experiment to Kaggle. This is usually a temporary API issue.",
                    )

                try:
                    poll = await self._wait_for_kernel_with_timeout(
                        push.kernel_slug,
                        method_name=method.name,
                    )
                    wait_error_detail = ""
                except Exception as e:
                    logger.warning(f"Kernel wait failed for {method.name}: {e}")
                    poll = None
                    wait_error_detail = str(e)

                if poll is None:
                    return await self._handle_failure_with_retry(
                        spec, report, method, push, notebook,
                        dataset_sources, competition_sources,
                        initial_error_context=wait_error_detail,
                    )
                return await self._collect_result(method, push, poll)

        tasks = [
            asyncio.create_task(run_kernel(i + 1, method, notebook))
            for i, (method, notebook) in enumerate(method_notebooks)
        ]
        results = await asyncio.gather(*tasks)

        if self.verbose:
            self._print_execution_summary(results)

        return results

    async def _api_call_with_backoff(self, call, *args, **kwargs):
        """Run blocking Kaggle API calls with bounded retries and backoff."""
        last_error = None
        for attempt in range(1, KAGGLE_API_MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(call, *args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == KAGGLE_API_MAX_RETRIES:
                    break
                delay = KAGGLE_API_BACKOFF_SECONDS * attempt
                logger.warning(
                    f"Kaggle API call failed (attempt {attempt}/{KAGGLE_API_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s."
                )
                await asyncio.sleep(delay)
        raise last_error

    async def _push_kernel_with_backoff(self, **kwargs):
        """Push kernel with retry/backoff."""
        return await self._api_call_with_backoff(self.kaggle.push_kernel, **kwargs)

    async def _wait_for_kernel_with_timeout(self, kernel_slug: str, method_name: str):
        """Wait for kernel completion with timeout protection."""
        last_error = None
        for attempt in range(1, KAGGLE_API_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.kaggle.wait_for_kernel(kernel_slug, method_name=method_name),
                    timeout=KERNEL_TIMEOUT_SECONDS,
                )
            except Exception as e:
                last_error = e
                if attempt == KAGGLE_API_MAX_RETRIES:
                    break
                delay = KAGGLE_API_BACKOFF_SECONDS * attempt
                logger.warning(
                    f"Kernel wait failed (attempt {attempt}/{KAGGLE_API_MAX_RETRIES}) for "
                    f"{method_name}: {e}. Retrying in {delay}s."
                )
                await asyncio.sleep(delay)
        raise last_error
    async def _collect_result(
        self,
        method: MethodSpec,
        push,
        poll,
    ) -> ExecutionResult:
        """Fetch output and build ExecutionResult from a completed kernel."""
        start = time.time()

        try:
            output = await self._api_call_with_backoff(self.kaggle.fetch_output, push.kernel_slug)
            raw_metrics = self._extract_metrics(output, method)
            primary_value, primary_name, all_metrics, train_metric, val_metric = self._normalize_metrics_fields(
                raw_metrics, method
            )

            # Find model artifact
            model_files = [
                f for f in output.files
                if any(f.endswith(ext) for ext in [".pkl", ".pt", ".h5", ".joblib"])
            ]
            artifact = None
            if model_files:
                artifact = ModelArtifact(
                    kaggle_output_path=f"/kaggle/working/{model_files[0]}",
                    format=model_files[0].split(".")[-1],
                    size_mb=len(output.files[model_files[0]]) / 1e6,
                    inference_script="inference.py",
                    requirements=[],
                )

            return ExecutionResult(
                method_id=method.id,
                method_name=method.name,
                status=ExecutionStatus.SUCCESS,
                kaggle_kernel_url=push.kernel_url,
                runtime_minutes=(time.time() - start) / 60,
                gpu_used=method.requires_gpu,
                primary_metric_value=primary_value,
                primary_metric_name=primary_name,
                all_metrics=all_metrics,
                train_metric=train_metric,
                val_metric=val_metric,
                model_artifact=artifact,
                results_plain_english=self._explain_results(method, {
                    "primary_value": primary_value,
                    "primary_metric": primary_name,
                    "all_metrics": all_metrics,
                    "train_metric": train_metric,
                    "val_metric": val_metric,
                }),
            )

        except Exception as e:
            return self._make_failed_result(
                method,
                str(e),
                f"Couldn't retrieve results for '{method.name}' from Kaggle.",
            )

    async def _handle_failure_with_retry(
        self,
        spec: ProblemSpec,
        report: DataHealthReport,
        method: MethodSpec,
        push,
        notebook,
        dataset_sources: list[str],
        competition_sources: list[str],
        initial_error_context: str = "",
    ) -> ExecutionResult:
        """
        When a kernel fails, read the error, ask CodeGen to fix it,
        and retry up to MAX_RETRIES times.
        """
        retry_changes = []

        kernel_slug = push.kernel_slug

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt == 1 and initial_error_context:
                error_msg = f"Kernel failed (attempt {attempt}): {initial_error_context}"
            else:
                error_msg = f"Kernel failed (attempt {attempt})"
            log_tail  = self._get_kernel_error_log(kernel_slug)
            log_tail  = _actionable_log_tail(log_tail)
            diagnosis = diagnose_error(error_msg, log_tail)

            if self.verbose:
                print(f"\n  âš ï¸  '{method.name}' failed (attempt {attempt}/{MAX_RETRIES})")
                print(f"     {diagnosis}")
                print(f"     AutoResearch is fixing the code and retrying...\n")

            retry_changes.append(f"Attempt {attempt}: {diagnosis}")

            try:
                # â”€â”€ Memory: inject similar past fixes into the prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                full_error = f"{error_msg}\n{log_tail}"
                past_fixes = recall(full_error)
                memory_section = format_for_prompt(past_fixes)
                if past_fixes and self.verbose:
                    print(f"  ðŸ§  Memory: found {len(past_fixes)} similar past fix(es) â€” injecting into prompt")

                # Extract full source of the notebook that just failed
                nb_code = _notebook_source(notebook)

                # â”€â”€ Token budget: allocate context window safely â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                model_name = getattr(self.codegen.llm, "model", "")
                budget = PromptBudget(model=model_name, reserved_output=4000)
                # Reserve for fixed boilerplate in the template (~300 tokens)
                budget.reserve("template", FIX_PROMPT_TEMPLATE[:1000])
                fitted = budget.fit([
                    ("error",    full_error,      True),   # required â€” always include
                    ("code",     nb_code,         True),   # required â€” full notebook
                    ("memory",   memory_section,  False),  # optional â€” drop if tight
                ])

                fix_prompt = FIX_PROMPT_TEMPLATE.format(
                    method_name=method.name,
                    error_message=error_msg,
                    log_tail=fitted["error"][-2000:],
                    plain_english_diagnosis=diagnosis,
                    notebook_code=fitted["code"],
                )
                fix_prompt = fix_prompt + "\n\n" + FIX_PROMPT_EXTRA_RULES
                if fitted["memory"]:
                    fix_prompt = fitted["memory"] + "\n\n" + fix_prompt

                fixed_notebook = self.codegen.generate(
                    spec, report, method, error_context=fix_prompt
                )

                contract_issues = _notebook_minimum_contract_issues(fixed_notebook)
                if contract_issues:
                    logger.warning(
                        f"Generated fix for {method.name} is incomplete: {contract_issues}"
                    )
                    retry_changes.append(
                        "Validation failed before push: " + "; ".join(contract_issues)
                    )
                    mark_failed(full_error, fix_prompt)
                    notebook = fixed_notebook
                    continue

                # Push the fixed notebook
                new_push = await self._push_kernel_with_backoff(
                    notebook=fixed_notebook,
                    kernel_slug_suffix=f"{method.id}-retry{attempt}",
                    dataset_sources=dataset_sources,
                    competition_sources=competition_sources,
                    force_cpu=not method.requires_gpu,
                )

                # Wait for it
                poll = await self._wait_for_kernel_with_timeout(
                    new_push.kernel_slug,
                    method_name=f"{method.name} (retry {attempt})",
                )

                # If it worked, collect and return â€” store fix in memory
                remember(full_error, fix_prompt, worked=True)
                if self.verbose:
                    print(f"  ðŸ§  Memory: fix worked â€” stored for future runs")
                result = await self._collect_result(method, new_push, poll)
                result.retry_count   = attempt
                result.retry_changes = retry_changes
                return result

            except (KernelExecutionError, KernelTimeoutError, asyncio.TimeoutError) as e:
                logger.warning(f"Retry {attempt} also failed for {method.name}: {e}")
                mark_failed(full_error, fix_prompt)
                # Use the fixed notebook as the base for the next retry attempt
                notebook = fixed_notebook
                kernel_slug = new_push.kernel_slug
                continue

            except Exception as e:
                logger.error(f"Unexpected error in retry {attempt} for {method.name}: {e}")
                break

        # All retries exhausted
        if self.verbose:
            print(f"\n  âŒ '{method.name}' failed after {MAX_RETRIES} attempts.")
            print(f"     AutoResearch will continue with the other methods.")
            print(f"     You can inspect the kernel at: {push.kernel_url}\n")

        return ExecutionResult(
            method_id=method.id,
            method_name=method.name,
            status=ExecutionStatus.FAILED,
            kaggle_kernel_url=push.kernel_url,
            retry_count=MAX_RETRIES,
            retry_changes=retry_changes,
            error_message=f"Failed after {MAX_RETRIES} retry attempts",
            error_plain_english=(
                f"'{method.name}' couldn't be completed after {MAX_RETRIES} attempts. "
                f"The most likely cause: {diagnose_error('', retry_changes[-1] if retry_changes else '')}. "
                f"You can inspect the full error at: {push.kernel_url}"
            ),
        )

    def _get_kernel_error_log(self, kernel_slug: str) -> str:
        """
        Fetch the actual error traceback from a failed Kaggle kernel.

        Kaggle kernels output includes the executed notebook (.ipynb) which
        contains the Python traceback inside the cell error outputs.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    ["kaggle", "kernels", "output", kernel_slug, "-p", tmpdir],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
                files = os.listdir(tmpdir)

                # Primary source: the executed output notebook contains the traceback
                # in cells that raised exceptions (cell output type "error")
                for fname in files:
                    if not fname.endswith(".ipynb"):
                        continue
                    fpath = os.path.join(tmpdir, fname)
                    try:
                        import json as _json
                        nb = _json.loads(
                            open(fpath, encoding="utf-8", errors="replace").read()
                        )
                        errors = []
                        for cell in nb.get("cells", []):
                            for output in cell.get("outputs", []):
                                if output.get("output_type") == "error":
                                    ename = output.get("ename", "")
                                    evalue = output.get("evalue", "")
                                    tb = "\n".join(output.get("traceback", []))
                                    # Strip ANSI colour codes
                                    tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                                    errors.append(f"{ename}: {evalue}\n{tb}")
                        if errors:
                            return "\n\n".join(errors)[-3000:]
                    except Exception:
                        pass

                # Fallback: plain text files that contain a traceback
                for fname in ["notebook.log", "output.log", "stderr.txt"]:
                    fpath = os.path.join(tmpdir, fname)
                    if os.path.exists(fpath):
                        return open(fpath, encoding="utf-8", errors="replace").read()[-3000:]

                for fname in files:
                    fpath = os.path.join(tmpdir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    content = open(fpath, encoding="utf-8", errors="replace").read()
                    if "traceback" in content.lower() or "error" in content.lower():
                        return content[-3000:]

                return f"No error log found. Files downloaded: {files}"
        except Exception as e:
            return f"Could not fetch error log: {e}"

    def _make_failed_result(
        self,
        method: MethodSpec,
        error: str,
        plain_english: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            method_id=method.id,
            method_name=method.name,
            status=ExecutionStatus.FAILED,
            error_message=error,
            error_plain_english=plain_english,
        )

    def _explain_results(self, method: MethodSpec, metrics: dict) -> str:
        """Generate a plain English explanation of the results."""
        primary_value = metrics.get("primary_value")
        primary_name  = metrics.get("primary_metric", "primary metric")

        if primary_value is None:
            return "Results could not be parsed."

        # Contextual interpretation by metric type
        interpretations = {
            "auc_roc":    (0.5, 0.7, 0.8, 0.9,  "AUC-ROC scores range from 0.5 (random) to 1.0 (perfect)"),
            "accuracy":   (0.5, 0.7, 0.8, 0.9,  "Accuracy is the % of correct predictions"),
            "f1_macro":   (0.0, 0.6, 0.75, 0.85, "F1 score balances precision and recall (1.0 = perfect)"),
            "rmse":       None,
            "r2":         (0.0, 0.5, 0.7, 0.85, "RÂ² measures how much variance the model explains (1.0 = perfect)"),
        }

        thresholds = interpretations.get(primary_name)
        if thresholds and thresholds is not None:
            low, ok, good, great, description = thresholds
            if primary_value >= great:
                quality = "excellent"
            elif primary_value >= good:
                quality = "good"
            elif primary_value >= ok:
                quality = "moderate"
            else:
                quality = "below baseline â€” worth investigating"

            explanation = (
                f"{method.name} achieved {primary_name.upper()} = {primary_value:.4f} ({quality}). "
                f"{description}. "
            )
        else:
            explanation = f"{method.name} achieved {primary_name} = {primary_value:.4f}. "

        # Overfitting check
        train = metrics.get("train_metric")
        val   = metrics.get("val_metric")
        if train and val:
            gap = abs(train - val)
            if gap > 0.1:
                explanation += f"Note: there's a {gap:.2f} gap between training ({train:.3f}) and validation ({val:.3f}) scores â€” this suggests the model may be overfitting."
            elif gap > 0.05:
                explanation += f"The training/validation gap ({gap:.2f}) is slightly high â€” worth monitoring."
            else:
                explanation += f"The training/validation gap ({gap:.2f}) looks healthy."

        return explanation

    def _print_execution_summary(self, results: list[ExecutionResult]) -> None:
        """Print a clean summary of all execution results."""
        successes = [r for r in results if r.status == ExecutionStatus.SUCCESS]
        failures  = [r for r in results if r.status == ExecutionStatus.FAILED]

        print("\n" + "â”" * 60)
        print("ðŸ“Š EXECUTION SUMMARY")
        print("â”" * 60)
        print(f"\n  {len(successes)}/{len(results)} experiments completed successfully\n")

        for r in successes:
            metric_str = f"{r.primary_metric_name} = {r.primary_metric_value:.4f}" if r.primary_metric_value else "no metrics"
            print(f"  âœ… {r.method_name}: {metric_str}")
            if r.results_plain_english:
                print(f"     {r.results_plain_english[:120]}")

        for r in failures:
            print(f"\n  âŒ {r.method_name}: {r.error_plain_english}")

        print("â”" * 60 + "\n")
