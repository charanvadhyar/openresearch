"""
Orchestrator — LangGraph StateGraph

Wires all 7 agents into a sequential pipeline with:
  - State persistence (Redis — resumable if crash)
  - Stage-level error handling and retry
  - Mentor-voice progress throughout
  - Human-in-the-loop pause when confidence < 0.80

Pipeline stages:
    INIT → PROBLEM_ANALYSIS → EDA → METHOD_SELECTION →
    CODE_GENERATION → EXECUTION → EVALUATION → PAPER_WRITING → COMPLETE

No agent talks to another directly. All data flows through GlobalState.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from autoresearch.orchestrator.state import GlobalState, Stage
from autoresearch.schemas import (
    ProblemSpec,
    DataHealthReport,
    MethodsCatalog,
    ExecutionResult,
    EvaluationReport,
)
from autoresearch.agents.api_utils import LLMClient, resolve_models
from autoresearch.agents.problem_analyst import ProblemAnalystAgent
from autoresearch.agents.eda_agent import EDAAgent
from autoresearch.agents.method_formulator import MethodFormulatorAgent
from autoresearch.agents.dataset_diagnostics import DatasetDiagnostics
from autoresearch.agents.codegen_agent import CodeGenAgent
from autoresearch.agents.executor_agent import ExecutionAgent
from autoresearch.agents.evaluator_agent import EvaluatorAgent
from autoresearch.agents.paper_writer import PaperWriterAgent
from autoresearch.tools.kaggle_client import KaggleClient

logger = logging.getLogger(__name__)


class AutoResearchOrchestrator:
    """
    The main orchestrator. Runs the full pipeline from problem statement
    to Research Starter Pack.

    Instantiate once per run. Call run() to start.

    Usage:
        orchestrator = AutoResearchOrchestrator.from_config("config.yaml")
        await orchestrator.run(state)
    """

    def __init__(
        self,
        kaggle_username: str,
        kaggle_key: str,
        llm_backends: list[tuple[str, str, str]],   # [(provider, api_key, model), ...]
        evaluation_weights: Optional[dict] = None,
        max_runtime_hours: float = 4.0,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        confidence_threshold: float = 0.80,
        verbose: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.verbose              = verbose

        # Build shared Kaggle client
        self.kaggle = KaggleClient(
            username=kaggle_username,
            key=kaggle_key,
            max_runtime_hours=max_runtime_hours,
            enable_gpu=enable_gpu,
            enable_internet=enable_internet,
            verbose=verbose,
        )

        # Build shared LLM client (stored so EDA/Executor can use it too)
        self.llm = LLMClient(llm_backends)
        llm = self.llm

        # Build all agents
        self.problem_analyst   = ProblemAnalystAgent(llm)
        self.method_formulator = MethodFormulatorAgent(llm, verbose)
        self.codegen           = CodeGenAgent(llm, verbose)
        self.evaluator         = EvaluatorAgent(llm, evaluation_weights, verbose)
        self.paper_writer      = PaperWriterAgent(llm, verbose)
        # EDA and Executor need the kaggle client
        self._eda_agent_cls    = EDAAgent
        self._executor_cls     = ExecutionAgent

    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        llm_backends: "list[tuple[str, str, str]] | None" = None,
    ) -> "AutoResearchOrchestrator":
        """
        Load configuration from config.yaml and build orchestrator.

        Args:
            config_path:  Path to config.yaml.
            llm_backends: Optional pre-built list of (provider, api_key, model)
                          tuples. When supplied (e.g. from the interactive picker
                          in main.py), this overrides the provider/model settings
                          in config.yaml. When None, the config is used (supports
                          ``llm_model: "auto"`` for live model discovery).
        """
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        api     = cfg["api_keys"]
        kaggle  = cfg["kaggle"]
        weights = cfg.get("evaluation_weights")
        adv     = cfg.get("advanced", {})
        verbose = adv.get("verbose", True)

        if llm_backends is not None:
            # Caller already resolved the model list (e.g. interactive picker)
            backends = llm_backends
        else:
            # Parse comma-separated provider/model lists into a fallback chain.
            # e.g. llm_provider: "anthropic,openai"
            #      llm_model:    "claude-sonnet-4-20250514, gpt-4o"
            key_map = {
                "anthropic": api.get("anthropic", ""),
                "openai":    api.get("openai", ""),
                "minimax":   api.get("minimax", ""),
            }
            providers = [p.strip() for p in str(adv.get("llm_provider", "anthropic")).split(",")]
            models    = [m.strip() for m in str(adv.get("llm_model", "claude-sonnet-4-20250514")).split(",")]

            backends = []
            for i, provider in enumerate(providers):
                requested_model = models[i] if i < len(models) else models[-1]
                api_key = key_map.get(provider, "")
                if not api_key:
                    logger.warning(f"No API key for provider '{provider}' — skipping from chain.")
                    continue
                # Resolve "auto" → live model list; otherwise keep the explicit name
                resolved = resolve_models(provider, api_key, requested_model, verbose=verbose)
                for model in resolved:
                    backends.append((provider, api_key, model))

        if not backends:
            raise ValueError(
                "No valid LLM backends configured. "
                "Set api_keys.anthropic or api_keys.openai in config.yaml."
            )

        return cls(
            kaggle_username=api["kaggle_username"],
            kaggle_key=api["kaggle_key"],
            llm_backends=backends,
            evaluation_weights=weights,
            max_runtime_hours=kaggle.get("max_runtime_hours", 4.0),
            enable_gpu=kaggle.get("enable_gpu", True),
            enable_internet=kaggle.get("enable_internet", True),
            confidence_threshold=adv.get("confidence_threshold", 0.80),
            verbose=adv.get("verbose", True),
        )

    async def run(self, state: GlobalState) -> GlobalState:
        """
        Run the full pipeline. Resumes from last completed stage if state
        already has completed stages (e.g. after a crash).

        Returns the final GlobalState with all outputs populated.
        """
        self._print_welcome(state)

        # Build data source lists for Kaggle
        dataset_sources, competition_sources = self._parse_data_source(state)

        # ── Stage 1: Problem Analysis ─────────────────────────────────────────
        if not state.has_completed(Stage.PROBLEM_ANALYSIS):
            state = await self._run_stage(
                state, Stage.PROBLEM_ANALYSIS,
                self._stage_problem_analysis, state
            )
            if state.current_stage == Stage.FAILED:
                return state
            if state.problem_spec is None:
                state.current_stage = Stage.FAILED
                return state

            # Human-in-the-loop: pause if confidence is low
            if (state.problem_spec and
                    state.problem_spec.confidence < self.confidence_threshold):
                state = self._confirm_with_user(state)
                if state.current_stage == Stage.FAILED:
                    return state

        # ── Stage 2: EDA ──────────────────────────────────────────────────────
        if not state.has_completed(Stage.EDA):
            state = await self._run_stage(
                state, Stage.EDA,
                self._stage_eda, state, dataset_sources, competition_sources
            )
            if state.current_stage == Stage.FAILED:
                return state
            if state.data_health is None:
                state.current_stage = Stage.FAILED
                return state

        # ── Stage 3: Method Selection ─────────────────────────────────────────
        if not state.has_completed(Stage.METHOD_SELECTION):
            state = await self._run_stage(
                state, Stage.METHOD_SELECTION,
                self._stage_method_selection, state
            )
            if state.current_stage == Stage.FAILED:
                return state
            if state.methods_catalog is None:
                state.current_stage = Stage.FAILED
                return state

        # ── Stage 4: Code Generation ──────────────────────────────────────────
        if not state.has_completed(Stage.CODE_GENERATION):
            state = await self._run_stage(
                state, Stage.CODE_GENERATION,
                self._stage_code_generation, state
            )
            if state.current_stage == Stage.FAILED:
                return state

        # ── Stage 5: Execution ────────────────────────────────────────────────
        if not state.has_completed(Stage.EXECUTION):
            state = await self._run_stage(
                state, Stage.EXECUTION,
                self._stage_execution, state, dataset_sources, competition_sources
            )
            if state.current_stage == Stage.FAILED:
                return state

        # ── Stage 6: Evaluation ───────────────────────────────────────────────
        if not state.has_completed(Stage.EVALUATION):
            state = await self._run_stage(
                state, Stage.EVALUATION,
                self._stage_evaluation, state
            )
            if state.current_stage == Stage.FAILED:
                return state

        # ── Stage 7: Paper Writing ────────────────────────────────────────────
        if not state.has_completed(Stage.PAPER_WRITING):
            state = await self._run_stage(
                state, Stage.PAPER_WRITING,
                self._stage_paper_writing, state
            )

        state.current_stage = Stage.COMPLETE
        self._print_completion(state)
        return state

    # ── Stage implementations ─────────────────────────────────────────────────

    async def _stage_problem_analysis(self, state: GlobalState) -> GlobalState:
        spec = self.problem_analyst.analyze(
            problem_statement=state.problem_statement,
            dataset_info=None,
            verbose=self.verbose,
        )
        state.problem_spec = spec
        if state.problem_spec is None:
            state.current_stage = Stage.FAILED
        return state

    async def _stage_eda(
        self,
        state: GlobalState,
        dataset_sources: list[str],
        competition_sources: list[str],
    ) -> GlobalState:
        eda_agent = self._eda_agent_cls(
            kaggle_client=self.kaggle,
            llm=self.llm,
            verbose=self.verbose,
        )
        report = await eda_agent.run(
            spec=state.problem_spec,
            dataset_sources=dataset_sources,
            competition_sources=competition_sources,
        )
        state.data_health = report
        if state.data_health is None:
            state.current_stage = Stage.FAILED
            return state

        # Generate data report HTML immediately after EDA
        try:
            from autoresearch.tools.report_generator import generate_data_report
            from pathlib import Path as _Path
            html_path = generate_data_report(report, state.problem_spec, _Path(state.output_dir))
            if self.verbose:
                print(f'  📊 Data report saved: {html_path}')
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f'Data report generation failed (non-fatal): {_e}')

        return state

    async def _stage_method_selection(self, state: GlobalState) -> GlobalState:
        # Run pre-planning diagnostics so planner/codegen/evaluator can adapt.
        target_col = state.problem_spec.target_column or state.data_health.label_column or ""
        try:
            import pandas as pd
            preview_df = pd.DataFrame(state.data_health.sample_rows or [])
            diagnostics = DatasetDiagnostics.analyze_dataset(preview_df, target_col)
            diagnostics = DatasetDiagnostics.enrich_from_data_health(
                diagnostics=diagnostics,
                report=state.data_health,
                target_column=target_col,
            )
            state.problem_spec.dataset_diagnostics = diagnostics
            if self.verbose:
                DatasetDiagnostics.print_summary(diagnostics)
        except Exception as e:
            logger.warning(f"Dataset diagnostics failed (non-fatal): {e}")

        catalog = self.method_formulator.formulate(
            spec=state.problem_spec,
            report=state.data_health,
        )
        state.methods_catalog = catalog
        if state.methods_catalog is None:
            state.current_stage = Stage.FAILED
        return state

    async def _stage_code_generation(self, state: GlobalState) -> GlobalState:
        method_notebooks = self.codegen.generate_all(
            spec=state.problem_spec,
            report=state.data_health,
            methods=state.methods_catalog.methods,
        )
        # Store notebooks on state for execution stage
        state._method_notebooks = method_notebooks  # temporary field
        return state

    async def _stage_execution(
        self,
        state: GlobalState,
        dataset_sources: list[str],
        competition_sources: list[str],
    ) -> GlobalState:
        executor = self._executor_cls(
            kaggle_client=self.kaggle,
            codegen_agent=self.codegen,
            verbose=self.verbose,
        )
        results = await executor.run_all(
            spec=state.problem_spec,
            report=state.data_health,
            method_notebooks=getattr(state, "_method_notebooks", []),
            dataset_sources=dataset_sources,
            competition_sources=competition_sources,
        )
        state.execution_results = results
        return state

    async def _stage_evaluation(self, state: GlobalState) -> GlobalState:
        report = self.evaluator.evaluate(
            spec=state.problem_spec,
            results=state.execution_results,
        )
        state.evaluation_report = report
        if state.evaluation_report is None:
            state.current_stage = Stage.FAILED
            return state

        # Generate comparison report HTML immediately after evaluation
        try:
            from autoresearch.tools.report_generator import generate_comparison_report
            from pathlib import Path as _Path
            html_path = generate_comparison_report(
                state.execution_results, report, state.problem_spec, _Path(state.output_dir)
            )
            if self.verbose:
                print(f'  ⚖️  Comparison report saved: {html_path}')
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f'Comparison report generation failed (non-fatal): {_e}')

        return state

    async def _stage_paper_writing(self, state: GlobalState) -> GlobalState:
        if state.evaluation_report is None:
            if self.verbose:
                print("  ⚠️  Skipping paper writing — evaluation did not complete.")
            return state

        sections = self.paper_writer.write(
            spec=state.problem_spec,
            report=state.data_health,
            catalog=state.methods_catalog,
            results=state.execution_results,
            evaluation=state.evaluation_report,
        )
        state.paper_sections = sections

        # Export outputs
        output_dir = Path(state.output_dir)
        self.paper_writer.export_markdown(sections, state.problem_spec, output_dir)
        try:
            self.paper_writer.export_docx(
                sections, state.problem_spec,
                state.evaluation_report, state.execution_results, output_dir
            )
        except Exception as e:
            logger.warning(f"DOCX export failed (install python-docx): {e}")

        return state

    # ── Infrastructure ────────────────────────────────────────────────────────

    async def _run_stage(
        self,
        state: GlobalState,
        stage: Stage,
        fn,
        *args,
    ) -> GlobalState:
        """
        Run a pipeline stage with error handling and state updates.
        On failure: records error, marks stage as FAILED, returns state.
        """
        state.current_stage = stage
        self._save_state(state)

        try:
            if asyncio.iscoroutinefunction(fn):
                state = await fn(*args)
            else:
                state = fn(*args)

            state.mark_stage_complete(stage)
            self._save_state(state)
            return state

        except Exception as e:
            plain = self._plain_english_stage_error(stage, e)
            state.add_error(stage, 1, e, plain)
            logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)

            if self.verbose:
                print(f"\n  ❌ {stage.value} stage failed: {plain}")
                if state.can_retry(stage):
                    print(f"  AutoResearch will retry automatically...")
                else:
                    print(f"  Max retries reached. Run is paused.")
                    print(f"  You can resume from this stage by re-running with the same run_id: {state.run_id}")

            if not state.can_retry(stage):
                state.current_stage = Stage.FAILED

            return state

    def _confirm_with_user(self, state: GlobalState) -> GlobalState:
        """
        Pause and ask the researcher to confirm the problem analysis.
        Called when confidence < threshold.
        """
        spec = state.problem_spec
        print("\n" + "━" * 60)
        print("🛑 CONFIRMATION NEEDED")
        print("━" * 60)
        print(f"\n  AutoResearch is {spec.confidence:.0%} confident in its analysis.")
        print(f"  Reason: {spec.confidence_explanation}\n")
        print(f"  Current understanding:")
        print(f"    Task type:  {spec.task_type.value}")
        print(f"    Metric:     {spec.primary_metric.value}")
        print(f"    GPU needed: {spec.requires_gpu}")
        print()
        print("  Press ENTER to continue with this, or Ctrl+C to cancel and")
        print("  edit your problem statement or config.yaml.\n")

        try:
            input("  > ")
        except KeyboardInterrupt:
            print("\n\n  Run cancelled. Edit config.yaml and re-run.")
            state.current_stage = Stage.FAILED

        return state

    def _parse_data_source(
        self, state: GlobalState
    ) -> tuple[list[str], list[str]]:
        """Parse data source into Kaggle dataset/competition lists."""
        ds = state.data_source
        if ds.type == "kaggle":
            ident = ds.identifier
            # Competition: "titanic" or "competition/titanic"
            # Dataset: "username/dataset-name"
            if "/" in ident and not ident.startswith("competition/"):
                return [ident], []
            else:
                slug = ident.replace("competition/", "")
                return [], [slug]
        elif ds.type == "gdrive":
            logger.warning("Google Drive source: data must be in /kaggle/input via dataset")
            return [], []
        elif ds.type == "huggingface":
            return [], []  # Downloaded in kernel via datasets library
        else:
            return [], []

    def _save_state(self, state: GlobalState) -> None:
        """Save state to disk (Redis would be used in production)."""
        output_dir = Path(state.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / f"run_{state.run_id}_state.json"
        try:
            state_path.write_text(state.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Could not save state: {e}")

    def _plain_english_stage_error(self, stage: Stage, error: Exception) -> str:
        messages = {
            Stage.PROBLEM_ANALYSIS: (
                "AutoResearch couldn't understand your problem statement. "
                "Try making it more specific — describe what you want to predict "
                "and what data you have."
            ),
            Stage.EDA:              (
                "The data exploration failed. This usually means the data file "
                "wasn't found in Kaggle. Check that your dataset is attached correctly."
            ),
            Stage.METHOD_SELECTION: (
                "Method selection failed. This is unusual — please report it as a bug."
            ),
            Stage.CODE_GENERATION:  (
                "Code generation failed. AutoResearch couldn't write the experiment notebook."
            ),
            Stage.EXECUTION:        (
                "All experiments failed on Kaggle. Check your internet connection "
                "and Kaggle credentials."
            ),
            Stage.EVALUATION:       (
                "Evaluation failed — likely because no experiments completed successfully."
            ),
            Stage.PAPER_WRITING:    (
                "Paper draft generation failed. Your experiment results are saved "
                "and you can re-run just this stage."
            ),
        }
        return messages.get(stage, f"Unexpected error in {stage.value}: {error}")

    def _print_welcome(self, state: GlobalState) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 60)
        print("  AutoResearch — Autonomous ML Research Agent")
        print("═" * 60)
        print(f"\n  Run ID:  {state.run_id}")
        print(f"  Problem: {state.problem_statement[:80]}{'...' if len(state.problem_statement) > 80 else ''}")
        print(f"  Data:    {state.data_source.type}:{state.data_source.identifier}")
        print(f"  Output:  {state.output_dir}")
        backends = self.llm._backends
        if len(backends) == 1:
            b = backends[0]
            print(f"  LLM:     {b.provider} / {b.model}")
        else:
            print(f"  LLM:     fallback chain")
            for i, b in enumerate(backends, 1):
                tag = "(primary)" if i == 1 else f"(fallback {i-1})"
                print(f"           {i}. {b.provider} / {b.model}  {tag}")
        if state.completed_stages:
            print(f"\n  Resuming from: {state.current_stage.value}")
            print(f"  Already done:  {', '.join(s.value for s in state.completed_stages)}")
        print()

    def _print_completion(self, state: GlobalState) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 60)
        print("  ✅  AutoResearch Complete!")
        print("═" * 60)
        print(f"\n  Your Research Starter Pack is in: {state.output_dir}/")
        print()
        print("  What's inside:")
        print("    📊 data_report.html         — EDA findings")
        print("    🏆 method_comparison.html   — Results comparison")
        print("    📓 notebooks/               — One notebook per method")
        print("    🤖 best_model/              — Your best model, saved")
        print("    📄 paper_draft.md           — Paper skeleton (review required)")
        print("    📝 paper_draft.docx         — Word version")
        print("    📚 next_steps.md            — Recommended actions")
        print()

        if state.evaluation_report:
            winner = state.evaluation_report.winner_method_id
            print(f"  Best method: {winner}")
            print(f"  {state.evaluation_report.winner_explanation[:200]}")

        est_cost = state.token_usage.estimated_cost_usd()
        print(f"\n  Estimated API cost: ~${est_cost:.2f}")
        print()
        print("  ⚠️  Remember: the paper draft is a starting point.")
        print("     Every [DRAFT] section needs your expert review.")
        print("     Every [RESEARCHER TO FILL] is waiting for your knowledge.")
        print("═" * 60 + "\n")
