"""
GlobalState — the single source of truth for an AutoResearch run.

Every agent reads from and writes to this via the Orchestrator.
Stored in Redis so runs are resumable if they crash mid-pipeline.

Design principle:
    No agent talks to another directly.
    All communication goes through GlobalState → Orchestrator → next agent.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Annotated
from pydantic import BaseModel, Field

from autoresearch.schemas import (
    ProblemSpec,
    DataHealthReport,
    MethodsCatalog,
    ExecutionResult,
    EvaluationReport,
)


class Stage(str, Enum):
    """The pipeline stages in order."""
    INIT             = "init"
    PROBLEM_ANALYSIS = "problem_analysis"
    EDA              = "eda"
    METHOD_SELECTION = "method_selection"
    CODE_GENERATION  = "code_generation"
    EXECUTION        = "execution"
    EVALUATION       = "evaluation"
    PAPER_WRITING    = "paper_writing"
    COMPLETE         = "complete"
    FAILED           = "failed"


class DataSource(BaseModel):
    """Where the data comes from."""
    type: str           # "kaggle", "gdrive", "huggingface", "csv_upload"
    identifier: str     # competition slug, file ID, dataset name, or file path
    description: str    # Plain English: what is this data?


class AgentError(BaseModel):
    """A captured error from any agent, with retry history."""
    stage: Stage
    attempt: int
    error_type: str
    error_message: str
    plain_english: str  # What went wrong, for non-experts
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TokenUsage(BaseModel):
    """Claude API token usage per agent — so the researcher knows what they spent."""
    problem_analyst:   int = 0
    eda_agent:         int = 0
    method_formulator: int = 0
    codegen_agent:     int = 0
    evaluator:         int = 0
    paper_writer:      int = 0

    @property
    def total(self) -> int:
        return sum([
            self.problem_analyst,
            self.eda_agent,
            self.method_formulator,
            self.codegen_agent,
            self.evaluator,
            self.paper_writer,
        ])

    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate based on claude-sonnet-4 pricing.
        Shown to researcher at end of run so they know what they spent.
        """
        # claude-sonnet-4: ~$3 per million tokens (blended input/output estimate)
        return (self.total / 1_000_000) * 3.0


class GlobalState(BaseModel):
    """
    The complete state of an AutoResearch run.

    Serialized to Redis after every stage completes.
    If the run crashes, it resumes from the last completed stage.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    problem_statement: str

    # ── Data source ────────────────────────────────────────────────────────
    data_source: DataSource

    # ── Pipeline state ─────────────────────────────────────────────────────
    current_stage: Stage = Stage.INIT
    completed_stages: list[Stage] = Field(default_factory=list)

    # ── Agent outputs (populated as the run progresses) ────────────────────
    problem_spec:      Optional[ProblemSpec]      = None
    data_health:       Optional[DataHealthReport] = None
    methods_catalog:   Optional[MethodsCatalog]   = None
    execution_results: list[ExecutionResult]       = Field(default_factory=list)
    evaluation_report: Optional[EvaluationReport] = None

    # Paper sections generated one at a time
    paper_sections: dict[str, str] = Field(
        default_factory=dict,
        description="section_name → generated text. Built up incrementally.",
    )

    # ── Kaggle kernel tracking ─────────────────────────────────────────────
    kaggle_kernel_ids: list[str] = Field(
        default_factory=list,
        description="Pushed kernel slugs — used for status polling.",
    )

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: list[AgentError] = Field(default_factory=list)
    max_retries: int = 3

    # ── Resource usage ─────────────────────────────────────────────────────
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # ── Output paths ──────────────────────────────────────────────────────
    output_dir: str = Field(
        default="./autoresearch_output",
        description="Where the Starter Pack files are saved.",
    )

    def mark_stage_complete(self, stage: Stage) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def has_completed(self, stage: Stage) -> bool:
        return stage in self.completed_stages

    def add_error(self, stage: Stage, attempt: int, error: Exception, plain_english: str) -> None:
        self.errors.append(AgentError(
            stage=stage,
            attempt=attempt,
            error_type=type(error).__name__,
            error_message=str(error),
            plain_english=plain_english,
        ))

    def can_retry(self, stage: Stage) -> bool:
        stage_errors = [e for e in self.errors if e.stage == stage]
        return len(stage_errors) < self.max_retries

    def summary(self) -> str:
        """Plain English run summary for the researcher."""
        lines = [
            f"Run ID: {self.run_id}",
            f"Stages complete: {len(self.completed_stages)}/{len(Stage) - 2}",
            f"Current stage: {self.current_stage.value}",
            f"Errors: {len(self.errors)}",
            f"Est. API cost: ~${self.token_usage.estimated_cost_usd():.2f}",
        ]
        return "\n".join(lines)
