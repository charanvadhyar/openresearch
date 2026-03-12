"""
AutoResearch Test Suite

Tests that don't need network access or API keys:
  - Schema validation (Pydantic contracts)
  - Evaluator scoring logic
  - Report generation (HTML output)
  - Knowledge base integrity
  - Data source resolver routing

Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Make sure autoresearch is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.schemas import (
    ProblemSpec,
    TaskType,
    EvalMetric,
    Constraint,
    DataHealthReport,
    FeatureInsight,
    DataQualityFlag,
    CorrelationPair,
    Severity,
    MethodSpec,
    MethodsCatalog,
    ComplexityLevel,
    ExecutionResult,
    ExecutionStatus,
    EvaluationReport,
    MethodScore,
    RiskFlag,
)
from autoresearch.orchestrator.state import GlobalState, Stage, DataSource, TokenUsage


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def heart_disease_spec():
    """A realistic classification ProblemSpec for testing."""
    return ProblemSpec(
        task_type=TaskType.CLASSIFICATION,
        domain="medical diagnosis",
        primary_metric=EvalMetric.AUC_ROC,
        secondary_metrics=[EvalMetric.F1_WEIGHTED],
        target_column="diagnosis",
        input_description="Tabular data with 14 cardiac features",
        estimated_row_count=303,
        has_class_imbalance=False,
        constraints=[
            Constraint(
                name="interpretability",
                description="Doctors need to understand predictions",
                is_hard=True,
            )
        ],
        confidence=0.95,
        confidence_explanation="Clear binary classification with named target column.",
        requires_gpu=False,
        plain_english_summary=(
            "You want to predict whether a patient has heart disease. "
            "This is a binary classification problem. "
            "Interpretability is required for clinical use."
        ),
    )


@pytest.fixture
def sample_data_health(heart_disease_spec):
    """A minimal DataHealthReport."""
    return DataHealthReport(
        row_count=303,
        column_count=14,
        duplicate_row_count=0,
        memory_usage_mb=0.04,
        target_distribution={"0": 0.46, "1": 0.54},
        target_plain_english="Nearly balanced: 54% positive, 46% negative.",
        features=[
            FeatureInsight(
                name="age",
                dtype="float64",
                missing_pct=0.0,
                unique_count=41,
                mean=54.4,
                std=9.0,
                min=29.0,
                max=77.0,
                skewness=0.23,
                flags=[],
                insight="Age ranges 29–77 with mean 54. Slightly right-skewed but fine for tree methods.",
                usefulness_estimate="likely useful",
            ),
            FeatureInsight(
                name="cholesterol",
                dtype="float64",
                missing_pct=0.0,
                unique_count=152,
                mean=246.3,
                std=51.8,
                min=126.0,
                max=564.0,
                skewness=1.1,
                flags=[DataQualityFlag.OUTLIERS_DETECTED],
                insight="Cholesterol has outliers (max 564 vs mean 246). Tree methods handle this naturally.",
                usefulness_estimate="likely useful",
            ),
        ],
        strong_correlations=[
            CorrelationPair(
                feature_a="thalach",
                feature_b="age",
                correlation=-0.72,
                plain_english="Max heart rate decreases with age (r=-0.72) — expected clinically.",
            )
        ],
        flags=[
            (DataQualityFlag.OUTLIERS_DETECTED, Severity.WARNING,
             "Outliers detected in cholesterol column. Tree-based methods handle this robustly."),
        ],
        health_score=82.0,
        health_score_explanation="Clean dataset with minimal missing values. One outlier concern.",
        recommendations=[
            "Consider capping cholesterol at 99th percentile if using linear models.",
        ],
        method_hints=[
            "Dataset is small (303 rows) — avoid deep learning.",
            "Strong outliers in cholesterol — tree methods preferred over linear.",
        ],
    )


@pytest.fixture
def xgboost_result():
    return ExecutionResult(
        method_id="xgboost",
        method_name="XGBoost",
        status=ExecutionStatus.SUCCESS,
        kaggle_kernel_url="https://kaggle.com/code/user/autoresearch-xgboost",
        runtime_minutes=3.2,
        gpu_used=False,
        primary_metric_value=0.847,
        primary_metric_name="auc_roc",
        all_metrics={"auc_roc": 0.847, "f1_weighted": 0.810},
        train_metric=0.891,
        val_metric=0.847,
        results_plain_english="XGBoost achieved AUC = 0.847 (good). Train/val gap 0.044 — healthy.",
    )


@pytest.fixture
def logreg_result():
    return ExecutionResult(
        method_id="logistic_regression",
        method_name="Logistic Regression",
        status=ExecutionStatus.SUCCESS,
        kaggle_kernel_url="https://kaggle.com/code/user/autoresearch-logreg",
        runtime_minutes=0.4,
        gpu_used=False,
        primary_metric_value=0.821,
        primary_metric_name="auc_roc",
        all_metrics={"auc_roc": 0.821, "f1_weighted": 0.790},
        train_metric=0.835,
        val_metric=0.821,
        results_plain_english="Logistic Regression achieved AUC = 0.821. Very healthy train/val gap.",
    )


@pytest.fixture
def failed_result():
    return ExecutionResult(
        method_id="tabnet",
        method_name="TabNet",
        status=ExecutionStatus.FAILED,
        error_message="CUDA out of memory",
        error_plain_english="TabNet ran out of GPU memory. Try reducing batch size.",
        retry_count=2,
    )


# ── Schema Tests ──────────────────────────────────────────────────────────────

class TestProblemSpec:
    def test_valid_spec(self, heart_disease_spec):
        assert heart_disease_spec.task_type == TaskType.CLASSIFICATION
        assert heart_disease_spec.confidence == 0.95
        assert heart_disease_spec.requires_gpu is False

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ProblemSpec(
                task_type=TaskType.CLASSIFICATION,
                domain="test",
                primary_metric=EvalMetric.AUC_ROC,
                target_column="y",
                input_description="test data",
                confidence=1.5,  # Invalid — must be ≤ 1.0
                confidence_explanation="too high",
                requires_gpu=False,
                plain_english_summary="test",
            )

    def test_low_confidence_flag(self, heart_disease_spec):
        spec = heart_disease_spec.model_copy(update={"confidence": 0.65})
        assert spec.confidence < 0.80  # Should trigger confirmation

    def test_constraint_hard_flag(self, heart_disease_spec):
        hard = [c for c in heart_disease_spec.constraints if c.is_hard]
        assert len(hard) == 1
        assert hard[0].name == "interpretability"

    def test_task_types_complete(self):
        types = [t.value for t in TaskType]
        assert "classification"   in types
        assert "regression"       in types
        assert "nlp"              in types
        assert "computer_vision"  in types
        assert "time_series"      in types
        assert "clustering"       in types


class TestDataHealthReport:
    def test_health_score_bounds(self, sample_data_health):
        assert 0 <= sample_data_health.health_score <= 100

    def test_flags_structure(self, sample_data_health):
        for flag, severity, explanation in sample_data_health.flags:
            assert isinstance(flag, DataQualityFlag)
            assert isinstance(severity, Severity)
            assert len(explanation) > 0

    def test_method_hints_present(self, sample_data_health):
        assert len(sample_data_health.method_hints) > 0
        # Should mention dataset size for a 303-row dataset
        hint_text = " ".join(sample_data_health.method_hints).lower()
        assert "303" in hint_text or "small" in hint_text


class TestExecutionResult:
    def test_success_has_metrics(self, xgboost_result):
        assert xgboost_result.status == ExecutionStatus.SUCCESS
        assert xgboost_result.primary_metric_value is not None
        assert xgboost_result.train_metric is not None
        assert xgboost_result.val_metric is not None

    def test_failed_has_explanation(self, failed_result):
        assert failed_result.status == ExecutionStatus.FAILED
        assert failed_result.error_plain_english is not None
        assert len(failed_result.error_plain_english) > 0

    def test_overfitting_detectable(self, xgboost_result):
        gap = abs(xgboost_result.train_metric - xgboost_result.val_metric)
        assert gap < 0.05  # XGBoost fixture has healthy gap


# ── Evaluator Logic Tests (no API calls) ─────────────────────────────────────

class TestEvaluatorLogic:
    """Test the scoring logic without calling Claude."""

    def test_normalize_scores_equal(self):
        from autoresearch.agents.evaluator_agent import normalize_scores
        result = normalize_scores([0.5, 0.5, 0.5])
        assert all(v == 1.0 for v in result)

    def test_normalize_scores_higher_better(self):
        from autoresearch.agents.evaluator_agent import normalize_scores
        result = normalize_scores([0.6, 0.8, 1.0], higher_is_better=True)
        assert result[2] > result[1] > result[0]
        assert result[2] == pytest.approx(1.0)
        assert result[0] == pytest.approx(0.0)

    def test_normalize_scores_lower_better(self):
        from autoresearch.agents.evaluator_agent import normalize_scores
        result = normalize_scores([10, 20, 30], higher_is_better=False)
        assert result[0] > result[1] > result[2]  # Lower time → higher score

    def test_interpretability_scores_ordered(self):
        from autoresearch.agents.evaluator_agent import get_interpretability_score
        linear_score  = get_interpretability_score("Linear")
        gb_score      = get_interpretability_score("Gradient Boosting")
        nn_score      = get_interpretability_score("Transformer")
        assert linear_score > gb_score > nn_score

    def test_interpretability_unknown_family(self):
        from autoresearch.agents.evaluator_agent import get_interpretability_score
        score = get_interpretability_score("SomeUnknownAlgorithm")
        assert score == 0.5  # Neutral default

    def test_risk_flag_overfitting(self):
        from autoresearch.agents.evaluator_agent import EvaluatorAgent
        # Create a result with extreme overfitting
        overfit_result = ExecutionResult(
            method_id="xgboost",
            method_name="XGBoost",
            status=ExecutionStatus.SUCCESS,
            primary_metric_value=0.85,
            primary_metric_name="auc_roc",
            all_metrics={},
            train_metric=0.98,  # Big gap!
            val_metric=0.72,
        )
        # _detect_risks doesn't call Claude — safe to test directly
        agent = object.__new__(EvaluatorAgent)
        flags = agent._detect_risks([overfit_result])
        critical_flags = [f for f in flags if f.flag_type == "overfitting" and f.severity == "critical"]
        assert len(critical_flags) == 1

    def test_risk_flag_suspicious_score(self):
        from autoresearch.agents.evaluator_agent import EvaluatorAgent
        perfect_result = ExecutionResult(
            method_id="model",
            method_name="Suspicious Model",
            status=ExecutionStatus.SUCCESS,
            primary_metric_value=0.999,
            primary_metric_name="auc_roc",
            all_metrics={},
        )
        agent = object.__new__(EvaluatorAgent)
        flags = agent._detect_risks([perfect_result])
        suspicious = [f for f in flags if f.flag_type == "suspicious_score"]
        assert len(suspicious) == 1
        assert suspicious[0].severity == "critical"

    def test_weight_validation(self):
        from autoresearch.agents.evaluator_agent import EvaluatorAgent
        # Weights that don't sum to 1.0 should be auto-normalized
        agent = object.__new__(EvaluatorAgent)
        agent.weights = {"performance": 2.0, "speed": 2.0, "interpretability": 2.0, "robustness": 2.0}
        agent._validate_weights()
        assert abs(sum(agent.weights.values()) - 1.0) < 0.01


# ── Knowledge Base Tests ──────────────────────────────────────────────────────

class TestKnowledgeBase:
    KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "ml_methods.json"

    def test_kb_exists(self):
        assert self.KB_PATH.exists(), "knowledge_base/ml_methods.json not found"

    def test_kb_valid_json(self):
        data = json.loads(self.KB_PATH.read_text())
        assert isinstance(data, dict)

    def test_all_task_types_present(self):
        data = json.loads(self.KB_PATH.read_text())
        required = ["classification", "regression", "nlp", "computer_vision", "time_series", "clustering"]
        for task in required:
            assert task in data, f"Missing task type in knowledge base: {task}"

    def test_all_methods_have_required_fields(self):
        data = json.loads(self.KB_PATH.read_text())
        required_fields = ["id", "name", "family", "requires_gpu", "min_rows",
                           "strengths", "weaknesses", "good_when", "bad_when",
                           "hyperparam_space", "feature_engineering", "complexity"]
        for task, methods in data.items():
            for m in methods:
                for field in required_fields:
                    assert field in m, f"Method '{m.get('id', '?')}' in {task} missing field: {field}"

    def test_classification_has_interpretable_baseline(self):
        data = json.loads(self.KB_PATH.read_text())
        clf_methods = data["classification"]
        interpretable = [m for m in clf_methods if m["family"] == "Linear"]
        assert len(interpretable) >= 1, "Need at least one interpretable baseline for classification"

    def test_gpu_methods_flagged(self):
        data = json.loads(self.KB_PATH.read_text())
        # Deep learning methods should require GPU
        for task, methods in data.items():
            for m in methods:
                if "neural" in m["family"].lower() or "transformer" in m["family"].lower():
                    assert m["requires_gpu"] is True, \
                        f"Neural method '{m['id']}' should have requires_gpu=True"


# ── State Tests ───────────────────────────────────────────────────────────────

class TestGlobalState:
    def test_stage_completion(self):
        state = GlobalState(
            problem_statement="test",
            data_source=DataSource(type="kaggle", identifier="test/data", description="test"),
        )
        assert not state.has_completed(Stage.EDA)
        state.mark_stage_complete(Stage.EDA)
        assert state.has_completed(Stage.EDA)

    def test_no_duplicate_completed_stages(self):
        state = GlobalState(
            problem_statement="test",
            data_source=DataSource(type="kaggle", identifier="test/data", description="test"),
        )
        state.mark_stage_complete(Stage.EDA)
        state.mark_stage_complete(Stage.EDA)
        assert state.completed_stages.count(Stage.EDA) == 1

    def test_retry_logic(self):
        state = GlobalState(
            problem_statement="test",
            data_source=DataSource(type="kaggle", identifier="test/data", description="test"),
            max_retries=2,
        )
        assert state.can_retry(Stage.EDA)
        state.add_error(Stage.EDA, 1, Exception("fail1"), "First failure")
        assert state.can_retry(Stage.EDA)
        state.add_error(Stage.EDA, 2, Exception("fail2"), "Second failure")
        assert not state.can_retry(Stage.EDA)

    def test_token_usage_cost_estimate(self):
        usage = TokenUsage(
            problem_analyst=5000,
            eda_agent=8000,
            method_formulator=6000,
            codegen_agent=15000,
            evaluator=5000,
            paper_writer=20000,
        )
        assert usage.total == 59000
        cost = usage.estimated_cost_usd()
        assert cost > 0
        assert cost < 1.0  # 59k tokens should be well under $1

    def test_state_summary(self):
        state = GlobalState(
            problem_statement="Predict heart disease",
            data_source=DataSource(type="kaggle", identifier="titanic", description="test"),
        )
        summary = state.summary()
        assert "Run ID" in summary
        assert "Stages" in summary


# ── Report Generator Tests ────────────────────────────────────────────────────

class TestReportGenerator:
    def test_data_report_creates_file(self, tmp_path, sample_data_health, heart_disease_spec):
        from autoresearch.tools.report_generator import generate_data_report
        output = generate_data_report(sample_data_health, heart_disease_spec, tmp_path)
        assert output.exists()
        html = output.read_text()
        assert "Data Intelligence Report" in html
        assert "303" in html  # Row count
        assert "82" in html   # Health score

    def test_data_report_contains_flags(self, tmp_path, sample_data_health, heart_disease_spec):
        from autoresearch.tools.report_generator import generate_data_report
        output = generate_data_report(sample_data_health, heart_disease_spec, tmp_path)
        html = output.read_text()
        assert "flag-warning" in html or "flag-critical" in html

    def test_comparison_report_creates_file(self, tmp_path, heart_disease_spec, xgboost_result, logreg_result):
        from autoresearch.tools.report_generator import generate_comparison_report

        evaluation = EvaluationReport(
            method_scores=[
                MethodScore(
                    method_id="xgboost",
                    method_name="XGBoost",
                    primary_metric=0.847,
                    score_performance=1.0,
                    score_speed=0.9,
                    score_interpretability=0.6,
                    score_robustness=0.95,
                    total_score=0.87,
                    summary="XGBoost scored best overall.",
                ),
                MethodScore(
                    method_id="logistic_regression",
                    method_name="Logistic Regression",
                    primary_metric=0.821,
                    score_performance=0.0,
                    score_speed=1.0,
                    score_interpretability=1.0,
                    score_robustness=1.0,
                    total_score=0.74,
                    summary="Fast and interpretable.",
                ),
            ],
            winner_method_id="xgboost",
            winner_explanation="XGBoost won on performance.",
            next_steps=["Add SHAP explanations.", "Collect more data."],
            research_directions=["Explore feature interactions."],
        )

        output = generate_comparison_report(
            [xgboost_result, logreg_result], evaluation, heart_disease_spec, tmp_path
        )
        assert output.exists()
        html = output.read_text()
        assert "XGBoost" in html
        assert "winner-row" in html
        assert "0.847" in html

    def test_comparison_report_shows_failed(self, tmp_path, heart_disease_spec,
                                             xgboost_result, failed_result):
        from autoresearch.tools.report_generator import generate_comparison_report

        evaluation = EvaluationReport(
            method_scores=[
                MethodScore(
                    method_id="xgboost", method_name="XGBoost",
                    primary_metric=0.847,
                    score_performance=1.0, score_speed=1.0,
                    score_interpretability=0.6, score_robustness=0.95,
                    total_score=0.9, summary="Won.",
                )
            ],
            winner_method_id="xgboost",
            winner_explanation="Only successful method.",
            next_steps=["Review TabNet failure."],
        )

        output = generate_comparison_report(
            [xgboost_result, failed_result], evaluation, heart_disease_spec, tmp_path
        )
        html = output.read_text()
        assert "TabNet" in html
        assert "Failed" in html or "GPU memory" in html


# ── Data Source Resolver Tests ────────────────────────────────────────────────

class TestDataSourceResolver:
    def test_kaggle_dataset_slug(self):
        from autoresearch.tools.data_sources import DataSourceResolver
        from autoresearch.orchestrator.state import DataSource

        resolver = DataSourceResolver(kaggle_client=None)
        ds = DataSource(type="kaggle", identifier="username/heart-disease", description="test")
        result = resolver.resolve(ds)
        assert "username/heart-disease" in result.dataset_sources
        assert len(result.competition_sources) == 0

    def test_kaggle_competition_slug(self):
        from autoresearch.tools.data_sources import DataSourceResolver
        from autoresearch.orchestrator.state import DataSource

        resolver = DataSourceResolver(kaggle_client=None)
        ds = DataSource(type="kaggle", identifier="titanic", description="Titanic competition")
        # Without a kaggle client, verify_competition_access is skipped
        result = resolver.resolve(ds)
        assert "titanic" in result.competition_sources

    def test_huggingface_generates_setup_cell(self):
        from autoresearch.tools.data_sources import DataSourceResolver
        from autoresearch.orchestrator.state import DataSource

        resolver = DataSourceResolver(kaggle_client=None)
        ds = DataSource(type="huggingface", identifier="imdb", description="IMDB sentiment")
        result = resolver.resolve(ds)
        assert len(result.kernel_setup_cells) > 0
        assert "load_dataset" in result.kernel_setup_cells[0]
        assert "imdb" in result.kernel_setup_cells[0]

    def test_gdrive_generates_setup_cell(self):
        from autoresearch.tools.data_sources import DataSourceResolver
        from autoresearch.orchestrator.state import DataSource

        resolver = DataSourceResolver(kaggle_client=None)
        ds = DataSource(
            type="gdrive",
            identifier="https://drive.google.com/file/d/ABC123XYZ/view",
            description="My CSV"
        )
        result = resolver.resolve(ds)
        assert len(result.kernel_setup_cells) > 0
        assert "ABC123XYZ" in result.kernel_setup_cells[0]
        assert "gdown" in result.kernel_setup_cells[0]

    def test_unknown_source_raises(self):
        from autoresearch.tools.data_sources import DataSourceResolver
        from autoresearch.orchestrator.state import DataSource

        resolver = DataSourceResolver(kaggle_client=None)
        ds = DataSource(type="dropbox", identifier="some/file", description="test")
        with pytest.raises(ValueError, match="Unknown data source type"):
            resolver.resolve(ds)
