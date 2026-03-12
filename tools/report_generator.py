"""
HTML Report Generator

Produces two standalone HTML reports included in the Research Starter Pack:

1. data_report.html   — EDA findings, data health, feature insights
2. comparison.html    — Side-by-side method comparison with scores

Both are single-file HTML (no external dependencies) so the researcher
can open them anywhere without a web server.

Design: clean, readable, no JavaScript frameworks.
Numbers are always accompanied by plain English interpretation.
"""

from pathlib import Path
from typing import Optional

from autoresearch.schemas import (
    DataHealthReport,
    EvaluationReport,
    ExecutionResult,
    ExecutionStatus,
    ProblemSpec,
    Severity,
)


# ── Shared CSS ────────────────────────────────────────────────────────────────

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 15px; line-height: 1.6;
    color: #1a1a2e; background: #f8f9fb;
    padding: 2rem;
}
.container { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.8rem; color: #16213e; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; color: #0f3460; margin: 2rem 0 0.75rem; border-bottom: 2px solid #e94560; padding-bottom: 0.3rem; }
h3 { font-size: 1rem; color: #16213e; margin: 1rem 0 0.3rem; }
.subtitle { color: #666; margin-bottom: 2rem; font-size: 0.9rem; }
.card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
.badge-green  { background: #d4edda; color: #155724; }
.badge-yellow { background: #fff3cd; color: #856404; }
.badge-red    { background: #f8d7da; color: #721c24; }
.badge-blue   { background: #d1ecf1; color: #0c5460; }
.score-bar { height: 8px; border-radius: 4px; background: #e9ecef; margin: 4px 0; }
.score-fill { height: 100%; border-radius: 4px; background: #0f3460; transition: width 0.3s; }
table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
th { background: #16213e; color: white; padding: 0.6rem 0.8rem; text-align: left; font-size: 0.85rem; }
td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }
tr:nth-child(even) { background: #fafafa; }
tr:hover { background: #f0f4ff; }
.winner-row { background: #e8f5e9 !important; font-weight: 600; }
.flag-critical { border-left: 4px solid #dc3545; padding: 0.75rem 1rem; background: #fff5f5; margin: 0.5rem 0; border-radius: 0 4px 4px 0; }
.flag-warning  { border-left: 4px solid #ffc107; padding: 0.75rem 1rem; background: #fffbf0; margin: 0.5rem 0; border-radius: 0 4px 4px 0; }
.flag-info     { border-left: 4px solid #17a2b8; padding: 0.75rem 1rem; background: #f0fafd; margin: 0.5rem 0; border-radius: 0 4px 4px 0; }
.watermark { color: #999; font-size: 0.8rem; margin-top: 3rem; text-align: center; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 600px) { .grid-2 { grid-template-columns: 1fr; } }
"""


def _health_badge(score: float) -> str:
    if score >= 70:
        return f'<span class="badge badge-green">{score:.0f}/100 Good</span>'
    elif score >= 50:
        return f'<span class="badge badge-yellow">{score:.0f}/100 Fair</span>'
    else:
        return f'<span class="badge badge-red">{score:.0f}/100 Needs Work</span>'


def _score_bar(value: float, max_val: float = 1.0) -> str:
    pct = min(100, (value / max_val) * 100)
    return f'<div class="score-bar"><div class="score-fill" style="width:{pct:.0f}%"></div></div>'


def generate_data_report(
    report: DataHealthReport,
    spec: ProblemSpec,
    output_dir: Path,
) -> Path:
    """Generate data_report.html from a DataHealthReport."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Flag HTML ─────────────────────────────────────────────────────────────
    flags_html = ""
    for flag, severity, explanation in report.flags:
        css_class = {
            Severity.CRITICAL: "flag-critical",
            Severity.WARNING:  "flag-warning",
            Severity.INFO:     "flag-info",
        }.get(severity, "flag-info")
        label = severity.value.upper()
        flags_html += f'<div class="{css_class}"><strong>[{label}]</strong> {explanation}</div>\n'

    if not flags_html:
        flags_html = '<p style="color:#666">No major data quality issues detected.</p>'

    # ── Feature table ─────────────────────────────────────────────────────────
    feature_rows = ""
    for feat in report.features:
        missing_color = "color:#dc3545" if feat.missing_pct > 20 else (
            "color:#ffc107" if feat.missing_pct > 5 else "color:#28a745"
        )
        flag_badges = " ".join(
            f'<span class="badge badge-yellow">{f.value}</span>'
            for f in feat.flags
        )
        feature_rows += f"""
        <tr>
            <td><strong>{feat.name}</strong></td>
            <td>{feat.dtype}</td>
            <td style="{missing_color}">{feat.missing_pct:.1f}%</td>
            <td>{feat.unique_count:,}</td>
            <td>{f"{feat.mean:.3f}" if feat.mean is not None else "—"}</td>
            <td>{feat.usefulness_estimate}</td>
            <td>{flag_badges}</td>
        </tr>"""

    # ── Correlations ──────────────────────────────────────────────────────────
    corr_rows = ""
    for pair in report.strong_correlations:
        strength = "strong" if abs(pair.correlation) > 0.85 else "moderate"
        corr_rows += f"""
        <tr>
            <td>{pair.feature_a}</td>
            <td>{pair.feature_b}</td>
            <td><strong>{pair.correlation:+.3f}</strong> ({strength})</td>
            <td style="font-size:0.85rem;color:#555">{pair.plain_english}</td>
        </tr>"""

    corr_section = f"""
    <h2>Strong Feature Correlations</h2>
    <div class="card">
        <p style="color:#555;margin-bottom:0.75rem">
            Correlated features carry similar information. You may not need both — removing one
            can simplify the model without losing predictive power.
        </p>
        <table>
            <tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Interpretation</th></tr>
            {corr_rows}
        </table>
    </div>""" if corr_rows else ""

    # ── Recommendations ───────────────────────────────────────────────────────
    rec_html = "".join(
        f'<li style="margin-bottom:0.5rem">{rec}</li>'
        for rec in report.recommendations
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoResearch — Data Report</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="container">

<h1>📊 Data Intelligence Report</h1>
<p class="subtitle">
    Generated by AutoResearch &nbsp;|&nbsp;
    Problem: {spec.domain} ({spec.task_type.value}) &nbsp;|&nbsp;
    Dataset: {report.row_count:,} rows × {report.column_count} columns
</p>

<div class="grid-2">
    <div class="card">
        <h3>Dataset Overview</h3>
        <table>
            <tr><td>Rows</td><td><strong>{report.row_count:,}</strong></td></tr>
            <tr><td>Columns</td><td><strong>{report.column_count}</strong></td></tr>
            <tr><td>Duplicate rows</td><td><strong>{report.duplicate_row_count:,}</strong></td></tr>
            <tr><td>Memory</td><td><strong>{report.memory_usage_mb:.1f} MB</strong></td></tr>
        </table>
    </div>
    <div class="card">
        <h3>Data Health Score</h3>
        {_health_badge(report.health_score)}
        {_score_bar(report.health_score, 100)}
        <p style="margin-top:0.75rem;color:#555;font-size:0.9rem">{report.health_score_explanation}</p>
    </div>
</div>

<h2>Target Variable</h2>
<div class="card">
    <p>{report.target_plain_english}</p>
</div>

<h2>Data Quality Flags</h2>
<div class="card">
{flags_html}
</div>

<h2>Feature Analysis</h2>
<div class="card" style="overflow-x:auto">
    <table>
        <tr>
            <th>Feature</th><th>Type</th><th>Missing</th>
            <th>Unique</th><th>Mean</th><th>Usefulness</th><th>Flags</th>
        </tr>
        {feature_rows}
    </table>
</div>

{corr_section}

<h2>Recommendations</h2>
<div class="card">
    <ol style="padding-left:1.5rem">{rec_html}</ol>
</div>

{"<h2>Method Hints</h2><div class='card'><ul style='padding-left:1.5rem'>" + "".join(f"<li>{h}</li>" for h in report.method_hints) + "</ul></div>" if report.method_hints else ""}

<p class="watermark">AutoResearch Data Report &nbsp;·&nbsp; Review all findings with domain expertise</p>
</div>
</body>
</html>"""

    output_path = output_dir / "data_report.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_comparison_report(
    results: list[ExecutionResult],
    evaluation: EvaluationReport,
    spec: ProblemSpec,
    output_dir: Path,
) -> Path:
    """Generate comparison.html from ExecutionResults and EvaluationReport."""
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in results if r.status == ExecutionStatus.SUCCESS]
    failed     = [r for r in results if r.status != ExecutionStatus.SUCCESS]

    # Score lookup
    score_map = {s.method_id: s for s in evaluation.method_scores}

    # ── Results table rows ────────────────────────────────────────────────────
    result_rows = ""
    for r in successful:
        is_winner = r.method_id == evaluation.winner_method_id
        row_class = "winner-row" if is_winner else ""
        crown     = " 🏆" if is_winner else ""
        s         = score_map.get(r.method_id)

        gap = ""
        if r.train_metric and r.val_metric:
            g = abs(r.train_metric - r.val_metric)
            color = "#dc3545" if g > 0.1 else ("#ffc107" if g > 0.05 else "#28a745")
            gap = f'<span style="color:{color}">{g:.3f}</span>'

        result_rows += f"""
        <tr class="{row_class}">
            <td><strong>{r.method_name}{crown}</strong></td>
            <td><strong>{r.primary_metric_value:.4f}</strong></td>
            <td>{r.train_metric:.4f if r.train_metric else "—"}</td>
            <td>{r.val_metric:.4f if r.val_metric else "—"}</td>
            <td>{gap}</td>
            <td>{f"{r.runtime_minutes:.1f} min" if r.runtime_minutes else "—"}</td>
            <td>{f"{s.total_score:.3f}" if s else "—"}</td>
        </tr>"""

    for r in failed:
        result_rows += f"""
        <tr style="color:#999">
            <td>{r.method_name}</td>
            <td colspan="5" style="font-style:italic">{r.error_plain_english or "Failed"}</td>
            <td>—</td>
        </tr>"""

    # ── Score breakdown ───────────────────────────────────────────────────────
    score_breakdown = ""
    for s in evaluation.method_scores:
        is_winner = s.method_id == evaluation.winner_method_id
        bg = "background:#e8f5e9;" if is_winner else ""
        score_breakdown += f"""
        <div class="card" style="{bg}">
            <h3>{s.method_name} {"🏆" if is_winner else ""} &nbsp; Total: {s.total_score:.3f}</h3>
            <table>
                <tr>
                    <td style="width:140px">Performance</td>
                    <td>{_score_bar(s.score_performance)}</td>
                    <td style="width:60px;text-align:right">{s.score_performance:.3f}</td>
                </tr>
                <tr>
                    <td>Speed</td>
                    <td>{_score_bar(s.score_speed)}</td>
                    <td style="text-align:right">{s.score_speed:.3f}</td>
                </tr>
                <tr>
                    <td>Interpretability</td>
                    <td>{_score_bar(s.score_interpretability)}</td>
                    <td style="text-align:right">{s.score_interpretability:.3f}</td>
                </tr>
                <tr>
                    <td>Robustness</td>
                    <td>{_score_bar(s.score_robustness)}</td>
                    <td style="text-align:right">{s.score_robustness:.3f}</td>
                </tr>
            </table>
            <p style="margin-top:0.5rem;color:#555;font-size:0.9rem">{s.summary}</p>
        </div>"""

    # ── Risk flags ────────────────────────────────────────────────────────────
    risk_html = ""
    for flag in evaluation.risk_flags:
        css = "flag-critical" if flag.severity == "critical" else "flag-warning"
        risk_html += f'<div class="{css}"><strong>[{flag.severity.upper()}] {flag.method_id}</strong>: {flag.explanation}</div>\n'

    # ── Next steps ────────────────────────────────────────────────────────────
    steps_html = "".join(
        f'<li style="margin-bottom:0.5rem">{step}</li>'
        for step in evaluation.next_steps
    )

    research_html = "".join(
        f'<li style="margin-bottom:0.5rem">{d}</li>'
        for d in evaluation.research_directions
    ) if evaluation.research_directions else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoResearch — Method Comparison</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="container">

<h1>⚖️ Method Comparison Report</h1>
<p class="subtitle">
    Generated by AutoResearch &nbsp;|&nbsp;
    {spec.domain} ({spec.task_type.value}) &nbsp;|&nbsp;
    Primary metric: {spec.primary_metric.value}
</p>

<h2>Winner</h2>
<div class="card" style="border-left:4px solid #28a745">
    <h3>🏆 {evaluation.winner_method_id}</h3>
    <p>{evaluation.winner_explanation}</p>
</div>

<h2>Results Summary</h2>
<div class="card" style="overflow-x:auto">
    <table>
        <tr>
            <th>Method</th>
            <th>{spec.primary_metric.value}</th>
            <th>Train</th>
            <th>Val</th>
            <th>Gap</th>
            <th>Runtime</th>
            <th>Total Score</th>
        </tr>
        {result_rows}
    </table>
    <p style="font-size:0.8rem;color:#999;margin-top:0.5rem">
        Gap = |train − val|. Red &gt; 0.1 (overfitting risk). Green ≤ 0.05 (healthy).
    </p>
</div>

<h2>Score Breakdown</h2>
<p style="color:#555;margin-bottom:1rem">
    Weights: Performance {evaluation.method_scores[0].score_performance:.0%}
    (if it existed they'd be shown). Each bar shows normalized score (0–1).
</p>
{score_breakdown}

{"<h2>⚠️ Risk Flags</h2><div class='card'>" + risk_html + "</div>" if risk_html else ""}

<h2>Recommended Next Steps</h2>
<div class="card">
    <ol style="padding-left:1.5rem">{steps_html}</ol>
</div>

{"<h2>🔭 Novel Research Directions</h2><div class='card'><p style='color:#555;margin-bottom:0.75rem'>AutoResearch ran the baselines. Here are directions for your original contribution:</p><ul style='padding-left:1.5rem'>" + research_html + "</ul></div>" if research_html else ""}

<p class="watermark">AutoResearch Comparison Report &nbsp;·&nbsp; Scores reflect the evaluation criteria in your config.yaml</p>
</div>
</body>
</html>"""

    output_path = output_dir / "comparison.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
