"""
AutoResearch CLI

The entry point. Run with:
    autoresearch run              — start a new run from config.yaml
    autoresearch init             — set up credentials interactively
    autoresearch resume RUN_ID    — resume a failed run
    autoresearch status RUN_ID    — check run status

Designed for first-time users who may never have used a CLI tool before.
Every message explains what's happening and why.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from autoresearch.orchestrator.graph import AutoResearchOrchestrator
from autoresearch.orchestrator.state import GlobalState, Stage, DataSource
from autoresearch.agents.api_utils import list_provider_models

app     = typer.Typer(help="AutoResearch — Autonomous ML Research Agent")
console = Console()


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    output: str = typer.Option("./autoresearch_output", "--output", "-o", help="Output directory"),
):
    """
    Start a new AutoResearch run.

    Reads your problem statement and data source from config.yaml,
    then runs the full pipeline: EDA → method selection → experiments →
    evaluation → paper draft.

    Takes 3–8 hours depending on dataset size and number of methods.
    You can Ctrl+C and resume later with: autoresearch resume RUN_ID
    """
    _check_config(config)

    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    problem_stmt = cfg.get("problem", {}).get("statement", "").strip()
    if not problem_stmt:
        console.print("[red]Error:[/red] No problem statement in config.yaml.")
        console.print("Open config.yaml and fill in: problem.statement")
        raise typer.Exit(1)

    ds_cfg = cfg.get("problem", {}).get("data_source", {})
    if not ds_cfg.get("identifier", "").strip():
        console.print("[red]Error:[/red] No data source in config.yaml.")
        console.print("Open config.yaml and fill in: problem.data_source.identifier")
        raise typer.Exit(1)

    data_source = DataSource(
        type=ds_cfg.get("type", "kaggle"),
        identifier=ds_cfg.get("identifier", ""),
        description=ds_cfg.get("description", ""),
    )

    state = GlobalState(
        problem_statement=problem_stmt,
        data_source=data_source,
        output_dir=output,
    )

    console.print(Panel(
        f"[bold]Run ID:[/bold] {state.run_id}\n"
        f"[bold]Problem:[/bold] {problem_stmt[:100]}\n"
        f"[bold]Data:[/bold] {data_source.type}:{data_source.identifier}\n\n"
        f"If this run is interrupted, resume it with:\n"
        f"  [cyan]autoresearch resume {state.run_id}[/cyan]",
        title="AutoResearch Starting",
        border_style="green",
    ))

    backends = _pick_models(config)
    orchestrator = AutoResearchOrchestrator.from_config(config, llm_backends=backends)

    try:
        final_state = asyncio.run(orchestrator.run(state))
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Run paused.[/yellow] Resume with:")
        console.print(f"  [cyan]autoresearch resume {state.run_id}[/cyan]")
        raise typer.Exit(0)

    if final_state.current_stage == Stage.COMPLETE:
        console.print(f"\n[green]✅ Complete![/green] Results in: {output}/")
    else:
        console.print(f"\n[red]Run ended with errors.[/red]")
        console.print(f"Resume with: [cyan]autoresearch resume {state.run_id}[/cyan]")
        raise typer.Exit(1)


@app.command()
def init():
    """
    Set up AutoResearch interactively.

    Walks you through:
      - Anthropic API key
      - Kaggle credentials
      - Creating your config.yaml

    Run this before your first `autoresearch run`.
    """
    console.print(Panel(
        "Welcome to AutoResearch!\n\n"
        "Let's get you set up. You'll need:\n"
        "  1. An Anthropic API key (console.anthropic.com)\n"
        "  2. A Kaggle account and API key (kaggle.com/settings → API)\n\n"
        "These stay on your machine and are never sent anywhere.",
        title="AutoResearch Setup",
        border_style="blue",
    ))

    anthropic_key = Prompt.ask("\n[bold]Anthropic API key[/bold] (starts with sk-ant-)")
    kaggle_user   = Prompt.ask("[bold]Kaggle username[/bold]")
    kaggle_key    = Prompt.ask("[bold]Kaggle API key[/bold]")

    # Validate keys aren't empty
    if not anthropic_key.startswith("sk-"):
        console.print("[yellow]Warning:[/yellow] Anthropic keys usually start with 'sk-'. Double-check.")

    # Write config.yaml
    config_path = Path("config.yaml")
    if config_path.exists():
        overwrite = Confirm.ask("config.yaml already exists. Overwrite?")
        if not overwrite:
            console.print("Setup cancelled. Edit config.yaml manually.")
            raise typer.Exit(0)

    # Read template
    template_path = Path(__file__).parent / "config.yaml"
    if template_path.exists():
        template = template_path.read_text()
    else:
        template = _default_config_template()

    # Inject credentials
    template = template.replace('anthropic: ""', f'anthropic: "{anthropic_key}"')
    template = template.replace('kaggle_username: ""', f'kaggle_username: "{kaggle_user}"')
    template = template.replace('kaggle_key: ""', f'kaggle_key: "{kaggle_key}"')

    config_path.write_text(template)

    console.print("\n[green]✅ config.yaml created![/green]")
    console.print("\nNext steps:")
    console.print("  1. Open config.yaml")
    console.print("  2. Fill in your [bold]problem.statement[/bold]")
    console.print("  3. Fill in your [bold]problem.data_source[/bold]")
    console.print("  4. Run: [cyan]autoresearch run[/cyan]")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="The run ID to resume (shown at start of previous run)"),
    output:  str = typer.Option("./autoresearch_output", "--output", "-o"),
    config:  str = typer.Option("config.yaml", "--config", "-c"),
):
    """
    Resume a failed or interrupted run.

    AutoResearch saves its progress after each stage. If a run is
    interrupted (Ctrl+C, crash, Kaggle timeout), you can resume it
    from where it left off.
    """
    _check_config(config)

    state_path = Path(output) / f"run_{run_id}_state.json"
    if not state_path.exists():
        console.print(f"[red]Error:[/red] No saved state found for run {run_id}")
        console.print(f"Expected at: {state_path}")
        raise typer.Exit(1)

    state_data = json.loads(state_path.read_text())
    state = GlobalState(**state_data)

    completed = [s.value for s in state.completed_stages]
    console.print(Panel(
        f"[bold]Run ID:[/bold] {run_id}\n"
        f"[bold]Completed stages:[/bold] {', '.join(completed) or 'none'}\n"
        f"[bold]Resuming from:[/bold] {state.current_stage.value}",
        title="Resuming AutoResearch",
        border_style="yellow",
    ))

    backends = _pick_models(config)
    orchestrator = AutoResearchOrchestrator.from_config(config, llm_backends=backends)

    try:
        final_state = asyncio.run(orchestrator.run(state))
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Run paused again.[/yellow] Resume with:")
        console.print(f"  [cyan]autoresearch resume {run_id}[/cyan]")
        raise typer.Exit(0)

    if final_state.current_stage == Stage.COMPLETE:
        console.print(f"\n[green]✅ Complete![/green] Results in: {output}/")
    else:
        console.print(f"\n[red]Run ended with errors.[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    run_id: str = typer.Argument(..., help="The run ID to check"),
    output:  str = typer.Option("./autoresearch_output", "--output", "-o"),
):
    """Show the current status of a run."""
    state_path = Path(output) / f"run_{run_id}_state.json"
    if not state_path.exists():
        console.print(f"[red]No state found for run {run_id}[/red]")
        raise typer.Exit(1)

    state = GlobalState(**json.loads(state_path.read_text()))

    console.print(Panel(
        state.summary(),
        title=f"Run {run_id} Status",
        border_style="blue",
    ))

    if state.errors:
        console.print("\n[red]Errors:[/red]")
        for err in state.errors[-3:]:  # Last 3 errors
            console.print(f"  [{err.stage.value}] {err.plain_english}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_live_models(
    key_map: dict[str, str],
) -> list[tuple[str, str, str]]:
    """
    Call each configured provider's API and return all available models as
    (provider, api_key, model_id) tuples.
    """
    entries: list[tuple[str, str, str]] = []
    for prov in ("anthropic", "openai", "minimax"):
        api_key = key_map.get(prov, "")
        if not api_key:
            continue
        try:
            console.print(f"  Fetching models from [bold]{prov}[/bold]...", end=" ")
            ids = list_provider_models(prov, api_key)
            console.print(f"[green]{len(ids)} found[/green]")
            for model_id in ids:
                entries.append((prov, api_key, model_id))
        except Exception as e:
            console.print(f"[yellow]failed ({e})[/yellow]")
    return entries


def _pick_models(config_path: str) -> list[tuple[str, str, str]] | None:
    """
    Interactively ask the user which model(s) to use.

    First asks whether to use live discovery or config.yaml defaults,
    then shows the full model list for selection.

    Returns a list of (provider, api_key, model) tuples, or None to use
    whatever is in config.yaml unchanged.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    api = cfg.get("api_keys", {})
    key_map = {
        "anthropic": api.get("anthropic", ""),
        "openai":    api.get("openai", ""),
        "minimax":   api.get("minimax", ""),
    }

    # ── Live discovery ─────────────────────────────────────────────────────────
    console.print()
    live_entries = _fetch_live_models(key_map)

    if not live_entries:
        console.print("[yellow]No models discovered — falling back to config.yaml.[/yellow]")
        return None

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    available: list[tuple[str, str, str]] = []   # (provider, api_key, model_id)
    for prov, key, model_id in live_entries:
        if (prov, model_id) not in seen:
            seen.add((prov, model_id))
            available.append((prov, key, model_id))

    table = Table(title="Live Models", show_header=True, header_style="bold cyan")
    table.add_column("#",        style="bold", width=4)
    table.add_column("Provider", width=12)
    table.add_column("Model ID", style="dim")

    for i, (prov, _key, model_id) in enumerate(available, 1):
        prov_color = "green" if prov == "anthropic" else "blue"
        table.add_row(str(i), f"[{prov_color}]{prov}[/{prov_color}]", model_id)

    last = len(available)
    table.add_row(str(last + 1), "[yellow]Custom[/yellow]", "enter your own model ID")
    table.add_row(str(last + 2), "[dim]config.yaml[/dim]",  "keep current settings")

    console.print()
    console.print(table)

    raw = Prompt.ask(
        f"\nChoose model(s) — single number or comma-separated fallback chain "
        f"(e.g. [cyan]1[/cyan] or [cyan]1,3,5[/cyan])",
        default=str(last + 2),
    ).strip()

    if raw == str(last + 2) or raw == "":
        return None

    backends: list[tuple[str, str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        try:
            idx = int(token)
        except ValueError:
            console.print(f"[yellow]Skipping invalid entry:[/yellow] {token}")
            continue

        if idx == last + 2:
            return None

        if idx == last + 1:
            prov = Prompt.ask("  Provider", choices=["anthropic", "openai", "minimax"], default="anthropic")
            model_id = Prompt.ask("  Model ID (exact API model string)").strip()
            api_key = key_map.get(prov, "")
            if not api_key:
                console.print(f"[red]No API key for {prov} in config.yaml — skipping.[/red]")
                continue
            backends.append((prov, api_key, model_id))
        elif 1 <= idx <= last:
            prov, api_key, model_id = available[idx - 1]
            backends.append((prov, api_key, model_id))
        else:
            console.print(f"[yellow]Skipping out-of-range choice:[/yellow] {idx}")

    if not backends:
        console.print("[yellow]No valid models selected — using config.yaml settings.[/yellow]")
        return None

    console.print("\n  [bold]Selected model chain:[/bold]")
    for i, (prov, _key, model) in enumerate(backends, 1):
        tag = "(primary)" if i == 1 else f"(fallback {i-1})"
        console.print(f"    {i}. {prov} / {model}  [dim]{tag}[/dim]")
    console.print()

    return backends


def _check_config(config_path: str) -> None:
    """Make sure config.yaml exists and has the required credentials."""
    if not Path(config_path).exists():
        console.print(f"[red]Error:[/red] {config_path} not found.")
        console.print("Run [cyan]autoresearch init[/cyan] to create it.")
        raise typer.Exit(1)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    api = cfg.get("api_keys", {})
    missing = []
    if not api.get("anthropic") and not api.get("openai") and not api.get("minimax"):
        missing.append("api_keys.anthropic  (or api_keys.openai or api_keys.minimax)")
    if not api.get("kaggle_username"):
        missing.append("api_keys.kaggle_username")
    if not api.get("kaggle_key"):
        missing.append("api_keys.kaggle_key")

    if missing:
        console.print(f"[red]Missing credentials in {config_path}:[/red]")
        for m in missing:
            console.print(f"  - {m}")
        console.print("\nRun [cyan]autoresearch init[/cyan] to set them up.")
        raise typer.Exit(1)


def _default_config_template() -> str:
    return """\
api_keys:
  anthropic: ""
  kaggle_username: ""
  kaggle_key: ""
  huggingface: ""

problem:
  statement: ""
  data_source:
    type: "kaggle"
    identifier: ""
    description: ""

methods:
  max_methods: 3
  task_type_override: ""

evaluation_weights:
  performance: 0.40
  speed: 0.20
  interpretability: 0.20
  robustness: 0.20

kaggle:
  enable_gpu: true
  max_runtime_hours: 4
  enable_internet: true

output:
  directory: "./autoresearch_output"
  paper_formats:
    docx: true
    latex: true
    pdf: false

advanced:
  confidence_threshold: 0.80
  max_retries: 3
  llm_model: "claude-sonnet-4-20250514"
  verbose: true
"""


if __name__ == "__main__":
    app()
