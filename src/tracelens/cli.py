"""TraceLens CLI — entry point for all user-facing commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="tracelens",
    help="Automated causal root-cause diagnostics for multi-agent AI systems.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)


@app.command()
def ingest(
    trace_file: Path = typer.Argument(..., help="Path to a trace JSON file"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to tracelens.toml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest a trace from a JSON file into the TraceLens database."""
    _setup_logging(verbose)
    from tracelens.config import load_config
    from tracelens.schema import Trace
    from tracelens.store import connect, save_trace

    cfg = load_config(config)

    if not trace_file.exists():
        typer.echo(f"Error: file {trace_file} not found", err=True)
        raise typer.Exit(1) from None

    raw = json.loads(trace_file.read_text(encoding="utf-8"))
    trace = Trace(**raw)
    conn = connect(cfg.db_path)
    try:
        save_trace(conn, trace)
        typer.echo(f"✓ Ingested trace {trace.trace_id!r} ({len(trace.steps)} steps)")
    finally:
        conn.close()


@app.command()
def diagnose(
    trace_id: str = typer.Argument(..., help="Trace ID to diagnose"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to tracelens.toml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run causal attribution on a stored trace and identify the root cause."""
    _setup_logging(verbose)
    from tracelens.attribute import diagnose_trace
    from tracelens.config import load_config
    from tracelens.schema import Trace
    from tracelens.store import connect, load_trace, save_diagnosis

    cfg = load_config(config)
    conn = connect(cfg.db_path)
    
    try:
        typer.echo(f"Loading trace {trace_id!r} from database '{cfg.db_path}'...")
        trace_dict = load_trace(conn, trace_id)
        trace = Trace.model_validate(trace_dict)
        
        typer.echo(f"Diagnosing trace {trace_id!r} (this requires LLM calls)...")
        diagnosis = diagnose_trace(trace, cfg)
        
        save_diagnosis(conn, diagnosis)
        
        typer.echo(f"\n{'=' * 60}")
        typer.echo("DIAGNOSIS COMPLETE")
        typer.echo(f"{'=' * 60}")
        typer.echo(f"\nSummary:\n  {diagnosis.summary}\n")
        
        if diagnosis.root_cause_step:
            typer.echo("Step Breakdown:")
            for attr in diagnosis.all_steps:
                mark = "👉" if attr.step_id == diagnosis.root_cause_step.step_id else "  "
                typer.echo(
                    f"{mark} {attr.agent_name:<15} | Score: {attr.attribution_score:<5.2f} "
                    f"| Novel Claims: {len(attr.novel_claims)} "
                    f"| Impact: {attr.downstream_impact:.2f}"
                )
                
                # Print the actual hallucinated claims for the root cause
                if attr.step_id == diagnosis.root_cause_step.step_id and attr.novel_claims:
                    typer.echo("\n    Hallucinated Claims Identified:")
                    for c in attr.novel_claims:
                        typer.echo(f"      - {c.text}")
                        typer.echo(f"        (Confidence: {c.confidence:.2f})")
        
        typer.echo(
            f"\nDiagnosis saved. Run 'tracelens report --trace-id {trace_id}' to view later."
        )
        
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1) from None
    finally:
        conn.close()


@app.command()
def report(
    trace_id: str = typer.Option(None, "--trace-id", "-t", help="Show report for a specific trace"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to tracelens.toml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Print a diagnostic report for stored traces."""
    _setup_logging(verbose)
    from tracelens.config import load_config
    from tracelens.store import connect, list_traces, load_diagnosis

    cfg = load_config(config)
    conn = connect(cfg.db_path)

    try:
        if trace_id:
            diag = load_diagnosis(conn, trace_id)
            if diag is None:
                typer.echo(f"No diagnosis found for trace {trace_id!r}")
                typer.echo("Run 'tracelens diagnose' first.")
                raise typer.Exit(1) from None
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"Trace: {trace_id}")
            typer.echo(
                f"Root Cause: {diag['root_cause_agent']} (step {diag['root_cause_step_id']})"
            )
            score = diag['attribution_score']
            score_str = f"{score:.2f}" if score is not None else "N/A"
            typer.echo(f"Attribution Score: {score_str}")
            typer.echo(f"Summary: {diag['summary']}")
            typer.echo(f"{'=' * 60}\n")
        else:
            traces = list_traces(conn)
            if not traces:
                typer.echo("No traces stored yet. Use 'tracelens ingest' to add traces.")
                raise typer.Exit(0) from None

            typer.echo(
                f"\n{'Trace ID':<25} {'Project':<15} {'Root Cause':<20} {'Score':<8} {'Date'}"
            )
            typer.echo("-" * 90)
            for t in traces:
                root_agent = t.get("root_cause_agent") or "—"
                score = t.get("attribution_score")
                score_str = f"{score:.2f}" if score is not None else "—"
                project_name = t.get("project_name", "default")
                
                trace_id_disp = (
                    (t['trace_id'][:22] + "...") if len(t['trace_id']) > 25 else t['trace_id']
                )
                proj_disp = (
                    (project_name[:12] + "...") if len(project_name) > 15 else project_name
                )
                root_disp = (
                    (root_agent[:17] + "...") if len(root_agent) > 20 else root_agent
                )
                
                typer.echo(
                    f"{trace_id_disp:<25} {proj_disp:<15} {root_disp:<20} {score_str:<8} "
                    f"{t['created_at'][:19]}"
                )
            typer.echo()
    finally:
        conn.close()


@app.command()
def dashboard(
    config: Path = typer.Option(None, "--config", "-c", help="Path to tracelens.toml"),
) -> None:
    """Launch the interactive Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "dashboard.py"
    if not dashboard_path.exists():
        typer.echo(f"Dashboard not found at {dashboard_path}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Launching dashboard at {dashboard_path}...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=False,
    )
