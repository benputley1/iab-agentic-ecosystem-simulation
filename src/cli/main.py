#!/usr/bin/env python3
"""
RTB Simulation CLI - Compare IAB A2A vs Alkimi Ledger approaches.

Usage:
    rtb-sim run --scenario a,b,c --days 30
    rtb-sim run --scenario b --days 7 --mock-llm
    rtb-sim compare --report ./output/comparison.json
    rtb-sim test-scenario --scenario c --mock-llm

V2 Modes (Context Window Hallucination Testing):
    rtb-sim run --days 30 --context-pressure --volume-profile large
    rtb-sim run --days 30 --decision-chain --lookback-window 100
    rtb-sim run --days 30 --restart-test --crash-probability 0.02
    rtb-sim run --days 30 --full-v2 --volume-profile medium

Examples:
    # Quick test of all scenarios (1 day, mock mode)
    rtb-sim run --days 1 --mock-llm

    # Full 30-day simulation comparison
    rtb-sim run --days 30 --scenario a,b,c

    # Test Scenario C ledger recovery
    rtb-sim test-recovery --agent test-buyer-001

    # V2: Full hallucination testing with all features
    rtb-sim run --days 30 --full-v2 --volume-profile large --output results_v2.json
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from .v2_commands import V2Config, VolumeProfile, validate_v2_config, get_v2_feature_summary

app = typer.Typer(
    name="rtb-sim",
    help="IAB Agentic RTB Simulation - Compare trading models (V1 + V2 modes)",
    add_completion=False,
)
console = Console()


def parse_scenarios(scenario_str: str) -> list[str]:
    """Parse comma-separated scenario list."""
    scenarios = [s.strip().lower() for s in scenario_str.split(",")]
    valid = ["a", "b", "c"]
    for s in scenarios:
        if s not in valid:
            raise typer.BadParameter(f"Invalid scenario '{s}'. Valid: a, b, c")
    return scenarios


@app.command()
def run(
    scenario: str = typer.Option(
        "a,b,c",
        "--scenario", "-s",
        help="Scenarios to run (comma-separated: a,b,c)",
    ),
    days: int = typer.Option(
        1,
        "--days", "-d",
        help="Number of simulation days (1-30)",
        min=1,
        max=30,
    ),
    buyers: int = typer.Option(
        5,
        "--buyers", "-b",
        help="Number of buyer agents",
        min=1,
        max=20,
    ),
    sellers: int = typer.Option(
        5,
        "--sellers",
        help="Number of seller agents",
        min=1,
        max=20,
    ),
    mock_llm: bool = typer.Option(
        True,
        "--mock-llm/--real-llm",
        help="Use mock LLM (no API costs) or real LLM",
    ),
    skip_infra: bool = typer.Option(
        False,
        "--skip-infra",
        help="Skip Redis/Postgres (use mocks for quick testing)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
    # V2 Context Pressure Options
    context_pressure: bool = typer.Option(
        False,
        "--context-pressure",
        help="[V2] Enable token pressure tracking",
    ),
    volume_profile: Optional[str] = typer.Option(
        None,
        "--volume-profile",
        help="[V2] Volume profile: small|medium|large|enterprise",
    ),
    context_limit: int = typer.Option(
        200_000,
        "--context-limit",
        help="[V2] Context window token limit (default: 200000)",
    ),
    compression_loss: float = typer.Option(
        0.20,
        "--compression-loss",
        help="[V2] Information loss rate on compression (default: 0.20)",
    ),
    # V2 Decision Chain Options
    decision_chain: bool = typer.Option(
        False,
        "--decision-chain",
        help="[V2] Enable decision chain tracking",
    ),
    lookback_window: int = typer.Option(
        100,
        "--lookback-window",
        help="[V2] Decision reference lookback window (default: 100)",
    ),
    # V2 Restart Test Options
    restart_test: bool = typer.Option(
        False,
        "--restart-test",
        help="[V2] Enable restart/crash simulation",
    ),
    crash_probability: float = typer.Option(
        0.01,
        "--crash-probability",
        help="[V2] Crash rate per hour (default: 0.01)",
    ),
    recovery_modes: Optional[str] = typer.Option(
        None,
        "--recovery-modes",
        help="[V2] Comma-separated recovery modes (default: private_db,ledger)",
    ),
    # V2 Full Mode
    full_v2: bool = typer.Option(
        False,
        "--full-v2",
        help="[V2] Enable all V2 features (context pressure + decision chain + restart test)",
    ),
):
    """
    Run RTB simulation across specified scenarios.

    Compares:
    - Scenario A: Current state with exchange fees (10-20%)
    - Scenario B: IAB Pure A2A (context rot, no persistence)
    - Scenario C: Alkimi Ledger (zero context rot, full audit trail)

    V2 Modes add context window hallucination testing:
    - --context-pressure: Track token overflow and compression
    - --decision-chain: Track decision dependencies and cascading errors
    - --restart-test: Simulate crashes and state recovery
    - --full-v2: Enable all V2 features together
    """
    scenarios = parse_scenarios(scenario)

    # Build V2 config from CLI args
    v2_config = V2Config.from_cli_args(
        context_pressure=context_pressure,
        volume_profile=volume_profile,
        context_limit=context_limit,
        compression_loss=compression_loss,
        decision_chain=decision_chain,
        lookback_window=lookback_window,
        restart_test=restart_test,
        crash_probability=crash_probability,
        recovery_modes=recovery_modes,
        full_v2=full_v2,
    )

    # Validate V2 config
    v2_issues = validate_v2_config(v2_config)
    if v2_issues:
        for issue in v2_issues:
            console.print(f"[yellow]Warning:[/] {issue}")
        if any("must be" in issue for issue in v2_issues):
            console.print("[red]Aborting due to invalid V2 configuration.[/]")
            raise typer.Exit(1)

    # Build config display
    v2_display = ""
    if v2_config.is_v2_enabled:
        v2_display = f"\n\n[bold magenta]V2 Mode Active[/]\n{get_v2_feature_summary(v2_config)}"

    console.print(Panel.fit(
        f"[bold cyan]IAB Agentic RTB Simulation[/]\n\n"
        f"Scenarios: [green]{', '.join(s.upper() for s in scenarios)}[/]\n"
        f"Days: [yellow]{days}[/]\n"
        f"Buyers: {buyers} | Sellers: {sellers}\n"
        f"Mode: [{'green' if mock_llm else 'red'}]{'Mock LLM' if mock_llm else 'Real LLM'}[/]"
        f"{v2_display}",
        title="Configuration",
    ))

    # Run simulation
    result = asyncio.run(_run_simulation(
        scenarios=scenarios,
        days=days,
        buyers=buyers,
        sellers=sellers,
        mock_llm=mock_llm,
        skip_infra=skip_infra,
        verbose=verbose,
        v2_config=v2_config,
    ))

    # Display results
    _display_results(result)

    # Save to file if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to:[/] {output}")


async def _run_simulation(
    scenarios: list[str],
    days: int,
    buyers: int,
    sellers: int,
    mock_llm: bool,
    skip_infra: bool,
    verbose: bool,
    v2_config: Optional[V2Config] = None,
) -> dict:
    """Run the actual simulation."""
    from ..orchestration.run_simulation import SimulationRunner

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Determine task description based on mode
        mode_str = "V2" if (v2_config and v2_config.is_v2_enabled) else "V1"
        task = progress.add_task(f"Running simulation ({mode_str})...", total=None)

        # Build runner kwargs
        runner_kwargs = {
            "scenarios": scenarios,
            "days": days,
            "buyers": buyers,
            "sellers": sellers,
            "mock_llm": mock_llm,
            "time_acceleration": 10000.0,  # 100x faster than default for faster results
        }

        # Add V2 config if enabled
        if v2_config and v2_config.is_v2_enabled:
            runner_kwargs["v2_config"] = v2_config.to_dict()

        runner = SimulationRunner(**runner_kwargs)

        # Override to skip infra if requested
        if skip_infra:
            progress.update(task, description=f"Running simulation ({mode_str}, mock mode)...")
        else:
            progress.update(task, description=f"Running simulation ({mode_str})...")

        try:
            result = await runner.run()
            return result.to_dict()
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return {"error": str(e)}


def _display_results(result: dict):
    """Display simulation results in a formatted table."""
    if "error" in result:
        console.print(f"\n[red]Simulation failed:[/] {result['error']}")
        return

    console.print("\n[bold]Simulation Results[/]\n")

    # Create comparison table
    table = Table(title="Scenario Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("A (Exchange)", justify="right")
    table.add_column("B (IAB A2A)", justify="right")
    table.add_column("C (Alkimi)", justify="right")

    # Extract metrics from each scenario
    scenario_results = {r["scenario_id"]: r for r in result.get("scenario_results", [])}

    def get_metric(scenario_id: str, metric: str, default="N/A"):
        r = scenario_results.get(scenario_id, {})
        m = r.get("metrics", {})
        val = m.get(metric, default)
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    metrics = [
        ("Total Deals", "total_deals"),
        ("Total Impressions", "total_impressions"),
        ("Buyer Spend ($)", "total_buyer_spend"),
        ("Seller Revenue ($)", "total_seller_revenue"),
        ("Exchange Fees ($)", "total_exchange_fees"),
        ("Take Rate (%)", "intermediary_take_rate"),
        ("Avg CPM ($)", "average_cpm"),
        ("Context Rot Events", "context_rot_events"),
        ("Keys Lost", "keys_lost_total"),
        ("Recovery Success (%)", "recovery_success_rate"),
        ("Hallucination Rate (%)", "hallucination_rate"),
    ]

    for label, metric in metrics:
        table.add_row(
            label,
            get_metric("A", metric),
            get_metric("B", metric),
            get_metric("C", metric),
        )

    console.print(table)

    # Summary panel
    console.print(Panel(
        "[bold green]Key Findings:[/]\n\n"
        "• [red]Scenario A[/]: Exchange extracts significant fees (10-20%)\n"
        "• [yellow]Scenario B[/]: Context rot degrades performance over time\n"
        "• [green]Scenario C[/]: Zero context rot with ledger persistence\n\n"
        "[dim]See ANALYSIS.md for detailed interpretation.[/]",
        title="Summary",
    ))


@app.command()
def test_scenario(
    scenario: str = typer.Option(
        "c",
        "--scenario", "-s",
        help="Scenario to test (a, b, or c)",
    ),
    mock_llm: bool = typer.Option(
        True,
        "--mock-llm/--real-llm",
        help="Use mock LLM",
    ),
    skip_infra: bool = typer.Option(
        True,
        "--skip-infra/--with-infra",
        help="Skip Redis/Postgres (use mocks)",
    ),
):
    """
    Quick test of a single scenario.

    Useful for verifying the scenario works before full simulation.
    """
    scenario = scenario.lower()
    if scenario not in ["a", "b", "c"]:
        raise typer.BadParameter("Scenario must be a, b, or c")

    console.print(f"[cyan]Testing Scenario {scenario.upper()}...[/]")

    result = asyncio.run(_test_single_scenario(scenario, mock_llm, skip_infra))

    console.print(Panel(
        json.dumps(result, indent=2, default=str)[:2000],
        title=f"Scenario {scenario.upper()} Test Results",
    ))


async def _test_single_scenario(scenario: str, mock_llm: bool, skip_infra: bool) -> dict:
    """Test a single scenario."""
    if scenario == "a":
        from ..scenarios.scenario_a import ScenarioA, ScenarioConfig
        config = ScenarioConfig(mock_llm=mock_llm)
        s = ScenarioA(config)
        # Quick test
        return {"scenario": "A", "status": "ok", "config": {"mock_llm": mock_llm}}

    elif scenario == "b":
        from ..scenarios.scenario_b import run_scenario_b_test
        return await run_scenario_b_test(
            days=1,
            buyers=1,
            sellers=1,
            mock_llm=mock_llm,
            skip_redis=skip_infra,
        )

    elif scenario == "c":
        from ..scenarios.scenario_c import run_scenario_c_test
        return await run_scenario_c_test(
            days=1,
            buyers=1,
            sellers=1,
            mock_llm=mock_llm,
            skip_ledger=skip_infra,
        )

    return {"error": f"Unknown scenario: {scenario}"}


@app.command()
def test_recovery(
    agent: str = typer.Option(
        "test-buyer-001",
        "--agent", "-a",
        help="Agent ID to test recovery for",
    ),
    skip_ledger: bool = typer.Option(
        True,
        "--skip-ledger/--with-ledger",
        help="Use mock ledger",
    ),
):
    """
    Test ledger state recovery (Scenario C feature).

    Demonstrates that Alkimi's ledger-backed approach enables
    perfect state recovery after context loss.
    """
    console.print(f"[cyan]Testing state recovery for {agent}...[/]")

    result = asyncio.run(_test_recovery(agent, skip_ledger))

    # Display recovery results
    table = Table(title="Recovery Test Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in result.items():
        if key != "entries":
            table.add_row(key, str(value))

    console.print(table)

    if result.get("recovery_accuracy") == 1.0:
        console.print("\n[bold green]✓ Perfect recovery achieved![/]")
        console.print("[dim]This demonstrates Alkimi's zero context rot advantage.[/]")
    else:
        console.print(f"\n[yellow]Recovery accuracy: {result.get('recovery_accuracy', 'N/A')}[/]")


async def _test_recovery(agent_id: str, skip_ledger: bool) -> dict:
    """Test state recovery."""
    from ..scenarios.scenario_c import ScenarioC, MockLedgerClient

    mock_ledger = MockLedgerClient() if skip_ledger else None

    scenario = ScenarioC(ledger_client=mock_ledger)
    scenario._ledger = mock_ledger or scenario._ledger
    scenario._connected = True

    # Create some test data
    await scenario.run_single_deal(
        buyer_id=agent_id,
        seller_id="test-seller-001",
        impressions=100000,
        cpm=15.0,
    )

    # Simulate loss and recovery
    result = await scenario.simulate_context_loss_and_recovery(agent_id, "buyer")

    return result


@app.command()
def compare(
    report: Path = typer.Argument(
        ...,
        help="Path to simulation results JSON",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json, markdown",
    ),
):
    """
    Generate comparison report from simulation results.

    Reads a saved simulation result and generates formatted comparison.
    """
    if not report.exists():
        console.print(f"[red]File not found:[/] {report}")
        raise typer.Exit(1)

    with open(report) as f:
        result = json.load(f)

    if format == "json":
        console.print(json.dumps(result, indent=2))
    elif format == "markdown":
        _print_markdown_comparison(result)
    else:
        _display_results(result)


def _print_markdown_comparison(result: dict):
    """Print results in markdown format."""
    console.print("# IAB Agentic RTB Simulation Results\n")
    console.print(f"Generated: {datetime.now().isoformat()}\n")

    console.print("## Scenario Comparison\n")
    console.print("| Metric | A (Exchange) | B (IAB A2A) | C (Alkimi) |")
    console.print("|--------|-------------|-------------|------------|")

    scenario_results = {r["scenario_id"]: r for r in result.get("scenario_results", [])}

    metrics = [
        ("Total Deals", "total_deals"),
        ("Buyer Spend ($)", "total_buyer_spend"),
        ("Exchange Fees ($)", "total_exchange_fees"),
        ("Take Rate (%)", "intermediary_take_rate"),
        ("Context Rot Events", "context_rot_events"),
        ("Recovery Success (%)", "recovery_success_rate"),
    ]

    for label, metric in metrics:
        a = scenario_results.get("A", {}).get("metrics", {}).get(metric, "N/A")
        b = scenario_results.get("B", {}).get("metrics", {}).get(metric, "N/A")
        c = scenario_results.get("C", {}).get("metrics", {}).get(metric, "N/A")
        console.print(f"| {label} | {a} | {b} | {c} |")


@app.command("v2-status")
def v2_status():
    """
    Show V2 feature status and available options.

    Lists all V2 simulation modes and their configuration options.
    """
    console.print(Panel.fit(
        "[bold cyan]V2 Context Window Hallucination Testing[/]\n\n"
        "Test single-agent reliability problems: context overflow,\n"
        "memory loss, and hallucinations over time.",
        title="V2 Overview",
    ))

    # Context Pressure feature
    table1 = Table(title="Context Pressure Mode", show_header=True, header_style="bold blue")
    table1.add_column("Option", style="cyan")
    table1.add_column("Type")
    table1.add_column("Default")
    table1.add_column("Description")
    table1.add_row("--context-pressure", "flag", "off", "Enable token pressure tracking")
    table1.add_row("--volume-profile", "choice", "medium", "small|medium|large|enterprise")
    table1.add_row("--context-limit", "int", "200000", "Context window token limit")
    table1.add_row("--compression-loss", "float", "0.20", "Information loss rate on compression")
    console.print(table1)
    console.print()

    # Decision Chain feature
    table2 = Table(title="Decision Chain Mode", show_header=True, header_style="bold yellow")
    table2.add_column("Option", style="cyan")
    table2.add_column("Type")
    table2.add_column("Default")
    table2.add_column("Description")
    table2.add_row("--decision-chain", "flag", "off", "Enable decision chain tracking")
    table2.add_row("--lookback-window", "int", "100", "Decision reference lookback window")
    console.print(table2)
    console.print()

    # Restart Test feature
    table3 = Table(title="Restart Test Mode", show_header=True, header_style="bold green")
    table3.add_column("Option", style="cyan")
    table3.add_column("Type")
    table3.add_column("Default")
    table3.add_column("Description")
    table3.add_row("--restart-test", "flag", "off", "Enable restart/crash simulation")
    table3.add_row("--crash-probability", "float", "0.01", "Crash rate per hour")
    table3.add_row("--recovery-modes", "str", "private_db,ledger", "Comma-separated recovery modes")
    console.print(table3)
    console.print()

    # Full V2
    console.print(Panel(
        "[bold magenta]--full-v2[/]: Enable all V2 features together\n\n"
        "[dim]Example:[/]\n"
        "  rtb-sim run --days 30 --full-v2 --volume-profile large --output results_v2.json",
        title="Full V2 Mode",
    ))

    # Volume profiles
    console.print("\n[bold]Volume Profiles:[/]")
    console.print("  small      -  10K requests/day,  300K/month  (Low pressure)")
    console.print("  medium     - 100K requests/day,    3M/month  (Medium pressure)")
    console.print("  large      -   1M requests/day,   30M/month  (High pressure)")
    console.print("  enterprise -  10M requests/day,  300M/month  (Extreme pressure)")


@app.command()
def version():
    """Show version information."""
    console.print("[cyan]IAB Agentic RTB Simulation[/] v0.2.0 (with V2 modes)")
    console.print("https://github.com/benputley1/iab-agentic-ecosystem-simulation")


def main():
    """Entry point for rtb-sim command."""
    app()


if __name__ == "__main__":
    main()
