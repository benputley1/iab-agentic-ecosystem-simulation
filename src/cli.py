"""
RTB Simulation CLI

Main entry point for running the IAB Agentic RTB Simulation.

Usage:
    python -m src.cli run --scenario a,b,c --days 30 --buyers 5 --sellers 5
    python -m src.cli status
    python -m src.cli report --format markdown
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rtb-sim",
    help="IAB Agentic RTB Simulation - Compare exchange-based, A2A, and ledger-backed scenarios",
)
console = Console()


@app.command()
def run(
    scenarios: str = typer.Option("a,b,c", help="Scenarios to run (comma-separated: a,b,c)"),
    days: int = typer.Option(30, help="Simulation duration in days"),
    buyers: int = typer.Option(5, help="Number of buyer agents"),
    sellers: int = typer.Option(5, help="Number of seller agents"),
    campaigns_per_buyer: int = typer.Option(10, help="Campaigns per buyer"),
    acceleration: float = typer.Option(100.0, help="Time acceleration factor"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use mock LLM (no API costs)"),
    enable_chaos: bool = typer.Option(False, "--chaos", help="Enable chaos testing"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Directory for checkpoints"),
):
    """Run the RTB simulation."""
    import asyncio
    from .orchestration import SimulationRunner

    console.print(f"[bold green]Starting RTB Simulation[/bold green]")
    console.print(f"  Scenarios: {scenarios}")
    console.print(f"  Duration: {days} days")
    console.print(f"  Buyers: {buyers}")
    console.print(f"  Sellers: {sellers}")
    console.print(f"  Campaigns per buyer: {campaigns_per_buyer}")
    console.print(f"  Time acceleration: {acceleration}x")
    console.print(f"  Mock LLM: {mock_llm}")
    console.print(f"  Chaos testing: {enable_chaos}")

    runner = SimulationRunner(
        scenarios=scenarios.split(","),
        days=days,
        buyers=buyers,
        sellers=sellers,
        campaigns_per_buyer=campaigns_per_buyer,
        time_acceleration=acceleration,
        mock_llm=mock_llm,
        enable_chaos=enable_chaos,
        checkpoint_dir=checkpoint_dir,
    )

    # Calculate estimated real-world duration
    real_duration = runner.time_controller.get_real_duration_for_sim_days(days)
    console.print(f"  Estimated real-world duration: {real_duration}")

    console.print("\n[bold]Running simulation...[/bold]")

    try:
        result = asyncio.run(runner.run())

        console.print(f"\n[bold green]Simulation completed![/bold green]")
        console.print(f"  State: {result.state.value}")
        console.print(f"  Events injected: {result.total_events_injected}")
        console.print(f"  Checkpoints: {result.checkpoints_created}")

        # Display comparison table
        if result.scenario_results:
            _display_comparison_table(result.scenario_results)

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Simulation failed: {e}[/red]")
        raise typer.Exit(code=1)


def _display_comparison_table(scenario_results: list) -> None:
    """Display comparison table for scenario results."""
    table = Table(title="Scenario Comparison")

    table.add_column("Metric", style="cyan")
    for result in scenario_results:
        table.add_column(f"Scenario {result.scenario_id}", style="green")

    # Key metrics to compare
    metrics = [
        ("Total Deals", "total_deals"),
        ("Total Impressions", "total_impressions"),
        ("Buyer Spend ($)", "total_buyer_spend"),
        ("Seller Revenue ($)", "total_seller_revenue"),
        ("Exchange Fees ($)", "total_exchange_fees"),
        ("Take Rate (%)", "intermediary_take_rate"),
        ("Avg CPM ($)", "average_cpm"),
        ("Hallucination Rate (%)", "hallucination_rate"),
    ]

    for label, key in metrics:
        row = [label]
        for result in scenario_results:
            value = result.metrics.get(key, 0)
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        table.add_row(*row)

    console.print(table)


@app.command()
def status():
    """Show simulation status."""
    console.print("[bold]Simulation Status[/bold]")

    # TODO: Query database for current state
    table = Table(title="Current State")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", "Not running")
    table.add_row("Simulation Day", "-")
    table.add_row("Active Campaigns", "-")
    table.add_row("Completed Deals", "-")

    console.print(table)


@app.command()
def report(
    format: str = typer.Option("markdown", help="Output format (markdown, html, json)"),
    output: str = typer.Option(None, help="Output file path"),
):
    """Generate comparative report."""
    console.print(f"[bold]Generating Report[/bold]")
    console.print(f"  Format: {format}")

    # TODO: Import and run ReportGenerator when implemented
    console.print("\n[yellow]ReportGenerator not yet implemented[/yellow]")


@app.command()
def init_db():
    """Initialize the database schema."""
    console.print("[bold]Initializing Database[/bold]")
    console.print("Run: docker-compose -f docker/docker-compose.yml up -d")
    console.print("Database will be initialized automatically from SQL scripts.")


@app.command()
def seed():
    """Seed the database with sample data."""
    console.print("[bold]Seeding Database[/bold]")
    # TODO: Import and run seed data generation
    console.print("\n[yellow]Seed data generation not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
