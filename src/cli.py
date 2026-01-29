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
):
    """Run the RTB simulation."""
    console.print(f"[bold green]Starting RTB Simulation[/bold green]")
    console.print(f"  Scenarios: {scenarios}")
    console.print(f"  Duration: {days} days")
    console.print(f"  Buyers: {buyers}")
    console.print(f"  Sellers: {sellers}")
    console.print(f"  Campaigns per buyer: {campaigns_per_buyer}")
    console.print(f"  Time acceleration: {acceleration}x")
    console.print(f"  Mock LLM: {mock_llm}")

    # TODO: Import and run SimulationRunner when implemented
    console.print("\n[yellow]SimulationRunner not yet implemented[/yellow]")


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
