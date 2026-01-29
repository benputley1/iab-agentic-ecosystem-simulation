"""
RTB Simulation CLI

Main entry point for running the IAB Agentic RTB Simulation.

Usage:
    python -m src.cli run --scenario a,b,c --days 30 --buyers 5 --sellers 5
    python -m src.cli status
    python -m src.cli report --format markdown
    python -m src.cli kpis
    python -m src.cli content --output content/
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

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
    title: str = typer.Option("RTB Simulation Comparative Report", help="Report title"),
    influx_url: str = typer.Option(None, help="InfluxDB URL"),
    influx_token: str = typer.Option(None, help="InfluxDB token"),
):
    """Generate comparative report across scenarios A, B, and C."""
    from .reports import ReportGenerator, ReportConfig, ReportFormat
    from .metrics.collector import InfluxConfig

    console.print(f"[bold]Generating Report[/bold]")
    console.print(f"  Format: {format}")
    console.print(f"  Title: {title}")

    # Configure InfluxDB connection
    influx_config = InfluxConfig()
    if influx_url:
        influx_config.url = influx_url
    if influx_token:
        influx_config.token = influx_token

    # Validate format
    try:
        report_format = ReportFormat(format.lower())
    except ValueError:
        console.print(f"[red]Invalid format: {format}. Use markdown, html, or json.[/red]")
        raise typer.Exit(1)

    # Generate report
    config = ReportConfig(
        influx_config=influx_config,
        output_format=report_format,
        output_path=output,
        title=title,
    )

    try:
        generator = ReportGenerator(config)

        with console.status("[bold green]Calculating KPIs..."):
            generated = generator.generate()

        if output:
            console.print(f"\n[green]Report saved to: {output}[/green]")
        else:
            # Display to console
            if format == "markdown":
                console.print(Panel(Markdown(generated.content), title=title))
            else:
                console.print(generated.content)

        console.print(f"\n[green]Report generated successfully![/green]")

    except Exception as e:
        console.print(f"\n[red]Error generating report: {e}[/red]")
        console.print("[yellow]Make sure InfluxDB is running and contains simulation data.[/yellow]")
        raise typer.Exit(1)


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


@app.command()
def kpis(
    scenario: str = typer.Option(None, help="Specific scenario (a, b, or c). Shows all if not specified."),
    influx_url: str = typer.Option(None, help="InfluxDB URL"),
    influx_token: str = typer.Option(None, help="InfluxDB token"),
):
    """Display KPI summary for scenarios."""
    from .reports import generate_kpi_summary
    from .metrics.collector import InfluxConfig

    console.print("[bold]KPI Summary[/bold]\n")

    influx_config = InfluxConfig()
    if influx_url:
        influx_config.url = influx_url
    if influx_token:
        influx_config.token = influx_token

    try:
        with console.status("[bold green]Querying InfluxDB..."):
            kpis = generate_kpi_summary(influx_config)

        # Fee Extraction Table
        fee_table = Table(title="Fee Extraction Comparison")
        fee_table.add_column("Scenario", style="cyan")
        fee_table.add_column("Gross Spend", justify="right")
        fee_table.add_column("Net to Publisher", justify="right")
        fee_table.add_column("Intermediary Take", justify="right", style="red")
        fee_table.add_column("Take Rate", justify="right")

        scenarios_to_show = [scenario.upper()] if scenario else ["A", "B", "C"]

        for s in scenarios_to_show:
            fee = kpis.fee_extraction.get(s)
            if fee:
                fee_table.add_row(
                    s,
                    f"${fee.gross_spend:,.2f}",
                    f"${fee.net_to_publisher:,.2f}",
                    f"${fee.intermediary_take:,.2f}",
                    f"{fee.take_rate_pct:.2f}%",
                )

        console.print(fee_table)
        console.print()

        # Goal Achievement Table
        goal_table = Table(title="Campaign Goal Achievement")
        goal_table.add_column("Scenario", style="cyan")
        goal_table.add_column("Campaigns", justify="right")
        goal_table.add_column("Goals Met", justify="right")
        goal_table.add_column("Success Rate", justify="right", style="green")
        goal_table.add_column("Avg Attainment", justify="right")

        for s in scenarios_to_show:
            goal = kpis.goal_achievement.get(s)
            if goal:
                goal_table.add_row(
                    s,
                    str(goal.total_campaigns),
                    str(goal.hit_impression_goal),
                    f"{goal.success_rate_pct:.1f}%",
                    f"{goal.avg_goal_attainment:.1f}%",
                )

        console.print(goal_table)
        console.print()

        # Context Rot Impact (B vs C)
        if not scenario or scenario.upper() in ["B", "C"]:
            rot_table = Table(title="Context Rot Impact (B vs C)")
            rot_table.add_column("Scenario", style="cyan")
            rot_table.add_column("Rot Events", justify="right")
            rot_table.add_column("Avg Keys Lost", justify="right")
            rot_table.add_column("Recovery Accuracy", justify="right", style="green")
            rot_table.add_column("Degradation", justify="right", style="red")

            for s in ["B", "C"]:
                if scenario and scenario.upper() != s:
                    continue
                rot = kpis.context_rot.get(s)
                if rot:
                    rot_table.add_row(
                        s,
                        str(rot.total_rot_events),
                        f"{rot.avg_keys_lost:.1f}",
                        f"{rot.avg_recovery_accuracy:.1%}",
                        f"{rot.degradation_pct:.1f}%",
                    )

            console.print(rot_table)
            console.print()

        # Blockchain Costs (C only)
        if (not scenario or scenario.upper() == "C") and kpis.blockchain_costs:
            bc = kpis.blockchain_costs
            bc_table = Table(title="Blockchain Infrastructure Costs (Scenario C)")
            bc_table.add_column("Metric", style="cyan")
            bc_table.add_column("Value", justify="right", style="green")

            bc_table.add_row("Total Transactions", f"{bc.total_transactions:,}")
            bc_table.add_row("Total Sui Gas", f"{bc.total_sui_gas:,.4f} SUI")
            bc_table.add_row("Total Walrus Storage", f"{bc.total_walrus_cost:,.4f} SUI")
            bc_table.add_row("Total USD Cost", f"${bc.total_usd:,.2f}")
            bc_table.add_row("Cost per 1k Impressions", f"${bc.cost_per_1k_impressions:,.4f}")
            bc_table.add_row("Exchange Fee per 1k (comparison)", f"${bc.comparison_exchange_fee_per_1k:,.2f}")

            console.print(bc_table)
            console.print()

        # Summary metrics
        if kpis.fee_reduction_pct > 0:
            console.print(Panel(
                f"[bold green]Fee Reduction (A→C):[/bold green] {kpis.fee_reduction_pct:.0f}%\n"
                f"[bold green]Savings per $100k:[/bold green] ${kpis.savings_per_100k:,.2f}\n"
                f"[bold green]Reliability Advantage (C vs B):[/bold green] {kpis.reliability_advantage:.1%}",
                title="Comparative Metrics",
            ))

    except Exception as e:
        console.print(f"\n[red]Error querying KPIs: {e}[/red]")
        console.print("[yellow]Make sure InfluxDB is running and contains simulation data.[/yellow]")
        raise typer.Exit(1)


@app.command()
def compare(
    day: int = typer.Option(None, help="Compare specific simulation day"),
    campaign: str = typer.Option(None, help="Compare specific campaign across scenarios"),
):
    """Compare scenarios for a specific day or campaign."""
    console.print("[bold]Scenario Comparison[/bold]")

    if day:
        console.print(f"\n[cyan]Day {day} Comparison:[/cyan]")
        console.print("[yellow]Requires event data from running simulation[/yellow]")

    elif campaign:
        console.print(f"\n[cyan]Campaign {campaign} Comparison:[/cyan]")
        console.print("[yellow]Requires event data from running simulation[/yellow]")

    else:
        console.print("\n[yellow]Specify --day or --campaign to compare[/yellow]")


@app.command()
def content(
    output: str = typer.Option("content", help="Output directory for articles"),
    days: int = typer.Option(30, help="Simulation days for sample data"),
    buyers: int = typer.Option(5, help="Number of buyers for sample data"),
    sellers: int = typer.Option(5, help="Number of sellers for sample data"),
    campaigns_per_buyer: int = typer.Option(10, help="Campaigns per buyer"),
    use_sample: bool = typer.Option(True, "--sample/--real", help="Use sample or real simulation data"),
):
    """Generate article series from simulation results."""
    console.print("[bold green]Generating Content Series[/bold green]")

    from .logging.content import generate_sample_simulation, ArticleGenerator, FindingExtractor

    # Generate or load event data
    if use_sample:
        console.print(f"  Generating sample simulation data...")
        console.print(f"    Days: {days}")
        console.print(f"    Buyers: {buyers}")
        console.print(f"    Sellers: {sellers}")
        console.print(f"    Campaigns per buyer: {campaigns_per_buyer}")
        event_index = generate_sample_simulation(
            days=days,
            buyers=buyers,
            sellers=sellers,
            campaigns_per_buyer=campaigns_per_buyer,
        )
        console.print(f"  [green]Generated {len(event_index.events)} events[/green]")
    else:
        # TODO: Load real simulation data from logs
        console.print("[yellow]Real data loading not yet implemented, using sample data[/yellow]")
        event_index = generate_sample_simulation(days=days)

    # Extract findings
    console.print("\n  Extracting findings...")
    extractor = FindingExtractor(event_index)
    findings = extractor.extract_top_findings(count=10)

    # Display findings summary
    findings_table = Table(title="Top Findings")
    findings_table.add_column("Rank", style="cyan", width=6)
    findings_table.add_column("Category", style="magenta", width=15)
    findings_table.add_column("Headline", style="green")
    findings_table.add_column("Significance", style="yellow", width=12)

    for i, finding in enumerate(findings, 1):
        findings_table.add_row(
            str(i),
            finding.category.value,
            finding.headline[:50] + "..." if len(finding.headline) > 50 else finding.headline,
            "★" * finding.significance,
        )

    console.print(findings_table)

    # Generate articles
    console.print("\n  Generating articles...")
    generator = ArticleGenerator(event_index)
    output_path = Path(output)
    written_files = generator.write_series(output_path)

    console.print(f"\n[bold green]Content generation complete![/bold green]")
    console.print(f"  Output directory: {output_path.absolute()}")
    console.print(f"  Files written: {len(written_files)}")
    for f in written_files:
        console.print(f"    - {f.name}")


@app.command()
def findings(
    count: int = typer.Option(5, help="Number of findings to display"),
    days: int = typer.Option(30, help="Simulation days for sample data"),
):
    """Display top findings from simulation."""
    console.print("[bold]Extracting Top Findings[/bold]")

    from .logging.content import generate_sample_simulation, FindingExtractor

    # Generate sample data
    console.print("  Generating sample simulation data...")
    event_index = generate_sample_simulation(days=days)
    console.print(f"  [green]Generated {len(event_index.events)} events[/green]")

    # Extract findings
    extractor = FindingExtractor(event_index)
    top_findings = extractor.extract_top_findings(count=count)

    # Display each finding
    for i, finding in enumerate(top_findings, 1):
        console.print(f"\n[bold cyan]Finding #{i}: {finding.headline}[/bold cyan]")
        console.print(f"[dim]Category: {finding.category.value} | Significance: {'★' * finding.significance}[/dim]")
        console.print()
        console.print(finding.summary)
        console.print()
        if finding.pull_quote:
            console.print(f'[italic]"{finding.pull_quote}"[/italic]')
        console.print("─" * 60)


if __name__ == "__main__":
    app()
