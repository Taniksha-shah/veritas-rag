"""
Veritas RAG Engine — Command Line Interface.
Usage:
  python veritas_cli.py ask "What is the relationship between clozapine and malignancy?"
  python veritas_cli.py index --docs_dir ./raw_papers --max_docs 50
  python veritas_cli.py benchmark --samples 8
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veritas.config import VeritasConfig
from veritas.pipeline.engine import VeritasEngine, VeritasResponse
from veritas.retrieval.indexer import CorpusIndexer
from benchmarks.evaluator import BenchmarkEvaluator


console = Console(safe_box=True)


def handle_ask(args):
    console.print(f"[bold cyan][Query][/bold cyan] {args.query}\n")

    config = VeritasConfig()
    config.max_verification_iterations = args.max_iter
    config.top_k_retrieval = args.top_k

    engine = VeritasEngine(config=config)

    with console.status("[bold green]Executing Veritas Recursive Verification Pipeline...[/bold green]"):
        response: VeritasResponse = engine.query(args.query)

    # Output Answer
    console.print(Panel(response.annotated_answer, title="[bold green]Verified Attested Answer[/bold green]", expand=False))

    # Metrics
    metrics_table = Table(title="Execution & Verification Metrics", show_header=False)
    metrics_table.add_row("Iterations Run", f"{response.iterations_run} (Converged: {response.converged})")
    metrics_table.add_row("Initial Hallucination Rate", f"{response.initial_hallucination_rate * 100:.1f}%")
    metrics_table.add_row("Final Hallucination Rate", f"{response.final_hallucination_rate * 100:.1f}%")
    metrics_table.add_row("Claim Attestation Rate", f"{response.final_attestation_rate * 100:.1f}%")
    metrics_table.add_row("Execution Time", f"{response.execution_time_seconds:.2f}s")
    console.print(metrics_table)
    console.print("\n")

    # Output Citations Table
    if response.citations:
        cit_table = Table(title="Sentence-to-Source-Span Citations")
        cit_table.add_column("Citation", style="bold cyan")
        cit_table.add_column("Document", style="yellow")
        cit_table.add_column("Span Offsets", style="magenta")
        cit_table.add_column("Evidence Snippet", style="white")
        cit_table.add_column("Confidence", justify="right", style="green")

        for i, c in enumerate(response.citations, start=1):
            snippet = c.get("matched_snippet", "")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            cit_table.add_row(
                f"[^{i}]",
                c.get("doc_id", "DOC"),
                f"[{c.get('span_start', 0)}:{c.get('span_end', 0)}]",
                snippet,
                f"{c.get('confidence', 0.0) * 100:.1f}%",
            )
        console.print(cit_table)


def handle_index(args):
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        console.print(f"[bold red]Directory not found: {docs_dir}[/bold red]")
        sys.exit(1)

    console.print(f"Indexing documents in [cyan]{docs_dir}[/cyan] (max docs: {args.max_docs or 'all'})...")
    indexer = CorpusIndexer()
    indexer.index_directory(docs_dir, max_docs=args.max_docs)
    console.print(f"Created [green]{len(indexer.chunks)}[/green] chunks.")

    config = VeritasConfig()
    indexer.build_and_save_index(config.index_dir)
    console.print(f"[bold green]Successfully saved hybrid index to: {config.index_dir}[/bold green]")


def handle_benchmark(args):
    dataset_path = Path("benchmarks/eval_dataset.json")
    if not dataset_path.exists():
        console.print(f"[bold red]Benchmark dataset not found: {dataset_path}[/bold red]")
        sys.exit(1)

    config = VeritasConfig()
    engine = VeritasEngine(config=config)
    evaluator = BenchmarkEvaluator(engine=engine)

    console.print(f"Running benchmarks on [green]{dataset_path}[/green]...")
    summary = evaluator.run_benchmark(dataset_path, max_samples=args.samples)

    table = Table(title="Veritas Benchmark Performance vs. Baselines")
    table.add_column("System", style="bold white")
    table.add_column("Hallucination Rate ↓", justify="right", style="red")
    table.add_column("Attestation Rate ↑", justify="right", style="green")
    table.add_column("Claims / Answer", justify="right", style="cyan")
    table.add_column("Iterations", justify="right", style="magenta")
    table.add_column("Latency (s)", justify="right", style="yellow")

    for k, v in summary.items():
        name = k.replace("_", " ")
        if "Veritas" in k:
            name = f"[bold green]{name} (Ours)[/bold green]"
        table.add_row(
            name,
            f"{v.mean_hallucination_rate * 100:.1f}%",
            f"{v.mean_attestation_rate * 100:.1f}%",
            f"{v.mean_claims_per_answer:.1f}",
            f"{v.mean_iterations:.1f}",
            f"{v.mean_latency_seconds:.2f}s",
        )

    console.print(table)

    # Notes: say what these numbers are, and what was not measured.
    console.print("[dim]Rates are judged by the engine's own verifier on a single gold passage per question. "
                  "Citation precision/recall are not measured.[/dim]")
    seeded = evaluator.checks.get("seeded_error_check", {})
    if seeded.get("enabled") and seeded.get("tested"):
        console.print(
            f"[dim]Seeded-error check: verifier flagged {seeded['detected']}/{seeded['tested']} answers "
            f"after one fabricated claim was injected into each.[/dim]"
        )


def main():
    parser = argparse.ArgumentParser(description="Veritas RAG Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ask
    ask_parser = subparsers.add_parser("ask", help="Query the Veritas engine")
    ask_parser.add_argument("query", type=str, help="Domain query string")
    ask_parser.add_argument("--top_k", type=int, default=5, help="Number of evidence passages to retrieve")
    ask_parser.add_argument("--max_iter", type=int, default=3, help="Max recursive verification iterations")

    # Index
    index_parser = subparsers.add_parser("index", help="Index document corpus")
    index_parser.add_argument("--docs_dir", type=str, default="raw_papers", help="Directory of text/PDF files")
    index_parser.add_argument("--max_docs", type=int, default=50, help="Max number of documents to index")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--samples", type=int, default=8, help="Number of benchmark samples to evaluate")

    args = parser.parse_args()

    if args.command == "ask":
        handle_ask(args)
    elif args.command == "index":
        handle_index(args)
    elif args.command == "benchmark":
        handle_benchmark(args)


if __name__ == "__main__":
    main()