"""
Benchmark Runner CLI for Veritas RAG Engine.
Executes the domain QA benchmark, prints a comparison table, and saves reports.

Everything written to the report is computed from the run. Nothing is
hardcoded as a finding, and metrics that are not measured are not shown.
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

# Make `veritas` and `benchmarks` importable when run as `python benchmarks/run_benchmarks.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from veritas.pipeline.engine import VeritasEngine
from veritas.config import VeritasConfig
from benchmarks.evaluator import BenchmarkEvaluator


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def pretty(name: str) -> str:
    return name.replace("_", " ")


# ----------------------------------------------------------------------
# Findings: computed from the results, never hardcoded
# ----------------------------------------------------------------------
def build_findings(summary, checks):
    n = next(iter(summary.values())).sample_count
    std = summary.get("Standard_RAG")
    sp = summary.get("Single_Pass_Verification")
    ver = summary.get("Veritas_Recursive")
    lines = []

    # Hallucination rate (as judged by the engine's own verifier)
    hal = {k: m.mean_hallucination_rate for k, m in summary.items()}
    if max(hal.values()) - min(hal.values()) < 1e-9:
        lines.append(
            f"- **Hallucination rate:** {pct(next(iter(hal.values())))} for every system. "
            "On this question set the three systems cannot be told apart."
        )
    elif ver is not None:
        for k, m in summary.items():
            if k == "Veritas_Recursive":
                continue
            diff = (ver.mean_hallucination_rate - m.mean_hallucination_rate) * 100
            lines.append(
                f"- **Hallucination rate vs {pretty(k)}:** Veritas {pct(ver.mean_hallucination_rate)} "
                f"vs {pct(m.mean_hallucination_rate)} ({diff:+.1f} percentage points)."
            )

    # Did the correction machinery actually run?
    if sp is not None and sp.revisions_triggered is not None:
        if sp.revisions_triggered == 0:
            lines.append(
                "- **Single-pass revision:** never ran (no claim was flagged on the first check), "
                "so that row is effectively identical to Standard RAG."
            )
        else:
            lines.append(f"- **Single-pass revision:** ran on {sp.revisions_triggered}/{n} samples.")
    if ver is not None and ver.revisions_triggered is not None:
        if ver.revisions_triggered == 0:
            lines.append(
                f"- **Recursive loop:** converged on the first pass for every sample "
                f"(mean iterations {ver.mean_iterations:.1f}), so its corrective behaviour was not exercised."
            )
        else:
            lines.append(
                f"- **Recursive loop:** needed more than one iteration on {ver.revisions_triggered}/{n} samples "
                f"(mean iterations {ver.mean_iterations:.1f})."
            )

    # Answer length
    parts = ", ".join(f"{pretty(k)} {m.mean_claims_per_answer:.1f}" for k, m in summary.items())
    note = ""
    if min(m.mean_claims_per_answer for m in summary.values()) < 2:
        note = " Very short answers leave few claims to be wrong, which limits what this set can show."
    lines.append(f"- **Answer length (mean claims per answer):** {parts}.{note}")

    # Seeded-error check on the verifier
    c = checks.get("seeded_error_check", {})
    if c.get("enabled") and c.get("tested"):
        rate = c["detected"] / c["tested"] * 100
        lines.append(
            f"- **Seeded-error check:** the verifier flagged {c['detected']}/{c['tested']} answers "
            f"({rate:.0f}%) after one fabricated claim was injected into each."
        )
        if c["detected"] < c["tested"]:
            missed = "every injected claim" if c["detected"] == 0 else "some injected claims"
            lines.append(
                f"  The verifier missed {missed}, so it can pass unsupported statements. "
                "Treat low hallucination rates with caution."
            )
        if c.get("errors"):
            lines.append(f"  {c['errors']} seeded checks errored and are excluded from this count.")

    # Latency
    if std is not None and ver is not None:
        lines.append(
            f"- **Latency:** Veritas {ver.mean_latency_seconds:.2f}s vs Standard RAG "
            f"{std.mean_latency_seconds:.2f}s per question."
        )
    return lines


def build_limitations(n):
    return [
        f"- Sample size is small ({n} questions).",
        "- Each question is answered from a single gold passage (open-book): no retrieval from the full "
        "corpus and no distractor passages.",
        "- Hallucination and attestation rates come from the engine's own verifier, "
        "which also drives Veritas. They are not judged by an independent model.",
        "- Citation precision and citation recall are not measured in this benchmark.",
    ]


def build_report(summary, checks, dataset_path):
    n = next(iter(summary.values())).sample_count
    md = [
        "# Veritas RAG Engine — Benchmark Evaluation Report",
        "",
        "## Setup",
        f"Dataset: `{dataset_path}` ({n} biomedical questions with PubMed context passages). "
        "Three configurations are compared:",
        "1. **Standard RAG**: single-pass retrieval and generation.",
        "2. **Single Pass Verification**: one check, then one revision if a claim was flagged.",
        "3. **Veritas Recursive**: iterative loop between generator and verifier with discrepancy feedback.",
        "",
        "## Results",
        "",
        "| System | Hallucination Rate ↓ | Attestation Rate ↑ | Claims / Answer | Mean Iterations | Mean Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, m in summary.items():
        md.append(
            f"| **{pretty(k)}** | `{pct(m.mean_hallucination_rate)}` | `{pct(m.mean_attestation_rate)}` | "
            f"`{m.mean_claims_per_answer:.1f}` | `{m.mean_iterations:.1f}` | `{m.mean_latency_seconds:.2f}s` |"
        )
    md.extend(["", "Citation precision/recall: not measured in this run.", "", "## Findings (computed from this run)"])
    md.extend(build_findings(summary, checks))
    md.extend(["", "## Limitations"])
    md.extend(build_limitations(n))
    return "\n".join(md)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Veritas RAG Engine Benchmarks")
    parser.add_argument("--dataset", type=str, default="benchmarks/eval_dataset.json",
                        help="Path to evaluation dataset JSON file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (optional)")
    parser.add_argument("--output_dir", type=str, default="benchmarks",
                        help="Directory to save benchmark results and report")
    parser.add_argument("--no_seeded_check", action="store_true",
                        help="Skip the seeded-error check on the verifier")
    args = parser.parse_args()

    console = Console()
    console.print("[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold cyan]       Veritas RAG Engine — Benchmark Evaluation Suite     [/bold cyan]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[bold red]Dataset not found at: {dataset_path}[/bold red]")
        return

    config = VeritasConfig()
    engine = VeritasEngine(config=config)
    evaluator = BenchmarkEvaluator(engine=engine, seeded_check=not args.no_seeded_check)

    console.print(f"Starting evaluation on dataset: [green]{dataset_path}[/green]")
    summary = evaluator.run_benchmark(dataset_path, max_samples=args.max_samples)
    checks = evaluator.checks
    n = next(iter(summary.values())).sample_count

    # ---- Console table (only measured columns) ----
    table = Table(title="Benchmark Comparative Performance (Veritas vs. Baselines)")
    table.add_column("System / Model", style="bold white")
    table.add_column("Hallucination Rate ↓", justify="right", style="red")
    table.add_column("Attestation Rate ↑", justify="right", style="green")
    table.add_column("Claims / Answer", justify="right", style="cyan")
    table.add_column("Mean Iterations", justify="right", style="magenta")
    table.add_column("Latency (s)", justify="right", style="yellow")

    for sys_name, m in summary.items():
        name_display = pretty(sys_name)
        if "Veritas" in sys_name:
            name_display = f"[bold green]{name_display} (Ours)[/bold green]"
        table.add_row(
            name_display,
            pct(m.mean_hallucination_rate),
            pct(m.mean_attestation_rate),
            f"{m.mean_claims_per_answer:.1f}",
            f"{m.mean_iterations:.1f}",
            f"{m.mean_latency_seconds:.2f}s",
        )

    console.print("\n")
    console.print(table)
    console.print("\n[bold]Findings (computed from this run)[/bold]")
    for line in build_findings(summary, checks):
        console.print(line.replace("**", ""), markup=False, highlight=False)
    console.print("\n[bold]Limitations[/bold]")
    for line in build_limitations(n):
        console.print(line.replace("**", ""), markup=False, highlight=False)
    console.print("")

    # ---- Save outputs ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in summary.items()}, f, indent=2)

    with open(out_dir / "benchmark_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    with open(out_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(build_report(summary, checks, dataset_path))

    console.print(f"[bold green]Saved benchmark report to:[/bold green] {out_dir / 'benchmark_report.md'}")
    console.print(f"[bold green]Saved raw results to:[/bold green] {out_dir / 'benchmark_results.json'}")
    console.print(f"[bold green]Saved verifier checks to:[/bold green] {out_dir / 'benchmark_checks.json'}")


if __name__ == "__main__":
    main()