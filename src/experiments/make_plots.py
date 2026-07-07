"""
make_plots.py — generate every plot in results/figures/
========================================================

Reads everything in ``results/data/`` and produces the standard set of
figures used in the report:

    results/figures/headline.svg              — 4-panel bar chart
    results/figures/coverage_over_time.svg    — cumulative identification curve
    results/figures/bandwidth_over_time.svg   — sliding-window bandwidth
    results/figures/polarization_timeline.svg — per-target polarization dots

CLI
---
    python src/experiments/make_plots.py
    python src/experiments/make_plots.py --results-dir other/dir
    python src/experiments/make_plots.py --only headline

Outputs are SVG by default (vector graphics for the report). Pass
``--png`` to additionally save PNGs at 200 dpi for slides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import _REPO_ROOT, DEFAULT_RESULTS_DIR  # noqa: E402

from stomatopod_vision.metrics import EventLog, MetricsReport  # noqa: E402
from stomatopod_vision.viz import (  # noqa: E402
    bandwidth_over_time,
    bandwidth_over_time_multi_seed,
    coverage_over_time,
    coverage_over_time_multi_seed,
    headline_bar_chart,
    headline_bar_chart_multi_seed,
    polarization_detection_timeline,
)

# Order baselines should be displayed in (left → right on bar charts)
BASELINE_ORDER = ("B1", "B2", "B3", "B3L")  # B3L = bonus, learned scoring


def _discover_results(data_dir: Path) -> tuple[
    list[MetricsReport],
    dict[str, EventLog],
]:
    """
    Load all ``<baseline>_metrics.json`` + ``<baseline>_log.json`` pairs
    from ``data_dir``, returning them in canonical baseline order.
    """
    reports: list[MetricsReport] = []
    logs: dict[str, EventLog] = {}

    for baseline in BASELINE_ORDER:
        metrics_path = data_dir / f"{baseline}_metrics.json"
        log_path = data_dir / f"{baseline}_log.json"
        if metrics_path.exists():
            reports.append(MetricsReport.load_json(metrics_path))
        if log_path.exists():
            logs[baseline] = EventLog.load_json(log_path)

    return reports, logs


def _discover_multi_seed(data_dir: Path) -> tuple[
    list[dict],
    dict[str, list[EventLog]],
]:
    """Load ``<baseline>_summary.json`` and per-seed log files.

    Returns ``(summaries, logs_by_baseline)`` where ``summaries`` is in
    canonical baseline order and ``logs_by_baseline[B]`` is the list of
    per-seed :class:`EventLog` objects ordered by the seed list inside
    the summary.

    Returns empty containers if no summaries are present (caller should
    then fall back to single-seed discovery).
    """
    import json

    summaries: list[dict] = []
    logs_by_baseline: dict[str, list[EventLog]] = {}

    for baseline in BASELINE_ORDER:
        summary_path = data_dir / f"{baseline}_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)

        per_seed_logs: list[EventLog] = []
        for seed in summary.get("seeds", []):
            lp = data_dir / f"{baseline}_seed{seed}_log.json"
            if lp.exists():
                per_seed_logs.append(EventLog.load_json(lp))
        logs_by_baseline[baseline] = per_seed_logs

    return summaries, logs_by_baseline


def _save_pair(fig, svg_path: Path, also_png: bool) -> list[Path]:
    """Save a figure to SVG and optionally also PNG. Returns paths written."""
    written = []
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, bbox_inches="tight")
    written.append(svg_path)
    if also_png:
        png_path = svg_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        written.append(png_path)
    return written



def _safe_rel(p: Path) -> Path:
    """Best-effort relative path against the repo root.

    Path.relative_to raises if p is not under _REPO_ROOT (which
    happens whenever the user passes an absolute --results-dir outside the
    repo, or runs the orchestrator with a CWD-relative path). Fall back
    to the absolute path in that case so we never crash before plotting.
    """
    try:
        return p.resolve().relative_to(_REPO_ROOT.resolve())
    except ValueError:
        return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help=f"Where to read JSON inputs from and write SVG outputs to "
             f"(default: {DEFAULT_RESULTS_DIR.relative_to(_REPO_ROOT)}).")
    parser.add_argument(
        "--png", action="store_true",
        help="Also save 200-dpi PNG copies (for slides).")
    parser.add_argument(
        "--only", choices=[
            "headline", "coverage", "bandwidth", "polarization", "all",
        ], default="all",
        help="Only render one specific figure (default: all).")
    args = parser.parse_args()

    data_dir = args.results_dir / "data"
    fig_dir = args.results_dir / "figures"

    if not data_dir.exists():
        raise SystemExit(
            f"No results directory at {data_dir}. "
            f"Did you run the baselines first?  e.g.\n"
            f"  python src/experiments/run_b1_fixed.py\n"
            f"  python src/experiments/run_b2_preprocessed.py"
        )

    # Prefer multi-seed (summary + per-seed logs) when present;
    # otherwise fall back to single-seed (one metrics + one log per baseline).
    summaries, logs_by_baseline = _discover_multi_seed(data_dir)
    multi_seed = bool(summaries)

    if multi_seed:
        n_seeds = summaries[0].get("n_seeds", "?")
        found = ", ".join(s["baseline"] for s in summaries)
        print(f"Loaded MULTI-SEED summaries for: {found}  (n={n_seeds} seeds each)")
    else:
        reports, logs = _discover_results(data_dir)
        if not reports:
            raise SystemExit(
                f"No <baseline>_metrics.json files found in {data_dir}. "
                f"Did you run any of the baseline scripts?"
            )
        found = ", ".join(r.baseline for r in reports)
        print(f"Loaded SINGLE-SEED results for: {found}")
    print(f"Writing figures to: {_safe_rel(fig_dir)}/\n")

    written: list[Path] = []

    # ---- headline (bar chart, mandatory) ----
    if args.only in ("headline", "all"):
        if multi_seed:
            fig = headline_bar_chart_multi_seed(summaries, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "headline.svg", args.png))
            print(f"  ✓ headline                ({len(summaries)} baselines × "
                  f"{summaries[0].get('n_seeds', '?')} seeds, mean ± std)")
        else:
            fig = headline_bar_chart(reports, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "headline.svg", args.png))
            print(f"  ✓ headline                ({len(reports)} baselines)")

    # ---- coverage_over_time ----
    if args.only in ("coverage", "all"):
        if multi_seed and any(logs_by_baseline.values()):
            fig = coverage_over_time_multi_seed(logs_by_baseline, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "coverage_over_time.svg",
                                      args.png))
            print(f"  ✓ coverage_over_time      ({len(logs_by_baseline)} baselines, "
                  f"mean + ribbon across seeds)")
        elif (not multi_seed) and logs:
            fig = coverage_over_time(logs, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "coverage_over_time.svg",
                                      args.png))
            print(f"  ✓ coverage_over_time      ({len(logs)} baselines)")
        else:
            print("  ⚠ coverage_over_time      skipped (no log files found)")

    # ---- bandwidth_over_time ----
    if args.only in ("bandwidth", "all"):
        if multi_seed and any(logs_by_baseline.values()):
            fig = bandwidth_over_time_multi_seed(logs_by_baseline, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "bandwidth_over_time.svg",
                                      args.png))
            print(f"  ✓ bandwidth_over_time     ({len(logs_by_baseline)} baselines, "
                  f"mean + ribbon across seeds)")
        elif (not multi_seed) and logs:
            fig = bandwidth_over_time(logs, output_path=None)
            written.extend(_save_pair(fig, fig_dir / "bandwidth_over_time.svg",
                                      args.png))
            print(f"  ✓ bandwidth_over_time     ({len(logs)} baselines)")
        else:
            print("  ⚠ bandwidth_over_time     skipped (no log files found)")

    # ---- polarization timeline (per-target chart; uses seed 0 in multi-seed mode) ----
    if args.only in ("polarization", "all"):
        if multi_seed and any(logs_by_baseline.values()):
            # Multi-seed timelines would be visually cluttered; use the first
            # seed as a representative trace and call that out in the title.
            first_logs = {b: lst[0] for b, lst in logs_by_baseline.items() if lst}
            fig = polarization_detection_timeline(
                first_logs, output_path=None,
                title="Polarization detection timeline  "
                      "(representative trace, seed 0)",
            )
            written.extend(_save_pair(fig,
                                      fig_dir / "polarization_timeline.svg",
                                      args.png))
            print(f"  ✓ polarization_timeline   ({len(first_logs)} baselines, "
                  f"seed-0 representative trace)")
        elif (not multi_seed) and logs:
            fig = polarization_detection_timeline(logs, output_path=None)
            written.extend(_save_pair(fig,
                                      fig_dir / "polarization_timeline.svg",
                                      args.png))
            print(f"  ✓ polarization_timeline   ({len(logs)} baselines)")
        else:
            print("  ⚠ polarization_timeline   skipped (no log files found)")

    print(f"\nWrote {len(written)} file(s):")
    for p in written:
        print(f"  {_safe_rel(p)}")


if __name__ == "__main__":
    main()
