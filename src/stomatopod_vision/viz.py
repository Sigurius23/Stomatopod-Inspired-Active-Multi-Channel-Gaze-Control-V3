"""
viz.py — visualisation helpers
==============================

Convenience plotters and renderers used by the experiment scripts and
exploratory notebooks. Two flavours:

  - **Offline plots** (matplotlib): comparison bar charts and time-series
    figures for the report. These are the production functions.
  - **Live overlays** (MuJoCo viewer extensions) + **video recording**:
    stubs reserved for week 5. They need the scheduler in place to be
    visually informative; using them today would just show static eyes.

The functions here are deliberately thin wrappers — they each save a
single figure file. The experiment scripts in ``src/experiments/``
orchestrate them.

All plotters return the matplotlib ``Figure`` they produced (in case
the caller wants to embed it in a notebook or compose with other axes)
and ALSO save it to disk if ``output_path`` is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol

if TYPE_CHECKING:
    from .gimbal_control import GimbalSetpoint

# Use matplotlib's non-interactive backend by default so the scripts
# can run headless without an X server.
import matplotlib
import numpy as np

if matplotlib.get_backend() not in ("Agg", "agg"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import EventLog, MetricsReport  # noqa: E402

# =====================================================================
# Style: house colours per baseline
# =====================================================================

#: Stable, colour-blind-safe colours so all plots agree.
BASELINE_COLORS: dict[str, str] = {
    "B1":  "#bdbdbd",   # grey   — "passive baseline"
    "B2":  "#1f77b4",   # blue   — "in-sensor preprocessing"
    "B3":  "#d62728",   # red    — "active scanning + preprocessing"
    "B3L": "#9467bd",   # purple — bonus: B3 with a LEARNED scoring sum
    "B3D": "#2ca02c",   # green  — bonus: B3 with a Hopf DYNAMICAL scanner
}

# Human-readable labels for axes/legends
BASELINE_LABELS: dict[str, str] = {
    "B1":  "B1 — fixed cameras\n(raw stream)",
    "B2":  "B2 — fixed cameras\n+ preprocessing",
    "B3":  "B3 — active scanning\n+ preprocessing",
    "B3L": "B3-Learned — active scanning\n+ LEARNED scoring",
    "B3D": "B3-Dynamical — Hopf\nlimit-cycle scanning",
}

# Compact labels for bar-chart x-ticks: the two-line ``BASELINE_LABELS``
# above collide horizontally when four baselines share one axis, so we
# use these shorter one-line variants there. Kept in a separate dict so
# callers can pick whichever form fits their layout.
SHORT_BASELINE_LABELS: dict[str, str] = {
    "B1":  "B1 (fixed, raw)",
    "B2":  "B2 (fixed, events)",
    "B3":  "B3 (active)",
    "B3L": "B3L (active, learned)",
    "B3D": "B3D (active, Hopf)",
}


# =====================================================================
# 1. The headline figure — four-panel bar chart
# =====================================================================

def headline_bar_chart(
    reports: Iterable[MetricsReport],
    output_path: str | Path | None = "results/figures/headline.svg",
    title: str | None = None,
) -> plt.Figure:
    """
    Generate the four-panel bar chart comparing baselines on:
        - Coverage             (0..1, higher is better)
        - Bandwidth            (bytes/s, log y, lower is better)
        - Polarization accuracy (0..1, higher is better)
        - Median latency       (seconds, lower is better)

    This is the project's headline figure. ``reports`` should contain
    one :class:`MetricsReport` per baseline you want to compare. Order
    is preserved.

    Parameters
    ----------
    reports :
        Iterable of :class:`MetricsReport`. Their ``baseline`` attribute
        determines colour and label.
    output_path :
        Where to save the figure. If ``None``, the figure is returned
        but not saved. Format inferred from extension (``.svg`` for the
        report, ``.png`` for slides).
    title :
        Optional suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (already saved if ``output_path`` was set).
    """
    reports = list(reports)
    if not reports:
        raise ValueError("headline_bar_chart needs at least one report")

    labels = [r.baseline for r in reports]
    colors = [BASELINE_COLORS.get(b, "#777777") for b in labels]
    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    (ax_cov, ax_bw), (ax_pol, ax_lat) = axs

    # --- coverage ---
    cov_vals = [r.coverage for r in reports]
    ax_cov.bar(x, cov_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_cov.set_ylim(0, 1.05)
    ax_cov.set_ylabel("Coverage")
    ax_cov.set_title("Coverage  (higher is better)")
    _annotate(ax_cov, x, cov_vals, fmt="{:.2f}")

    # --- bandwidth (log) ---
    bw_vals = [max(r.bandwidth_bps, 1e-3) for r in reports]   # avoid log(0)
    ax_bw.bar(x, bw_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_bw.set_yscale("log")
    if bw_vals:
        ax_bw.set_ylim(bottom=1, top=max(bw_vals) * 10)
    ax_bw.set_ylabel("Bandwidth (bytes/s, log)")
    ax_bw.set_title("Bandwidth  (lower is better)")
    _annotate(ax_bw, x, bw_vals, fmt=_log_fmt, color="black")

    # --- polarization accuracy ---
    pol_vals = [r.polarization_accuracy for r in reports]
    ax_pol.bar(x, pol_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_pol.set_ylim(0, 1.05)
    ax_pol.set_ylabel("Linear polarization accuracy")
    ax_pol.set_title("Linear polarization accuracy  (higher is better)")
    _annotate(ax_pol, x, pol_vals, fmt="{:.2f}")

    # --- latency ---
    lat_vals = [r.median_latency_s for r in reports]
    ax_lat.bar(x, lat_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_lat.set_ylabel("Median latency (s)")
    ax_lat.set_title("Median latency  (lower is better)")
    _annotate(ax_lat, x, lat_vals, fmt="{:.2f}s")
    lat_top = max(lat_vals) if lat_vals else 0.0
    if lat_top <= 1e-6:
        ax_lat.set_ylim(0, 0.10)
        ax_lat.text(0.5, 0.55,
                    "all baselines identified their target within\n"
                    "the first controller step (10 ms)",
                    transform=ax_lat.transAxes,
                    ha="center", va="center",
                    fontsize=9, style="italic", color="#555",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="#bbb",
                              linewidth=0.6))
    else:
        ax_lat.set_ylim(0, max(lat_top * 1.20, 0.10))

    short_pretty = [SHORT_BASELINE_LABELS.get(b, b) for b in labels]
    for ax in (ax_cov, ax_bw, ax_pol, ax_lat):
        ax.set_xticks(x)
        ax.set_xticklabels(short_pretty, fontsize=9, rotation=20, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# =====================================================================
# 2. Coverage over time — line plot
# =====================================================================

def coverage_over_time(
    logs: dict[str, EventLog],
    output_path: str | Path | None = "results/figures/coverage_over_time.svg",
    identification_window_s: float = 0.5,
    title: str | None = None,
) -> plt.Figure:
    """
    Per-baseline curve of cumulative interesting-target identifications
    versus simulation time.

    Each line shows the number of *distinct* interesting targets that
    have been correctly identified (matching ``dominant_class`` within
    ``identification_window_s`` of FoV entry) at each point in time.
    A line that reaches the total number of interesting targets early
    is better.

    Parameters
    ----------
    logs :
        ``{"B1": log_b1, "B2": log_b2, ...}``.
    output_path :
        Where to save the figure (None = don't save).
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # Total interesting targets (consistent across baselines on the same scene)
    total_interesting = max(
        (len(log.interesting_targets) for log in logs.values()),
        default=0,
    )

    for baseline, log in logs.items():
        times, cumulative = _cumulative_identifications(
            log, identification_window_s,
        )
        if len(times) == 0:
            # No identifications at all — still draw a flat zero line so
            # the legend includes this baseline.
            times = np.array([0.0, log.duration_s])
            cumulative = np.array([0, 0])
        ax.step(
            times, cumulative,
            where="post",
            color=BASELINE_COLORS.get(baseline, "#777777"),
            label=BASELINE_LABELS.get(baseline, baseline).replace("\n", " "),
            linewidth=2.0,
        )

    if total_interesting > 0:
        ax.axhline(total_interesting, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5,
                   label=f"total interesting = {total_interesting}")
        ax.axhspan(total_interesting, max(total_interesting + 1.0, 1.5) + 1.0,
                   color="black", alpha=0.05, zorder=0)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Cumulative interesting targets identified")
    ax.set_ylim(-0.4, max(total_interesting + 1.0, 1.5))
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3, zorder=1)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Coverage over time")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# =====================================================================
# 3. Bandwidth over time — sliding-window line plot
# =====================================================================

def bandwidth_over_time(
    logs: dict[str, EventLog],
    output_path: str | Path | None = "results/figures/bandwidth_over_time.svg",
    window_s: float = 1.0,
    n_eyes: int = 2,
    raw_sample_rate_hz: float = 500.0,
    title: str | None = None,
) -> plt.Figure:
    """
    Sliding-window bandwidth usage (bytes/s) for each baseline over the
    run. Shows the bursty-vs-steady character of the streams.

    For ``B1`` we draw a flat horizontal line equal to the steady-state
    rate ``n_eyes × bytes_per_sample × sample_rate``. For ``B2``/``B3``
    we count events emitted in each sliding window of width ``window_s``.

    Parameters
    ----------
    logs :
        ``{"B1": log_b1, "B2": log_b2, ...}``.
    output_path :
        Where to save the figure (None = don't save).
    window_s :
        Width of the sliding window (seconds). Default 1 s gives
        bytes-per-second directly.
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for baseline, log in logs.items():
        duration = log.duration_s
        if duration <= 0:
            continue
        color = BASELINE_COLORS.get(baseline, "#777777")
        label = BASELINE_LABELS.get(baseline, baseline).replace("\n", " ")

        _floor = 0.5
        if baseline == "B1":
            rate = n_eyes * log.raw_stream_bytes_per_sample * raw_sample_rate_hz
            ax.plot([0, log.duration_s], [rate, rate], color=color, linewidth=2.0, label=label)
        else:
            t, bw = _sliding_window_bandwidth(log, window_s)
            never_emitted = (len(bw) == 0 or float(np.max(bw)) < _floor)
            lbl = label + ("  — 0 B/s (drawn at floor)" if never_emitted else "")
            ax.plot(t, np.maximum(bw, _floor),
                    color=color, linewidth=1.6, label=lbl)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Bandwidth (bytes/s, log)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3)   # so the floor-line stays inside the plot
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")
    ax.grid(which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Bandwidth over time  ({window_s:g}-s sliding window)")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# =====================================================================
# 4. Polarization detection timeline — Gantt-style chart
# =====================================================================

def polarization_detection_timeline(
    logs: dict[str, EventLog],
    output_path: str | Path | None = "results/figures/polarization_timeline.svg",
    tolerance_rad: float = np.deg2rad(15.0),
    title: str | None = None,
) -> plt.Figure:
    """
    For each polarized target, show a horizontal stripe of dots: each dot
    marks an event whose decoded polarization is within ``tolerance_rad``
    of the true value. One stripe per (baseline × target) pair.

    Lets the reader see at a glance that only B3 produces consistent
    polarization decodings.

    Parameters
    ----------
    logs :
        ``{"B1": log_b1, "B2": log_b2, ...}``.
    """
    # Collect the set of polarized targets across all baselines
    polarized: dict[str, float] = {}
    circular: dict[str, str] = {}
    for log in logs.values():
        polarized.update(log.polarized_targets)
        circular.update(log.circular_targets)

    if not polarized and not circular:
        # Nothing to plot — produce a blank figure with an explanatory note
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        ax.text(0.5, 0.5,
                "No polarized targets in scene\n— nothing to plot.",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
        return fig

    baselines = list(logs.keys())
    target_names = sorted(list(polarized.keys()) + list(circular.keys()))
    max_duration = max(log.duration_s for log in logs.values())

    # First pass: compute (row_data) for every (target, baseline) pair,
    # but skip rows where the baseline never emitted an event for the
    # target. This is what makes the hard-scene plot readable — B1 and
    # B2 have no events at all, so their 20 empty rows were pure noise.
    rows: list[dict] = []
    silent_baselines: set[str] = set(baselines)
    for tgt_name in target_names:
        is_circular = tgt_name not in polarized
        true_angle = polarized.get(tgt_name, 0.0)
        true_hand = circular.get(tgt_name, "")

        for baseline in baselines:
            log = logs[baseline]
            ts_ok: list[float] = []
            ts_bad: list[float] = []
            for ev in log.preprocessed_events:
                if ev.target_name != tgt_name:
                    continue
                if is_circular:
                    if ev.circular_handedness is None:
                        continue
                    if ev.circular_handedness == true_hand:
                        ts_ok.append(ev.time)
                    else:
                        ts_bad.append(ev.time)
                else:
                    if ev.polarization_angle is None:
                        continue
                    err = _wrap_pi(ev.polarization_angle, true_angle)
                    if err <= tolerance_rad:
                        ts_ok.append(ev.time)
                    else:
                        ts_bad.append(ev.time)
            if not ts_ok and not ts_bad:
                continue  # skip — this baseline is silent on this target
            silent_baselines.discard(baseline)
            rows.append(dict(
                tgt=tgt_name, baseline=baseline,
                ts_ok=ts_ok, ts_bad=ts_bad,
                color=BASELINE_COLORS.get(baseline, "#777777"),
            ))

    # Size the figure to the actual (non-empty) row count, with a
    # sensible per-row height.
    n_rows = len(rows)
    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
        ax.text(0.5, 0.5,
                "No polarization decodings in any log — nothing to plot.",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
        return fig
    fig, ax = plt.subplots(
        figsize=(9, max(2.2, 0.32 * n_rows + 1.2)),
        constrained_layout=True,
    )

    # Alternating row-band shading, for readability
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#f4f4f4",
                       edgecolor="none", zorder=0)

    y_labels = []
    for i, r in enumerate(rows):
        ax.scatter(r["ts_ok"], [i] * len(r["ts_ok"]), marker="o",
                   color=r["color"], s=22, zorder=3)
        ax.scatter(r["ts_bad"], [i] * len(r["ts_bad"]), marker="x",
                   color=r["color"], s=32, linewidth=1.4, zorder=4)
        y_labels.append(f"{r['tgt']}  /  {r['baseline']}")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Simulation time (s)")
    ax.set_xlim(0, max_duration)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()  # top → bottom reads top-target first, more natural
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend: two markers, plus a note about which baselines are silent
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="k",
                   markersize=8, label="correct (≤15° or handedness)"),
        plt.Line2D([0], [0], marker="x", color="k", linestyle="None",
                   markersize=8, markeredgewidth=1.4, label="incorrect"),
    ]
    if silent_baselines:
        silent_str = ", ".join(sorted(silent_baselines))
        legend_handles.append(
            plt.Line2D([], [], marker="", color="none",
                       label=f"(rows omitted for silent baselines: {silent_str})")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Polarization detection timeline")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# =====================================================================
# Helpers (internal)
# =====================================================================

def _annotate(ax, x, vals, fmt="{:.2f}", color="black"):
    """Write each value above its bar."""
    for xi, v in zip(x, vals, strict=True):
        text = fmt(v) if callable(fmt) else fmt.format(v)
        ax.annotate(text, xy=(xi, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=color)


def _log_fmt(v: float) -> str:
    """Format a number for log axis: '12', '480', '1.2k', '105k'."""
    if v >= 1_000:
        return f"{v / 1_000:,.1f}k"
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:.2g}"


def _cumulative_identifications(
    log: EventLog,
    identification_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times, cumulative_count) for distinct interesting targets
    identified over the run. Reuses the same definition as
    :func:`metrics.coverage`.
    """
    id_times: list[float] = []
    for name in log.interesting_targets:
        first_seen = log.first_fov_entry(name)
        if first_seen is None:
            continue
        true_class = log.target_true_class.get(name)
        if true_class is None:
            continue
        deadline = first_seen + identification_window_s
        for ev in log.preprocessed_events:
            if ev.target_name != name:
                continue
            if ev.time > deadline:
                break
            from .world import SPECTRAL_CLASSES
            idx = SPECTRAL_CLASSES.index(true_class) if true_class in SPECTRAL_CLASSES else 0
            pat = [0]*len(SPECTRAL_CLASSES)
            pat[idx] = 10
            if ev.spectral_pattern == tuple(pat):
                id_times.append(ev.time)
                break
    id_times.sort()
    times = np.asarray(id_times, dtype=float)
    cumulative = np.arange(1, len(times) + 1)
    return times, cumulative


def _sliding_window_bandwidth(
    log: EventLog,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (t, bytes_per_second) by counting events in sliding windows
    of width ``window_s`` over the run.
    """
    duration = log.duration_s
    if duration <= 0 or window_s <= 0:
        return np.array([]), np.array([])

    # Sample every ~window/10 for a smooth curve
    step = max(window_s / 10.0, 1e-4)
    t = np.arange(0.0, duration + step, step)

    event_times = np.array([e.time for e in log.preprocessed_events])
    if event_times.size == 0:
        return t, np.zeros_like(t)

    counts = np.zeros_like(t)
    for i, ti in enumerate(t):
        lo, hi = ti - window_s, ti
        counts[i] = np.sum((event_times >= lo) & (event_times <= hi))

    bw = counts * log.event_bytes_per_event / window_s
    return t, bw


def _wrap_pi(a: float, b: float) -> float:
    """Doubled-angle distance between two polarization angles ∈ [0, π)."""
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)


# =====================================================================
# Multi-seed variants — error bars + ribbons (used when run_all.py is
# invoked with --seeds N N N ... and a <baseline>_summary.json exists)
# =====================================================================

def headline_bar_chart_multi_seed(
    summaries: "Iterable[dict]",
    output_path: str | Path | None = "results/figures/headline.svg",
    title: str | None = None,
) -> plt.Figure:
    """Multi-seed version of :func:`headline_bar_chart` with error bars.

    Parameters
    ----------
    summaries :
        Iterable of summary dicts as written by
        ``_common.write_summary_if_multi_seed``. Each must contain
        ``baseline``, ``n_seeds``, and keys ``coverage``,
        ``bandwidth_bps``, ``polarization_accuracy``,
        ``median_latency_s`` mapping to dicts ``{"mean", "std",
        "min", "max", "values"}``.

    The bars show the mean across seeds; the error bars show ±1
    population standard deviation. For the bandwidth panel (log
    y-axis) the error bars are clipped to keep the lower bound
    positive.
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError("headline_bar_chart_multi_seed needs at least one summary")

    labels = [s["baseline"] for s in summaries]
    colors = [BASELINE_COLORS.get(b, "#777777") for b in labels]
    n_seeds = summaries[0].get("n_seeds", "?")
    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    (ax_cov, ax_bw), (ax_pol, ax_lat) = axs

    def _bar_with_err(ax, key, *, log=False, ylim=None, ylabel="", title="",
                      fmt="{:.2f}"):
        means = np.array([s[key]["mean"] for s in summaries])
        stds  = np.array([s[key]["std"]  for s in summaries])
        if log:
            # For log axes, clip the lower error bar so we don't dip
            # to ≤0. Use the bar value itself as the floor.
            lower = np.minimum(stds, means * 0.999)
            yerr = np.vstack([lower, stds])
            plot_means = np.maximum(means, 1e-3)
        else:
            yerr = np.vstack([stds, stds])
            plot_means = means
        ax.bar(x, plot_means, color=colors, edgecolor="black", linewidth=0.6,
               yerr=yerr, capsize=4,
               error_kw={"elinewidth": 1.0, "ecolor": "#222"})
        if log:
            ax.set_yscale("log")
            if len(plot_means) > 0:
                top_val = max(m + sd for m, sd in zip(means, stds, strict=True))
                ax.set_ylim(bottom=1, top=top_val * 10)
            else:
                ax.set_ylim(bottom=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Annotate each bar with mean±std
        for xi, m, sd in zip(x, means, stds, strict=True):
            text = _log_fmt(m) if fmt is _log_fmt else fmt.format(m)
            if sd > 0:
                text += "\n±" + (_log_fmt(sd) if fmt is _log_fmt else fmt.format(sd))
            # Place text above the bar (in linear) or just above (in log).
            top = (plot_means[xi] + (stds[xi] if not log else 0)) if not log else plot_means[xi]
            ax.annotate(text, (xi, top),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")

    _bar_with_err(ax_cov, "coverage",
                  ylim=(0, 1.15), ylabel="Coverage",
                  title="Coverage  (higher is better)")
    _bar_with_err(ax_bw, "bandwidth_bps",
                  log=True, ylabel="Bandwidth (bytes/s, log)",
                  title="Bandwidth  (lower is better)", fmt=_log_fmt)
    _bar_with_err(ax_pol, "polarization_accuracy",
                  ylim=(0, 1.15), ylabel="Linear polarization accuracy",
                  title="Linear polarization accuracy  (higher is better)")

    # Latency: pick a sensible y-limit so bars are always visible.
    lat_means = np.array([s_["median_latency_s"]["mean"] for s_ in summaries])
    lat_stds  = np.array([s_["median_latency_s"]["std"]  for s_ in summaries])
    lat_top   = float(np.max(lat_means + lat_stds))
    if lat_top <= 1e-6:
        # Everyone identified their target within the first controller step:
        # bars would be invisible. Set a small y-window and annotate.
        _bar_with_err(ax_lat, "median_latency_s",
                      ylim=(0, 0.10),
                      ylabel="Median latency (s)",
                      title="Median latency  (lower is better)", fmt="{:.2f}s")
        ax_lat.text(0.5, 0.55,
                    "all baselines identified their target within\n"
                    "the first controller step (10 ms)",
                    transform=ax_lat.transAxes,
                    ha="center", va="center",
                    fontsize=9, style="italic", color="#555",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="#bbb",
                              linewidth=0.6))
    else:
        _bar_with_err(ax_lat, "median_latency_s",
                      ylim=(0, max(lat_top * 1.20, 0.10)),
                      ylabel="Median latency (s)",
                      title="Median latency  (lower is better)", fmt="{:.2f}s")

    # Short, non-colliding x-tick labels (see SHORT_BASELINE_LABELS docstring).
    short_pretty = [SHORT_BASELINE_LABELS.get(b, b) for b in labels]
    for ax in (ax_cov, ax_bw, ax_pol, ax_lat):
        ax.set_xticks(x)
        ax.set_xticklabels(short_pretty, fontsize=9, rotation=20, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    if title is None:
        title = f"Mean ± 1 std across {n_seeds} seeds"
    fig.suptitle(title, fontsize=11, y=1.02)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


def coverage_over_time_multi_seed(
    logs_by_baseline: dict[str, list[EventLog]],
    output_path: str | Path | None = "results/figures/coverage_over_time.svg",
    identification_window_s: float = 0.5,
    title: str | None = None,
) -> plt.Figure:
    """Mean coverage curve per baseline, with ±1 std ribbon across seeds.

    ``logs_by_baseline`` maps each baseline name to a list of
    :class:`EventLog` (one per seed). Each log is converted to a
    cumulative-identifications curve, the curves are resampled onto a
    common time grid, and the per-time mean ± std is plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    total_interesting = 0
    for logs in logs_by_baseline.values():
        for log in logs:
            total_interesting = max(total_interesting, len(log.interesting_targets))

    # Common time grid (200 samples spanning the longest run).
    duration = max(
        (log.duration_s for logs in logs_by_baseline.values() for log in logs),
        default=1.0,
    )
    t_grid = np.linspace(0.0, duration, 200)

    for baseline, logs in logs_by_baseline.items():
        if not logs:
            continue
        color = BASELINE_COLORS.get(baseline, "#777777")
        label = BASELINE_LABELS.get(baseline, baseline).replace("\n", " ")

        # Resample each seed's step curve onto t_grid.
        curves = np.zeros((len(logs), len(t_grid)))
        for i, log in enumerate(logs):
            t_steps, cum = _cumulative_identifications(log, identification_window_s)
            if len(t_steps) == 0:
                continue
            # Step interpolation: for each grid time, the cumulative
            # count is the count at the most recent step ≤ that time.
            idx = np.searchsorted(t_steps, t_grid, side="right") - 1
            idx = np.clip(idx, 0, len(cum) - 1)
            curves[i] = np.where(idx >= 0, cum[idx], 0)
            curves[i, t_grid < t_steps[0]] = 0

        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(t_grid, mean, color=color, linewidth=2.0,
                label=f"{label}  (n={len(logs)})")
        ax.fill_between(t_grid, mean - std, mean + std,
                        color=color, alpha=0.20, linewidth=0)

    if total_interesting > 0:
        ax.axhline(total_interesting, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5,
                   label=f"total interesting = {total_interesting}")
        ax.axhspan(total_interesting, max(total_interesting + 1.0, 1.5) + 1.0,
                   color="black", alpha=0.05, zorder=0)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Cumulative interesting targets identified")
    # Extend y-limit slightly below 0 so flat-zero baselines (B1/B2 on the
    # hard scene) are visibly drawn at y=0 instead of being hidden by the
    # bottom axis line.
    y_top = max(total_interesting + 1.0, 1.5)
    ax.set_ylim(-0.4, y_top)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3, zorder=1)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title or "Coverage over time  (mean ± 1 std across seeds)")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


def bandwidth_over_time_multi_seed(
    logs_by_baseline: dict[str, list[EventLog]],
    output_path: str | Path | None = "results/figures/bandwidth_over_time.svg",
    window_s: float = 1.0,
    n_eyes: int = 2,
    raw_sample_rate_hz: float = 500.0,
    title: str | None = None,
) -> plt.Figure:
    """Mean sliding-window bandwidth per baseline with ±1 std ribbon."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    duration = max(
        (log.duration_s for logs in logs_by_baseline.values() for log in logs),
        default=1.0,
    )
    t_grid = np.linspace(window_s, duration, 200)

    for baseline, logs in logs_by_baseline.items():
        if not logs:
            continue
        color = BASELINE_COLORS.get(baseline, "#777777")
        label = BASELINE_LABELS.get(baseline, baseline).replace("\n", " ")

        if baseline == "B1":
            # B1 = flat rate; identical across seeds. Draw single line.
            rate = n_eyes * logs[0].raw_stream_bytes_per_sample * raw_sample_rate_hz
            ax.plot([0, duration], [rate, rate], color=color, linewidth=2.0,
                    label=f"{label}  (n={len(logs)})")
            continue

        curves = np.zeros((len(logs), len(t_grid)))
        for i, log in enumerate(logs):
            t, bw = _sliding_window_bandwidth(log, window_s)
            if len(t) == 0:
                continue
            # Linear-interpolate the per-seed curve onto the common grid.
            curves[i] = np.interp(t_grid, t, bw, left=0.0, right=bw[-1])

        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        # Floor zero-valued means so log-y doesn't send the line to -inf
        # (and vanish off the plot). We use a floor of 0.5 B/s, well below
        # the sub-1 B/s regime that is effectively "no events".
        _floor = 0.5
        display_mean = np.maximum(mean, _floor)
        display_lo   = np.maximum(mean - std, _floor)
        display_hi   = np.maximum(mean + std, _floor)
        never_emitted = float(mean.max()) < _floor
        lbl = f"{label}  (n={len(logs)})"
        if never_emitted:
            lbl += "  — 0 B/s (drawn at floor)"
        ax.plot(t_grid, display_mean, color=color, linewidth=1.6, label=lbl)
        ax.fill_between(t_grid, display_lo, display_hi,
                        color=color, alpha=0.20, linewidth=0)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Bandwidth (bytes/s, log)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3)
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")
    ax.grid(which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title or
                 f"Bandwidth over time  ({window_s:g}-s sliding window, "
                 f"mean ± 1 std across seeds)")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# =====================================================================
# Live MuJoCo overlays + video recording  (Bonus #8)
# =====================================================================
#
# All overlays operate on a populated MuJoCo MjvScene *before* it is
# rendered. They append small primitive geoms (cones / spheres / lines)
# to scene.geoms[scene.ngeom] and bump scene.ngeom. This is more
# efficient than post-processing the rendered RGB frame and gives us
# correct depth ordering for free.
#
# record_run() ties everything together: it runs a simulation while
# repeatedly calling the supplied overlay functions and writing the
# rendered frames to an MP4 with mediapy.

import mujoco  # noqa: E402

# RGB tuples for overlay primitives (alpha set per-call)
_FOV_COLOR_L = (0.85, 0.20, 0.20)
_FOV_COLOR_R = (0.20, 0.20, 0.85)
_SALIENCY_COLOR = (1.0, 0.85, 0.10)
_SIGHTING_COLOR = (0.15, 0.85, 0.15)


def _push_geom(
    scene: "mujoco.MjvScene",
    *,
    geom_type: int,
    pos: np.ndarray,
    size: np.ndarray,
    mat: np.ndarray | None = None,
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5),
) -> bool:
    """Append one primitive geom to ``scene``. Returns False if full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    if mat is None:
        mat = np.eye(3, dtype=np.float64)
    mujoco.mjv_initGeom(
        g,
        type=int(geom_type),
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.asarray(mat, dtype=np.float64).reshape(9),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def _eye_forward_axis_world(model: "mujoco.MjModel", data: "mujoco.MjData",
                            eye: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eye_centre_world, forward_dir_world) for eye \"L\" or \"R\".

    Uses the sites ``eye_<EYE>_center`` and ``eye_<EYE>_axis`` defined
    in the XML; their difference gives the forward direction.
    """
    centre_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_center")
    axis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_axis")
    centre = data.site_xpos[centre_id].copy()
    axis = data.site_xpos[axis_id].copy()
    fwd = axis - centre
    n = np.linalg.norm(fwd)
    if n > 1e-9:
        fwd /= n
    return centre, fwd


def render_eye_fov_overlay(
    scene: "mujoco.MjvScene",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    *,
    cone_length: float = 1.2,
    yaw_half_angle: float = np.deg2rad(60.0),
    pitch_half_angle: float = np.deg2rad(5.0),
    alpha: float = 0.25,
) -> None:
    """Append two coloured cones to ``scene`` showing each eye\'s FoV.

    The mid-band FoV is very wide in azimuth (~60°) and narrow in
    pitch (~5°), so we approximate it with a thin ellipse-base cone
    drawn from the eye centre out to ``cone_length`` metres ahead.
    Left eye = red, right eye = blue.
    """
    for eye, colour in (("L", _FOV_COLOR_L), ("R", _FOV_COLOR_R)):
        centre, fwd = _eye_forward_axis_world(model, data, eye)
        # Place an ellipsoid representing the (squashed) FoV cone halfway
        # along the eye\'s forward axis. radius_az = L * tan(half_yaw),
        # radius_el = L * tan(half_pitch).
        L = float(cone_length)
        rx = L * float(np.tan(yaw_half_angle))
        rz = L * float(np.tan(pitch_half_angle))
        # Build a rotation matrix from the eye\'s local frame (z = up,
        # y = forward) to world. The simplest robust construction is a
        # full orthonormal basis from `fwd` plus a world-up reference.
        y_axis = fwd
        z_ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_axis, z_ref)) > 0.99:
            z_ref = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(y_axis, z_ref)
        x_axis /= max(np.linalg.norm(x_axis), 1e-9)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(np.linalg.norm(z_axis), 1e-9)
        mat = np.column_stack([x_axis, y_axis, z_axis])  # (3,3) world<-local
        pos = centre + 0.5 * L * fwd
        # Ellipsoid half-sizes (x = azimuth radius, y = depth/2, z = pitch radius)
        size = np.array([rx, 0.5 * L, rz])
        _push_geom(
            scene,
            geom_type=int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            pos=pos, size=size, mat=mat,
            rgba=(*colour, float(alpha)),
        )


def render_saliency_map_overlay(
    scene: "mujoco.MjvScene",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    candidates_per_eye: "dict[str, np.ndarray] | None" = None,
    scores_per_eye: "dict[str, np.ndarray] | None" = None,
    *,
    max_dots: int = 30,
    distance: float = 0.6,
    base_size: float = 0.015,
) -> None:
    """Paint the scheduler\'s candidate scores as little spheres in space.

    For each eye, a coloured dot is placed at the world position the
    candidate (yaw, pitch) setpoint would point at (relative to the
    eye centre), at a fixed ``distance`` ahead. Dot size scales with
    score so the chosen candidate visibly stands out.

    Accepts the optional pre-sampled candidates + scores; if either is
    None, the function silently no-ops (caller has nothing to draw).
    """
    if candidates_per_eye is None or scores_per_eye is None:
        return
    for eye in ("L", "R"):
        cands = candidates_per_eye.get(eye)
        scores = scores_per_eye.get(eye)
        if cands is None or scores is None or len(cands) == 0:
            continue
        # Normalise scores → [0,1]
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        if s_max - s_min < 1e-9:
            s_norm = np.full_like(scores, 0.5, dtype=np.float64)
        else:
            s_norm = (scores - s_min) / (s_max - s_min)

        # Sort by score so the brightest dot wins the depth fight
        order = np.argsort(s_norm)[-max_dots:]
        centre, _ = _eye_forward_axis_world(model, data, eye)
        for i in order:
            yaw, pitch, _ = cands[i]
            # Spherical to Cartesian relative to the head's forward
            # direction. We deliberately project candidates onto the
            # head\'s coordinate frame (not the rotated eye frame) so
            # the dots stay stable as the eye moves.
            #   yaw rotates around z, pitch around the rotated y.
            # Sign for eye_L mirrors yaw (see scheduler._EYE_AZIMUTH_SIGN).
            sign = -1.0 if eye == "L" else 1.0
            world_dir = np.array([
                sign * np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                -np.sin(pitch),
            ])
            pos = centre + distance * world_dir
            sphere_size = base_size * (0.5 + 1.5 * float(s_norm[i]))
            alpha = 0.25 + 0.7 * float(s_norm[i])
            _push_geom(
                scene,
                geom_type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                pos=pos,
                size=np.array([sphere_size, sphere_size, sphere_size]),
                rgba=(*_SALIENCY_COLOR, float(alpha)),
            )


def render_recent_sightings(
    scene: "mujoco.MjvScene",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    log: "EventLog | None",
    time_now: float,
    *,
    lookback_s: float = 1.0,
    ring_size: float = 0.06,
) -> None:
    """Paint a translucent green ring around each recently-seen target.

    ``log`` is the same :class:`EventLog` populated by
    ``_common.run_simulation``. Targets that received a preprocessed
    event in the last ``lookback_s`` seconds get a halo.
    """
    if log is None:
        return
    recent_names: set[str] = set()
    for ev in reversed(log.preprocessed_events):
        if time_now - ev.time > lookback_s:
            break
        recent_names.add(ev.target_name)
    for name in recent_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        pos = data.xpos[bid].copy()
        # Halo: a slightly larger translucent sphere co-located with target
        _push_geom(
            scene,
            geom_type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
            pos=pos,
            size=np.array([ring_size, ring_size, ring_size]),
            rgba=(*_SIGHTING_COLOR, 0.30),
        )



# ---- typed protocols used by record_run --------------------------------
class _StepCallable(Protocol):
    """Anything with a `.step(...)` method — covers MovingTargetController, etc."""
    def step(self, *args, **kwargs): ...


class _PipelineLike(Protocol):
    """Subset of PreprocessingPipeline that record_run uses."""
    def step(self, raws, *, time_now, roll_angles): ...


# ---------------------------------------------------------------------
# Camera-pan helper for record_run
# ---------------------------------------------------------------------

# Module-level state used only by _apply_camera_pan. ``_pan_cache`` stores
# the initial (pos, lookat) snapshot per (renderer, camera) so subsequent
# frames can rotate about a consistent axis. ``_pan_data`` is a small
# indirection so the helper can call ``renderer.update_scene(data, ...)``
# without threading MjData through every overlay signature.
_pan_cache: "dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]" = {}
_pan_data: "dict[int, object]" = {}


def _apply_camera_pan(
    renderer: "Any",
    model: "Any",
    camera_name: str,
    *,
    total_pan_deg: float,
    progress: float,
) -> None:
    """Rotate the camera around the world +z axis about its own lookat.

    Called once per rendered frame. We snapshot the named camera's
    body-frame position + orientation from the model on the first call
    (cached in module-level ``_pan_cache`` keyed by
    ``(id(renderer), camera_name)``), then apply a per-frame rotation
    about a lookat point derived from the camera's forward ray.

    The pan is symmetric around the mid-clip: at progress=0 the camera
    is at ``-total_pan_deg/2``, at progress=1 it is at ``+total_pan_deg/2``.
    The mid-clip position matches the XML's canonical framing exactly.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        # Named camera not found — fall back to the free camera.
        data = _pan_data.get(id(renderer))
        if data is not None:
            renderer.update_scene(data, camera=-1)
        return

    key = (id(renderer), camera_name)
    if key not in _pan_cache:
        pos0 = model.cam_pos[cam_id].copy()
        mat0 = model.cam_mat0[cam_id].reshape(3, 3).copy()
        forward = -mat0[:, 2]                         # world-frame forward
        # Lookat = where the forward ray crosses z = 1.05 m (scene centre).
        z_target = 1.05
        if abs(forward[2]) > 1e-6:
            t = (z_target - pos0[2]) / forward[2]
        else:
            t = 2.0
        lookat = pos0 + t * forward
        _pan_cache[key] = (pos0, lookat)
    pos0, lookat = _pan_cache[key]

    # Rotate (pos0 - lookat) about world +z by theta radians, add back lookat.
    theta = float(np.deg2rad(total_pan_deg * (progress - 0.5)))
    rel = pos0 - lookat
    c, s_ = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    new_pos = lookat + rot @ rel

    # Also update the camera orientation so it still points at the lookat.
    fwd = lookat - new_pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= r_norm
    up = np.cross(right, fwd)
    # MuJoCo camera matrix columns: [right, up, -forward] (camera looks -z).
    mat = np.column_stack([right, up, -fwd])
    model.cam_pos[cam_id] = new_pos
    model.cam_mat0[cam_id] = mat.reshape(9)

    # Update the scene against the mutated camera (uses the mj_data stashed
    # by record_run in _pan_data).
    data = _pan_data.get(id(renderer))
    if data is not None:
        renderer.update_scene(data, camera=cam_id)


def record_run(
    model_path: "str | Path",
    setpoint_at: "Callable[[float], \"GimbalSetpoint\"]",
    duration_s: float,
    output_path: "str | Path",
    *,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: "str | None" = None,
    pipeline: "_PipelineLike | None" = None,
    on_events: "Callable[[list, float], None] | None" = None,
    overlays: "list[Callable[..., None]] | None" = None,
    scheduler_for_overlays: "object | None" = None,
    motion_controller: "_StepCallable | None" = None,
    camera_pan_deg: float = 0.0,
    quiet: bool = True,
) -> "Path":
    """Run a simulation and write an MP4 with optional live overlays.

    Parameters
    ----------
    model_path :
        Path to the MuJoCo XML.
    setpoint_at :
        Same callback shape as ``_common.run_simulation`` —
        ``(time_now: float) -> GimbalSetpoint``.
    duration_s :
        Simulated seconds to record.
    output_path :
        Where to write the MP4 (parent dirs created if missing).
    fps :
        Output video frames per second (default 30).
    width, height :
        Output resolution in pixels.
    camera :
        Named MuJoCo camera. ``None`` → use the free default camera.
    pipeline :
        Optional :class:`PreprocessingPipeline` (for B2/B3-style runs
        that need to feed the scheduler).
    on_events :
        Optional ``(events, time_now) -> None`` callback, same as
        ``run_simulation``. Used to update scheduler memory.
    overlays :
        List of overlay callables. Each is invoked as
        ``f(scene, model, data, **kwargs)`` *after* ``update_scene()``
        and *before* ``render()``. ``kwargs`` contains ``log`` (the
        populated :class:`EventLog`), ``time_now``, ``scheduler``
        (==scheduler_for_overlays).
    scheduler_for_overlays :
        Passed through to each overlay as the ``scheduler`` kwarg, so
        ``render_saliency_map_overlay`` can sample fresh candidates if
        it wants to.
    camera_pan_deg :
        If nonzero, slowly rotate the camera around its lookat point
        over the full clip duration (total sweep in degrees). Purely
        cosmetic: useful for B1/B2 clips where the physics is static
        so an otherwise-frozen frame reads as "video is broken". A
        gentle 8–15° pan keeps the clip visibly alive without
        distracting from the "eyes locked" story. Only applies when
        ``camera`` names an existing MuJoCo camera (needs the camera's
        pos + xyaxes to rotate about the lookat).

    Returns
    -------
    output_path : Path
        The MP4 file written.
    """
    # Local imports keep the top-of-file import surface small.
    from pathlib import Path as _Path

    from stomatopod_vision.gimbal_control import GimbalPD
    from stomatopod_vision.metrics import EventLog
    from stomatopod_vision.sensor import make_eye_pair
    from stomatopod_vision.world import Scene

    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scene_world = Scene.from_xml(model_path)
    scene_world.reset()
    eye_L, eye_R = make_eye_pair(scene_world)
    pd = GimbalPD(scene_world.model)
    log = EventLog()
    log.populate_targets_from_scene(scene_world)

    timestep = scene_world.model.opt.timestep
    n_steps = int(duration_s / timestep)
    steps_per_frame = max(1, int(round((1.0 / fps) / timestep)))
    frames: list[np.ndarray] = []

    def _now() -> float:
        return float(scene_world.data.time)

    # Use the Renderer as a context manager so the EGL context is
    # always released (even on exception). Without this we get noisy
    # EGL teardown errors when the GC eventually collects the renderer
    # in a thread that no longer holds the GL context.
    with mujoco.Renderer(scene_world.model, height=height,
                         width=width) as renderer:
        # Stash the MjData for _apply_camera_pan (which needs it inside
        # its own update_scene call). Cleared after the with-block exits.
        _pan_data[id(renderer)] = scene_world.data
        for step_idx in range(n_steps):
            # 1. Setpoint → PD → (optional) move targets → physics
            sp = setpoint_at(_now())
            pd.step(scene_world.data, sp)
            if motion_controller is not None:
                motion_controller.step(_now())
            mujoco.mj_step(scene_world.model, scene_world.data)
            t = _now()

            # 2. Sensors + (optional) preprocessing
            raws = eye_L.step() + eye_R.step()
            for r in raws:
                log.log_raw_sighting(t, r)
            if pipeline is not None:
                events = pipeline.step(
                    raws, time_now=t,
                    roll_angles={"L": eye_L.roll_angle(),
                                 "R": eye_R.roll_angle()},
                )
                for ev in events:
                    log.log_event(ev)
                if on_events is not None:
                    on_events(events, t)

            # 3. Render this frame, but only at the output frame rate
            if step_idx % steps_per_frame != 0:
                continue

            # ----- cosmetic camera pan (optional) -----
            # For "frozen scene" clips (B1/B2), a gentle orbit around
            # the lookat prevents the video from looking like a still.
            if camera_pan_deg and camera:
                _apply_camera_pan(
                    renderer, scene_world.model, camera,
                    total_pan_deg=camera_pan_deg,
                    progress=step_idx / max(1, n_steps - 1),
                )
            else:
                renderer.update_scene(scene_world.data,
                                      camera=camera if camera else -1)
            if overlays:
                for fn in overlays:
                    fn(
                        renderer.scene, scene_world.model, scene_world.data,
                        log=log, time_now=t,
                        scheduler=scheduler_for_overlays,
                    )
            frames.append(renderer.render())

            if not quiet and step_idx % (steps_per_frame * 30) == 0:
                print(f"  recording … t={t:5.2f}s  frames={len(frames):4d}")

    _pan_data.pop(id(renderer), None)
    _write_video_frames(out, frames, fps=fps)
    if not quiet:
        print(f"  wrote {len(frames)} frames → {out}  ({fps} fps, {width}×{height})")
    return out


def _write_video_frames(out: "Path", frames: "list[np.ndarray]",
                        *, fps: int) -> None:
    """Encode ``frames`` (list of HxWx3 uint8) to the file at ``out``.

    Tries ``imageio`` (which ships ffmpeg via the ``imageio-ffmpeg``
    package), then falls back to ``mediapy`` (system ffmpeg), and
    finally to a per-frame PNG dump if no encoder is available.
    """
    out_str = str(out)
    try:
        import imageio.v3 as iio
        # `pyav` plugin requires extra deps; the default ffmpeg plugin
        # works with imageio-ffmpeg out of the box.
        iio.imwrite(out_str, frames, fps=fps)
        return
    except Exception as e_iio:
        try:
            import mediapy
            mediapy.write_video(out_str, np.stack(frames, axis=0), fps=fps)
            return
        except Exception as e_mp:
            # Last-ditch fallback: per-frame PNGs in a folder next to ``out``.
            png_dir = out.with_suffix("").parent / (out.stem + "_frames")
            png_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                # Use matplotlib for a no-dep PNG writer.
                plt.imsave(png_dir / f"frame_{i:05d}.png", fr)
            print(f"  ⚠ no video encoder found; wrote {len(frames)} PNGs to "
                  f"{png_dir} (imageio error: {e_iio}; mediapy: {e_mp})")
