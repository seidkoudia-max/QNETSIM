from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .config import StbdConstellationConfig
from .metrics_collector import CandidateRecord, MetricsSummary
from .qnetsim_adapter import LaunchResult, TimeSeriesRow


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "savefig.bbox": "tight",
    }
)


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_timeseries_csv(path: Path, rows: Sequence[TimeSeriesRow]) -> None:
    data = [
        {
            "time_s": float(row.time_s),
            "throughput_pairs_per_s": float(row.throughput_pairs_per_s),
            "fidelity": float(row.fidelity),
            "delayed_fidelity": float(row.delayed_fidelity),
            "hops": float(row.hops),
            "availability": float(row.availability),
        }
        for row in rows
    ]
    _write_csv(
        path,
        data,
        [
            "time_s",
            "throughput_pairs_per_s",
            "fidelity",
            "delayed_fidelity",
            "hops",
            "availability",
        ],
    )


def write_candidate_metrics_csv(path: Path, candidates: Sequence[CandidateRecord]) -> None:
    rows = [candidate.to_flat_dict() for candidate in candidates]
    if not rows:
        _write_csv(path, [], [])
        return
    fieldnames = list(rows[0].keys())
    _write_csv(path, rows, fieldnames)


def plot_throughput_vs_time(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    pair_label: str,
) -> None:
    xs = [float(row.time_s) for row in rows]
    ys = [float(row.throughput_pairs_per_s) for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(xs, ys, color="#0b4fa8", linewidth=2.5, marker="o", markersize=3.5)
    ax.set_title(f"Ground-to-Ground ({pair_label}) – Throughput")
    ax.set_xlabel("Transmission duration (s)")
    ax.set_ylabel("Throughput (entangled pairs / s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fidelity_vs_time(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    pair_label: str,
) -> None:
    xs = [float(row.time_s) for row in rows]
    stbd = [float(row.fidelity) for row in rows]
    delayed = [float(row.delayed_fidelity) for row in rows]
    zeros = [0.0 for _row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(
        xs,
        stbd,
        color="#0b4fa8",
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        label="STBD (segmented nesting)",
    )
    ax.plot(
        xs,
        delayed,
        color="#f97316",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3.0,
        label="Wait until full path is ready",
    )
    ax.plot(
        xs,
        zeros,
        color="#9ca3af",
        linewidth=1.8,
        linestyle=":",
        label="No entanglement path (fidelity = 0)",
    )
    ax.set_title(f"Ground-to-Ground ({pair_label}) – Fidelity")
    ax.set_xlabel("Transmission duration (s)")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hops_vs_time(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    pair_label: str,
) -> None:
    xs = [float(row.time_s) for row in rows]
    ys = [float(row.hops) for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.step(xs, ys, where="mid", color="#7c3aed", linewidth=2.3)
    ax.set_title(f"Ground-to-Ground ({pair_label}) – Hops")
    ax.set_xlabel("Transmission duration (s)")
    ax.set_ylabel("Average hops in active route")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_bucketed_delivery_profile(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    pair_label: str,
    bin_s: float,
) -> None:
    xs = [float(row.time_s) for row in rows]
    throughput = [float(row.throughput_pairs_per_s) for row in rows]
    fidelity = [float(row.fidelity) for row in rows]
    availability = [float(row.availability) for row in rows]
    if not xs:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.text(
            0.5,
            0.5,
            "No bucketed delivery samples are available for this run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Ground-to-Ground ({pair_label}) – Bucketed Delivery Profile")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
    ax0, ax1 = axes
    ax0.step(xs, throughput, where="mid", color="#0b4fa8", linewidth=2.5)
    ax0.plot(xs, throughput, color="#0b4fa8", marker="o", markersize=3.2, linewidth=0.0)
    ax0.fill_between(
        xs,
        0.0,
        throughput,
        where=[value > 0.0 for value in throughput],
        step="mid",
        alpha=0.16,
        color="#93c5fd",
    )
    ax0.set_ylabel("Throughput (pairs / s)")
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1.step(xs, fidelity, where="mid", color="#dc2626", linewidth=2.3)
    ax1.plot(xs, fidelity, color="#dc2626", marker="o", markersize=3.0, linewidth=0.0)
    ax1.fill_between(
        xs,
        0.0,
        1.0,
        where=[value > 0.0 for value in availability],
        step="mid",
        alpha=0.08,
        color="#86efac",
        label="Delivery-active bucket",
    )
    ax1.set_xlabel("Transmission duration (s)")
    ax1.set_ylabel("Fidelity")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right")

    fig.suptitle(
        f"Ground-to-Ground ({pair_label}) – Bucketed Delivery Profile ({int(round(bin_s))} s bins)"
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_availability_timeline(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    pair_label: str,
) -> None:
    xs = [float(row.time_s) for row in rows]
    ys = [float(row.availability) for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.fill_between(xs, ys, step="mid", alpha=0.35, color="#10b981")
    ax.step(xs, ys, where="mid", color="#047857", linewidth=2.2)
    ax.set_title(f"Ground-to-Ground ({pair_label}) – Path Availability")
    ax.set_xlabel("Transmission duration (s)")
    ax.set_ylabel("Availability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _active_window_bounds(rows: Sequence[TimeSeriesRow]) -> tuple[int, int] | None:
    active_indices = [
        idx
        for idx, row in enumerate(rows)
        if float(row.availability) > 0.0 or float(row.throughput_pairs_per_s) > 0.0
    ]
    if not active_indices:
        return None
    start = min(active_indices)
    end = max(active_indices)
    return start, end


def _series_bin_s(rows: Sequence[TimeSeriesRow]) -> float:
    if len(rows) > 1:
        return max(1e-9, float(rows[1].time_s) - float(rows[0].time_s))
    return 1.0


def _rasterize_launch_service_window_envelope(
    *,
    launches: Sequence[LaunchResult],
    raw_event_rows: Sequence[dict],
    total_duration_s: float,
    bin_s: float,
) -> List[TimeSeriesRow]:
    rows: List[TimeSeriesRow] = []
    active_launches = [
        launch
        for launch in launches
        if bool(launch.route_found)
        and not any(
            value != value
            for value in (
                float(launch.route_start_s),
                float(launch.route_end_s),
                float(launch.mean_fidelity_success),
                float(launch.delayed_fidelity),
            )
        )
    ]

    def _service_bounds(launch: LaunchResult) -> tuple[float, float]:
        if launch.segment_start_s and launch.segment_end_s:
            return float(launch.segment_start_s[-1]), float(launch.segment_end_s[-1])
        return float(launch.route_start_s), float(launch.route_end_s)

    launch_event_stats: Dict[float, tuple[int, int]] = {}
    for event in raw_event_rows:
        launch_s = float(event.get("launch_s", 0.0))
        count = int(event.get("count", 0))
        total_pairs, total_events = launch_event_stats.get(launch_s, (0, 0))
        launch_event_stats[launch_s] = (total_pairs + count, total_events + 1)

    def _average_pairs_per_event(launch: LaunchResult) -> float:
        total_pairs, total_events = launch_event_stats.get(float(launch.launch_s), (0, 0))
        if total_events > 0:
            return float(total_pairs) / float(total_events)
        return float(launch.success_count) if float(launch.success_count) > 0.0 else 0.0

    t = 0.0
    while t < total_duration_s - 1e-12:
        t_mid = t + 0.5 * bin_s
        window_launches = [
            launch
            for launch in active_launches
            if _service_bounds(launch)[0] - 1e-12
            <= t_mid
            < _service_bounds(launch)[1] + 1e-12
        ]
        if window_launches:
            throughput = max(_average_pairs_per_event(launch) for launch in window_launches)
            weights = [max(1.0, float(launch.success_count)) for launch in window_launches]
            total_weight = sum(weights)
            fidelity = (
                sum(
                    float(launch.mean_fidelity_success) * weight
                    for launch, weight in zip(window_launches, weights)
                )
                / total_weight
                if total_weight > 0.0
                else 0.0
            )
            delayed_fidelity = (
                sum(
                    float(launch.delayed_fidelity) * weight
                    for launch, weight in zip(window_launches, weights)
                )
                / total_weight
                if total_weight > 0.0
                else 0.0
            )
            hops = (
                sum(float(launch.hops) * weight for launch, weight in zip(window_launches, weights))
                / total_weight
                if total_weight > 0.0
                else 0.0
            )
            availability = 1.0
        else:
            throughput = 0.0
            fidelity = 0.0
            delayed_fidelity = 0.0
            hops = 0.0
            availability = 0.0
        rows.append(
            TimeSeriesRow(
                time_s=float(t_mid),
                throughput_pairs_per_s=float(throughput),
                fidelity=float(fidelity),
                delayed_fidelity=float(delayed_fidelity),
                hops=float(hops),
                availability=float(availability),
            )
        )
        t += float(bin_s)
    return rows


def plot_service_window_zoom(
    out_path: Path,
    *,
    rows: Sequence[TimeSeriesRow],
    launch_rows: Sequence[LaunchResult],
    raw_event_rows: Sequence[dict],
    pair_label: str,
    throughput_ylim: tuple[float, float] | None = None,
    fidelity_ylim: tuple[float, float] | None = None,
    title_suffix: str = "Availability Window Zoom",
) -> None:
    envelope_rows = _rasterize_launch_service_window_envelope(
        launches=launch_rows,
        raw_event_rows=raw_event_rows,
        total_duration_s=max((float(row.time_s) for row in rows), default=0.0)
        + (_series_bin_s(rows) if rows else 0.0),
        bin_s=_series_bin_s(rows),
    )
    window = _active_window_bounds(envelope_rows)
    if window is None:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.text(
            0.5,
            0.5,
            "No end-to-end service window was found in this run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Ground-to-Ground ({pair_label}) – {title_suffix}")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    start_idx, end_idx = window
    pad = max(2, int(round(0.15 * max(1, end_idx - start_idx + 1))))
    lo = max(0, start_idx - pad)
    hi = min(len(envelope_rows), end_idx + pad + 1)
    active_rows = envelope_rows[lo:hi]

    xs = [float(row.time_s) for row in active_rows]
    throughput = [float(row.throughput_pairs_per_s) for row in active_rows]
    fidelity = [float(row.fidelity) for row in active_rows]
    availability = [float(row.availability) for row in active_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
    ax0, ax1 = axes

    ax0.plot(
        xs,
        throughput,
        color="#0b4fa8",
        linewidth=2.5,
        marker="o",
        markersize=3.6,
        label="STBD avg pairs / event",
    )
    ax0.fill_between(
        xs,
        0.0,
        throughput,
        where=[avail > 0.0 for avail in availability],
        alpha=0.18,
        color="#93c5fd",
        step="mid",
    )
    ax0.step(
        xs,
        throughput,
        where="mid",
        color="#1d4ed8",
        linewidth=1.6,
        alpha=0.8,
    )
    ax0.set_ylabel("Average delivered pairs / event")
    if throughput_ylim is not None:
        y_lo, y_hi = throughput_ylim
        data_hi = max(throughput) if throughput else y_hi
        if data_hi > y_hi:
            y_hi = max(y_hi, 1.10 * data_hi)
        ax0.set_ylim(y_lo, y_hi)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="upper right")

    ax1.plot(
        xs,
        fidelity,
        color="#dc2626",
        linewidth=2.4,
        marker="o",
        markersize=3.4,
        label="Delivered fidelity",
    )
    ax1.step(
        xs,
        fidelity,
        where="mid",
        color="#b91c1c",
        linewidth=1.4,
        alpha=0.85,
    )
    ax1.fill_between(
        xs,
        0.0,
        1.0,
        where=[avail > 0.0 for avail in availability],
        alpha=0.10,
        color="#86efac",
        step="mid",
        label="Availability window",
    )
    ax1.set_xlabel("Transmission duration (s)")
    ax1.set_ylabel("Fidelity")
    ax1.set_ylim(*(fidelity_ylim if fidelity_ylim is not None else (0.0, 1.0)))
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right")

    fig.suptitle(f"Ground-to-Ground ({pair_label}) – {title_suffix}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _aggregate_raw_events(raw_event_rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[float, dict] = {}
    for event in raw_event_rows:
        t_s = float(event.get("abs_time_s", 0.0))
        count = int(event.get("count", 0))
        fidelity = float(event.get("fidelity", 0.0))
        bucket = grouped.setdefault(
            t_s,
            {
                "abs_time_s": t_s,
                "count": 0,
                "weighted_fidelity_sum": 0.0,
                "launches": set(),
            },
        )
        bucket["count"] += count
        bucket["weighted_fidelity_sum"] += fidelity * max(0, count)
        launch_s = event.get("launch_s")
        if launch_s is not None:
            bucket["launches"].add(float(launch_s))
    rows: List[dict] = []
    for t_s in sorted(grouped):
        bucket = grouped[t_s]
        total_count = int(bucket["count"])
        rows.append(
            {
                "abs_time_s": float(t_s),
                "count": total_count,
                "fidelity": (
                    float(bucket["weighted_fidelity_sum"]) / float(total_count)
                    if total_count > 0
                    else 0.0
                ),
                "launch_count": len(bucket["launches"]),
            }
        )
    return rows


def plot_event_delivery_timeline(
    out_path: Path,
    *,
    raw_event_rows: Sequence[dict],
    pair_label: str,
) -> None:
    aggregated = _aggregate_raw_events(raw_event_rows)
    if not aggregated:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.text(
            0.5,
            0.5,
            "No delivered end-to-end events were recorded in this run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Ground-to-Ground ({pair_label}) – Event-Level Delivery Timeline")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    xs = [float(row["abs_time_s"]) for row in aggregated]
    counts = [int(row["count"]) for row in aggregated]
    fidelities = [float(row["fidelity"]) for row in aggregated]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
    ax0, ax1 = axes

    ax0.vlines(xs, [0.0] * len(xs), counts, color="#0b4fa8", linewidth=2.0, alpha=0.95)
    ax0.scatter(xs, counts, color="#0b4fa8", s=22, zorder=3)
    ax0.set_ylabel("Delivered pairs / event")
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1.plot(
        xs,
        fidelities,
        color="#dc2626",
        linewidth=2.2,
        marker="o",
        markersize=3.2,
        label="Delivered fidelity",
    )
    ax1.scatter(xs, fidelities, color="#dc2626", s=18, zorder=3)
    ax1.set_xlabel("Transmission duration (s)")
    ax1.set_ylabel("Fidelity")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right")

    fig.suptitle(f"Ground-to-Ground ({pair_label}) – Event-Level Delivery Timeline")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _spread_duplicate_event_times(rows: Sequence[dict]) -> List[float]:
    if not rows:
        return []
    unique_times = sorted({float(row["abs_time_s"]) for row in rows})
    if len(unique_times) > 1:
        gaps = [
            max(1e-9, unique_times[idx + 1] - unique_times[idx])
            for idx in range(len(unique_times) - 1)
        ]
        base_gap = min(gaps)
    else:
        base_gap = 30.0

    # Visual-only separation for same-timestamp events. Keep offsets small
    # relative to the smallest actual time gap so we don't imply a different
    # snapshot structure.
    spread_span = max(1.0, min(12.0, 0.35 * base_gap))

    grouped: Dict[float, List[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(float(row["abs_time_s"]), []).append(idx)

    xs = [float(row["abs_time_s"]) for row in rows]
    for t_s, indices in grouped.items():
        if len(indices) <= 1:
            continue
        if len(indices) == 2:
            offsets = [-0.5 * spread_span, 0.5 * spread_span]
        else:
            step = spread_span / max(1, len(indices) - 1)
            start = -0.5 * spread_span
            offsets = [start + i * step for i in range(len(indices))]
        for idx, offset in zip(indices, offsets):
            xs[idx] = float(t_s) + float(offset)
    return xs


def _service_bounds_by_launch(launches: Sequence[LaunchResult]) -> Dict[float, tuple[float, float]]:
    bounds: Dict[float, tuple[float, float]] = {}
    for launch in launches:
        if launch.segment_start_s and launch.segment_end_s:
            start_s = float(launch.segment_start_s[-1])
            end_s = float(launch.segment_end_s[-1])
        else:
            start_s = float(launch.route_start_s)
            end_s = float(launch.route_end_s)
        if end_s > start_s:
            bounds[float(launch.launch_s)] = (start_s, end_s)
    return bounds


def _spread_events_within_launch_windows(
    rows: Sequence[dict],
    *,
    launches: Sequence[LaunchResult],
) -> List[float]:
    if not rows:
        return []

    service_bounds = _service_bounds_by_launch(launches)
    xs = [float(row["abs_time_s"]) for row in rows]
    launch_groups: Dict[float, List[int]] = {}
    for idx, row in enumerate(rows):
        launch_groups.setdefault(float(row["launch_s"]), []).append(idx)

    for launch_s, indices in launch_groups.items():
        bounds = service_bounds.get(float(launch_s))
        if bounds is None or len(indices) <= 1:
            continue

        start_s, end_s = bounds
        window_s = max(1e-9, end_s - start_s)

        original_times = [float(rows[idx]["abs_time_s"]) for idx in indices]
        if len(set(original_times)) == len(original_times):
            continue

        # Spread events visually across the active final service window for
        # this launch. This preserves ordering while avoiding the misleading
        # vertical pile-up of same-timestamp deliveries.
        usable_start = start_s + 0.08 * window_s
        usable_end = end_s - 0.08 * window_s
        if usable_end <= usable_start:
            usable_start = start_s
            usable_end = end_s
        span = max(1e-9, usable_end - usable_start)
        step = span / float(len(indices) + 1)
        for order, idx in enumerate(indices, start=1):
            xs[idx] = usable_start + step * float(order)

    return xs


def plot_all_events_timeline(
    out_path: Path,
    *,
    raw_event_rows: Sequence[dict],
    launches: Sequence[LaunchResult],
    pair_label: str,
) -> None:
    rows = sorted(
        (
            {
                "abs_time_s": float(event.get("abs_time_s", 0.0)),
                "count": int(event.get("count", 0)),
                "fidelity": float(event.get("fidelity", 0.0)),
                "launch_s": float(event.get("launch_s", 0.0)),
            }
            for event in raw_event_rows
        ),
        key=lambda row: (row["abs_time_s"], row["launch_s"]),
    )
    if not rows:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.text(
            0.5,
            0.5,
            "No raw end-to-end delivery events were recorded in this run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Ground-to-Ground ({pair_label}) – All Delivery Events")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    xs = _spread_events_within_launch_windows(rows, launches=launches)
    if not xs:
        xs = _spread_duplicate_event_times(rows)
    counts = [int(row["count"]) for row in rows]
    fidelities = [float(row["fidelity"]) for row in rows]
    launches = [float(row["launch_s"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.8), sharex=True)
    ax0, ax1 = axes

    ax0.vlines(xs, [0.0] * len(xs), counts, color="#0b4fa8", linewidth=1.6, alpha=0.65)
    ax0.scatter(xs, counts, c=launches, cmap="viridis", s=18, zorder=3)
    ax0.set_ylabel("Delivered pairs / event")
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1.scatter(xs, fidelities, c=launches, cmap="viridis", s=18, zorder=3)
    ax1.plot(xs, fidelities, color="#dc2626", linewidth=0.8, alpha=0.25)
    ax1.set_xlabel("Transmission duration (s)")
    ax1.set_ylabel("Fidelity")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"Ground-to-Ground ({pair_label}) – All Delivery Events")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _short_node_label(name: str) -> str:
    text = str(name)
    if "Luxembourg" in text:
        return "Lux"
    if "Singapore" in text:
        return "SG"
    return text.replace("OPT-", "S")


def plot_path_timeline(
    out_path: Path,
    *,
    launches: Sequence[LaunchResult],
    pair_label: str,
) -> None:
    active_launches = [
        launch
        for launch in launches
        if bool(launch.route_found)
        and launch.segment_start_s
        and launch.segment_end_s
    ]
    if not active_launches:
        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        ax.text(
            0.5,
            0.5,
            "No STBD route timeline is available for this run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Ground-to-Ground ({pair_label}) – STBD Path Timeline")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    active_launches = sorted(active_launches, key=lambda launch: float(launch.launch_s))
    unique_snapshots = sorted(
        {
            int(snapshot_idx)
            for launch in active_launches
            for snapshot_idx in launch.segment_snapshot_indices
        }
    )
    color_cycle = list(plt.cm.tab20.colors)
    snapshot_colors = {
        snapshot_idx: color_cycle[idx % len(color_cycle)]
        for idx, snapshot_idx in enumerate(unique_snapshots)
    }

    route_end_max = max(float(launch.route_end_s) for launch in active_launches)
    label_margin = max(250.0, 0.22 * max(1.0, route_end_max))
    fig_height = max(5.2, 0.62 * len(active_launches) + 2.4)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))

    row_height = 0.8
    row_gap = 0.35
    yticks: List[float] = []
    ylabels: List[str] = []

    for row_idx, launch in enumerate(active_launches):
        y_base = row_idx * (row_height + row_gap)
        center_y = y_base + 0.5 * row_height
        yticks.append(center_y)
        ylabels.append(
            f"{int(round(float(launch.launch_s)))} s | "
            f"{int(launch.hops)} hops | {int(launch.segment_count)} seg"
        )

        full_window = max(0.0, float(launch.route_end_s) - float(launch.route_start_s))
        ax.broken_barh(
            [(float(launch.route_start_s), full_window)],
            (y_base, row_height),
            facecolors="#e5e7eb",
            edgecolors="none",
            alpha=0.42,
        )

        for snapshot_idx, route_text, seg_start, seg_end in zip(
            launch.segment_snapshot_indices,
            launch.segment_routes,
            launch.segment_start_s,
            launch.segment_end_s,
        ):
            duration = max(0.0, float(seg_end) - float(seg_start))
            color = snapshot_colors.get(int(snapshot_idx), "#2563eb")
            ax.broken_barh(
                [(float(seg_start), duration)],
                (y_base + 0.08, row_height - 0.16),
                facecolors=color,
                edgecolors="white",
                linewidth=1.0,
                alpha=0.95,
            )
            short_route = " → ".join(_short_node_label(node) for node in str(route_text).split(" -> "))
            if duration >= 260.0:
                ax.text(
                    float(seg_start) + 0.5 * duration,
                    center_y,
                    short_route,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="white",
                    fontweight="bold",
                )

        route_summary = " → ".join(_short_node_label(node) for node in launch.physical_nodes)
        ax.text(
            route_end_max + 40.0,
            center_y,
            route_summary,
            ha="left",
            va="center",
            fontsize=8.6,
            color="#111827",
        )

    ax.set_title(f"Ground-to-Ground ({pair_label}) – STBD Path Timeline")
    ax.set_xlabel("Transmission duration (s)")
    ax.set_ylabel("Launch / route summary")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(-0.02 * max(1.0, route_end_max), route_end_max + label_margin)
    ax.set_ylim(-0.25, yticks[-1] + row_height)
    ax.invert_yaxis()

    handles = [
        Patch(facecolor=snapshot_colors[idx], edgecolor="white", label=f"Snapshot {idx}")
        for idx in unique_snapshots
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=min(4, len(handles)))

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pareto(
    out_path: Path,
    *,
    candidates: Sequence[CandidateRecord],
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    xs = [candidate.metrics.average_end_to_end_fidelity for candidate in candidates]
    ys = [candidate.metrics.average_entanglement_throughput for candidate in candidates]
    colors = [candidate.proposal.total_satellites for candidate in candidates]
    scatter = ax.scatter(
        xs,
        ys,
        c=colors,
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_xlabel("Average end-to-end fidelity")
    ax.set_ylabel("Average entanglement throughput")
    ax.set_title("Pareto-style Fidelity / Throughput / Satellite Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of satellites")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_result_artifacts(
    out_dir: Path,
    *,
    cfg: StbdConstellationConfig,
    pair_label: str,
    metrics: MetricsSummary,
    timeseries: Sequence[TimeSeriesRow],
    launch_results: Sequence[LaunchResult],
    raw_event_rows: Sequence[dict],
    candidates: Sequence[CandidateRecord],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timeseries_csv = out_dir / "timeseries.csv"
    candidate_csv = out_dir / "candidate_metrics.csv"
    throughput_png = out_dir / "throughput_vs_time.png"
    fidelity_png = out_dir / "fidelity_vs_time.png"
    hops_png = out_dir / "hops_vs_time.png"
    availability_png = out_dir / "availability_timeline.png"
    zoom_png = out_dir / "availability_window_zoom.png"
    zoom_detail_png = out_dir / "availability_window_zoom_detail.png"
    bucketed_delivery_png = out_dir / "bucketed_delivery_profile.png"
    event_delivery_png = out_dir / "event_delivery_timeline.png"
    all_events_png = out_dir / "all_events_timeline.png"
    path_timeline_png = out_dir / "path_timeline.png"
    pareto_png = out_dir / "pareto_fidelity_throughput_satellites.png"
    envelope_csv = out_dir / "service_envelope_timeseries.csv"
    raw_events_json = out_dir / "raw_event_rows.json"

    write_timeseries_csv(timeseries_csv, timeseries)
    write_candidate_metrics_csv(candidate_csv, candidates)
    raw_events_json.write_text(json.dumps(list(raw_event_rows), indent=2), encoding="utf-8")
    envelope_rows = _rasterize_launch_service_window_envelope(
        launches=launch_results,
        raw_event_rows=raw_event_rows,
        total_duration_s=max((float(row.time_s) for row in timeseries), default=0.0)
        + (_series_bin_s(timeseries) if timeseries else 0.0),
        bin_s=_series_bin_s(timeseries),
    )
    write_timeseries_csv(envelope_csv, envelope_rows)
    # Use the final-segment service-window envelope for the primary STBD plots.
    # A multi-snapshot STBD path is not end-to-end visible during its earlier
    # snapshots; only the last segment corresponds to end-to-end service.
    plot_throughput_vs_time(throughput_png, rows=envelope_rows, pair_label=pair_label)
    plot_fidelity_vs_time(fidelity_png, rows=envelope_rows, pair_label=pair_label)
    plot_hops_vs_time(hops_png, rows=envelope_rows, pair_label=pair_label)
    plot_availability_timeline(availability_png, rows=envelope_rows, pair_label=pair_label)
    plot_service_window_zoom(
        zoom_png,
        rows=envelope_rows,
        launch_rows=launch_results,
        raw_event_rows=raw_event_rows,
        pair_label=pair_label,
    )
    plot_service_window_zoom(
        zoom_detail_png,
        rows=envelope_rows,
        launch_rows=launch_results,
        raw_event_rows=raw_event_rows,
        pair_label=pair_label,
        throughput_ylim=(0.0, 5.0),
        fidelity_ylim=(0.7, 0.9),
        title_suffix="Availability Window Zoom (Detail)",
    )
    plot_bucketed_delivery_profile(
        bucketed_delivery_png,
        rows=timeseries,
        pair_label=pair_label,
        bin_s=_series_bin_s(timeseries),
    )
    plot_event_delivery_timeline(
        event_delivery_png,
        raw_event_rows=raw_event_rows,
        pair_label=pair_label,
    )
    plot_all_events_timeline(
        all_events_png,
        raw_event_rows=raw_event_rows,
        launches=launch_results,
        pair_label=pair_label,
    )
    plot_path_timeline(path_timeline_png, launches=launch_results, pair_label=pair_label)
    plot_pareto(pareto_png, candidates=candidates)

    return {
        "timeseries_csv": timeseries_csv,
        "service_envelope_csv": envelope_csv,
        "candidate_metrics_csv": candidate_csv,
        "raw_event_rows_json": raw_events_json,
        "throughput_vs_time_png": throughput_png,
        "fidelity_vs_time_png": fidelity_png,
        "hops_vs_time_png": hops_png,
        "availability_timeline_png": availability_png,
        "availability_window_zoom_png": zoom_png,
        "availability_window_zoom_detail_png": zoom_detail_png,
        "bucketed_delivery_profile_png": bucketed_delivery_png,
        "event_delivery_timeline_png": event_delivery_png,
        "all_events_timeline_png": all_events_png,
        "path_timeline_png": path_timeline_png,
        "pareto_png": pareto_png,
    }
