from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms.stbd import STBDResult, plan_stbd_route
from channels.fso import fidelity_to_werner, werner_to_fidelity
from framework.core.des import DESEnv, Entity, EventHandler
from des_physics_no_sdn_simulator import (
    PS_PER_S,
    build_bsm_model,
    build_fso_model,
    build_memory_model,
    build_parser as build_des_parser,
)
from mobility.geometry import GeometryEngine
from mobility.pass_planner import PassPlannerConfig, rank_satellite_candidates
from mobility.propagator import GroundStation, SkyfieldPropagator
from mobility.synthetic_constellation import generate_walker_delta_constellation
from mobility.tle_loader import (
    SatelliteEphemeris,
    load_ephemerides,
    representative_epoch_unix_s,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
        "lines.linewidth": 2.3,
        "legend.framealpha": 0.92,
        "savefig.bbox": "tight",
    }
)


def _sample_poisson(rng: random.Random, lam: float) -> int:
    lam = max(0.0, float(lam))
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        threshold = math.exp(-lam)
        k = 0
        p = 1.0
        while p > threshold:
            k += 1
            p *= rng.random()
        return k - 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def _sample_binomial(rng: random.Random, n: int, p: float) -> int:
    n = max(0, int(n))
    p = max(0.0, min(1.0, float(p)))
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n <= 128:
        return sum(1 for _ in range(n) if rng.random() < p)
    mean = n * p
    variance = n * p * (1.0 - p)
    if mean < 30.0 and p < 0.05:
        return min(n, _sample_poisson(rng, mean))
    return max(0, min(n, int(round(rng.gauss(mean, math.sqrt(variance))))))


GROUND_PRESETS: Dict[str, GroundStation] = {
    "luxembourg": GroundStation(
        "Luxembourg", lat_deg=49.6116, lon_deg=6.1319, alt_m=300.0
    ),
    "singapore": GroundStation(
        "Singapore", lat_deg=1.3521, lon_deg=103.8198, alt_m=15.0
    ),
    "capetown": GroundStation(
        "Cape Town", lat_deg=-33.9249, lon_deg=18.4241, alt_m=25.0
    ),
    "santiago": GroundStation(
        "Santiago", lat_deg=-33.4489, lon_deg=-70.6693, alt_m=520.0
    ),
    "sydney": GroundStation(
        "Sydney", lat_deg=-33.8688, lon_deg=151.2093, alt_m=58.0
    ),
    "buenosaires": GroundStation(
        "Buenos Aires", lat_deg=-34.6037, lon_deg=-58.3816, alt_m=25.0
    ),
    "tokyo": GroundStation(
        "Tokyo", lat_deg=35.6762, lon_deg=139.6503, alt_m=40.0
    ),
    "delhi": GroundStation(
        "Delhi", lat_deg=28.6139, lon_deg=77.2090, alt_m=216.0
    ),
    "dubai": GroundStation(
        "Dubai", lat_deg=25.2048, lon_deg=55.2708, alt_m=16.0
    ),
    "norway": GroundStation(
        "Norway", lat_deg=59.9139, lon_deg=10.7522, alt_m=23.0
    ),
    "paris": GroundStation("Paris", lat_deg=48.8566, lon_deg=2.3522, alt_m=35.0),
    "vienna": GroundStation("Vienna", lat_deg=48.2082, lon_deg=16.3738, alt_m=171.0),
    "berlin": GroundStation("Berlin", lat_deg=52.5200, lon_deg=13.4050, alt_m=34.0),
    "lisbon": GroundStation("Lisbon", lat_deg=38.7223, lon_deg=-9.1393, alt_m=2.0),
    "warsaw": GroundStation("Warsaw", lat_deg=52.2297, lon_deg=21.0122, alt_m=100.0),
    "london": GroundStation("London", lat_deg=51.5074, lon_deg=-0.1278, alt_m=11.0),
    "nice": GroundStation("Nice", lat_deg=43.7102, lon_deg=7.2620, alt_m=10.0),
    "madrid": GroundStation("Madrid", lat_deg=40.4168, lon_deg=-3.7038, alt_m=667.0),
    "brussels": GroundStation("Brussels", lat_deg=50.8503, lon_deg=4.3517, alt_m=13.0),
}


def _default_tle_path() -> str | None:
    candidate = Path(__file__).with_name("starlink_tle.txt")
    return str(candidate) if candidate.exists() else None


def _endpoint_visibility_ranked_names(
    *,
    satellites: Sequence[SatelliteEphemeris],
    ground_a: GroundStation,
    ground_b: GroundStation,
    epoch_unix_s: float,
    duration_s: float,
    step_s: float,
    limit: int,
) -> List[str]:
    propagator = SkyfieldPropagator(
        satellites=satellites,
        ground_nodes=[ground_a, ground_b],
        epoch_unix_s=epoch_unix_s,
    )
    stats: List[Tuple[float, float, float, str]] = []
    end_ps = int(max(1.0, float(duration_s)) * PS_PER_S)
    step_ps = int(max(1.0, float(step_s)) * PS_PER_S)
    for sat in satellites:
        best_a = -90.0
        best_b = -90.0
        t_ps = 0
        while t_ps <= end_ps:
            best_a = max(best_a, propagator.elevation_deg(sat.name, ground_a.name, t_ps))
            best_b = max(best_b, propagator.elevation_deg(sat.name, ground_b.name, t_ps))
            t_ps += step_ps
        stats.append((best_a, best_b, min(best_a, best_b), sat.name))

    src_rank = sorted(stats, key=lambda item: (item[0], item[2], item[1]), reverse=True)
    dst_rank = sorted(stats, key=lambda item: (item[1], item[2], item[0]), reverse=True)
    bal_rank = sorted(stats, key=lambda item: (item[2], max(item[0], item[1])), reverse=True)

    ordered: List[str] = []
    seen = set()
    rank_lists = [src_rank, dst_rank, bal_rank]
    idx = 0
    while len(ordered) < min(int(limit), len(satellites)):
        progressed = False
        for rank in rank_lists:
            if idx >= len(rank):
                continue
            name = rank[idx][3]
            if name not in seen:
                ordered.append(name)
                seen.add(name)
            progressed = True
            if len(ordered) >= min(int(limit), len(satellites)):
                break
        if not progressed:
            break
        idx += 1
    return ordered


@dataclass(frozen=True)
class EdgeConfig:
    u: str
    v: str
    label: str
    source_node: str


@dataclass(frozen=True)
class LinkState:
    label: str
    t_ps: int
    visible: bool
    distance_m: float
    elevation_deg: float
    p_link: float
    werner_w: float
    fidelity: float


@dataclass
class TokenBatch:
    label: str
    count: int
    ready_ps: int
    werner_w: float
    fidelity: float


@dataclass(frozen=True)
class SwapOp:
    op_id: str
    middle: str
    left_label: str
    right_label: str
    out_label: str
    store_node: str


@dataclass(frozen=True)
class MetricRow:
    t_s: float
    active_segments: int
    active_elementary_links: int
    visible_elementary_links: int
    mean_active_p_link: float
    mean_active_fidelity: float
    successes: int
    intermediate_swaps: int
    failures: int
    mean_fidelity_success: float
    throughput_pairs_per_s: float
    cumulative_successes: int
    cumulative_mean_fidelity_success: float
    cumulative_throughput_pairs_per_s: float
    total_memory_occupancy: int
    fail_link: int
    fail_recall: int
    fail_bsm: int
    fail_storage_full: int
    fail_threshold: int


@dataclass(frozen=True)
class EndToEndEvent:
    t_s: float
    count: int
    fidelity: float
    kind: str


@dataclass(frozen=True)
class SwapEvent:
    t_s: float
    op_id: str
    middle: str
    left_label: str
    right_label: str
    out_label: str
    attempts: int
    recalled: int
    bsm_successes: int
    accepted: int
    dropped: int
    fidelity: float
    final: int


def _edge_label_for_adjacent(u: str, v: str, edges: Sequence[EdgeConfig]) -> Optional[str]:
    for edge in edges:
        if (edge.u == u and edge.v == v) or (edge.u == v and edge.v == u):
            return edge.label
    return None


def _out_label(a: str, b: str) -> str:
    return f"{a}__{b}"


def build_segmented_nesting_plan(
    path: Sequence[str],
    edges: Sequence[EdgeConfig],
) -> List[SwapOp]:
    if len(path) < 3:
        return []

    def build(i: int, j: int, sink: str) -> Tuple[str, List[SwapOp]]:
        if j - i == 1:
            label = _edge_label_for_adjacent(path[i], path[j], edges)
            if label is None:
                raise KeyError(f"missing edge label for adjacent path pair {path[i]}-{path[j]}")
            return label, []

        mid = (i + j) // 2
        mid_node = str(path[mid])
        left_label, left_ops = build(i, mid, mid_node)
        right_label, right_ops = build(mid, j, mid_node)
        op = SwapOp(
            op_id=f"swap_{path[i]}_{mid_node}_{path[j]}",
            middle=mid_node,
            left_label=left_label,
            right_label=right_label,
            out_label=_out_label(str(path[i]), str(path[j])),
            store_node=str(sink),
        )
        return op.out_label, left_ops + right_ops + [op]

    global_mid = str(path[(len(path) - 1) // 2])
    _final_label, ops = build(0, len(path) - 1, global_mid)
    return ops


def build_snapshot_segmented_execution_plan(
    stbd_result: STBDResult,
    route_edges: Sequence[EdgeConfig],
) -> Tuple[List[SwapOp], str]:
    all_ops: List[SwapOp] = []
    segment_outputs: List[Tuple[str, str, str]] = []

    for seg_idx, segment in enumerate(stbd_result.segments):
        seg_nodes = [str(node) for node in segment.nodes]
        if len(seg_nodes) < 2:
            continue
        seg_ops = build_segmented_nesting_plan(seg_nodes, route_edges)
        if seg_ops:
            seg_out = str(seg_ops[-1].out_label)
        else:
            seg_out = _edge_label_for_adjacent(seg_nodes[0], seg_nodes[1], route_edges)
            if seg_out is None:
                raise KeyError(
                    f"missing segment edge label for {seg_nodes[0]}-{seg_nodes[1]}"
                )
        all_ops.extend(seg_ops)
        segment_outputs.append((seg_nodes[0], seg_nodes[-1], seg_out))

    if not segment_outputs:
        return [], ""

    prefix_start, prefix_end, current_label = segment_outputs[0]
    for seg_idx, (seg_start, seg_end, seg_label) in enumerate(segment_outputs[1:], start=1):
        if str(prefix_end) != str(seg_start):
            raise ValueError(
                f"non-contiguous STBD segments: expected boundary {prefix_end}, got {seg_start}"
            )
        bridge = SwapOp(
            op_id=f"bridge_{seg_idx}_{prefix_start}_{seg_start}_{seg_end}",
            middle=str(seg_start),
            left_label=str(current_label),
            right_label=str(seg_label),
            out_label=_out_label(str(prefix_start), str(seg_end)),
            store_node=str(seg_end),
        )
        all_ops.append(bridge)
        current_label = bridge.out_label
        prefix_end = seg_end

    return all_ops, str(current_label)


def build_route_edges(path_nodes: Sequence[str]) -> List[EdgeConfig]:
    if len(path_nodes) < 2:
        return []

    host_by_index: Dict[int, str] = {}

    def assign(i: int, j: int, sink: str) -> None:
        if j - i == 1:
            host_by_index[i] = sink
            return
        mid = (i + j) // 2
        mid_node = str(path_nodes[mid])
        assign(i, mid, mid_node)
        assign(mid, j, mid_node)

    assign(0, len(path_nodes) - 1, str(path_nodes[(len(path_nodes) - 1) // 2]))
    edges: List[EdgeConfig] = []
    for idx, (u, v) in enumerate(zip(path_nodes, path_nodes[1:])):
        edges.append(
            EdgeConfig(
                u=str(u),
                v=str(v),
                label=f"L{idx}:{u}->{v}",
                source_node=str(host_by_index[idx]),
            )
        )
    return edges


def _segment_labels(
    segment_nodes: Sequence[str],
    edges: Sequence[EdgeConfig],
) -> List[str]:
    labels: List[str] = []
    for u, v in zip(segment_nodes, segment_nodes[1:]):
        label = _edge_label_for_adjacent(str(u), str(v), edges)
        if label is None:
            raise KeyError(f"missing route edge for segment hop {u}-{v}")
        labels.append(label)
    return labels


def _load_satellite_subset(args: argparse.Namespace) -> List[SatelliteEphemeris]:
    constellation_source = str(getattr(args, "constellation_source", "tle"))
    if constellation_source == "walker":
        total_walker_sats = int(args.walker_planes) * int(args.walker_sats_per_plane)
        if int(args.num_sats) > total_walker_sats:
            raise ValueError(
                f"requested {int(args.num_sats)} satellites, but walker shell only has {total_walker_sats}"
            )
        sats = generate_walker_delta_constellation(
            planes=int(args.walker_planes),
            sats_per_plane=int(args.walker_sats_per_plane),
            altitude_km=float(args.walker_altitude_km),
            inclination_deg=float(args.walker_inclination_deg),
            phase=int(args.walker_phase),
            epoch_unix_s=getattr(args, "epoch_unix_s", None),
        )
    else:
        sats = load_ephemerides(
            tle_path=args.tle,
            omm_json_path=args.omm_json,
            use_demo_if_missing=(args.tle is None and args.omm_json is None),
        )
    if len(sats) < int(args.num_sats):
        raise ValueError(
            f"requested {int(args.num_sats)} satellites, but only {len(sats)} are available"
        )
    selected = list(sats[: int(args.num_sats)])
    selection_mode = str(getattr(args, "satellite_selection", "sequential"))
    if selection_mode in {"ranked_dual_visibility", "ranked_endpoint_visibility"} and str(getattr(args, "scenario", "")) == "ground-ground":
        gs_src = _clone_ground(str(args.ground_src), min_elevation_deg=float(args.min_elevation_deg))
        gs_dst = _clone_ground(str(args.ground_dst), min_elevation_deg=float(args.min_elevation_deg))
        epoch_unix_s = representative_epoch_unix_s(sats)
        if selection_mode == "ranked_dual_visibility":
            ranked = rank_satellite_candidates(
                satellites=sats,
                ground_a=gs_src,
                ground_b=gs_dst,
                epoch_unix_s=epoch_unix_s,
                cfg=PassPlannerConfig(
                    scan_end_ps=int(float(args.duration_s) * PS_PER_S),
                    fine_step_ps=int(max(1.0, float(args.link_update_dt_s)) * PS_PER_S),
                    coarse_step_ps=int(max(60.0, 10.0 * float(args.link_update_dt_s)) * PS_PER_S),
                    candidate_limit=max(int(args.num_sats), 24),
                    objective="memory_aware",
                    memory_coherence_s=float(_resolve_planning_coherence_s(args)),
                ),
            )
            if ranked:
                selected = ranked[: int(args.num_sats)]
        else:
            ordered_names = _endpoint_visibility_ranked_names(
                satellites=sats,
                ground_a=gs_src,
                ground_b=gs_dst,
                epoch_unix_s=epoch_unix_s,
                duration_s=float(args.duration_s),
                step_s=max(30.0, 10.0 * float(args.link_update_dt_s)),
                limit=int(args.num_sats),
            )
            if ordered_names:
                by_name = {sat.name: sat for sat in sats}
                selected = [by_name[name] for name in ordered_names if name in by_name]

    subset: List[SatelliteEphemeris] = []
    for idx, sat in enumerate(selected, start=1):
        metadata = dict(sat.metadata or {})
        metadata["original_name"] = sat.name
        metadata["selection_mode"] = selection_mode
        metadata["constellation_source"] = constellation_source
        subset.append(
            SatelliteEphemeris(
                name=f"SAT-{idx:03d}",
                tle_line1=sat.tle_line1,
                tle_line2=sat.tle_line2,
                source=sat.source,
                metadata=metadata,
            )
        )
    return subset


def _clone_ground(name: str, *, min_elevation_deg: float) -> GroundStation:
    base = GROUND_PRESETS[name]
    return GroundStation(
        name=base.name,
        lat_deg=base.lat_deg,
        lon_deg=base.lon_deg,
        alt_m=base.alt_m,
        min_elevation_deg=float(min_elevation_deg),
    )


def _scenario_endpoints(
    args: argparse.Namespace,
    satellites: Sequence[SatelliteEphemeris],
) -> Tuple[str, str, List[GroundStation]]:
    if str(args.scenario) == "ground-ground":
        gs_src = _clone_ground(
            str(args.ground_src),
            min_elevation_deg=float(args.min_elevation_deg),
        )
        gs_dst = _clone_ground(
            str(args.ground_dst),
            min_elevation_deg=float(args.min_elevation_deg),
        )
        return gs_src.name, gs_dst.name, [gs_src, gs_dst]

    src_idx = int(args.src_index)
    dst_idx = int(args.dst_index)
    if src_idx < 1 or src_idx > len(satellites):
        raise ValueError(f"src-index must be in [1, {len(satellites)}]")
    if dst_idx < 1 or dst_idx > len(satellites):
        raise ValueError(f"dst-index must be in [1, {len(satellites)}]")
    return satellites[src_idx - 1].name, satellites[dst_idx - 1].name, []


def _resolve_planning_coherence_s(args: argparse.Namespace) -> float:
    if getattr(args, "planner_coherence_s", None) is not None:
        return float(args.planner_coherence_s)
    if getattr(args, "coherence_time_s", None) is not None:
        return float(args.coherence_time_s)
    return float(args.mem_t2)


def _resolve_planning_start_s(args: argparse.Namespace) -> float:
    return max(0.0, float(getattr(args, "planner_start_s", 0.0)))


def _resolve_destination_horizon_s(args: argparse.Namespace) -> float:
    if getattr(args, "planner_horizon_s", None) is not None:
        return max(0.0, float(args.planner_horizon_s))
    return max(0.0, float(args.duration_s))


def _label_windows(
    result: STBDResult,
    route_edges: Sequence[EdgeConfig],
) -> Dict[str, List[Tuple[float, float]]]:
    windows: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for segment in result.segments:
        for label in _segment_labels(segment.nodes, route_edges):
            windows[label].append((float(segment.start_s), float(segment.end_s)))
    return dict(windows)


class STBDPhysicsDES(Entity):
    """Persistent stochastic STBD-over-DES simulator over a planned constellation path."""

    def __init__(
        self,
        *,
        name: str,
        env: DESEnv,
        args: argparse.Namespace,
        geometry: GeometryEngine,
        fso,
        memory,
        bsm,
        stbd_result: STBDResult,
        route_edges: Sequence[EdgeConfig],
        controller_client=None,
    ):
        super().__init__(name=name, env=env)
        self.args = args
        self.geometry = geometry
        self.fso = fso
        self.memory = memory
        self.bsm = bsm
        self.stbd_result = stbd_result
        self.route_edges = list(route_edges)
        self.route_edge_by_label = {edge.label: edge for edge in self.route_edges}
        self.elementary_labels = [edge.label for edge in self.route_edges]
        self.execution_mode = str(getattr(args, "stbd_execution_mode", "basic")).lower().strip()
        self.controller_backend = str(getattr(args, "stbd_controller_backend", "des_only")).lower().strip()
        self.controller_client = controller_client
        self._controller_cache_t_ps: int | None = None
        self._controller_cache: Dict[str, object] | None = None
        if self.execution_mode in {"segmented", "paced"}:
            self.plan, self.final_label = build_snapshot_segmented_execution_plan(
                stbd_result, self.route_edges
            )
        else:
            self.plan = build_segmented_nesting_plan(stbd_result.physical_nodes, self.route_edges)
            self.final_label = (
                self.plan[-1].out_label if self.plan else (self.elementary_labels[0] if self.elementary_labels else "")
            )
        self.label_windows = _label_windows(stbd_result, self.route_edges)
        self.segment_labels = [
            _segment_labels(segment.nodes, self.route_edges) for segment in stbd_result.segments
        ]
        self.final_ops = [op for op in self.plan if op.out_label == self.final_label]
        final_segment = stbd_result.segments[-1] if stbd_result.segments else None
        self.final_segment_start_s = (
            float(final_segment.start_s) if final_segment is not None else 0.0
        )
        self.final_segment_end_s = (
            float(final_segment.end_s) if final_segment is not None else 0.0
        )
        self.rng = random.Random(int(args.seed))
        self.time_offset_ps = int(max(0.0, float(getattr(args, "time_offset_s", 0.0))) * PS_PER_S)
        self.link_states: Dict[str, LinkState] = {}
        self.tokens: Dict[str, List[TokenBatch]] = defaultdict(list)
        self.metrics: List[MetricRow] = []
        self.end_to_end_events: List[EndToEndEvent] = []
        self.swap_events: List[SwapEvent] = []

        self.successes_bin = 0
        self.intermediate_swaps_bin = 0
        self.failures_bin = 0
        self.fidelity_sum_bin = 0.0
        self.paced_tick_successes = 0
        self.paced_tick_fidelity_sum = 0.0
        self.fail_link_bin = 0
        self.fail_recall_bin = 0
        self.fail_bsm_bin = 0
        self.fail_storage_full_bin = 0
        self.fail_threshold_bin = 0
        self.cumulative_successes = 0
        self.cumulative_fidelity_sum = 0.0

    def init(self) -> None:
        if self.controller_client is not None and self.controller_backend != "des_only":
            segments = [
                {
                    "snapshot_index": int(segment.snapshot_index),
                    "start_s": float(segment.start_s),
                    "end_s": float(segment.end_s),
                    "labels": list(self.segment_labels[idx]),
                    "nodes": list(segment.nodes),
                }
                for idx, segment in enumerate(self.stbd_result.segments)
            ]
            route_start_s = (
                min(float(segment.start_s) for segment in self.stbd_result.segments)
                if self.stbd_result.segments
                else 0.0
            )
            route_end_s = (
                max(float(segment.end_s) for segment in self.stbd_result.segments)
                if self.stbd_result.segments
                else 0.0
            )
            self.controller_client.reset()
            self.controller_client.install_request(
                req_id=1,
                execution_mode=self.execution_mode,
                final_label=self.final_label,
                route_start_s=float(route_start_s),
                route_end_s=float(route_end_s),
                final_segment_start_s=float(self.final_segment_start_s),
                final_segment_end_s=float(self.final_segment_end_s),
                segments=segments,
            )
        self.scheduler.schedule_now(EventHandler(self, "orbit_update"))
        self.scheduler.schedule_after(
            int(float(self.args.gen_delay_s) * PS_PER_S),
            EventHandler(self, "generation_tick"),
        )
        self.scheduler.schedule_after(
            int(float(self.args.swap_dt_s) * PS_PER_S),
            EventHandler(self, "swap_tick"),
        )
        self.scheduler.schedule_after(
            int(float(self.args.metrics_dt_s) * PS_PER_S),
            EventHandler(self, "metrics_tick"),
        )
        if self.execution_mode == "paced":
            self.scheduler.schedule_after(
                int(float(getattr(self.args, "paced_delivery_dt_s", 30.0)) * PS_PER_S),
                EventHandler(self, "delivery_tick"),
            )

    def _schedule_next(self, handler: str, delay_s: float) -> None:
        next_t = int(self.env.now) + int(max(0.0, float(delay_s)) * PS_PER_S)
        if next_t <= int(float(self.args.duration_s) * PS_PER_S):
            self.scheduler.schedule_at(next_t, EventHandler(self, handler))

    def _absolute_now_ps(self) -> int:
        return int(self.env.now) + int(self.time_offset_ps)

    def _absolute_now_s(self) -> float:
        return float(self._absolute_now_ps()) / PS_PER_S

    def _controller_state(self) -> Dict[str, object] | None:
        if self.controller_client is None or self.controller_backend == "des_only":
            return None
        now_ps = self._absolute_now_ps()
        if self._controller_cache_t_ps == now_ps and self._controller_cache is not None:
            return self._controller_cache
        payload = self.controller_client.poll(req_id=1, t_s=self._absolute_now_s())
        self._controller_cache_t_ps = now_ps
        self._controller_cache = payload if isinstance(payload, dict) else {}
        return self._controller_cache

    def _controller_allows_label(self, label: str, t_s: float) -> bool:
        del t_s
        state = self._controller_state()
        if not state:
            return False
        active_labels = {
            str(item)
            for item in state.get("active_labels", [])
        }
        return str(label) in active_labels

    def _controller_final_segment_active(self) -> bool:
        state = self._controller_state()
        if not state:
            return False
        return bool(state.get("final_segment_active", False))

    def _is_label_active(self, label: str, t_s: float) -> bool:
        if self.controller_client is not None and self.controller_backend != "des_only":
            return self._controller_allows_label(label, t_s)
        for start_s, end_s in self.label_windows.get(label, []):
            if float(start_s) - 1e-12 <= t_s < float(end_s) + 1e-12:
                return True
        return False

    def _active_segment_count(self, t_s: float) -> int:
        return sum(
            1
            for segment in self.stbd_result.segments
            if float(segment.start_s) - 1e-12 <= t_s < float(segment.end_s) + 1e-12
        )

    def _all_elementary_links_visible(self) -> bool:
        if not self.elementary_labels:
            return False
        for label in self.elementary_labels:
            state = self.link_states.get(label) or self.compute_link(label)
            if not state.visible:
                return False
        return True

    def compute_link(self, label: str) -> LinkState:
        edge = self.route_edge_by_label[label]
        now_ps = self._absolute_now_ps()
        now_s = float(now_ps) / PS_PER_S
        geom = self.geometry.contact_geometry(edge.u, edge.v, now_ps)
        visible = geom.visible and self._is_label_active(label, now_s)
        if bool(self.args.force_visible):
            visible = self._is_label_active(label, now_s)
        intersat = self.geometry.is_satellite(edge.u) and self.geometry.is_satellite(edge.v)
        if visible:
            if bool(getattr(self.args, "perfect_isl", False)) and intersat:
                p_link, w, fid = 1.0, 1.0, 1.0
            elif bool(getattr(self.args, "stochastic_link_state", True)):
                p_link, w, fid = self.fso.link_metrics(
                    rng=self.rng,
                    distance_m=float(geom.distance_m),
                    elevation_deg=float(geom.elevation_deg),
                    intersatellite=intersat,
                )
            else:
                p_link, w, fid = self.fso.mean_link_metrics(
                    distance_m=float(geom.distance_m),
                    elevation_deg=float(geom.elevation_deg),
                    intersatellite=intersat,
                )
        else:
            p_link, w, fid = 0.0, 0.0, 0.25
        return LinkState(
            label=label,
            t_ps=int(self.env.now),
            visible=visible,
            distance_m=float(geom.distance_m),
            elevation_deg=float(geom.elevation_deg),
            p_link=float(p_link),
            werner_w=float(w),
            fidelity=float(fid),
        )

    def orbit_update(self) -> None:
        for label in self.elementary_labels:
            self.link_states[label] = self.compute_link(label)
        self._schedule_next("orbit_update", float(self.args.link_update_dt_s))

    def generation_tick(self) -> None:
        dt_s = max(0.0, float(self.args.gen_delay_s))
        attempts = max(1, int(round(max(0.0, float(self.args.source_rate_hz)) * dt_s)))
        direct_final = len(self.elementary_labels) == 1 and not self.plan
        now_s = self._absolute_now_s()

        if (
            bool(getattr(self.args, "direct_swap_on_full_visibility", True))
            and self.plan
            and self._all_elementary_links_visible()
        ):
            self.execute_full_visibility_direct(attempts)
            self._schedule_next("generation_tick", float(self.args.gen_delay_s))
            return

        for label in self.elementary_labels:
            if not self._is_label_active(label, now_s):
                continue
            link = self.link_states.get(label) or self.compute_link(label)
            capacity = max(0, int(self.args.max_modes_per_label) - self.count_tokens(label))
            if not direct_final and capacity <= 0:
                continue
            if not link.visible:
                self.fail_link_bin += attempts
                self.failures_bin += attempts
                continue

            link_success = _sample_binomial(self.rng, attempts, link.p_link)
            stored = _sample_binomial(self.rng, link_success, self.memory.write_success_prob())
            self.fail_link_bin += attempts - link_success
            self.fail_recall_bin += link_success - stored
            self.failures_bin += (attempts - link_success) + (link_success - stored)

            if direct_final:
                accepted = stored
                if link.fidelity < float(self.args.f_min):
                    self.fail_threshold_bin += accepted
                    self.failures_bin += accepted
                    accepted = 0
                if accepted > 0:
                    self.successes_bin += accepted
                    self.cumulative_successes += accepted
                    self.fidelity_sum_bin += accepted * float(link.fidelity)
                    self.cumulative_fidelity_sum += accepted * float(link.fidelity)
                    self.end_to_end_events.append(
                        EndToEndEvent(
                            t_s=now_s,
                            count=int(accepted),
                            fidelity=float(link.fidelity),
                            kind="direct_link",
                        )
                    )
                continue

            accepted = min(stored, capacity)
            dropped = max(0, stored - accepted)
            self.fail_storage_full_bin += dropped
            self.failures_bin += dropped
            if accepted > 0:
                self.tokens[label].append(
                    TokenBatch(
                        label=label,
                        count=int(accepted),
                        ready_ps=int(self.env.now),
                        werner_w=float(link.werner_w),
                        fidelity=float(link.fidelity),
                    )
                )
        self._schedule_next("generation_tick", float(self.args.gen_delay_s))

    def execute_full_visibility_direct(self, attempts: int) -> None:
        direct_tokens: Dict[str, List[TokenBatch]] = defaultdict(list)
        for label in self.elementary_labels:
            link = self.link_states.get(label) or self.compute_link(label)
            if not link.visible:
                self.fail_link_bin += attempts
                self.failures_bin += attempts
                return
            link_success = _sample_binomial(self.rng, attempts, link.p_link)
            self.fail_link_bin += attempts - link_success
            self.failures_bin += attempts - link_success
            if link_success > 0:
                direct_tokens[label].append(
                    TokenBatch(
                        label=label,
                        count=int(link_success),
                        ready_ps=int(self.env.now),
                        werner_w=float(link.werner_w),
                        fidelity=float(link.fidelity),
                    )
                )

        made_progress = True
        while made_progress:
            made_progress = False
            for op in self.plan:
                count = min(
                    self._count_pool(direct_tokens, op.left_label),
                    self._count_pool(direct_tokens, op.right_label),
                )
                if count <= 0:
                    continue
                left_batches = self._pop_pool(direct_tokens, op.left_label, count)
                right_batches = self._pop_pool(direct_tokens, op.right_label, count)
                for n, left, right in self.pair_batches(left_batches, right_batches):
                    self.execute_direct_swap_batch(op, n, left, right, direct_tokens)
                made_progress = True

    def execute_direct_swap_batch(
        self,
        op: SwapOp,
        n: int,
        left: TokenBatch,
        right: TokenBatch,
        token_pool: Dict[str, List[TokenBatch]],
    ) -> None:
        w_out, f_out = self.swap_state(
            max(0.0, min(1.0, float(left.werner_w))),
            max(0.0, min(1.0, float(right.werner_w))),
        )
        if f_out < float(self.args.f_min):
            self.fail_threshold_bin += int(n)
            self.failures_bin += int(n)
            self.swap_events.append(
                SwapEvent(
                    t_s=float(self.env.now) / PS_PER_S,
                    op_id=f"{op.op_id}:direct",
                    middle=op.middle,
                    left_label=op.left_label,
                    right_label=op.right_label,
                    out_label=op.out_label,
                    attempts=int(n),
                    recalled=int(n),
                    bsm_successes=0,
                    accepted=0,
                    dropped=0,
                    fidelity=float(f_out),
                    final=int(op.out_label == self.final_label),
                )
            )
            return

        bsm_successes = _sample_binomial(self.rng, n, self.bsm.p_succ_bsm(1.0, 1.0))
        self.fail_bsm_bin += int(n) - int(bsm_successes)
        self.failures_bin += int(n) - int(bsm_successes)
        if bsm_successes <= 0:
            self.swap_events.append(
                SwapEvent(
                    t_s=float(self.env.now) / PS_PER_S,
                    op_id=f"{op.op_id}:direct",
                    middle=op.middle,
                    left_label=op.left_label,
                    right_label=op.right_label,
                    out_label=op.out_label,
                    attempts=int(n),
                    recalled=int(n),
                    bsm_successes=0,
                    accepted=0,
                    dropped=0,
                    fidelity=float(f_out),
                    final=int(op.out_label == self.final_label),
                )
            )
            return

        is_final = op.out_label == self.final_label
        now_s = self._absolute_now_s()
        if is_final:
            self.successes_bin += int(bsm_successes)
            self.cumulative_successes += int(bsm_successes)
            self.fidelity_sum_bin += int(bsm_successes) * float(f_out)
            self.cumulative_fidelity_sum += int(bsm_successes) * float(f_out)
            self.end_to_end_events.append(
                EndToEndEvent(
                    t_s=now_s,
                    count=int(bsm_successes),
                    fidelity=float(f_out),
                    kind="direct_swap",
                )
            )
        else:
            token_pool[op.out_label].append(
                TokenBatch(
                    label=op.out_label,
                    count=int(bsm_successes),
                    ready_ps=int(self.env.now),
                    werner_w=float(w_out),
                    fidelity=float(f_out),
                )
            )

        self.swap_events.append(
            SwapEvent(
                t_s=now_s,
                op_id=f"{op.op_id}:direct",
                middle=op.middle,
                left_label=op.left_label,
                right_label=op.right_label,
                out_label=op.out_label,
                attempts=int(n),
                recalled=int(n),
                bsm_successes=int(bsm_successes),
                accepted=int(bsm_successes),
                dropped=0,
                fidelity=float(f_out),
                final=int(is_final),
            )
        )

    def swap_tick(self) -> None:
        if not self.plan:
            self._schedule_next("swap_tick", float(self.args.swap_dt_s))
            return

        made_progress = True
        while made_progress:
            made_progress = False
            for op in self.plan:
                if self.execution_mode == "paced" and op.out_label == self.final_label:
                    continue
                count_available = min(
                    self.count_tokens(op.left_label),
                    self.count_tokens(op.right_label),
                    int(self.args.max_bsm_per_cycle),
                )
                if count_available <= 0:
                    continue
                left_batches = self.pop_tokens(op.left_label, count_available)
                right_batches = self.pop_tokens(op.right_label, count_available)
                for n, left, right in self.pair_batches(left_batches, right_batches):
                    self.execute_swap_batch(op, n, left, right)
                made_progress = True
        self._schedule_next("swap_tick", float(self.args.swap_dt_s))

    def _resolve_paced_delivery_limit(self, count_available: int, now_s: float) -> int:
        count_available = max(0, int(count_available))
        if count_available <= 0:
            return 0
        if self.execution_mode != "paced":
            return count_available
        if now_s + 1e-12 < self.final_segment_start_s:
            return 0
        cadence_s = max(1e-9, float(getattr(self.args, "paced_delivery_dt_s", 30.0)))
        remaining_window_s = max(0.0, self.final_segment_end_s - now_s)
        slots_remaining = max(1, int(math.ceil(remaining_window_s / cadence_s - 1e-12)))
        allowed = int(math.ceil(float(count_available) / float(slots_remaining)))
        return min(count_available, max(1, allowed))

    def delivery_tick(self) -> None:
        if self.execution_mode != "paced" or not self.final_ops:
            self._schedule_next("delivery_tick", float(getattr(self.args, "paced_delivery_dt_s", 30.0)))
            return

        now_s = self._absolute_now_s()
        self.paced_tick_successes = 0
        self.paced_tick_fidelity_sum = 0.0
        final_segment_active = (
            self._controller_final_segment_active()
            if self.controller_client is not None and self.controller_backend != "des_only"
            else (self.final_segment_start_s - 1e-12 <= now_s < self.final_segment_end_s - 1e-12)
        )
        if final_segment_active:
            for op in self.final_ops:
                count_available = min(
                    self.count_tokens(op.left_label),
                    self.count_tokens(op.right_label),
                    int(self.args.max_bsm_per_cycle),
                )
                if count_available <= 0:
                    continue
                count = self._resolve_paced_delivery_limit(count_available, now_s)
                if count <= 0:
                    continue
                left_batches = self.pop_tokens(op.left_label, count)
                right_batches = self.pop_tokens(op.right_label, count)
                for n, left, right in self.pair_batches(left_batches, right_batches):
                    self.execute_swap_batch(op, n, left, right)
        if self.paced_tick_successes > 0:
            mean_fidelity = self.paced_tick_fidelity_sum / float(self.paced_tick_successes)
            self.successes_bin += int(self.paced_tick_successes)
            self.cumulative_successes += int(self.paced_tick_successes)
            self.fidelity_sum_bin += float(self.paced_tick_fidelity_sum)
            self.cumulative_fidelity_sum += float(self.paced_tick_fidelity_sum)
            self.end_to_end_events.append(
                EndToEndEvent(
                    t_s=now_s,
                    count=int(self.paced_tick_successes),
                    fidelity=float(mean_fidelity),
                    kind="paced_swap",
                )
            )

        self._schedule_next(
            "delivery_tick",
            float(getattr(self.args, "paced_delivery_dt_s", 30.0)),
        )

    def _resolve_swap_batch_limit(self, op: SwapOp, count_available: int) -> int:
        count_available = max(0, int(count_available))
        if count_available <= 0:
            return count_available
        return count_available

    def execute_swap_batch(
        self,
        op: SwapOp,
        n: int,
        left: TokenBatch,
        right: TokenBatch,
    ) -> None:
        wait_left_s = (int(self.env.now) - int(left.ready_ps)) / PS_PER_S + float(
            self.args.memory_effective_wait_s
        )
        wait_right_s = (int(self.env.now) - int(right.ready_ps)) / PS_PER_S + float(
            self.args.memory_effective_wait_s
        )
        recall_prob = self.memory.conditional_recall_prob(wait_left_s) * self.memory.conditional_recall_prob(
            wait_right_s
        )
        recalled = _sample_binomial(self.rng, n, recall_prob)
        self.fail_recall_bin += n - recalled
        self.failures_bin += n - recalled
        if recalled <= 0:
            self.swap_events.append(
                SwapEvent(
                    t_s=self._absolute_now_s(),
                    op_id=op.op_id,
                    middle=op.middle,
                    left_label=op.left_label,
                    right_label=op.right_label,
                    out_label=op.out_label,
                    attempts=int(n),
                    recalled=0,
                    bsm_successes=0,
                    accepted=0,
                    dropped=0,
                    fidelity=0.25,
                    final=int(op.out_label == self.final_label),
                )
            )
            return

        w_left = max(0.0, min(1.0, float(left.werner_w) * self.memory.coherence_factor(wait_left_s)))
        w_right = max(0.0, min(1.0, float(right.werner_w) * self.memory.coherence_factor(wait_right_s)))
        w_out, f_out = self.swap_state(w_left, w_right)
        bsm_successes = _sample_binomial(self.rng, recalled, self.bsm.p_succ_bsm(1.0, 1.0))
        self.fail_bsm_bin += recalled - bsm_successes
        self.failures_bin += recalled - bsm_successes
        if bsm_successes <= 0:
            self.swap_events.append(
                SwapEvent(
                    t_s=self._absolute_now_s(),
                    op_id=op.op_id,
                    middle=op.middle,
                    left_label=op.left_label,
                    right_label=op.right_label,
                    out_label=op.out_label,
                    attempts=int(n),
                    recalled=int(recalled),
                    bsm_successes=0,
                    accepted=0,
                    dropped=0,
                    fidelity=float(f_out),
                    final=int(op.out_label == self.final_label),
                )
            )
            return

        accepted = int(bsm_successes)
        dropped = 0
        is_final = op.out_label == self.final_label
        now_s = self._absolute_now_s()
        if is_final and f_out < float(self.args.f_min):
            self.fail_threshold_bin += accepted
            self.failures_bin += accepted
            accepted = 0
        elif not is_final:
            capacity = max(0, int(self.args.max_modes_per_label) - self.count_tokens(op.out_label))
            accepted = min(accepted, capacity)
            dropped = bsm_successes - accepted
            self.fail_storage_full_bin += dropped
            self.failures_bin += dropped

        if accepted > 0:
            if is_final:
                if self.execution_mode == "paced":
                    self.paced_tick_successes += int(accepted)
                    self.paced_tick_fidelity_sum += int(accepted) * float(f_out)
                else:
                    self.successes_bin += accepted
                    self.cumulative_successes += accepted
                    self.fidelity_sum_bin += accepted * float(f_out)
                    self.cumulative_fidelity_sum += accepted * float(f_out)
                    self.end_to_end_events.append(
                        EndToEndEvent(
                            t_s=now_s,
                            count=int(accepted),
                            fidelity=float(f_out),
                            kind="swap",
                        )
                    )
            else:
                self.intermediate_swaps_bin += accepted
                self.tokens[op.out_label].append(
                    TokenBatch(
                        label=op.out_label,
                        count=int(accepted),
                        ready_ps=int(self.env.now),
                        werner_w=float(w_out),
                        fidelity=float(f_out),
                    )
                )

        self.swap_events.append(
            SwapEvent(
                t_s=now_s,
                op_id=op.op_id,
                middle=op.middle,
                left_label=op.left_label,
                right_label=op.right_label,
                out_label=op.out_label,
                attempts=int(n),
                recalled=int(recalled),
                bsm_successes=int(bsm_successes),
                accepted=int(accepted),
                dropped=int(dropped),
                fidelity=float(f_out),
                final=int(is_final),
            )
            )

    def swap_state(self, w_left: float, w_right: float) -> Tuple[float, float]:
        f_left = werner_to_fidelity(max(0.0, min(1.0, float(w_left))))
        f_right = werner_to_fidelity(max(0.0, min(1.0, float(w_right))))
        f_out = float(self.bsm.swapped_fidelity(f_left, f_right))
        return fidelity_to_werner(f_out), f_out

    def metrics_tick(self) -> None:
        t_s = self._absolute_now_s()
        active_states = [
            self.link_states[label]
            for label in self.elementary_labels
            if self._is_label_active(label, t_s)
        ]
        active_count = len(active_states)
        visible_count = sum(1 for state in active_states if state.visible)
        mean_p_link = (
            sum(float(state.p_link) for state in active_states) / active_count
            if active_count > 0
            else 0.0
        )
        mean_fidelity = (
            sum(float(state.fidelity) for state in active_states) / active_count
            if active_count > 0
            else 0.25
        )
        successes = int(self.successes_bin)
        mean_success_fidelity = self.fidelity_sum_bin / successes if successes > 0 else float("nan")
        cumulative_mean_success_fidelity = (
            self.cumulative_fidelity_sum / self.cumulative_successes
            if self.cumulative_successes > 0
            else float("nan")
        )
        self.metrics.append(
            MetricRow(
                t_s=t_s,
                active_segments=self._active_segment_count(t_s),
                active_elementary_links=int(active_count),
                visible_elementary_links=int(visible_count),
                mean_active_p_link=float(mean_p_link),
                mean_active_fidelity=float(mean_fidelity),
                successes=successes,
                intermediate_swaps=int(self.intermediate_swaps_bin),
                failures=int(self.failures_bin),
                mean_fidelity_success=float(mean_success_fidelity),
                throughput_pairs_per_s=successes / max(1e-12, float(self.args.metrics_dt_s)),
                cumulative_successes=int(self.cumulative_successes),
                cumulative_mean_fidelity_success=float(cumulative_mean_success_fidelity),
                cumulative_throughput_pairs_per_s=int(self.cumulative_successes)
                / max(1e-12, t_s),
                total_memory_occupancy=int(sum(self.count_tokens(label) for label in self.tokens)),
                fail_link=int(self.fail_link_bin),
                fail_recall=int(self.fail_recall_bin),
                fail_bsm=int(self.fail_bsm_bin),
                fail_storage_full=int(self.fail_storage_full_bin),
                fail_threshold=int(self.fail_threshold_bin),
            )
        )
        self.successes_bin = 0
        self.intermediate_swaps_bin = 0
        self.failures_bin = 0
        self.fidelity_sum_bin = 0.0
        self.fail_link_bin = 0
        self.fail_recall_bin = 0
        self.fail_bsm_bin = 0
        self.fail_storage_full_bin = 0
        self.fail_threshold_bin = 0
        self._schedule_next("metrics_tick", float(self.args.metrics_dt_s))

    def count_tokens(self, label: str) -> int:
        return sum(batch.count for batch in self.tokens[label])

    @staticmethod
    def _count_pool(pool: Dict[str, List[TokenBatch]], label: str) -> int:
        return sum(batch.count for batch in pool[label])

    def pop_tokens(self, label: str, count: int) -> List[TokenBatch]:
        out: List[TokenBatch] = []
        remaining = int(count)
        batches = self.tokens[label]
        while remaining > 0 and batches:
            batch = batches[0]
            take = min(remaining, batch.count)
            out.append(
                TokenBatch(
                    label=batch.label,
                    count=int(take),
                    ready_ps=int(batch.ready_ps),
                    werner_w=float(batch.werner_w),
                    fidelity=float(batch.fidelity),
                )
            )
            batch.count -= take
            remaining -= take
            if batch.count <= 0:
                batches.pop(0)
        return out

    @staticmethod
    def _pop_pool(
        pool: Dict[str, List[TokenBatch]],
        label: str,
        count: int,
    ) -> List[TokenBatch]:
        out: List[TokenBatch] = []
        remaining = int(count)
        batches = pool[label]
        while remaining > 0 and batches:
            batch = batches[0]
            take = min(remaining, batch.count)
            out.append(
                TokenBatch(
                    label=batch.label,
                    count=int(take),
                    ready_ps=int(batch.ready_ps),
                    werner_w=float(batch.werner_w),
                    fidelity=float(batch.fidelity),
                )
            )
            batch.count -= take
            remaining -= take
            if batch.count <= 0:
                batches.pop(0)
        return out

    @staticmethod
    def pair_batches(
        left: List[TokenBatch],
        right: List[TokenBatch],
    ) -> List[Tuple[int, TokenBatch, TokenBatch]]:
        pairs: List[Tuple[int, TokenBatch, TokenBatch]] = []
        i = j = 0
        while i < len(left) and j < len(right):
            n = min(left[i].count, right[j].count)
            if n > 0:
                pairs.append((n, left[i], right[j]))
            left[i].count -= n
            right[j].count -= n
            if left[i].count <= 0:
                i += 1
            if right[j].count <= 0:
                j += 1
        return pairs


def _metrics_overall_mean_fidelity(metrics: Sequence[MetricRow]) -> float:
    successes = sum(int(row.successes) for row in metrics)
    if successes <= 0:
        return float("nan")
    weighted = sum(int(row.successes) * float(row.mean_fidelity_success) for row in metrics if not math.isnan(float(row.mean_fidelity_success)))
    return weighted / successes


def _write_metrics_csv(out_dir: Path, metrics: Sequence[MetricRow]) -> None:
    fieldnames = list(MetricRow.__dataclass_fields__.keys())
    with (out_dir / "stbd_des_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in metrics:
            writer.writerow([getattr(row, field) for field in fieldnames])


def _write_end_to_end_csv(out_dir: Path, events: Sequence[EndToEndEvent]) -> None:
    fieldnames = list(EndToEndEvent.__dataclass_fields__.keys())
    with (out_dir / "stbd_des_end_to_end_events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in events:
            writer.writerow([getattr(row, field) for field in fieldnames])


def _write_swap_csv(out_dir: Path, events: Sequence[SwapEvent]) -> None:
    fieldnames = list(SwapEvent.__dataclass_fields__.keys())
    with (out_dir / "stbd_des_swap_events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in events:
            writer.writerow([getattr(row, field) for field in fieldnames])


def _shade_segments(ax, result: STBDResult) -> None:
    for idx, segment in enumerate(result.segments):
        ax.axvspan(
            float(segment.start_s),
            float(segment.end_s),
            color="#0b6e4f",
            alpha=0.08,
            label="active STBD segment" if idx == 0 else None,
        )


def _plot_fidelity_over_time(
    out_dir: Path,
    *,
    result: STBDResult,
    metrics: Sequence[MetricRow],
) -> None:
    xs = [float(row.t_s) for row in metrics]
    ys = [
        float(row.cumulative_mean_fidelity_success)
        if row.cumulative_successes > 0
        and not math.isnan(float(row.cumulative_mean_fidelity_success))
        else float("nan")
        for row in metrics
    ]
    finite_points = [
        (x, y)
        for x, y in zip(xs, ys)
        if not math.isnan(float(y))
    ]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    _shade_segments(ax, result)
    ax.plot(
        xs,
        ys,
        color="#c44536",
        marker="o",
        markersize=5.0,
        label="cumulative end-to-end fidelity",
    )
    if finite_points:
        ax.scatter(
            [point[0] for point in finite_points],
            [point[1] for point in finite_points],
            color="#c44536",
            s=36,
            zorder=3,
        )
    ax.set_title("End-to-End Entanglement Fidelity vs Transmission Duration")
    ax.set_xlabel("Entanglement Transmission Duration (s)")
    ax.set_ylabel("End-to-End Entanglement Fidelity")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "stbd_des_fidelity_over_time.png", dpi=200)
    plt.close(fig)


def _plot_throughput_over_time(
    out_dir: Path,
    *,
    result: STBDResult,
    metrics: Sequence[MetricRow],
) -> None:
    xs = [float(row.t_s) for row in metrics]
    ys = [float(row.cumulative_throughput_pairs_per_s) for row in metrics]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    _shade_segments(ax, result)
    ax.plot(
        xs,
        ys,
        color="#0b6e4f",
        marker="o",
        markersize=4.0,
        label="cumulative end-to-end throughput",
    )
    ax.set_title("End-to-End Throughput vs Transmission Duration")
    ax.set_xlabel("Entanglement Transmission Duration (s)")
    ax.set_ylabel("Throughput (pairs/s)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "stbd_des_throughput_over_time.png", dpi=200)
    plt.close(fig)


def summarize_run(
    sim: STBDPhysicsDES,
    *,
    stbd_result: STBDResult,
    route_edges: Sequence[EdgeConfig],
) -> Dict[str, object]:
    return {
        "mode": "stbd_physical_des",
        "route": {
            "physical_nodes": list(stbd_result.physical_nodes),
            "segments": [
                {
                    "snapshot_index": int(segment.snapshot_index),
                    "start_s": float(segment.start_s),
                    "end_s": float(segment.end_s),
                    "nodes": list(segment.nodes),
                }
                for segment in stbd_result.segments
            ],
            "edges": [
                {
                    "label": edge.label,
                    "u": edge.u,
                    "v": edge.v,
                    "source_node": edge.source_node,
                }
                for edge in route_edges
            ],
            "planned_segmented_fidelity": float(stbd_result.segmented_fidelity),
            "planned_delayed_fidelity": float(stbd_result.delayed_fidelity),
        },
        "execution": {
            "cumulative_successes": int(sim.cumulative_successes),
            "mean_fidelity_success": float(_metrics_overall_mean_fidelity(sim.metrics)),
            "throughput_pairs_per_s": int(sim.cumulative_successes)
            / max(1e-12, float(sim.args.duration_s)),
            "end_to_end_event_count": int(len(sim.end_to_end_events)),
            "swap_event_count": int(len(sim.swap_events)),
        },
        "artifacts": {
            "tables": [
                "stbd_des_metrics.csv",
                "stbd_des_end_to_end_events.csv",
                "stbd_des_swap_events.csv",
            ],
            "plots": [
                "stbd_des_fidelity_over_time.png",
                "stbd_des_throughput_over_time.png",
            ],
        },
    }


def write_outputs(
    sim: STBDPhysicsDES,
    *,
    stbd_result: STBDResult,
    route_edges: Sequence[EdgeConfig],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(out_dir, sim.metrics)
    _write_end_to_end_csv(out_dir, sim.end_to_end_events)
    _write_swap_csv(out_dir, sim.swap_events)
    if not bool(sim.args.no_plots):
        _plot_fidelity_over_time(out_dir, result=stbd_result, metrics=sim.metrics)
        _plot_throughput_over_time(out_dir, result=stbd_result, metrics=sim.metrics)
    (out_dir / "stbd_des_summary.json").write_text(
        json.dumps(
            summarize_run(sim, stbd_result=stbd_result, route_edges=route_edges),
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_des_parser()
    parser.description = "Physical STBD-over-DES simulator for LEO constellations."
    parser.set_defaults(
        out="stbd_des_out",
        tle=_default_tle_path(),
        duration_s=3600.0,
        metrics_dt_s=30.0,
        link_update_dt_s=1.0,
        gen_delay_s=0.2,
        swap_dt_s=0.2,
        source_rate_hz=1_000_000.0,
        max_modes_per_label=8192,
        max_bsm_per_cycle=4096,
        f_min=0.25,
        no_event_log=True,
        no_plots=False,
        mem_platform="COLD_ATOM",
        mem_t2=300.0,
        min_elevation_deg=10.0,
    )
    parser.add_argument("--scenario", choices=["sat-sat", "ground-ground"], default="sat-sat")
    parser.add_argument(
        "--constellation-source",
        choices=["tle", "walker"],
        default="tle",
        help="Use sampled TLE ephemerides or a synthetic Walker-delta shell.",
    )
    parser.add_argument("--num-sats", type=int, default=24)
    parser.add_argument("--walker-planes", type=int, default=20)
    parser.add_argument("--walker-sats-per-plane", type=int, default=8)
    parser.add_argument("--walker-altitude-km", type=float, default=550.0)
    parser.add_argument("--walker-inclination-deg", type=float, default=60.0)
    parser.add_argument("--walker-phase", type=int, default=1)
    parser.add_argument(
        "--satellite-selection",
        choices=["sequential", "ranked_dual_visibility", "ranked_endpoint_visibility"],
        default="sequential",
        help="How to choose the satellite pool before STBD planning.",
    )
    parser.add_argument("--src-index", type=int, default=2)
    parser.add_argument("--dst-index", type=int, default=13)
    parser.add_argument("--ground-src", choices=sorted(GROUND_PRESETS.keys()), default="luxembourg")
    parser.add_argument("--ground-dst", choices=sorted(GROUND_PRESETS.keys()), default="norway")
    parser.add_argument("--snapshot-s", type=float, default=60.0)
    parser.add_argument(
        "--paced-delivery-dt-s",
        type=float,
        default=30.0,
        help="For STBD paced execution, cadence used to release end-to-end deliveries inside the final active segment.",
    )
    parser.add_argument(
        "--coherence-time-s",
        type=float,
        default=None,
        help="Legacy alias that sets the physical memory coherence and, unless overridden, the STBD planning coherence.",
    )
    parser.add_argument(
        "--planner-coherence-s",
        type=float,
        default=None,
        help="STBD planning coherence used to build holdover edges. Defaults to --coherence-time-s, then --mem-t2.",
    )
    parser.add_argument(
        "--planner-horizon-s",
        type=float,
        default=None,
        help="Destination horizon for STBD planning. Defaults to the full simulation duration.",
    )
    parser.add_argument("--max-link-distance-km", type=float, default=5000.0)
    parser.add_argument("--planner-fidelity-threshold", type=float, default=0.25)
    parser.add_argument("--w-fid", type=float, default=0.5)
    parser.add_argument("--w-mem", type=float, default=0.5)
    parser.add_argument("--fid-utility-mode", choices=["product", "probability", "fidelity", "werner"], default="product")
    parser.add_argument("--bsm-round-duration-s", type=float, default=1e-3)
    parser.add_argument(
        "--perfect-isl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat inter-satellite elementary links as perfect in both STBD planning and DES execution.",
    )
    parser.add_argument(
        "--direct-swap-on-full-visibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When all elementary links on the planned route are simultaneously active, perform BSM directly without memory buffering.",
    )
    parser.add_argument(
        "--stochastic-link-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample stochastic FSO link states during DES execution instead of relying only on mean link metrics.",
    )
    return parser


def prepare_run(
    args: argparse.Namespace,
) -> Tuple[GeometryEngine, STBDResult, List[EdgeConfig]]:
    satellites = _load_satellite_subset(args)
    src, dst, ground_nodes = _scenario_endpoints(args, satellites)
    epoch_unix_s = representative_epoch_unix_s(satellites)
    propagator = SkyfieldPropagator(
        satellites=satellites,
        ground_nodes=ground_nodes,
        epoch_unix_s=epoch_unix_s,
    )
    geometry = GeometryEngine(propagator)
    planning_coherence_s = _resolve_planning_coherence_s(args)
    planning_start_s = _resolve_planning_start_s(args)
    route_result = plan_stbd_route(
        geometry=geometry,
        channel_model=build_fso_model(args),
        node_names=[sat.name for sat in satellites] + [gs.name for gs in ground_nodes],
        src=src,
        dst=dst,
        start_s=float(planning_start_s),
        duration_s=float(args.duration_s),
        snapshot_s=float(args.snapshot_s),
        coherence_time_s=float(planning_coherence_s),
        max_link_distance_m=float(args.max_link_distance_km) * 1e3,
        fidelity_threshold=float(args.planner_fidelity_threshold),
        w_fid=float(args.w_fid),
        w_mem=float(args.w_mem),
        fid_utility_mode=str(args.fid_utility_mode),
        bsm_round_duration_s=float(args.bsm_round_duration_s),
        destination_horizon_s=float(_resolve_destination_horizon_s(args)),
        perfect_intersatellite=bool(getattr(args, "perfect_isl", False)),
    )
    if len(route_result.physical_nodes) < 2:
        raise ValueError("STBD planning did not produce a usable route.")
    route_edges = build_route_edges(route_result.physical_nodes)
    return geometry, route_result, route_edges


def simulate_prepared_run(
    args: argparse.Namespace,
    *,
    geometry: GeometryEngine,
    route_result: STBDResult,
    route_edges: Sequence[EdgeConfig],
) -> STBDPhysicsDES:
    if len(route_result.physical_nodes) < 2:
        raise ValueError("STBD planning did not produce a usable route.")

    if getattr(args, "coherence_time_s", None) is not None:
        args.mem_t2 = float(args.coherence_time_s)

    env = DESEnv(name="stbd_physical_des", default=True)
    sim = STBDPhysicsDES(
        name="stbd_physics",
        env=env,
        args=args,
        geometry=geometry,
        fso=build_fso_model(args),
        memory=build_memory_model(args),
        bsm=build_bsm_model(args),
        stbd_result=route_result,
        route_edges=route_edges,
    )
    env.init()
    env.run(end_time=int(float(args.duration_s) * PS_PER_S), logging=False, summary=False)
    return sim


def run_once(
    args: argparse.Namespace,
) -> Tuple[STBDPhysicsDES, STBDResult, List[EdgeConfig]]:
    geometry, route_result, route_edges = prepare_run(args)
    sim = simulate_prepared_run(
        args,
        geometry=geometry,
        route_result=route_result,
        route_edges=route_edges,
    )
    return sim, route_result, route_edges


def main() -> None:
    args = build_parser().parse_args()
    sim, route_result, route_edges = run_once(args)
    out_dir = Path(args.out)
    write_outputs(sim, stbd_result=route_result, route_edges=route_edges, out_dir=out_dir)
    summary = summarize_run(sim, stbd_result=route_result, route_edges=route_edges)
    print("STBD physical DES run")
    print(f"cumulative_successes={summary['execution']['cumulative_successes']}")
    print(f"mean_fidelity_success={summary['execution']['mean_fidelity_success']}")
    print(f"throughput_pairs_per_s={summary['execution']['throughput_pairs_per_s']}")
    print(f"planned_path={' -> '.join(route_result.physical_nodes)}")
    print(f"outputs={out_dir.resolve()}")


if __name__ == "__main__":
    main()
