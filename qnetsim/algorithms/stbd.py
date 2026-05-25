from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from algorithms.nesting import (
    NestingSegment,
    binary_nesting_round_durations,
    late_nesting_werner,
    segmented_nesting_werner,
)
from channels.fso import (
    FSOAnalyticalChannelModel,
    fidelity_to_werner,
    werner_to_fidelity,
)
from mobility.geometry import GeometryEngine


Vertex = Tuple[int, str]


@dataclass(frozen=True)
class SnapshotWindow:
    index: int
    start_s: float
    end_s: float

    @property
    def mid_s(self) -> float:
        return 0.5 * (float(self.start_s) + float(self.end_s))


@dataclass(frozen=True)
class SnapshotEdge:
    src: Vertex
    dst: Vertex
    fid_utility: float
    wait_s: float
    coherence_time_s: float
    fidelity: float = 1.0
    metadata: str = ""
    kind: str = "link"
    probability: float = 1.0
    distance_m: float = 0.0
    snapshot_index: int = -1
    start_s: float = 0.0
    end_s: float = 0.0


@dataclass(frozen=True)
class SpaceTimePath:
    vertices: List[Vertex]
    edges: List[SnapshotEdge]
    utility: float
    min_fidelity: float
    total_cost: float


@dataclass(frozen=True)
class SegmentedSubpath:
    snapshot_index: int
    start_s: float
    end_s: float
    nodes: List[str]
    edges: List[SnapshotEdge]
    utility: float
    min_fidelity: float


@dataclass(frozen=True)
class STBDResult:
    src: str
    dst: str
    snapshot_windows: List[SnapshotWindow]
    graph_edges: List[SnapshotEdge]
    path: SpaceTimePath
    physical_nodes: List[str]
    segments: List[SegmentedSubpath]
    transition_points: List[Vertex]
    transition_waits_s: List[float]
    segmented_fidelity: float
    delayed_fidelity: float


class ExponentialMemoryModel:
    """
    Paper-aligned exponential coherence helper used for Eq. (5), (22), and (25).
    """

    def __init__(self, coherence_time_s: float):
        self.coherence_time_s = max(1e-12, float(coherence_time_s))

    def write_success_prob(self) -> float:
        return 1.0

    def p_succ(self, t_s: float) -> float:
        return 1.0

    def coherence_factor(self, t_s: float) -> float:
        t_s = max(0.0, float(t_s))
        return math.exp(-t_s / self.coherence_time_s)


def memory_utility(wait_s: float, coherence_time_s: float) -> float:
    wait_s = max(0.0, float(wait_s))
    coherence_time_s = max(1e-12, float(coherence_time_s))
    return math.exp(-wait_s / coherence_time_s)


def edge_utility(
    *,
    fid_utility: float,
    wait_s: float,
    coherence_time_s: float,
    w_fid: float = 0.5,
    w_mem: float = 0.5,
) -> float:
    fid_term = max(1e-12, min(1.0, float(fid_utility)))
    mem_term = max(1e-12, min(1.0, memory_utility(wait_s, coherence_time_s)))
    w_fid = max(0.0, float(w_fid))
    w_mem = max(0.0, float(w_mem))
    total = w_fid + w_mem
    if total <= 0.0:
        w_fid, w_mem = 0.5, 0.5
        total = 1.0
    w_fid /= total
    w_mem /= total
    return (fid_term**w_fid) * (mem_term**w_mem)


def build_snapshot_windows(
    *,
    start_s: float,
    duration_s: float,
    snapshot_s: float,
) -> List[SnapshotWindow]:
    start_s = float(start_s)
    duration_s = max(0.0, float(duration_s))
    snapshot_s = max(1e-9, float(snapshot_s))

    if duration_s <= 0.0:
        return [SnapshotWindow(index=0, start_s=start_s, end_s=start_s + snapshot_s)]

    windows: List[SnapshotWindow] = []
    cur = start_s
    end = start_s + duration_s
    idx = 0
    while cur < end - 1e-12:
        nxt = min(end, cur + snapshot_s)
        windows.append(SnapshotWindow(index=idx, start_s=cur, end_s=nxt))
        cur = nxt
        idx += 1
    return windows


def _communication_utility(
    *,
    probability: float,
    fidelity: float,
    mode: str,
) -> float:
    probability = max(0.0, min(1.0, float(probability)))
    fidelity = max(0.25, min(1.0, float(fidelity)))
    mode = str(mode).lower().strip()

    if mode == "probability":
        return probability
    if mode == "fidelity":
        return fidelity
    if mode == "werner":
        return fidelity_to_werner(fidelity)
    return probability * fidelity


def build_space_time_edges(
    *,
    geometry: GeometryEngine,
    channel_model: FSOAnalyticalChannelModel,
    node_names: Sequence[str],
    snapshot_windows: Sequence[SnapshotWindow],
    coherence_time_s: float,
    max_link_distance_m: float | None = None,
    fidelity_threshold: float = 0.0,
    fid_utility_mode: str = "product",
    perfect_intersatellite: bool = False,
) -> List[SnapshotEdge]:
    edges: List[SnapshotEdge] = []
    max_link_distance_m = (
        None
        if max_link_distance_m is None
        else max(0.0, float(max_link_distance_m))
    )

    for window in snapshot_windows:
        t_ps = int(window.mid_s * 1e12)
        for u, v in combinations(node_names, 2):
            if geometry.is_ground(u) and geometry.is_ground(v):
                continue

            geom = geometry.contact_geometry(u, v, t_ps)
            if not geom.visible:
                continue
            if max_link_distance_m is not None and geom.distance_m > max_link_distance_m:
                continue

            intersat = geometry.is_satellite(u) and geometry.is_satellite(v)
            if perfect_intersatellite and intersat:
                p_link, fidelity = 1.0, 1.0
            else:
                p_link, _werner_w, fidelity = channel_model.mean_link_metrics(
                    distance_m=float(geom.distance_m),
                    elevation_deg=float(geom.elevation_deg),
                    intersatellite=intersat,
                )
            if float(fidelity) < float(fidelity_threshold):
                continue

            fid_utility = _communication_utility(
                probability=p_link,
                fidelity=fidelity,
                mode=fid_utility_mode,
            )
            if fid_utility <= 0.0:
                continue

            common = {
                "fid_utility": float(fid_utility),
                "wait_s": 0.0,
                "coherence_time_s": float(coherence_time_s),
                "fidelity": float(fidelity),
                "kind": "link",
                "probability": float(p_link),
                "distance_m": float(geom.distance_m),
                "snapshot_index": int(window.index),
                "start_s": float(window.start_s),
                "end_s": float(window.end_s),
            }
            edges.append(
                SnapshotEdge(
                    src=(window.index, u),
                    dst=(window.index, v),
                    metadata=f"{u}->{v}@S{window.index}",
                    **common,
                )
            )
            edges.append(
                SnapshotEdge(
                    src=(window.index, v),
                    dst=(window.index, u),
                    metadata=f"{v}->{u}@S{window.index}",
                    **common,
                )
            )

    for prev_window, next_window in zip(snapshot_windows, snapshot_windows[1:]):
        wait_s = max(0.0, float(next_window.start_s) - float(prev_window.start_s))
        for node in node_names:
            edges.append(
                SnapshotEdge(
                    src=(prev_window.index, node),
                    dst=(next_window.index, node),
                    fid_utility=1.0,
                    wait_s=wait_s,
                    coherence_time_s=float(coherence_time_s),
                    fidelity=1.0,
                    metadata=f"wait:{node}:S{prev_window.index}->S{next_window.index}",
                    kind="wait",
                    probability=1.0,
                    distance_m=0.0,
                    snapshot_index=int(prev_window.index),
                    start_s=float(prev_window.start_s),
                    end_s=float(next_window.start_s),
                )
            )

    return edges


def best_space_time_path(
    *,
    edges: Iterable[SnapshotEdge],
    src: Vertex,
    dst_candidates: Sequence[Vertex],
    fidelity_threshold: float = 0.0,
    w_fid: float = 0.5,
    w_mem: float = 0.5,
) -> SpaceTimePath:
    adjacency: Dict[Vertex, List[Tuple[Vertex, SnapshotEdge, float]]] = {}
    for edge in edges:
        if float(edge.fidelity) < float(fidelity_threshold):
            continue
        utility = edge_utility(
            fid_utility=edge.fid_utility,
            wait_s=edge.wait_s,
            coherence_time_s=edge.coherence_time_s,
            w_fid=w_fid,
            w_mem=w_mem,
        )
        cost = -math.log(max(1e-12, utility))
        adjacency.setdefault(edge.src, []).append((edge.dst, edge, cost))

    dst_set = set(dst_candidates)
    pq: List[Tuple[float, Vertex]] = [(0.0, src)]
    dist: Dict[Vertex, float] = {src: 0.0}
    prev_v: Dict[Vertex, Vertex] = {}
    prev_e: Dict[Vertex, SnapshotEdge] = {}

    reached: Vertex | None = None
    while pq:
        cost, cur = heapq.heappop(pq)
        if cost > dist.get(cur, float("inf")):
            continue
        if cur in dst_set:
            reached = cur
            break
        for nxt, edge, edge_cost in adjacency.get(cur, []):
            nd = cost + edge_cost
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev_v[nxt] = cur
                prev_e[nxt] = edge
                heapq.heappush(pq, (nd, nxt))

    if reached is None:
        raise ValueError("no feasible space-time path found")

    vertices_rev = [reached]
    edges_rev: List[SnapshotEdge] = []
    min_fidelity = 1.0
    cur = reached
    while cur != src:
        edge = prev_e[cur]
        edges_rev.append(edge)
        min_fidelity = min(min_fidelity, float(edge.fidelity))
        cur = prev_v[cur]
        vertices_rev.append(cur)

    vertices = list(reversed(vertices_rev))
    path_edges = list(reversed(edges_rev))
    utility = math.exp(-dist[reached])
    return SpaceTimePath(
        vertices=vertices,
        edges=path_edges,
        utility=utility,
        min_fidelity=min_fidelity,
        total_cost=dist[reached],
    )


def _segment_utility(
    edges: Sequence[SnapshotEdge],
    *,
    w_fid: float,
    w_mem: float,
) -> float:
    utility = 1.0
    for edge in edges:
        utility *= edge_utility(
            fid_utility=edge.fid_utility,
            wait_s=edge.wait_s,
            coherence_time_s=edge.coherence_time_s,
            w_fid=w_fid,
            w_mem=w_mem,
        )
    return utility


def segment_path_by_snapshot(
    path: SpaceTimePath,
    *,
    w_fid: float = 0.5,
    w_mem: float = 0.5,
) -> Tuple[List[SegmentedSubpath], List[Vertex], List[float]]:
    segments: List[SegmentedSubpath] = []
    transition_points: List[Vertex] = []
    transition_waits: List[float] = []
    cur_edges: List[SnapshotEdge] = []

    def flush() -> None:
        if not cur_edges:
            return
        nodes = [cur_edges[0].src[1]]
        for edge in cur_edges:
            nodes.append(edge.dst[1])
        segments.append(
            SegmentedSubpath(
                snapshot_index=int(cur_edges[0].snapshot_index),
                start_s=float(cur_edges[0].start_s),
                end_s=float(cur_edges[0].end_s),
                nodes=nodes,
                edges=list(cur_edges),
                utility=_segment_utility(cur_edges, w_fid=w_fid, w_mem=w_mem),
                min_fidelity=min(float(edge.fidelity) for edge in cur_edges),
            )
        )
        cur_edges.clear()

    for edge in path.edges:
        if edge.kind == "wait":
            flush()
            transition_points.append(edge.src)
            transition_waits.append(float(edge.wait_s))
            continue
        cur_edges.append(edge)
    flush()
    return segments, transition_points, transition_waits


def estimate_segmented_nesting_fidelity(
    *,
    path: SpaceTimePath,
    segments: Sequence[SegmentedSubpath],
    transition_waits_s: Sequence[float],
    coherence_time_s: float,
    bsm_round_duration_s: float,
) -> Tuple[float, float]:
    memory = ExponentialMemoryModel(coherence_time_s=float(coherence_time_s))
    segment_inputs: List[NestingSegment] = []

    for segment in segments:
        link_werners = [
            fidelity_to_werner(edge.fidelity)
            for edge in segment.edges
            if edge.kind == "link"
        ]
        round_durations = binary_nesting_round_durations(
            max(0, len(link_werners) - 1),
            float(bsm_round_duration_s),
        )
        segment_inputs.append(
            NestingSegment(
                link_werners=link_werners,
                round_durations_s=round_durations,
            )
        )

    delayed_link_werners = [
        fidelity_to_werner(edge.fidelity) for edge in path.edges if edge.kind == "link"
    ]
    delayed_round_durations = binary_nesting_round_durations(
        max(0, len(delayed_link_werners) - 1),
        float(bsm_round_duration_s),
    )

    segmented_werner = segmented_nesting_werner(
        segments=segment_inputs,
        memory=memory,
        transition_waits_s=transition_waits_s,
    )
    delayed_werner = late_nesting_werner(
        link_werners=delayed_link_werners,
        memory=memory,
        round_durations_s=delayed_round_durations,
        extra_wait_s=sum(float(wait_s) for wait_s in transition_waits_s),
    )
    return (
        werner_to_fidelity(segmented_werner),
        werner_to_fidelity(delayed_werner),
    )


def _physical_nodes_from_segments(segments: Sequence[SegmentedSubpath]) -> List[str]:
    physical_nodes: List[str] = []
    for segment in segments:
        if not segment.nodes:
            continue
        if not physical_nodes:
            physical_nodes.extend(segment.nodes)
            continue
        if physical_nodes[-1] == segment.nodes[0]:
            physical_nodes.extend(segment.nodes[1:])
        else:
            physical_nodes.extend(segment.nodes)
    return physical_nodes


def plan_stbd_route(
    *,
    geometry: GeometryEngine,
    channel_model: FSOAnalyticalChannelModel,
    node_names: Sequence[str],
    src: str,
    dst: str,
    start_s: float,
    duration_s: float,
    snapshot_s: float,
    coherence_time_s: float,
    max_link_distance_m: float | None = None,
    fidelity_threshold: float = 0.0,
    w_fid: float = 0.5,
    w_mem: float = 0.5,
    fid_utility_mode: str = "product",
    bsm_round_duration_s: float = 1e-3,
    destination_horizon_s: float | None = None,
    perfect_intersatellite: bool = False,
) -> STBDResult:
    snapshot_windows = build_snapshot_windows(
        start_s=start_s,
        duration_s=duration_s,
        snapshot_s=snapshot_s,
    )
    graph_edges = build_space_time_edges(
        geometry=geometry,
        channel_model=channel_model,
        node_names=node_names,
        snapshot_windows=snapshot_windows,
        coherence_time_s=coherence_time_s,
        max_link_distance_m=max_link_distance_m,
        fidelity_threshold=fidelity_threshold,
        fid_utility_mode=fid_utility_mode,
        perfect_intersatellite=perfect_intersatellite,
    )
    src_vertex = (snapshot_windows[0].index, str(src))
    horizon_s = (
        float(start_s) + float(coherence_time_s)
        if destination_horizon_s is None
        else float(start_s) + max(0.0, float(destination_horizon_s))
    )
    dst_candidates = [
        (window.index, str(dst))
        for window in snapshot_windows
        if float(window.start_s) <= horizon_s + 1e-12
    ]
    if not dst_candidates:
        raise ValueError("the coherence-time horizon does not include any destination snapshots")

    path = best_space_time_path(
        edges=graph_edges,
        src=src_vertex,
        dst_candidates=dst_candidates,
        fidelity_threshold=fidelity_threshold,
        w_fid=w_fid,
        w_mem=w_mem,
    )
    segments, transition_points, transition_waits_s = segment_path_by_snapshot(
        path,
        w_fid=w_fid,
        w_mem=w_mem,
    )
    segmented_fidelity, delayed_fidelity = estimate_segmented_nesting_fidelity(
        path=path,
        segments=segments,
        transition_waits_s=transition_waits_s,
        coherence_time_s=coherence_time_s,
        bsm_round_duration_s=bsm_round_duration_s,
    )
    return STBDResult(
        src=str(src),
        dst=str(dst),
        snapshot_windows=list(snapshot_windows),
        graph_edges=graph_edges,
        path=path,
        physical_nodes=_physical_nodes_from_segments(segments),
        segments=segments,
        transition_points=transition_points,
        transition_waits_s=list(transition_waits_s),
        segmented_fidelity=float(segmented_fidelity),
        delayed_fidelity=float(delayed_fidelity),
    )


__all__ = [
    "ExponentialMemoryModel",
    "STBDResult",
    "SegmentedSubpath",
    "SnapshotEdge",
    "SnapshotWindow",
    "SpaceTimePath",
    "best_space_time_path",
    "build_snapshot_windows",
    "build_space_time_edges",
    "edge_utility",
    "estimate_segmented_nesting_fidelity",
    "memory_utility",
    "plan_stbd_route",
    "segment_path_by_snapshot",
]
