from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sdn_controller.link_state_db import LinkStateDB, LinkStateRecord


# ============================================================
# Dynamic path engine for the Ryu controller
# ============================================================
# Purpose:
# - Build a controller-side routing graph from dynamic link-state.
# - Support utility-aware path selection for entanglement routing.
# - Stay compatible with the original STBD-style idea, while using
#   the latest visible links and utilities from LinkStateDB.
#
# Design notes:
# - This is intentionally lightweight and pure Python.
# - It treats the network as an undirected graph.
# - Edge cost is derived from link utility / probability / fidelity.
# - Hidden links are excluded.


@dataclass(frozen=True)
class PathEngineConfig:
    metric: str = "utility"
    fid_threshold: float = 0.70
    p_link_threshold: float = 0.0
    prefer_shorter_distance_tie_break: bool = True
    epsilon: float = 1e-12


@dataclass(frozen=True)
class PathResult:
    path_nodes: List[str]
    path_labels: List[str]
    total_cost: float
    min_fidelity: float
    min_probability: float


class DynamicPathEngine:
    def __init__(self, *, link_db: LinkStateDB, cfg: Optional[PathEngineConfig] = None):
        self.link_db = link_db
        self.cfg = cfg or PathEngineConfig()

    # ------------------------------------------------------------
    # Edge filtering / scoring
    # ------------------------------------------------------------
    def is_edge_usable(self, rec: LinkStateRecord) -> bool:
        if not rec.visible:
            return False
        if float(rec.fidelity) < float(self.cfg.fid_threshold):
            return False
        if float(rec.p_link) < float(self.cfg.p_link_threshold):
            return False
        return True

    def edge_cost(self, rec: LinkStateRecord) -> float:
        eps = float(self.cfg.epsilon)
        metric = str(self.cfg.metric).lower().strip()

        if metric == "probability":
            score = max(eps, min(1.0, float(rec.p_link)))
        elif metric == "distance":
            # Direct geometric cost.
            return max(0.0, float(rec.distance_m))
        elif metric == "product":
            score = max(eps, min(1.0, float(rec.p_link) * float(rec.fidelity)))
        else:
            # Default: controller-provided utility proxy.
            score = max(eps, min(1.0, float(rec.u_fid)))

        cost = -math.log(score)
        if self.cfg.prefer_shorter_distance_tie_break:
            cost += 1e-15 * max(0.0, float(rec.distance_m))
        return cost

    # ------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------
    def adjacency(self) -> Dict[str, List[Tuple[str, LinkStateRecord, float]]]:
        adj: Dict[str, List[Tuple[str, LinkStateRecord, float]]] = {}
        for rec in self.link_db.records():
            if not self.is_edge_usable(rec):
                continue
            cost = self.edge_cost(rec)
            adj.setdefault(rec.u, []).append((rec.v, rec, cost))
            adj.setdefault(rec.v, []).append((rec.u, rec, cost))
        return adj

    # ------------------------------------------------------------
    # Shortest-path search (Dijkstra)
    # ------------------------------------------------------------
    def pick_path(self, *, src: str, dst: str) -> PathResult:
        if src == dst:
            return PathResult(
                path_nodes=[src],
                path_labels=[],
                total_cost=0.0,
                min_fidelity=1.0,
                min_probability=1.0,
            )

        adj = self.adjacency()
        if src not in adj or dst not in adj:
            return PathResult(
                path_nodes=[src, dst],
                path_labels=[],
                total_cost=float("inf"),
                min_fidelity=0.0,
                min_probability=0.0,
            )

        dist: Dict[str, float] = {src: 0.0}
        prev_node: Dict[str, str] = {}
        prev_edge: Dict[str, LinkStateRecord] = {}
        visited: set[str] = set()

        while True:
            cur: Optional[str] = None
            cur_d = float("inf")
            for n, d in dist.items():
                if n in visited:
                    continue
                if d < cur_d:
                    cur = n
                    cur_d = d
            if cur is None:
                break
            if cur == dst:
                break

            visited.add(cur)
            for nxt, rec, cost in adj.get(cur, []):
                nd = cur_d + cost
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev_node[nxt] = cur
                    prev_edge[nxt] = rec

        if dst not in dist:
            return PathResult(
                path_nodes=[src, dst],
                path_labels=[],
                total_cost=float("inf"),
                min_fidelity=0.0,
                min_probability=0.0,
            )

        # Backtrack path.
        nodes_rev: List[str] = [dst]
        labels_rev: List[str] = []
        min_fid = 1.0
        min_prob = 1.0
        cur = dst
        while cur != src:
            rec = prev_edge[cur]
            labels_rev.append(rec.label)
            min_fid = min(min_fid, float(rec.fidelity))
            min_prob = min(min_prob, float(rec.p_link))
            cur = prev_node[cur]
            nodes_rev.append(cur)

        nodes_rev.reverse()
        labels_rev.reverse()
        return PathResult(
            path_nodes=nodes_rev,
            path_labels=labels_rev,
            total_cost=float(dist[dst]),
            min_fidelity=float(min_fid),
            min_probability=float(min_prob),
        )

    # ------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------
    def best_path_nodes(self, *, src: str, dst: str) -> List[str]:
        return self.pick_path(src=src, dst=dst).path_nodes

    def best_path_labels(self, *, src: str, dst: str) -> List[str]:
        return self.pick_path(src=src, dst=dst).path_labels


__all__ = [
    "DynamicPathEngine",
    "PathEngineConfig",
    "PathResult",
]
