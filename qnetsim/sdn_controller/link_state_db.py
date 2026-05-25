from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ============================================================
# Dynamic link-state database for the Ryu controller
# ============================================================
# Purpose:
# - Store the latest LINK_STATE_UPDATE received from the simulator.
# - Provide query helpers for the path engine and topology manager.
# - Keep the controller-side dynamic view of visibility, fidelity,
#   probability, distance, and timestamp per edge label.
#
# Design notes:
# - Keys are primarily edge labels, since the simulator and controller
#   already coordinate using labels like A-R1, R1-R2, etc.
# - A secondary undirected (u, v) index is also maintained.
# - This module is pure Python and independent of Ryu internals.


@dataclass(frozen=True)
class LinkStateRecord:
    label: str
    u: str
    v: str
    t_ps: int
    distance_m: float
    visible: bool
    elevation_deg: float
    p_link: float
    fidelity: float
    u_fid: float
    source_node: str = ""

    def edge_key(self) -> Tuple[str, str]:
        return tuple(sorted((self.u, self.v)))


@dataclass
class LinkStateDB:
    _by_label: Dict[str, LinkStateRecord] = field(default_factory=dict)
    _label_by_edge: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # ------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------
    def upsert(
        self,
        *,
        label: str,
        u: str,
        v: str,
        t_ps: int,
        distance_m: float,
        visible: bool,
        elevation_deg: float,
        p_link: float,
        fidelity: float,
        u_fid: float,
        source_node: str = "",
    ) -> LinkStateRecord:
        rec = LinkStateRecord(
            label=str(label),
            u=str(u),
            v=str(v),
            t_ps=int(t_ps),
            distance_m=float(distance_m),
            visible=bool(visible),
            elevation_deg=float(elevation_deg),
            p_link=float(p_link),
            fidelity=float(fidelity),
            u_fid=float(u_fid),
            source_node=str(source_node),
        )
        self._by_label[rec.label] = rec
        self._label_by_edge[rec.edge_key()] = rec.label
        return rec

    def update_from_payload(self, payload: Dict[str, object]) -> LinkStateRecord:
        return self.upsert(
            label=str(payload["label"]),
            u=str(payload["u"]),
            v=str(payload["v"]),
            t_ps=int(payload["t_ps"]),
            distance_m=float(payload["distance_m"]),
            visible=bool(payload["visible"]),
            elevation_deg=float(payload.get("elevation_deg", 90.0)),
            p_link=float(payload.get("p_link", 0.0)),
            fidelity=float(payload.get("fidelity", 0.25)),
            u_fid=float(payload.get("u_fid", payload.get("fidelity", 0.25))),
            source_node=str(payload.get("node", "")),
        )

    # ------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------
    def get_by_label(self, label: str) -> Optional[LinkStateRecord]:
        return self._by_label.get(str(label))

    def get_by_edge(self, u: str, v: str) -> Optional[LinkStateRecord]:
        label = self._label_by_edge.get(tuple(sorted((str(u), str(v)))))
        if label is None:
            return None
        return self._by_label.get(label)

    def require_by_label(self, label: str) -> LinkStateRecord:
        rec = self.get_by_label(label)
        if rec is None:
            raise KeyError(f"unknown link-state label: {label}")
        return rec

    def labels(self) -> List[str]:
        return list(self._by_label.keys())

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._label_by_edge.keys())

    def records(self) -> List[LinkStateRecord]:
        return list(self._by_label.values())

    # ------------------------------------------------------------
    # Filtering / views
    # ------------------------------------------------------------
    def visible_records(self) -> List[LinkStateRecord]:
        return [rec for rec in self._by_label.values() if rec.visible]

    def hidden_records(self) -> List[LinkStateRecord]:
        return [rec for rec in self._by_label.values() if not rec.visible]

    def as_adjacency(self) -> Dict[str, List[LinkStateRecord]]:
        adj: Dict[str, List[LinkStateRecord]] = {}
        for rec in self._by_label.values():
            adj.setdefault(rec.u, []).append(rec)
            adj.setdefault(rec.v, []).append(rec)
        return adj

    def as_visible_adjacency(self) -> Dict[str, List[LinkStateRecord]]:
        adj: Dict[str, List[LinkStateRecord]] = {}
        for rec in self.visible_records():
            adj.setdefault(rec.u, []).append(rec)
            adj.setdefault(rec.v, []).append(rec)
        return adj

    # ------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------
    def best_record_for_edge(self, u: str, v: str) -> Optional[LinkStateRecord]:
        return self.get_by_edge(u, v)

    def utility(self, label: str, default: float = 0.0) -> float:
        rec = self.get_by_label(label)
        if rec is None:
            return float(default)
        return float(rec.u_fid)

    def probability(self, label: str, default: float = 0.0) -> float:
        rec = self.get_by_label(label)
        if rec is None:
            return float(default)
        return float(rec.p_link)

    def fidelity(self, label: str, default: float = 0.25) -> float:
        rec = self.get_by_label(label)
        if rec is None:
            return float(default)
        return float(rec.fidelity)

    def distance(self, label: str, default: float = 0.0) -> float:
        rec = self.get_by_label(label)
        if rec is None:
            return float(default)
        return float(rec.distance_m)

    # ------------------------------------------------------------
    # Mutation / maintenance
    # ------------------------------------------------------------
    def remove_label(self, label: str) -> None:
        rec = self._by_label.pop(str(label), None)
        if rec is None:
            return
        key = rec.edge_key()
        if self._label_by_edge.get(key) == rec.label:
            self._label_by_edge.pop(key, None)

    def clear(self) -> None:
        self._by_label.clear()
        self._label_by_edge.clear()

    def seed_from_static_edges(
        self,
        edges: Iterable[object],
        *,
        default_t_ps: int = 0,
    ) -> None:
        """
        Convenience method for bootstrapping the DB from static EdgeConfig-like
        objects that expose: u, v, label, source_node, u_fid.
        """
        for e in edges:
            self.upsert(
                label=str(getattr(e, "label")),
                u=str(getattr(e, "u")),
                v=str(getattr(e, "v")),
                t_ps=int(default_t_ps),
                distance_m=0.0,
                visible=True,
                elevation_deg=90.0,
                p_link=float(getattr(e, "u_fid", 1.0)),
                fidelity=float(getattr(e, "u_fid", 1.0)),
                u_fid=float(getattr(e, "u_fid", 1.0)),
                source_node=str(getattr(e, "source_node", "")),
            )


__all__ = [
    "LinkStateDB",
    "LinkStateRecord",
]
