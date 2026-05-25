from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sdn_controller.link_state_db import LinkStateDB
from sdn_controller.path_engine import DynamicPathEngine, PathEngineConfig


@dataclass(frozen=True)
class EdgeConfig:
    u: str
    v: str
    label: str
    source_node: str
    u_fid: float = 0.92


DEFAULT_EDGES: List[EdgeConfig] = [
    EdgeConfig("SAT_END", "R1", label="S-R1", source_node="R1", u_fid=0.92),
    EdgeConfig("R1", "GROUND_END", label="R1-G", source_node="R1", u_fid=0.92),
]


@dataclass
class RequestState:
    req_id: int
    src: str
    dst: str
    f_min: float
    timeout_ps: int
    start_ps: int
    deadline_ps: int
    target_pairs: int = 0
    delivered_pairs: int = 0
    buffered_counts: Dict[str, int] = field(default_factory=dict)


class GroundRepeaterSdnControllerCore:
    """
    Lightweight SDN control loop for the Ground repeater scenario.

    The route is still a single repeater chain, but the controller keeps a
    dynamic link-state view of the satellite FSO edge and the local fiber edge,
    decides when each elementary link should be scheduled, and triggers swaps
    once both OGS-side memories hold buffered pairs.
    """

    def __init__(
        self,
        *,
        max_modes_per_label: int,
        max_bsm_per_cycle: int,
        edges: Optional[List[EdgeConfig]] = None,
    ) -> None:
        self.edges = list(edges or DEFAULT_EDGES)
        self.max_modes_per_label = max(1, int(max_modes_per_label))
        self.max_bsm_per_cycle = max(1, int(max_bsm_per_cycle))
        self.link_db = LinkStateDB()
        self.link_db.seed_from_static_edges(self.edges)
        self.path_engine = DynamicPathEngine(
            link_db=self.link_db,
            cfg=PathEngineConfig(
                metric="utility",
                fid_threshold=0.25,
                p_link_threshold=0.0,
            ),
        )
        self._requests: Dict[int, RequestState] = {}

    def reset(self) -> None:
        self.link_db.clear()
        self.link_db.seed_from_static_edges(self.edges)
        self._requests.clear()

    def update_link_state(self, payload: Dict[str, Any]) -> None:
        self.link_db.update_from_payload(payload)

    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None:
        deadline_ps = int(t_ps) + int(timeout_ps) if int(timeout_ps) > 0 else 0
        self._requests[int(req_id)] = RequestState(
            req_id=int(req_id),
            src=str(src),
            dst=str(dst),
            f_min=float(f_min),
            timeout_ps=int(timeout_ps),
            start_ps=int(t_ps),
            deadline_ps=int(deadline_ps),
            target_pairs=max(0, int(target_pairs)),
            buffered_counts={"S-R1": 0, "R1-G": 0},
        )

    def poll(self, node: str, t_ps: int) -> List[Dict[str, Any]]:
        node = str(node)
        now_ps = int(t_ps)
        actions: List[Dict[str, Any]] = []
        for st in self._requests.values():
            if not self._request_active(st, now_ps):
                continue
            if node != "R1":
                continue
            actions.extend(self._schedule_actions(st))
            swap = self._swap_action(st)
            if swap is not None:
                actions.append(swap)
        return actions

    def report_storage(
        self,
        *,
        req_id: int,
        label: str,
        accepted_count: int,
    ) -> None:
        st = self._requests.get(int(req_id))
        if st is None:
            return
        label = str(label)
        count = max(0, int(accepted_count))
        if count <= 0:
            return
        cur = int(st.buffered_counts.get(label, 0))
        st.buffered_counts[label] = min(self.max_modes_per_label, cur + count)

    def report_swap_result(
        self,
        *,
        req_id: int,
        success_count: int,
    ) -> None:
        st = self._requests.get(int(req_id))
        if st is None:
            return
        st.delivered_pairs += max(0, int(success_count))

    def _request_active(self, st: RequestState, now_ps: int) -> bool:
        if st.target_pairs > 0 and st.delivered_pairs >= st.target_pairs:
            return False
        if st.timeout_ps > 0 and int(now_ps) > int(st.deadline_ps):
            return False
        return True

    def _usable_local_edges(self, st: RequestState) -> List[EdgeConfig]:
        usable_labels: set[str] = set()
        best_path = self.path_engine.pick_path(src=st.src, dst=st.dst)
        usable_labels.update(best_path.path_labels)
        for edge in self.edges:
            rec = self.link_db.get_by_label(edge.label)
            if rec is None:
                continue
            if not rec.visible:
                continue
            if float(rec.fidelity) < float(st.f_min):
                continue
            if float(rec.p_link) <= 0.0:
                continue
            usable_labels.add(edge.label)
        return [edge for edge in self.edges if edge.label in usable_labels]

    def _schedule_actions(self, st: RequestState) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for edge in self._usable_local_edges(st):
            if edge.source_node != "R1":
                continue
            actions.append(
                {
                    "msg_type": "StartSchedule",
                    "req_id": st.req_id,
                    "label": edge.label,
                    "f_min": st.f_min,
                }
            )
        return actions

    def _swap_action(self, st: RequestState) -> Optional[Dict[str, Any]]:
        left = int(st.buffered_counts.get("S-R1", 0))
        right = int(st.buffered_counts.get("R1-G", 0))
        n = min(left, right, self.max_bsm_per_cycle)
        if n <= 0:
            return None
        st.buffered_counts["S-R1"] = max(0, left - n)
        st.buffered_counts["R1-G"] = max(0, right - n)
        return {
            "msg_type": "InstructSwap",
            "req_id": st.req_id,
            "middle": "R1",
            "left_label": "S-R1",
            "right_label": "R1-G",
            "count": int(n),
            "f_min": st.f_min,
        }


__all__ = [
    "DEFAULT_EDGES",
    "EdgeConfig",
    "GroundRepeaterSdnControllerCore",
    "RequestState",
]
