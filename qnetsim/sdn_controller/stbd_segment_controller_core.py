from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class STBDSegmentWindow:
    snapshot_index: int
    start_s: float
    end_s: float
    labels: tuple[str, ...]
    nodes: tuple[str, ...]


@dataclass
class STBDRequestState:
    req_id: int
    execution_mode: str
    final_label: str
    route_start_s: float
    route_end_s: float
    final_segment_start_s: float
    final_segment_end_s: float
    segments: List[STBDSegmentWindow] = field(default_factory=list)


class STBDSegmentControllerCore:
    """
    Thin STBD control-plane core for the constellation GUI.

    This controller does not re-plan routes. It mirrors the already planned
    space-time route and exposes the currently active labels / final service
    window through a Ryu REST surface so the DES executor can delegate segment
    activation decisions to an SDN controller backend.
    """

    def __init__(self) -> None:
        self._requests: Dict[int, STBDRequestState] = {}

    def reset(self) -> None:
        self._requests.clear()

    def install_request(
        self,
        *,
        req_id: int,
        execution_mode: str,
        final_label: str,
        route_start_s: float,
        route_end_s: float,
        final_segment_start_s: float,
        final_segment_end_s: float,
        segments: List[Dict[str, Any]],
    ) -> None:
        built_segments = [
            STBDSegmentWindow(
                snapshot_index=int(segment.get("snapshot_index", 0)),
                start_s=float(segment.get("start_s", 0.0)),
                end_s=float(segment.get("end_s", 0.0)),
                labels=tuple(str(label) for label in segment.get("labels", [])),
                nodes=tuple(str(node) for node in segment.get("nodes", [])),
            )
            for segment in segments
        ]
        self._requests[int(req_id)] = STBDRequestState(
            req_id=int(req_id),
            execution_mode=str(execution_mode).lower().strip(),
            final_label=str(final_label),
            route_start_s=float(route_start_s),
            route_end_s=float(route_end_s),
            final_segment_start_s=float(final_segment_start_s),
            final_segment_end_s=float(final_segment_end_s),
            segments=built_segments,
        )

    def poll(self, *, req_id: int, t_s: float) -> Dict[str, Any]:
        state = self._requests.get(int(req_id))
        if state is None:
            return {
                "msg_type": "ControlSnapshot",
                "req_id": int(req_id),
                "active_labels": [],
                "active_segments": [],
                "final_segment_active": False,
                "route_active": False,
            }

        now_s = float(t_s)
        active_segments = [
            segment
            for segment in state.segments
            if float(segment.start_s) - 1e-12 <= now_s < float(segment.end_s) + 1e-12
        ]
        active_labels = sorted(
            {
                str(label)
                for segment in active_segments
                for label in segment.labels
            }
        )
        final_segment_active = bool(
            float(state.final_segment_start_s) - 1e-12
            <= now_s
            < float(state.final_segment_end_s) + 1e-12
        )
        route_active = bool(
            float(state.route_start_s) - 1e-12
            <= now_s
            < float(state.route_end_s) + 1e-12
        )
        return {
            "msg_type": "ControlSnapshot",
            "req_id": int(req_id),
            "execution_mode": str(state.execution_mode),
            "active_labels": active_labels,
            "active_segments": [
                {
                    "snapshot_index": int(segment.snapshot_index),
                    "start_s": float(segment.start_s),
                    "end_s": float(segment.end_s),
                    "labels": list(segment.labels),
                    "nodes": list(segment.nodes),
                }
                for segment in active_segments
            ],
            "final_segment_active": bool(final_segment_active),
            "route_active": bool(route_active),
        }


__all__ = ["STBDSegmentControllerCore"]
