from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from channels.fso import FSOAnalyticalModel, FSOParams
from mobility.geometry import ContactGeometry, GeometryEngine


# ============================================================
# Dynamic mobility-aware link model
# ============================================================
# Purpose:
# - Bind orbital/contact geometry to the FSO channel physics.
# - Expose a simulator-friendly interface for time-varying link quality.
# - Provide compact telemetry snapshots for SDN link-state reporting.


@dataclass(frozen=True)
class DynamicLinkMetrics:
    label: str
    u: str
    v: str
    t_ps: int
    distance_m: float
    visible: bool
    elevation_deg: float
    intersatellite: bool
    p_link: float
    werner_w: float
    fidelity: float
    utility_fid: float


class DynamicFSOLink:
    """
    Mobility-aware FSO link.

    Main flow:
    - Query geometry at time t_ps
    - If not visible, return a dead link
    - Otherwise compute analytical FSO metrics for the instantaneous geometry

    This object is intentionally stateless except for its configuration.
    """

    def __init__(
        self,
        *,
        label: str,
        u: str,
        v: str,
        geometry: GeometryEngine,
        params: FSOParams,
        utility_mode: str = "fidelity",
    ):
        self.label = str(label)
        self.u = str(u)
        self.v = str(v)
        self.geometry = geometry
        self.model = FSOAnalyticalModel(params)
        self.utility_mode = str(utility_mode).lower().strip()

    # ------------------------------------------------------------
    # Node-type helpers
    # ------------------------------------------------------------
    def is_intersatellite(self) -> bool:
        return self.geometry.is_satellite(self.u) and self.geometry.is_satellite(self.v)

    def is_ground_satellite(self) -> bool:
        return (
            self.geometry.is_ground(self.u) and self.geometry.is_satellite(self.v)
        ) or (self.geometry.is_ground(self.v) and self.geometry.is_satellite(self.u))

    # ------------------------------------------------------------
    # Utility scoring
    # ------------------------------------------------------------
    def utility_score(self, *, p_link: float, fidelity: float) -> float:
        p = max(0.0, min(1.0, float(p_link)))
        f = max(0.0, min(1.0, float(fidelity)))

        if self.utility_mode == "product":
            return p * f
        if self.utility_mode == "plink":
            return p
        return f

    # ------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------
    def contact_geometry(self, t_ps: int) -> ContactGeometry:
        return self.geometry.contact_geometry(self.u, self.v, int(t_ps))

    def instantaneous_metrics(self, *, t_ps: int) -> DynamicLinkMetrics:
        geom = self.contact_geometry(int(t_ps))
        intersat = self.is_intersatellite()

        if not geom.visible:
            return DynamicLinkMetrics(
                label=self.label,
                u=self.u,
                v=self.v,
                t_ps=int(t_ps),
                distance_m=float(geom.distance_m),
                visible=False,
                elevation_deg=float(geom.elevation_deg),
                intersatellite=intersat,
                p_link=0.0,
                werner_w=0.0,
                fidelity=0.25,
                utility_fid=0.0,
            )

        # Telemetry should expose the deterministic mean snapshot quality.
        p_link, w, f = self.model.mean_link_metrics(
            distance_m=float(geom.distance_m),
            elevation_deg=float(geom.elevation_deg),
            intersatellite=intersat,
        )
        util = self.utility_score(p_link=p_link, fidelity=f)

        return DynamicLinkMetrics(
            label=self.label,
            u=self.u,
            v=self.v,
            t_ps=int(t_ps),
            distance_m=float(geom.distance_m),
            visible=True,
            elevation_deg=float(geom.elevation_deg),
            intersatellite=intersat,
            p_link=float(p_link),
            werner_w=float(w),
            fidelity=float(f),
            utility_fid=float(util),
        )

    def sample_attempt(
        self,
        rng: random.Random,
        *,
        t_ps: int,
    ) -> Tuple[bool, float, float]:
        metrics = self.instantaneous_metrics(t_ps=int(t_ps))
        ok = rng.random() < max(0.0, min(1.0, metrics.p_link))
        if not ok:
            return False, 0.0, 0.25
        return True, metrics.werner_w, metrics.fidelity

    def telemetry_payload(self, *, t_ps: int, node: str = "") -> Dict[str, object]:
        m = self.instantaneous_metrics(t_ps=int(t_ps))
        return {
            "type": "LINK_STATE_UPDATE",
            "node": str(node),
            "label": m.label,
            "u": m.u,
            "v": m.v,
            "t_ps": m.t_ps,
            "distance_m": m.distance_m,
            "visible": m.visible,
            "elevation_deg": m.elevation_deg,
            "p_link": m.p_link,
            "fidelity": m.fidelity,
            "u_fid": m.utility_fid,
        }


class DynamicLinkRegistry:
    """
    Small convenience container for managing multiple dynamic links by label.
    Useful for link-state reporters and demo wiring.
    """

    def __init__(self):
        self._links: Dict[str, DynamicFSOLink] = {}

    def add(self, link: DynamicFSOLink) -> None:
        self._links[link.label] = link

    def get(self, label: str) -> DynamicFSOLink:
        if label not in self._links:
            raise KeyError(f"unknown dynamic link label: {label}")
        return self._links[label]

    def labels(self):
        return list(self._links.keys())

    def items(self):
        return self._links.items()

    def metrics_snapshot(self, *, t_ps: int) -> Dict[str, DynamicLinkMetrics]:
        return {
            label: link.instantaneous_metrics(t_ps=int(t_ps))
            for label, link in self._links.items()
        }

    def telemetry_snapshot(
        self, *, t_ps: int, node: str = ""
    ) -> Dict[str, Dict[str, object]]:
        return {
            label: link.telemetry_payload(t_ps=int(t_ps), node=node)
            for label, link in self._links.items()
        }


__all__ = [
    "DynamicFSOLink",
    "DynamicLinkMetrics",
    "DynamicLinkRegistry",
]
