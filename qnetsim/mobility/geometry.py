from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from mobility.propagator import (
    EARTH_RADIUS_M,
    GroundStation,
    SkyfieldPropagator,
    euclidean_distance,
    vector_dot,
    vector_norm,
    vector_sub,
)


# ============================================================
# Contact / visibility geometry
# ============================================================
# Purpose:
# - Convert propagated positions into communication-relevant geometry.
# - Provide visibility, elevation, slant range, Earth occultation, and
#   simple contact-window sampling.
#
# Notes:
# - This layer is policy-aware (e.g. min elevation masks).
# - It is intentionally independent from channel physics.


@dataclass(frozen=True)
class ContactGeometry:
    t_ps: int
    t_s: float
    distance_m: float
    visible: bool
    elevation_deg: float


@dataclass(frozen=True)
class ContactWindow:
    a: str
    b: str
    start_ps: int
    end_ps: int
    duration_ps: int
    peak_elevation_deg: float
    min_distance_m: float


@dataclass(frozen=True)
class LinkGeometrySample:
    a: str
    b: str
    t_ps: int
    distance_m: float
    visible: bool
    elevation_deg: float


class GeometryEngine:
    def __init__(self, propagator: SkyfieldPropagator):
        self.propagator = propagator

    # ------------------------------------------------------------
    # Node-type helpers
    # ------------------------------------------------------------
    def is_ground(self, name: str) -> bool:
        return self.propagator.has_ground_node(name)

    def is_satellite(self, name: str) -> bool:
        return self.propagator.has_satellite(name)

    # ------------------------------------------------------------
    # Primitive geometric checks
    # ------------------------------------------------------------
    def elevation_deg(self, sat_name: str, gs_name: str, t_ps: int) -> float:
        return self.propagator.elevation_deg(sat_name, gs_name, t_ps)

    def slant_range_m(self, a: str, b: str, t_ps: int) -> float:
        return self.propagator.range_m(a, b, t_ps)

    def sat_sat_los_clear(self, sat_a: str, sat_b: str, t_ps: int) -> bool:
        """
        Earth occultation test in ECEF.

        For segment p(tau) = pa + tau(pb-pa), tau in [0,1], find the point of
        closest approach to Earth's center. If that minimum radius is > Earth
        radius, the line of sight is clear.
        """
        pa = self.propagator.node_ecef(sat_a, t_ps).as_tuple()
        pb = self.propagator.node_ecef(sat_b, t_ps).as_tuple()
        d = vector_sub(pb, pa)
        dd = vector_dot(d, d)
        if dd <= 0.0:
            return False
        tau = max(0.0, min(1.0, -vector_dot(pa, d) / dd))
        pmin = (pa[0] + tau * d[0], pa[1] + tau * d[1], pa[2] + tau * d[2])
        return vector_norm(pmin) > EARTH_RADIUS_M

    def ground_sat_visible(self, gs_name: str, sat_name: str, t_ps: int) -> bool:
        gs = self.propagator.require_ground(gs_name)
        elev = self.elevation_deg(sat_name, gs_name, t_ps)
        return elev >= float(gs.min_elevation_deg)

    def visible1(self, a: str, b: str, t_ps: int) -> bool:
        a_is_ground = self.is_ground(a)
        b_is_ground = self.is_ground(b)
        a_is_sat = self.is_satellite(a)
        b_is_sat = self.is_satellite(b)

        if a_is_ground and b_is_ground:
            return True
        if a_is_ground and b_is_sat:
            return self.ground_sat_visible(a, b, t_ps)
        if b_is_ground and a_is_sat:
            return self.ground_sat_visible(b, a, t_ps)
        if a_is_sat and b_is_sat:
            return self.sat_sat_los_clear(a, b, t_ps)
        raise KeyError(f"unknown node pairing: {a}, {b}")

    def visible(self, a: str, b: str, t_ps: int) -> bool:
        return self.visible1(a, b, t_ps)

    def contact_geometry(self, a: str, b: str, t_ps: int) -> ContactGeometry:
        dist = self.slant_range_m(a, b, t_ps)
        if self.is_ground(a) and self.is_satellite(b):
            elev = self.elevation_deg(b, a, t_ps)
        elif self.is_ground(b) and self.is_satellite(a):
            elev = self.elevation_deg(a, b, t_ps)
        else:
            elev = 90.0
        vis = self.visible(a, b, t_ps)
        return ContactGeometry(
            t_ps=int(t_ps),
            t_s=float(t_ps) / 1e12,
            distance_m=float(dist),
            visible=bool(vis),
            elevation_deg=float(elev),
        )

    # ------------------------------------------------------------
    # Sampling / windows
    # ------------------------------------------------------------
    def sample_link(
        self,
        a: str,
        b: str,
        *,
        start_ps: int,
        end_ps: int,
        step_ps: int,
    ) -> List[LinkGeometrySample]:
        if step_ps <= 0:
            raise ValueError("step_ps must be > 0")
        samples: List[LinkGeometrySample] = []
        t_ps = int(start_ps)
        while t_ps <= int(end_ps):
            g = self.contact_geometry(a, b, t_ps)
            samples.append(
                LinkGeometrySample(
                    a=a,
                    b=b,
                    t_ps=t_ps,
                    distance_m=g.distance_m,
                    visible=g.visible,
                    elevation_deg=g.elevation_deg,
                )
            )
            t_ps += int(step_ps)
        return samples

    def contact_windows(
        self,
        a: str,
        b: str,
        *,
        start_ps: int,
        end_ps: int,
        step_ps: int,
    ) -> List[ContactWindow]:
        samples = self.sample_link(
            a, b, start_ps=start_ps, end_ps=end_ps, step_ps=step_ps
        )
        if not samples:
            return []

        windows: List[ContactWindow] = []
        in_window = False
        win_start_ps = 0
        peak_elev = -1e18
        min_dist = 1e30

        for idx, s in enumerate(samples):
            if s.visible and not in_window:
                in_window = True
                win_start_ps = s.t_ps
                peak_elev = s.elevation_deg
                min_dist = s.distance_m
            elif s.visible and in_window:
                peak_elev = max(peak_elev, s.elevation_deg)
                min_dist = min(min_dist, s.distance_m)
            elif (not s.visible) and in_window:
                win_end_ps = samples[max(0, idx - 1)].t_ps
                windows.append(
                    ContactWindow(
                        a=a,
                        b=b,
                        start_ps=win_start_ps,
                        end_ps=win_end_ps,
                        duration_ps=win_end_ps - win_start_ps,
                        peak_elevation_deg=peak_elev,
                        min_distance_m=min_dist,
                    )
                )
                in_window = False

        if in_window:
            win_end_ps = samples[-1].t_ps
            windows.append(
                ContactWindow(
                    a=a,
                    b=b,
                    start_ps=win_start_ps,
                    end_ps=win_end_ps,
                    duration_ps=win_end_ps - win_start_ps,
                    peak_elevation_deg=peak_elev,
                    min_distance_m=min_dist,
                )
            )

        return windows

    # ------------------------------------------------------------
    # Convenience summaries
    # ------------------------------------------------------------
    def nearest_approach(
        self,
        a: str,
        b: str,
        *,
        start_ps: int,
        end_ps: int,
        step_ps: int,
    ) -> Tuple[int, float]:
        samples = self.sample_link(
            a, b, start_ps=start_ps, end_ps=end_ps, step_ps=step_ps
        )
        if not samples:
            raise ValueError("no samples available")
        best = min(samples, key=lambda s: s.distance_m)
        return best.t_ps, best.distance_m

    def peak_elevation(
        self,
        gs_name: str,
        sat_name: str,
        *,
        start_ps: int,
        end_ps: int,
        step_ps: int,
    ) -> Tuple[int, float]:
        samples = self.sample_link(
            gs_name, sat_name, start_ps=start_ps, end_ps=end_ps, step_ps=step_ps
        )
        if not samples:
            raise ValueError("no samples available")
        best = max(samples, key=lambda s: s.elevation_deg)
        return best.t_ps, best.elevation_deg


# ============================================================
# Standalone helpers
# ============================================================
def earth_occults_segment(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> bool:
    d = vector_sub(b, a)
    dd = vector_dot(d, d)
    if dd <= 0.0:
        return True
    tau = max(0.0, min(1.0, -vector_dot(a, d) / dd))
    pmin = (a[0] + tau * d[0], a[1] + tau * d[1], a[2] + tau * d[2])
    return vector_norm(pmin) <= EARTH_RADIUS_M


def line_of_sight_clear(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> bool:
    return not earth_occults_segment(a, b)


__all__ = [
    "ContactGeometry",
    "ContactWindow",
    "GeometryEngine",
    "LinkGeometrySample",
    "earth_occults_segment",
    "line_of_sight_clear",
]
