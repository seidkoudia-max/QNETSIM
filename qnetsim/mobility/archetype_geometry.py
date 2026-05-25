from __future__ import annotations

import math
from dataclasses import dataclass

from mobility.geometry import ContactGeometry
from mobility.propagator import EARTH_RADIUS_M, PS_PER_S


MU_EARTH_M3_S2 = 3.986004418e14


@dataclass(frozen=True)
class OverpassScenario:
    key: str
    title: str
    baseline_km: float
    track_angle_deg: float
    delta_km: float
    altitude_km: float = 500.0
    half_window_s: float = 220.0


class ParametricOverpassGeometry:
    """
    Paper-style two-OGS / one-LEO overpass geometry parameterized by {Delta, phi}.

    The geometry follows the representative construction in arXiv:2604.16165:
    - Alice and Bob lie on an OGS baseline that is modeled as a great-circle arc
      of length d on a spherical Earth.
    - The satellite is in a circular orbit of altitude h.
    - The overpass is parameterized by the crossing angle phi between the
      satellite ground track and the OGS baseline, together with the offset
      Delta of the crossing point relative to the OGS-baseline midpoint.

    The resulting slant range and elevation are then consumed by the repository
    FSO, memory, and BSM models through the DES engine.
    """

    def __init__(self, scenario: OverpassScenario, min_elevation_deg: float):
        self.scenario = scenario
        self.min_elevation_deg = float(min_elevation_deg)
        self._earth_radius_km = float(EARTH_RADIUS_M) / 1e3
        self._earth_radius_m = float(EARTH_RADIUS_M)
        self._sat_radius_m = self._earth_radius_m + float(self.scenario.altitude_km) * 1e3
        self._omega_rad_s = math.sqrt(MU_EARTH_M3_S2 / (self._sat_radius_m**3))
        self._station_units = self._station_unit_vectors()
        self._u0, self._t0 = self._crossing_frame()

    @staticmethod
    def _unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            raise ValueError("cannot normalize a zero vector")
        return (vec[0] / norm, vec[1] / norm, vec[2] / norm)

    @staticmethod
    def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _surface_unit(lon_rad: float, lat_rad: float = 0.0) -> tuple[float, float, float]:
        c_lat = math.cos(lat_rad)
        return (
            c_lat * math.cos(lon_rad),
            c_lat * math.sin(lon_rad),
            math.sin(lat_rad),
        )

    def _station_unit_vectors(self) -> dict[str, tuple[float, float, float]]:
        half_sep_rad = 0.5 * float(self.scenario.baseline_km) / self._earth_radius_km
        return {
            "Alice": self._surface_unit(-half_sep_rad, 0.0),
            "Bob": self._surface_unit(half_sep_rad, 0.0),
        }

    def _crossing_frame(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lon0_rad = float(self.scenario.delta_km) / self._earth_radius_km
        u0 = self._surface_unit(lon0_rad, 0.0)
        east = self._unit((-math.sin(lon0_rad), math.cos(lon0_rad), 0.0))
        north = (0.0, 0.0, 1.0)
        phi = math.radians(float(self.scenario.track_angle_deg))
        t0 = self._unit(
            (
                math.cos(phi) * east[0] + math.sin(phi) * north[0],
                math.cos(phi) * east[1] + math.sin(phi) * north[1],
                math.cos(phi) * east[2] + math.sin(phi) * north[2],
            )
        )
        return u0, t0

    def _satellite_unit(self, t_centered_s: float) -> tuple[float, float, float]:
        alpha = self._omega_rad_s * float(t_centered_s)
        return self._unit(
            (
                math.cos(alpha) * self._u0[0] + math.sin(alpha) * self._t0[0],
                math.cos(alpha) * self._u0[1] + math.sin(alpha) * self._t0[1],
                math.cos(alpha) * self._u0[2] + math.sin(alpha) * self._t0[2],
            )
        )

    def _contact(self, station: str, t_centered_s: float, t_ps: int) -> ContactGeometry:
        gs_hat = self._station_units[station]
        sat_hat = self._satellite_unit(t_centered_s)
        gs_vec_m = tuple(self._earth_radius_m * v for v in gs_hat)
        sat_vec_m = tuple(self._sat_radius_m * v for v in sat_hat)
        los_vec_m = (
            sat_vec_m[0] - gs_vec_m[0],
            sat_vec_m[1] - gs_vec_m[1],
            sat_vec_m[2] - gs_vec_m[2],
        )
        distance_m = math.sqrt(sum(v * v for v in los_vec_m))
        los_hat = self._unit(los_vec_m)
        elevation_rad = math.asin(max(-1.0, min(1.0, self._dot(los_hat, gs_hat))))
        elevation_deg = math.degrees(elevation_rad)
        visible = elevation_deg >= self.min_elevation_deg
        return ContactGeometry(
            t_ps=int(t_ps),
            t_s=float(t_centered_s),
            distance_m=float(distance_m),
            visible=bool(visible),
            elevation_deg=float(elevation_deg),
        )

    def contact_geometry(self, a: str, b: str, t_ps: int) -> ContactGeometry:
        if {a, b} not in ({"Alice", "R1"}, {"Bob", "R1"}):
            raise KeyError(f"unsupported link: {a}, {b}")
        station = a if a != "R1" else b
        t_centered_s = float(int(t_ps)) / PS_PER_S - self.scenario.half_window_s
        return self._contact(station, t_centered_s, t_ps)


# Backward-compatible alias used by some older figure scripts.
SyntheticOverpassGeometry = ParametricOverpassGeometry


__all__ = [
    "OverpassScenario",
    "ParametricOverpassGeometry",
    "SyntheticOverpassGeometry",
]
