from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

from mobility.tle_loader import SatelliteEphemeris


# ============================================================
# Shared geometry / time primitives
# ============================================================
EARTH_RADIUS_M = 6_378_137.0
PS_PER_S = 1_000_000_000_000


@dataclass(frozen=True)
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    min_elevation_deg: float = 10.0


@dataclass(frozen=True)
class ECEFPosition:
    x_m: float
    y_m: float
    z_m: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)


@dataclass(frozen=True)
class GeodeticPoint:
    lat_deg: float
    lon_deg: float
    alt_m: float


@dataclass(frozen=True)
class TimeSample:
    t_ps: int
    unix_s: float
    utc_iso: str


# ============================================================
# Skyfield-backed propagator
# ============================================================
class SkyfieldPropagator:
    """
    Thin propagation layer around Skyfield + SGP4.

    Responsibilities:
    - Hold normalized TLE-backed EarthSatellite objects.
    - Convert simulation time in picoseconds to Skyfield time.
    - Provide satellite and ground positions in ECEF/ITRS.
    - Provide geodetic subpoints and simple vector helpers.

    Notes:
    - This module intentionally does NOT compute visibility/contact policy.
      That belongs in geometry.py.
    - The import path for SatelliteEphemeris assumes the generated
      tle_loader.py module is available on the Python path.
    """

    def __init__(
        self,
        *,
        satellites: Iterable[SatelliteEphemeris],
        ground_nodes: Optional[Iterable[GroundStation]] = None,
        epoch_unix_s: Optional[float] = None,
    ):
        self.ts = load.timescale()
        self.epoch_unix_s = float(
            epoch_unix_s if epoch_unix_s is not None else time.time()
        )

        self.satellites: Dict[str, EarthSatellite] = {}
        for sat in satellites:
            self.satellites[sat.name] = EarthSatellite(
                sat.tle_line1,
                sat.tle_line2,
                sat.name,
                self.ts,
            )

        self.ground_nodes: Dict[str, GroundStation] = {}
        self.ground_topos = {}
        for gs in ground_nodes or []:
            self.ground_nodes[gs.name] = gs
            self.ground_topos[gs.name] = wgs84.latlon(
                latitude_degrees=gs.lat_deg,
                longitude_degrees=gs.lon_deg,
                elevation_m=gs.alt_m,
            )

    # ------------------------------------------------------------
    # Time conversion
    # ------------------------------------------------------------
    def time_sample(self, t_ps: int) -> TimeSample:
        unix_s = self.epoch_unix_s + float(int(t_ps)) / PS_PER_S
        utc_dt = dt.datetime.fromtimestamp(unix_s, tz=dt.timezone.utc)
        return TimeSample(
            t_ps=int(t_ps),
            unix_s=unix_s,
            utc_iso=utc_dt.isoformat(),
        )

    def skyfield_time(self, t_ps: int):
        sample = self.time_sample(t_ps)
        utc_dt = dt.datetime.fromtimestamp(sample.unix_s, tz=dt.timezone.utc)
        return self.ts.from_datetime(utc_dt)

    # ------------------------------------------------------------
    # Node lookup helpers
    # ------------------------------------------------------------
    def has_satellite(self, name: str) -> bool:
        return name in self.satellites

    def has_ground_node(self, name: str) -> bool:
        return name in self.ground_nodes

    def require_satellite(self, name: str) -> EarthSatellite:
        if name not in self.satellites:
            raise KeyError(f"unknown satellite: {name}")
        return self.satellites[name]

    def require_ground(self, name: str) -> GroundStation:
        if name not in self.ground_nodes:
            raise KeyError(f"unknown ground node: {name}")
        return self.ground_nodes[name]

    # ------------------------------------------------------------
    # ECEF/ITRS positions
    # ------------------------------------------------------------
    def satellite_ecef(self, sat_name: str, t_ps: int) -> ECEFPosition:
        sat = self.require_satellite(sat_name)
        t = self.skyfield_time(t_ps)
        xyz_km = sat.at(t).frame_xyz(itrs).km
        return ECEFPosition(
            x_m=float(xyz_km[0]) * 1e3,
            y_m=float(xyz_km[1]) * 1e3,
            z_m=float(xyz_km[2]) * 1e3,
        )

    def ground_ecef(self, gs_name: str, t_ps: int) -> ECEFPosition:
        topo = self.ground_topos[gs_name]
        t = self.skyfield_time(t_ps)
        xyz_km = topo.at(t).frame_xyz(itrs).km
        return ECEFPosition(
            x_m=float(xyz_km[0]) * 1e3,
            y_m=float(xyz_km[1]) * 1e3,
            z_m=float(xyz_km[2]) * 1e3,
        )

    def node_ecef(self, node_name: str, t_ps: int) -> ECEFPosition:
        if self.has_ground_node(node_name):
            return self.ground_ecef(node_name, t_ps)
        if self.has_satellite(node_name):
            return self.satellite_ecef(node_name, t_ps)
        raise KeyError(f"unknown node: {node_name}")

    # ------------------------------------------------------------
    # Geodetic helpers
    # ------------------------------------------------------------
    def satellite_subpoint(self, sat_name: str, t_ps: int) -> GeodeticPoint:
        sat = self.require_satellite(sat_name)
        t = self.skyfield_time(t_ps)
        sp = wgs84.subpoint(sat.at(t))
        return GeodeticPoint(
            lat_deg=float(sp.latitude.degrees),
            lon_deg=float(sp.longitude.degrees),
            alt_m=float(sp.elevation.m),
        )

    def ground_geodetic(self, gs_name: str) -> GeodeticPoint:
        gs = self.require_ground(gs_name)
        return GeodeticPoint(
            lat_deg=float(gs.lat_deg),
            lon_deg=float(gs.lon_deg),
            alt_m=float(gs.alt_m),
        )

    # ------------------------------------------------------------
    # Observer-relative quantities
    # ------------------------------------------------------------
    def elevation_deg(self, sat_name: str, gs_name: str, t_ps: int) -> float:
        sat = self.require_satellite(sat_name)
        topo = self.ground_topos[gs_name]
        t = self.skyfield_time(t_ps)
        difference = sat - topo
        alt, _az, _distance = difference.at(t).altaz()
        return float(alt.degrees)

    def range_m(self, a: str, b: str, t_ps: int) -> float:
        pa = self.node_ecef(a, t_ps)
        pb = self.node_ecef(b, t_ps)
        return euclidean_distance(pa.as_tuple(), pb.as_tuple())

    def relative_vector_ecef(
        self, a: str, b: str, t_ps: int
    ) -> Tuple[float, float, float]:
        pa = self.node_ecef(a, t_ps).as_tuple()
        pb = self.node_ecef(b, t_ps).as_tuple()
        return vector_sub(pb, pa)

    # ------------------------------------------------------------
    # Batch sampling helpers
    # ------------------------------------------------------------
    def sample_satellite_track(
        self,
        sat_name: str,
        *,
        start_ps: int,
        stop_ps: int,
        step_ps: int,
    ) -> List[Tuple[TimeSample, GeodeticPoint]]:
        if step_ps <= 0:
            raise ValueError("step_ps must be > 0")
        out: List[Tuple[TimeSample, GeodeticPoint]] = []
        t_ps = int(start_ps)
        while t_ps <= int(stop_ps):
            out.append(
                (self.time_sample(t_ps), self.satellite_subpoint(sat_name, t_ps))
            )
            t_ps += int(step_ps)
        return out

    def sample_ranges(
        self,
        a: str,
        b: str,
        *,
        start_ps: int,
        stop_ps: int,
        step_ps: int,
    ) -> List[Tuple[TimeSample, float]]:
        if step_ps <= 0:
            raise ValueError("step_ps must be > 0")
        out: List[Tuple[TimeSample, float]] = []
        t_ps = int(start_ps)
        while t_ps <= int(stop_ps):
            out.append((self.time_sample(t_ps), self.range_m(a, b, t_ps)))
            t_ps += int(step_ps)
        return out


# ============================================================
# Vector helpers
# ============================================================
def vector_sub(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(vector_dot(v, v))


def euclidean_distance(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> float:
    return vector_norm(vector_sub(a, b))


# ============================================================
# Demo defaults
# ============================================================
def default_ground_stations() -> List[GroundStation]:
    return [
        GroundStation(
            "Alice",
            lat_deg=49.6116,
            lon_deg=6.1319,
            alt_m=300.0,
            min_elevation_deg=10.0,
        ),
        GroundStation(
            "Bob", lat_deg=40.7128, lon_deg=-74.0060, alt_m=10.0, min_elevation_deg=10.0
        ),
    ]


__all__ = [
    "EARTH_RADIUS_M",
    "ECEFPosition",
    "GeodeticPoint",
    "GroundStation",
    "PS_PER_S",
    "SkyfieldPropagator",
    "TimeSample",
    "default_ground_stations",
    "euclidean_distance",
    "vector_dot",
    "vector_norm",
    "vector_sub",
]
