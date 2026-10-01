"""Dated, illustrative circular overpasses on a rotating spherical Earth.

This is a two-body design study, not a fitted TLE or a mission ephemeris.
The inertial axes coincide with ECEF at the specified midpoint timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

import numpy as np

from mobility.geometry import ContactGeometry
from mobility.propagator import EARTH_RADIUS_M, PS_PER_S

MU_EARTH = 3.986004418e14
EARTH_ROTATION_RAD_S = 7.2921150e-5


def unit(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vec / norm


@dataclass(frozen=True)
class MapStation:
    name: str
    latitude_deg: float
    longitude_deg: float

    def unit_vector(self):
        lat, lon = np.radians([self.latitude_deg, self.longitude_deg])
        return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


class DatedCircularPass:
    def __init__(self, a, b, *, midpoint_utc, altitude_km=360.,
                 crossing_angle_deg=90., min_elevation_deg=10., half_window_s=300.):
        if midpoint_utc.tzinfo is None or midpoint_utc.utcoffset() is None:
            raise ValueError("midpoint_utc requires an explicit timezone")
        if altitude_km <= 0 or half_window_s <= 0:
            raise ValueError("altitude and half-window must be positive")
        if not 0 <= min_elevation_deg < 90:
            raise ValueError("elevation mask must be in [0, 90)")
        self.stations = {"Alice": a, "Bob": b}
        self.midpoint_utc = midpoint_utc
        self.altitude_km = float(altitude_km)
        self.crossing_angle_deg = float(crossing_angle_deg)
        self.min_elevation_deg = float(min_elevation_deg)
        self.half_window_s = float(half_window_s)
        self.radius_m = EARTH_RADIUS_M + altitude_km*1000.
        self.omega = math.sqrt(MU_EARTH / self.radius_m**3)
        self.period_s = 2*math.pi / self.omega
        av, bv = a.unit_vector(), b.unit_vector()
        self.origin = unit(av + bv)
        self.baseline_tangent = unit(bv - av)
        normal = unit(np.cross(self.origin, self.baseline_tangent))
        phi = math.radians(crossing_angle_deg)
        self.ground_direction = math.cos(phi)*self.baseline_tangent + math.sin(phi)*normal
        rotation = EARTH_ROTATION_RAD_S*np.cross([0., 0., 1.], self.origin)
        along_rotation = np.dot(self.ground_direction, rotation)
        # Correct the inertial velocity so the requested angle is an ECEF ground-track angle.
        discriminant = along_rotation**2 + self.omega**2 - np.dot(rotation, rotation)
        if discriminant <= 0:
            raise ValueError("requested ground-track direction unavailable at this altitude")
        speed = -along_rotation + math.sqrt(discriminant)
        self.tangent = unit(speed*self.ground_direction + rotation)
        self.inclination_deg = math.degrees(math.acos(float(np.clip(np.cross(self.origin, self.tangent)[2], -1., 1.))))

    def utc_at(self, centered_s):
        return self.midpoint_utc + timedelta(seconds=float(centered_s))

    def satellite_ecef(self, centered_s):
        t = float(centered_s)
        inertial = self.radius_m*(math.cos(self.omega*t)*self.origin + math.sin(self.omega*t)*self.tangent)
        angle = -EARTH_ROTATION_RAD_S*t
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c*inertial[0]-s*inertial[1], s*inertial[0]+c*inertial[1], inertial[2]])

    def subpoint(self, centered_s):
        x, y, z = self.satellite_ecef(centered_s)
        return math.degrees(math.atan2(z, math.hypot(x,y))), math.degrees(math.atan2(y,x))

    def contact_at(self, station, centered_s):
        gs = self.stations[station].unit_vector()
        los = self.satellite_ecef(centered_s) - EARTH_RADIUS_M*gs
        distance = float(np.linalg.norm(los))
        elevation = math.degrees(math.asin(float(np.clip(np.dot(los, gs)/distance, -1., 1.))))
        return ContactGeometry(t_ps=round((centered_s+self.half_window_s)*PS_PER_S),
            t_s=float(centered_s), distance_m=distance, elevation_deg=elevation,
            visible=elevation >= self.min_elevation_deg)

    def contact_geometry(self, a, b, t_ps):
        if {a,b} not in ({"Alice","R1"}, {"Bob","R1"}):
            raise KeyError((a,b))
        station = a if a != "R1" else b
        return self.contact_at(station, int(t_ps)/PS_PER_S-self.half_window_s)
