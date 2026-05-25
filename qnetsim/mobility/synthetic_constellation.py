from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional

from sgp4.api import WGS72, Satrec, jday
from sgp4.exporter import export_tle

from mobility.tle_loader import SatelliteEphemeris


EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.8


def _epoch_days_since_1949(epoch_utc: dt.datetime) -> float:
    jd_whole, jd_frac = jday(
        epoch_utc.year,
        epoch_utc.month,
        epoch_utc.day,
        epoch_utc.hour,
        epoch_utc.minute,
        epoch_utc.second + epoch_utc.microsecond / 1e6,
    )
    return (jd_whole - 2433281.5) + jd_frac


def _walker_mean_motion_rad_per_min(altitude_km: float) -> float:
    semi_major_axis_km = EARTH_RADIUS_KM + float(altitude_km)
    mean_motion_rad_per_s = math.sqrt(EARTH_MU_KM3_S2 / (semi_major_axis_km**3))
    return mean_motion_rad_per_s * 60.0


def generate_walker_delta_constellation(
    *,
    planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    phase: int = 1,
    epoch_unix_s: Optional[float] = None,
    eccentricity: float = 1e-4,
) -> List[SatelliteEphemeris]:
    if int(planes) <= 0:
        raise ValueError("planes must be > 0")
    if int(sats_per_plane) <= 0:
        raise ValueError("sats_per_plane must be > 0")

    total_sats = int(planes) * int(sats_per_plane)
    epoch_utc = dt.datetime.fromtimestamp(
        float(epoch_unix_s) if epoch_unix_s is not None else 1767225600.0,
        tz=dt.timezone.utc,
    )
    epoch_days = _epoch_days_since_1949(epoch_utc)
    inclination_rad = math.radians(float(inclination_deg))
    mean_motion_rad_per_min = _walker_mean_motion_rad_per_min(float(altitude_km))
    phase_term_deg = 360.0 * int(phase) / float(total_sats)

    satellites: List[SatelliteEphemeris] = []
    for plane_idx in range(int(planes)):
        raan_deg = 360.0 * plane_idx / float(planes)
        for slot_idx in range(int(sats_per_plane)):
            sat_idx = plane_idx * int(sats_per_plane) + slot_idx + 1
            mean_anomaly_deg = (
                360.0 * slot_idx / float(sats_per_plane)
                + plane_idx * phase_term_deg
            ) % 360.0

            satrec = Satrec()
            satrec.sgp4init(
                WGS72,
                "i",
                sat_idx,
                epoch_days,
                0.0,
                0.0,
                0.0,
                float(eccentricity),
                0.0,
                inclination_rad,
                math.radians(mean_anomaly_deg),
                mean_motion_rad_per_min,
                math.radians(raan_deg),
            )
            satrec.classification = "U"
            satrec.intldesg = "26001A"
            satrec.elnum = 1
            satrec.revnum = 1
            line1, line2 = export_tle(satrec)
            satellites.append(
                SatelliteEphemeris(
                    name=f"WALKER-{sat_idx:03d}",
                    tle_line1=line1,
                    tle_line2=line2,
                    source="SYNTHETIC_WALKER_TLE",
                    metadata={
                        "walker_planes": int(planes),
                        "walker_sats_per_plane": int(sats_per_plane),
                        "walker_altitude_km": float(altitude_km),
                        "walker_inclination_deg": float(inclination_deg),
                        "walker_phase": int(phase),
                        "plane_index": int(plane_idx),
                        "slot_index": int(slot_idx),
                        "epoch_unix_s": float(epoch_utc.timestamp()),
                    },
                )
            )
    return satellites


__all__ = ["generate_walker_delta_constellation"]
