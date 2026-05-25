from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence

from sgp4.api import WGS72, Satrec, jday
from sgp4.exporter import export_tle

from mobility.tle_loader import SatelliteEphemeris

from .config import SearchRanges


EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.8


@dataclass(frozen=True)
class ConstellationProposal:
    planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    raan_spacing_deg: float
    phasing_factor: int
    max_isl_range_km: float
    min_elevation_deg: float
    memory_key: str
    bsm_key: str
    fso_key: str

    @property
    def total_satellites(self) -> int:
        return int(self.planes) * int(self.sats_per_plane)

    def to_dict(self) -> dict:
        return asdict(self)


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


def sample_random_proposal(
    *,
    rng: random.Random,
    search: SearchRanges,
    memory_keys: Sequence[str],
    bsm_keys: Sequence[str],
    fso_keys: Sequence[str],
) -> ConstellationProposal:
    planes = rng.randint(int(search.planes_min), int(search.planes_max))
    sats_per_plane = rng.randint(
        int(search.sats_per_plane_min), int(search.sats_per_plane_max)
    )
    phasing_max = max(0, planes - 1)
    return ConstellationProposal(
        planes=planes,
        sats_per_plane=sats_per_plane,
        altitude_km=rng.uniform(search.altitude_km_min, search.altitude_km_max),
        inclination_deg=rng.uniform(
            search.inclination_deg_min, search.inclination_deg_max
        ),
        raan_spacing_deg=rng.uniform(
            search.raan_spacing_deg_min, search.raan_spacing_deg_max
        ),
        phasing_factor=rng.randint(0, phasing_max),
        max_isl_range_km=rng.uniform(
            search.max_isl_range_km_min, search.max_isl_range_km_max
        ),
        min_elevation_deg=rng.uniform(
            search.min_elevation_deg_min, search.min_elevation_deg_max
        ),
        memory_key=str(rng.choice(list(memory_keys))),
        bsm_key=str(rng.choice(list(bsm_keys))),
        fso_key=str(rng.choice(list(fso_keys))),
    )


def crossover_proposals(
    a: ConstellationProposal,
    b: ConstellationProposal,
    *,
    rng: random.Random,
) -> ConstellationProposal:
    return ConstellationProposal(
        planes=rng.choice([a.planes, b.planes]),
        sats_per_plane=rng.choice([a.sats_per_plane, b.sats_per_plane]),
        altitude_km=rng.choice([a.altitude_km, b.altitude_km]),
        inclination_deg=rng.choice([a.inclination_deg, b.inclination_deg]),
        raan_spacing_deg=rng.choice([a.raan_spacing_deg, b.raan_spacing_deg]),
        phasing_factor=rng.choice([a.phasing_factor, b.phasing_factor]),
        max_isl_range_km=rng.choice([a.max_isl_range_km, b.max_isl_range_km]),
        min_elevation_deg=rng.choice([a.min_elevation_deg, b.min_elevation_deg]),
        memory_key=rng.choice([a.memory_key, b.memory_key]),
        bsm_key=rng.choice([a.bsm_key, b.bsm_key]),
        fso_key=rng.choice([a.fso_key, b.fso_key]),
    )


def mutate_proposal(
    proposal: ConstellationProposal,
    *,
    rng: random.Random,
    search: SearchRanges,
    memory_keys: Sequence[str],
    bsm_keys: Sequence[str],
    fso_keys: Sequence[str],
    mutation_rate: float,
) -> ConstellationProposal:
    data = proposal.to_dict()
    if rng.random() < mutation_rate:
        data["planes"] = rng.randint(int(search.planes_min), int(search.planes_max))
    if rng.random() < mutation_rate:
        data["sats_per_plane"] = rng.randint(
            int(search.sats_per_plane_min), int(search.sats_per_plane_max)
        )
    if rng.random() < mutation_rate:
        data["altitude_km"] = rng.uniform(
            search.altitude_km_min, search.altitude_km_max
        )
    if rng.random() < mutation_rate:
        data["inclination_deg"] = rng.uniform(
            search.inclination_deg_min, search.inclination_deg_max
        )
    if rng.random() < mutation_rate:
        data["raan_spacing_deg"] = rng.uniform(
            search.raan_spacing_deg_min, search.raan_spacing_deg_max
        )
    if rng.random() < mutation_rate:
        data["phasing_factor"] = rng.randint(0, max(0, int(data["planes"]) - 1))
    if rng.random() < mutation_rate:
        data["max_isl_range_km"] = rng.uniform(
            search.max_isl_range_km_min, search.max_isl_range_km_max
        )
    if rng.random() < mutation_rate:
        data["min_elevation_deg"] = rng.uniform(
            search.min_elevation_deg_min, search.min_elevation_deg_max
        )
    if rng.random() < mutation_rate and memory_keys:
        data["memory_key"] = str(rng.choice(list(memory_keys)))
    if rng.random() < mutation_rate and bsm_keys:
        data["bsm_key"] = str(rng.choice(list(bsm_keys)))
    if rng.random() < mutation_rate and fso_keys:
        data["fso_key"] = str(rng.choice(list(fso_keys)))
    return ConstellationProposal(**data)


def build_constellation_ephemerides(
    proposal: ConstellationProposal,
    *,
    epoch_unix_s: float,
    eccentricity: float = 1e-4,
) -> List[SatelliteEphemeris]:
    total_sats = int(proposal.planes) * int(proposal.sats_per_plane)
    epoch_utc = dt.datetime.fromtimestamp(float(epoch_unix_s), tz=dt.timezone.utc)
    epoch_days = _epoch_days_since_1949(epoch_utc)
    inclination_rad = math.radians(float(proposal.inclination_deg))
    mean_motion_rad_per_min = _walker_mean_motion_rad_per_min(
        float(proposal.altitude_km)
    )
    phase_term_deg = 360.0 * int(proposal.phasing_factor) / float(max(1, total_sats))

    satellites: List[SatelliteEphemeris] = []
    sat_idx = 1
    for plane_idx in range(int(proposal.planes)):
        raan_deg = (plane_idx * float(proposal.raan_spacing_deg)) % 360.0
        for slot_idx in range(int(proposal.sats_per_plane)):
            mean_anomaly_deg = (
                360.0 * slot_idx / float(proposal.sats_per_plane)
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
                    name=f"OPT-{sat_idx:03d}",
                    tle_line1=line1,
                    tle_line2=line2,
                    source="STBD_OPTIMIZED_WALKER",
                    metadata={
                        "planes": int(proposal.planes),
                        "sats_per_plane": int(proposal.sats_per_plane),
                        "altitude_km": float(proposal.altitude_km),
                        "inclination_deg": float(proposal.inclination_deg),
                        "raan_spacing_deg": float(proposal.raan_spacing_deg),
                        "phasing_factor": int(proposal.phasing_factor),
                        "max_isl_range_km": float(proposal.max_isl_range_km),
                        "min_elevation_deg": float(proposal.min_elevation_deg),
                        "plane_index": int(plane_idx),
                        "slot_index": int(slot_idx),
                        "epoch_unix_s": float(epoch_unix_s),
                    },
                )
            )
            sat_idx += 1
    return satellites


def satellite_penalty(
    proposal: ConstellationProposal, *, search: SearchRanges
) -> float:
    return float(proposal.total_satellites) / float(max(1, search.max_satellites))
