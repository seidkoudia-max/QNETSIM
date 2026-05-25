from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ============================================================
# TLE / OMM loading helpers
# ============================================================
# Purpose:
# - Provide a clean input layer for satellite ephemerides.
# - Support either classic 3-line TLE records or JSON OMM-like records.
# - Normalize everything into a single internal representation that the
#   propagator layer can consume.
#
# Notes:
# - This module does not perform propagation.
# - It only parses, validates, and normalizes ephemeris inputs.
# - OMM support here is intentionally lightweight: it accepts a practical
#   JSON schema and converts it into a TLE-like carrier object. If you later
#   want full CCSDS OMM ingestion, this module is the right place to extend.


# ============================================================
# Normalized ephemeris model
# ============================================================
@dataclass(frozen=True)
class SatelliteEphemeris:
    name: str
    tle_line1: str
    tle_line2: str
    source: str = "TLE"
    metadata: Optional[Dict[str, Any]] = None


# ============================================================
# Validation helpers
# ============================================================
def _is_tle_line1(line: str) -> bool:
    s = line.strip()
    return len(s) >= 2 and s.startswith("1 ")



def _is_tle_line2(line: str) -> bool:
    s = line.strip()
    return len(s) >= 2 and s.startswith("2 ")



def validate_tle_pair(line1: str, line2: str) -> None:
    if not _is_tle_line1(line1):
        raise ValueError("invalid TLE line 1")
    if not _is_tle_line2(line2):
        raise ValueError("invalid TLE line 2")



def validate_ephemeris(ephem: SatelliteEphemeris) -> None:
    if not ephem.name.strip():
        raise ValueError("satellite name must be non-empty")
    validate_tle_pair(ephem.tle_line1, ephem.tle_line2)


def tle_epoch_datetime(ephem: SatelliteEphemeris) -> dt.datetime:
    line1 = ephem.tle_line1
    if len(line1) < 32:
        raise ValueError("TLE line 1 is too short to contain an epoch")
    token = line1[18:32].strip()
    if not token:
        raise ValueError("TLE line 1 is missing epoch information")
    year_2d = int(token[:2])
    year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
    day_of_year = float(token[2:])
    day_index = int(day_of_year)
    frac_day = day_of_year - day_index
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    return start + dt.timedelta(days=day_index - 1, seconds=frac_day * 86400.0)


def tle_epoch_unix_s(ephem: SatelliteEphemeris) -> float:
    return tle_epoch_datetime(ephem).timestamp()


def representative_epoch_unix_s(ephems: Sequence[SatelliteEphemeris]) -> float:
    if not ephems:
        raise ValueError("no ephemerides provided")
    epochs = [tle_epoch_unix_s(ephem) for ephem in ephems]
    epochs.sort()
    return float(epochs[len(epochs) // 2])


# ============================================================
# TLE parsing
# ============================================================
def parse_tle_triplets(lines: Sequence[str]) -> List[SatelliteEphemeris]:
    """
    Parse a sequence of lines in classic 3-line TLE form:

        NAME
        1 ...
        2 ...

    Blank lines are ignored.
    """
    cleaned = [line.rstrip("\n") for line in lines if line.strip()]
    out: List[SatelliteEphemeris] = []
    i = 0
    while i < len(cleaned):
        line0 = cleaned[i].strip()

        if (
            i + 2 < len(cleaned)
            and (not _is_tle_line1(line0))
            and _is_tle_line1(cleaned[i + 1].strip())
            and _is_tle_line2(cleaned[i + 2].strip())
        ):
            name = line0
            line1 = cleaned[i + 1].strip()
            line2 = cleaned[i + 2].strip()
            validate_tle_pair(line1, line2)
            out.append(
                SatelliteEphemeris(
                    name=name,
                    tle_line1=line1,
                    tle_line2=line2,
                    source="TLE",
                    metadata=None,
                )
            )
            i += 3
            continue

        if i + 1 < len(cleaned) and _is_tle_line1(line0) and _is_tle_line2(
            cleaned[i + 1].strip()
        ):
            line1 = line0
            line2 = cleaned[i + 1].strip()
            out.append(
                SatelliteEphemeris(
                    name=f"SAT-{len(out) + 1:05d}",
                    tle_line1=line1,
                    tle_line2=line2,
                    source="TLE",
                    metadata=None,
                )
            )
            i += 2
            continue

        i += 1

    if not out:
        raise ValueError("no valid TLE records found")
    return out



def load_tle_file(path: str | Path) -> List[SatelliteEphemeris]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    sats = parse_tle_triplets(text.splitlines())
    for sat in sats:
        validate_ephemeris(sat)
    return sats


# ============================================================
# Lightweight OMM JSON parsing
# ============================================================
# Expected practical schema, one object or a list of objects:
# {
#   "OBJECT_NAME": "R1",
#   "TLE_LINE1": "1 ...",
#   "TLE_LINE2": "2 ...",
#   ... optional metadata ...
# }
#
# or:
# {
#   "name": "R1",
#   "tle_line1": "1 ...",
#   "tle_line2": "2 ..."
# }


def _normalize_omm_record(record: Dict[str, Any]) -> SatelliteEphemeris:
    name = str(
        record.get("OBJECT_NAME")
        or record.get("object_name")
        or record.get("name")
        or ""
    ).strip()
    line1 = str(record.get("TLE_LINE1") or record.get("tle_line1") or "").strip()
    line2 = str(record.get("TLE_LINE2") or record.get("tle_line2") or "").strip()

    if not name:
        raise ValueError("OMM record missing OBJECT_NAME/name")
    validate_tle_pair(line1, line2)

    metadata = dict(record)
    return SatelliteEphemeris(
        name=name,
        tle_line1=line1,
        tle_line2=line2,
        source="OMM",
        metadata=metadata,
    )



def load_omm_json(path: str | Path) -> List[SatelliteEphemeris]:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("OMM JSON must be an object or list of objects")

    sats = [_normalize_omm_record(rec) for rec in records]
    for sat in sats:
        validate_ephemeris(sat)
    return sats


# ============================================================
# In-memory constructors
# ============================================================
def from_tle_strings(name: str, line1: str, line2: str) -> SatelliteEphemeris:
    ephem = SatelliteEphemeris(
        name=str(name).strip(),
        tle_line1=str(line1).strip(),
        tle_line2=str(line2).strip(),
        source="TLE",
        metadata=None,
    )
    validate_ephemeris(ephem)
    return ephem



def from_many_tles(records: Iterable[tuple[str, str, str]]) -> List[SatelliteEphemeris]:
    out: List[SatelliteEphemeris] = []
    for name, line1, line2 in records:
        out.append(from_tle_strings(name, line1, line2))
    return out


# ============================================================
# Demo defaults
# ============================================================
# These are example Starlink-era style TLEs for development/demo only.
# Replace them with fresher TLEs for serious simulation.
_DEMO_TLES: List[tuple[str, str, str]] = [
    (
        "R1",
        "1 44713U 19074A   26092.50000000  .00001234  00000+0  10270-3 0  9993",
        "2 44713  53.0000  10.0000 0001000  90.0000 270.0000 15.05500000234567",
    ),
    (
        "R2",
        "1 44714U 19074B   26092.50000000  .00001234  00000+0  10270-3 0  9991",
        "2 44714  53.0000 130.0000 0001000 120.0000 240.0000 15.05500000234568",
    ),
    (
        "R3",
        "1 44715U 19074C   26092.50000000  .00001234  00000+0  10270-3 0  9998",
        "2 44715  53.0000 250.0000 0001000 150.0000 210.0000 15.05500000234569",
    ),
]



def load_demo_leo_repeaters() -> List[SatelliteEphemeris]:
    sats = from_many_tles(_DEMO_TLES)
    for sat in sats:
        validate_ephemeris(sat)
    return sats


# ============================================================
# Generic loader
# ============================================================
def load_ephemerides(
    *,
    tle_path: str | Path | None = None,
    omm_json_path: str | Path | None = None,
    use_demo_if_missing: bool = False,
) -> List[SatelliteEphemeris]:
    supplied = [x is not None for x in (tle_path, omm_json_path)]
    if sum(supplied) > 1:
        raise ValueError("provide at most one of tle_path or omm_json_path")

    if tle_path is not None:
        return load_tle_file(tle_path)
    if omm_json_path is not None:
        return load_omm_json(omm_json_path)
    if use_demo_if_missing:
        return load_demo_leo_repeaters()
    raise ValueError("no ephemeris source provided")


__all__ = [
    "SatelliteEphemeris",
    "from_many_tles",
    "from_tle_strings",
    "load_demo_leo_repeaters",
    "load_ephemerides",
    "load_omm_json",
    "load_tle_file",
    "parse_tle_triplets",
    "representative_epoch_unix_s",
    "tle_epoch_datetime",
    "tle_epoch_unix_s",
    "validate_ephemeris",
    "validate_tle_pair",
]
