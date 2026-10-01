from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
import zipfile

from channels.polarization import PolarizationParams

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "studies" / "quasar_matera_helmos" / "inputs.json"
DETECTOR_TECHNOLOGIES = {810: "Si-APD", 1550: "SNSPD"}


def default_inputs():
    return json.loads(INPUTS.read_text())


def values_from_inputs(provenance):
    return {row["cell"]: row["value"] for row in provenance["rows"]}


def read_workbook(data: bytes):
    try:
        return _read_workbook(data)
    except (zipfile.BadZipFile, ET.ParseError, KeyError, IndexError, StopIteration, OverflowError) as exc:
        raise ValueError("Invalid workbook. Use the supplied QUASAR Sheet1 template with explicit values.") from exc


def _read_workbook(data: bytes):
    """Read this fixed reviewer template; never execute formulas or macros.

    Only standard ZIP/XML is needed by the GUI. The original template's labels
    are validated before values are accepted, rather than guessing moved cells.
    """
    if len(data) > 5_000_000:
        raise ValueError("Workbook exceeds the 5 MB limit.")
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(x.file_size for x in z.infolist()) > 20_000_000:
            raise ValueError("Expanded workbook exceeds the 20 MB limit.")
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        if workbook.find("s:workbookPr", ns) is not None and workbook.find("s:workbookPr", ns).get("date1904") in {"1", "true"}:
            raise ValueError("Use the supplied template's 1900 date system.")
        sheet = next((x for x in workbook.findall("s:sheets/s:sheet", ns) if x.get("name") == "Sheet1"), None)
        if sheet is None:
            raise ValueError("Expected the supplied template's Sheet1.")
        rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(x.get("Target") for x in rels if x.get("Id") == rel_id)
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        cells = {}
        for cell in ET.fromstring(z.read(path)).findall(".//s:sheetData/s:row/s:c", ns):
            address = cell.get("r")
            if cell.find("s:f", ns) is not None:
                raise ValueError(f"Formula in {address}: provide explicit parameter values.")
            raw = cell.findtext("s:v", default="", namespaces=ns)
            kind = cell.get("t")
            if kind == "s":
                value = strings[int(raw)]
            elif kind == "inlineStr":
                value = "".join(cell.find("s:is", ns).itertext())
            elif kind in {"str", "d"}:
                value = raw
            else:
                value = float(raw) if raw else None
            cells[address] = value
    original = default_inputs()
    rows = []
    for expected in original["rows"]:
        number = expected["cell"][1:]
        if str(cells.get("B" + number, "")).strip() != expected["parameter"].strip():
            raise ValueError(f"Template label mismatch at B{number}.")
        value = cells.get(expected["cell"])
        if value is None or value == "":
            raise ValueError(f"Missing value at {expected['cell']}.")
        if expected["cell"] == "C33" and isinstance(value, (int, float)):
            value = (dt.datetime(1899, 12, 30) + dt.timedelta(days=value)).isoformat()
        rows.append({**expected, "value": value, "notes": cells.get("D" + number)})
    return {"source_name": "QUASAR simulation parameters.xlsx", "sha256": hashlib.sha256(data).hexdigest(), "rows": rows}


@dataclass(frozen=True)
class Optics:
    eta_tx: float = .8
    eta_rx: float = .5
    eta_detector: float = .6
    zenith_transmission: float = .9
    m2: float = 1.
    extra_loss_db_per_arm: float = 0.
    sky_matera: float = 1e-6
    sky_helmos: float = 1e-6
    bandwidth_nm: float = 1.
    fov_half_angle_urad: float = 10.


@dataclass(frozen=True)
class DetectorSettings:
    """Receiver assumptions, shared by both OGSs at one wavelength.

    Technology is specified by the user; numerical defaults are retained
    placeholders, not manufacturer specifications. The gate is a receiver
    processing choice, and jitter is single-detector Gaussian RMS, not FWHM.
    Detection efficiency remains in the wavelength's Optics configuration.
    """
    technology: str
    model: str = ""
    gate_s: float = 1e-9
    jitter_s: float = 50e-12
    dark_per_detector_s: float = 100.
    detectors: int = 4
    deadtime_s: float = 50e-9
    preset_id: str = ""


@dataclass(frozen=True)
class RunConfig:
    mode: str = "batched"
    wavelengths_nm: Tuple[int, ...] = (810, 1550)
    seed: int = 20260910
    auto_pass: bool = True
    start_s: float = 780.
    duration_s: float = 1200.
    bin_s: float = 1.
    frame: str = "J2000"
    include_j2: bool = False
    gate_s: float = 1e-9
    jitter_s: float = 50e-12
    dark_per_detector_s: float = 100.
    detectors: int = 4
    deadtime_s: float = 50e-9
    max_photon_events: int = 200000
    optics: Dict[int, Optics] = field(default_factory=lambda: {810: Optics(), 1550: Optics()})
    polarization: Dict[int, PolarizationParams] = field(default_factory=lambda: {810: PolarizationParams(), 1550: PolarizationParams()})
    detector_settings: Dict[int, DetectorSettings] = field(default_factory=dict)

    def detector_for(self, wavelength):
        """Explicit wavelength settings override the legacy shared defaults."""
        if wavelength in self.detector_settings:
            return self.detector_settings[wavelength]
        return DetectorSettings(technology=DETECTOR_TECHNOLOGIES[wavelength], gate_s=self.gate_s,
                                jitter_s=self.jitter_s, dark_per_detector_s=self.dark_per_detector_s,
                                detectors=self.detectors, deadtime_s=self.deadtime_s)

    def detector_audit(self, wavelength):
        from quasar.detector_presets import audit_detector
        return dict(wavelength_nm=wavelength,
                    **audit_detector(self.detector_for(wavelength), self.optics[wavelength].eta_detector))

    def as_dict(self):
        return asdict(self)

    def validate(self, v):
        def finite(name, x, lower, upper):
            if not math.isfinite(float(x)) or not lower <= float(x) <= upper:
                raise ValueError(f"{name} must be between {lower:g} and {upper:g}.")
        if self.mode not in {"batched", "photons"} or self.frame not in {"J2000", "MOD"}:
            raise ValueError("Unsupported DES mode or orbital frame.")
        if not self.wavelengths_nm or len(set(self.wavelengths_nm)) != len(self.wavelengths_nm) or any(w not in {810, 1550} for w in self.wavelengths_nm):
            raise ValueError("Select 810 nm, 1550 nm, or both once each.")
        finite("source MHz", v["C21"], 0, 10000)
        finite("source fidelity", v["C22"], .25, 1)
        finite("minimum elevation", v["C14"], 1, 89)
        finite("semi-major axis (km)", v["C26"], 6578, 42164)
        finite("eccentricity", v["C29"], 0, 1e-8)
        finite("inclination", v["C30"], 0, 180)
        for key in ["C27", "C31", "C32"]:
            finite(key, v[key], 0, 360)
        for row in [3, 9]:
            finite("latitude", v[f"C{row}"], -90, 90)
            finite("longitude", v[f"C{row+1}"], -180, 180)
            finite("station height", v[f"C{row+2}"], 0, 10000)
            finite("receiver diameter", v[f"C{row+3}"], .01, 20)
            finite("obscuration area", v[f"C{row+4}"], 0, .99)
        for key in ["C17", "C19"]:
            finite(key, v[key], .01, 5)
        for key in ["C18", "C20"]:
            finite(key, v[key], 0, 1000)
        finite("Cn2", float(str(v["C36"]).replace(" ", "")), 0, 1e-10)
        finite("wind", v["C37"], 0, 200)
        epoch = dt.datetime.fromisoformat(str(v["C33"]).replace("Z", "+00:00"))
        if epoch.tzinfo is not None and epoch.utcoffset() != dt.timedelta(0):
            raise ValueError("Enter the epoch in UTC, not a nonzero timezone offset.")
        finite("start", self.start_s, 0, 86400)
        finite("duration", self.duration_s, 1e-8, 86400)
        finite("bin width", self.bin_s, 1e-8, 60)
        finite("coincidence window", self.gate_s, 1e-12, 1e-5)
        finite("timing jitter", self.jitter_s, 0, 1e-5)
        finite("dark counts", self.dark_per_detector_s, 0, 1e8)
        finite("dead time", self.deadtime_s, 0, .01)
        if not 1 <= self.detectors <= 64 or int(self.detectors) != self.detectors:
            raise ValueError("Detector count must be an integer between 1 and 64.")
        if not 0 <= self.seed <= 2**32-1 or int(self.seed) != self.seed:
            raise ValueError("Seed must be a nonnegative 32-bit integer.")
        if not 1 <= self.max_photon_events <= 500000 or int(self.max_photon_events) != self.max_photon_events:
            raise ValueError("Photon event limit must be 1 to 500,000.")
        if self.auto_pass and self.mode == "photons":
            raise ValueError("Timestamp-resolved DES requires an explicit short interval.")
        if math.ceil(self.duration_s/self.bin_s) > 10000:
            raise ValueError("Use at most 10,000 reporting bins.")
        if any(w not in DETECTOR_TECHNOLOGIES for w in self.detector_settings):
            raise ValueError("Detector settings support only 810 and 1550 nm.")
        for wavelength in self.wavelengths_nm:
            detector = self.detector_for(wavelength)
            if not isinstance(detector, DetectorSettings) or detector.technology != DETECTOR_TECHNOLOGIES[wavelength]:
                raise ValueError(f"Use {DETECTOR_TECHNOLOGIES[wavelength]} detector settings for {wavelength} nm.")
            if not isinstance(detector.model, str) or len(detector.model) > 200:
                raise ValueError("Detector model must be a name of at most 200 characters.")
            from quasar.detector_presets import PRESETS
            if detector.preset_id and (detector.preset_id not in PRESETS or PRESETS[detector.preset_id]["wavelength_nm"] != wavelength):
                raise ValueError("Detector preset does not match this wavelength.")
            finite(f"{wavelength} nm coincidence window", detector.gate_s, 1e-12, 1e-5)
            finite(f"{wavelength} nm timing jitter RMS", detector.jitter_s, 0, 1e-5)
            finite(f"{wavelength} nm dark counts", detector.dark_per_detector_s, 0, 1e8)
            finite(f"{wavelength} nm dead time", detector.deadtime_s, 0, .01)
            finite(f"{wavelength} nm detectors per OGS", detector.detectors, 1, 64)
            if int(detector.detectors) != detector.detectors:
                raise ValueError("Detector count must be an integer.")
            if self.mode == "batched" and self.bin_s < 100*max(detector.gate_s, detector.deadtime_s, detector.jitter_s):
                raise ValueError(f"Batched DES requires bins much wider than detector timing at {wavelength} nm; use timestamp-resolved DES.")
            if wavelength not in self.polarization:
                raise ValueError(f"Missing polarization configuration for {wavelength} nm.")
            pol = self.polarization[wavelength]
            pol.validate(self.start_s+self.duration_s+1)
            if pol.enabled and self.mode == "batched":
                resolution = min(pol.wander_grid_s, pol.update_interval_s) if pol.compensate else pol.wander_grid_s
                if self.bin_s > resolution:
                    raise ValueError("Reduce the DES batch interval to resolve the polarization wander grid and correction interval.")
            if wavelength not in self.optics:
                raise ValueError(f"Missing optical assumptions for {wavelength} nm.")
            o = self.optics[wavelength]
            for key in ["eta_tx", "eta_rx", "eta_detector", "zenith_transmission"]:
                finite(key, getattr(o, key), .000001, 1)
            finite("M2", o.m2, 1, 20)
            finite("extra per-arm loss", o.extra_loss_db_per_arm, 0, 100)
            finite("Matera radiance", o.sky_matera, 0, 1)
            finite("Helmos radiance", o.sky_helmos, 0, 1)
            finite("filter bandwidth", o.bandwidth_nm, .001, 100)
            finite("FOV half-angle", o.fov_half_angle_urad, .01, 10000)
