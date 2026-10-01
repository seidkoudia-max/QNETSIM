"""Versioned vendor-based sensitivity inputs, not calibrated receiver models."""
from dataclasses import asdict, replace
import math

from quasar.config import DetectorSettings, DETECTOR_TECHNOLOGIES

FWHM_TO_RMS = 1 / math.sqrt(8 * math.log(2))
THORLABS = "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5255&partnumber=SPDMH3"
THORLABS_MANUAL = "https://manuals.plus/m/61ed70d5d283173a4e6f0e6ad81e7e0bdbf3dc1619016d6611b09929dcc7b1c5.pdf"
SINGLE_QUANTUM = "https://www.singlequantum.com/wp-content/uploads/2026/03/SQ_combined_brochure_with_specs.pdf"
JITTER_DEFINITION = "https://www.singlequantum.com/technology/interleaved-detectors/"

PRESETS = {
    "thorlabs_spdmh2_v1": dict(
        wavelength_nm=810, label="Thorlabs SPDMH2 - datasheet sensitivity",
        model="Thorlabs SPDMH2 (free space)", efficiency=.60, jitter_fwhm_ps=1000.,
        dark_per_detector_s=100., deadtime_s=45e-9,
        basis=dict(eta_detector="Typical PDE at 810 nm for free-space SPDMH2; not the FC/PC variant.",
                   jitter_s="1000 ps typical timing resolution, treated as FWHM per manufacturer manual; Gaussian shape is an assumption. Not a calibrated 810 nm timing histogram.",
                   dark_per_detector_s="100 counts/s maximum specification, used as a sensitivity value.",
                   deadtime_s="45 ns typical dead time; approximated by a nonparalyzable step."),
        sources=[THORLABS, THORLABS_MANUAL],
        caveats=["The manufacturer's SPDMHx manual (MTN028160-D02, mirrored PDF) describes wavelength, spot-position and count-rate dependence of jitter. A unit production report is needed.",
                 "Thorlabs lists 0.2% typical afterpulsing; correlated afterpulses are NOT simulated.",
                 "The catalog maximum count rate is 20 MHz per detector; the model does not replace measured saturation curves."],
        jitter_rate_warning_s=1e6, catalog_max_count_s=20e6),
    "single_quantum_1550_v1": dict(
        wavelength_nm=1550, label="Single Quantum 1550 nm - specification-bound sensitivity",
        model="Single Quantum Eos portfolio, 1550 nm specification bounds", efficiency=.90,
        jitter_fwhm_ps=15., dark_per_detector_s=1., deadtime_s=50e-9,
        basis=dict(eta_detector="90% SDE, the stated >=90% bound at 1550 nm; not intrinsic nanowire efficiency.",
                   jitter_s="15 ps FWHM, the stated <=15 ps bound; converted to Gaussian RMS. Not a measured timing histogram.",
                   dark_per_detector_s="1 count/s, the stated <=1 cps bound at 1550 nm.",
                   deadtime_s="ASSUMED 50 ns effective nonparalyzable dead time, retained from baseline. Not a Single Quantum recovery specification."),
        sources=[SINGLE_QUANTUM, JITTER_DEFINITION],
        caveats=["Single Quantum explicitly states that not all specifications can be achieved simultaneously. This combined sensitivity case is not a guaranteed purchasable configuration.",
                 "The brochure's >=50 MHz maximum count rate is not inverted to invent a recovery time. The assumed 50 ns step model is not fitted to that count-rate specification.",
                 "Interleaved detector / ultra-high-count-rate specifications are not mixed into this preset. Efficiency recovery and latching are NOT simulated.",
                 "SDE includes losses up to the vendor's system reference plane. RX optics must include only additional upstream losses; actual telescope-to-detector coupling remains uncalibrated."],
        jitter_rate_warning_s=None, catalog_max_count_s=None),
}
DEFAULT_PRESET = {810: "thorlabs_spdmh2_v1", 1550: "single_quantum_1550_v1"}


def preset_settings(preset_id):
    spec = PRESETS[preset_id]
    return DetectorSettings(technology=DETECTOR_TECHNOLOGIES[spec["wavelength_nm"]],
                            model=spec["model"], preset_id=preset_id,
                            jitter_s=spec["jitter_fwhm_ps"]*1e-12*FWHM_TO_RMS,
                            dark_per_detector_s=spec["dark_per_detector_s"],
                            deadtime_s=spec["deadtime_s"])


def with_vendor_detectors(cfg):
    """Explicit opt-in, preserving optics other than detector PDE/SDE."""
    settings, optics = dict(cfg.detector_settings), dict(cfg.optics)
    for nm in cfg.wavelengths_nm:
        key = DEFAULT_PRESET[nm]
        settings[nm] = preset_settings(key)
        optics[nm] = replace(optics[nm], eta_detector=PRESETS[key]["efficiency"])
    return replace(cfg, detector_settings=settings, optics=optics)


def audit_detector(settings, efficiency):
    effective = {**asdict(settings), "eta_detector": efficiency}
    spec = PRESETS.get(settings.preset_id)
    audit = dict(effective, numerical_specifications_status="assumed_not_calibrated",
                 jitter_convention="Single-detector Gaussian RMS; independent between stations",
                 jitter_fwhm_equivalent_ps=settings.jitter_s/FWHM_TO_RMS/1e-12,
                 station_scope="Both OGSs; identical receiver parameters at this wavelength",
                 response_model="Nonparalyzable step dead time; no afterpulsing, latching or measured recovery curve")
    if spec is None:
        audit.update(sources=[], caveats=["Custom/legacy numeric inputs, not vendor specifications."],
                     parameter_basis={k:"User/legacy assumption" for k in effective if k not in {"technology", "model", "preset_id"}})
        return audit
    reference = {**asdict(preset_settings(settings.preset_id)), "eta_detector": spec["efficiency"]}
    def matches(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0) if isinstance(a, (int, float)) else a == b
    overrides = {k:dict(reference=reference[k], effective=v) for k,v in effective.items() if not matches(v, reference[k])}
    basis = {**spec["basis"], "gate_s":"ASSUMED 1 ns full software coincidence window; not detector gating hardware.",
             "detectors":"ASSUMED four analyzer detector channels per OGS; not the number of nanowires in an SNSPD."}
    for key in overrides:
        basis[key] = "User override; see preset reference and effective value."
    audit.update(numerical_specifications_status="vendor_sensitivity_with_overrides" if overrides else "vendor_sensitivity_not_calibrated",
                 preset_revision=1, sources_checked="2026-09-11", sources=spec["sources"],
                 parameter_basis=basis, preset_reference=reference, overrides=overrides, caveats=spec["caveats"])
    return audit


def operating_warnings(settings, peak_counts_per_detector_s):
    spec = PRESETS.get(settings.preset_id)
    warnings = []
    if spec:
        threshold = spec["jitter_rate_warning_s"]
        if threshold and peak_counts_per_detector_s > threshold:
            warnings.append("SPDMH2 exceeds 1 MHz per detector (bin-averaged, uniform-allocation estimate). The manual warns of increased timing FWHM at high count rates; the constant-jitter result needs calibration.")
        limit = spec["catalog_max_count_s"]
        if limit and peak_counts_per_detector_s > limit:
            warnings.append("Predicted mean per-detector count rate exceeds the SPDMH2 catalog 20 MHz maximum; this operating point is outside the catalog envelope.")
    return warnings
