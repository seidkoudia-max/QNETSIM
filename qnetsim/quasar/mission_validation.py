"""Evidence/readiness reporting, not an empirical validation engine.

Numerical agreement or a user acknowledgment cannot promote a run to a
mission-validated prediction. Independent reference data are not yet supplied.
"""
import json

from quasar.config import ROOT, values_from_inputs

EVIDENCE_PATH = ROOT / "quasar/mission_evidence.json"
STATUS = "assumption_based_not_mission_validated"
LABEL = "Assumption-based simulation | not mission-validated"


def source_interpretation(values, provenance):
    evidence = json.loads(EVIDENCE_PATH.read_text())["source_rate_interpretation"]
    original = values_from_inputs(provenance)
    matches = (provenance.get("sha256") == evidence["workbook_sha256"]
               and original.get(evidence["cell"]) == evidence["value_mhz"]
               and values.get(evidence["cell"]) == evidence["value_mhz"])
    return dict(status="user_confirmed" if matches else "not_confirmed_for_this_input",
                applies_to_run=matches, effective_rate_pairs_s=float(values["C21"])*1e6,
                recorded_evidence=evidence)


def assessment(cfg, values, provenance):
    source = source_interpretation(values, provenance)
    detectors = [dict(cfg.detector_audit(nm), mean_emitted_pairs_per_gate=float(values["C21"])*1e6*cfg.detector_for(nm).gate_s)
                 for nm in cfg.wavelengths_nm]
    gates = {d["gate_s"] for d in detectors}
    common_gate = next(iter(gates)) if len(gates) == 1 else None
    areas = [
        ("source_statistics", "Source statistics and state",
         "CW Poisson emissions; assumed isotropic Bell-state family from scalar source fidelity.",
         "CW or pulsed operation; measured pair-number/timing statistics at the operating brightness; source spectrum, target state/density matrix and uncertainty. Confirm separate wavelengths versus a non-degenerate pair."),
        ("receiver", "Receiver and coincidence processing",
         "Si-APD at 810 nm and SNSPD at 1550 nm. Vendor sensitivity presets or custom efficiencies/timing, not unit calibration; uniform independent detector allocation; all-pairs matching; I/4 accidentals.",
         "Per-station, per-wavelength optical/filter throughput, PDE, dark/background counts, timing histogram, dead time/saturation and afterpulsing; analyzer settings; actual coincidence matching and accidental-subtraction policy."),
        ("atmosphere", "Atmosphere, pointing and coupling",
         "Gaussian/pointing/extinction model; fixed assumed radiance; turbulence diagnostics do not change transmission.",
         "Measured or independently calibrated extinction, sky radiance, divergence/pointing and aperture/coupling/fading statistics for representative elevations, weather, field of view and filter band. Supply AO details if applicable."),
        ("polarization", "Polarization and compensation",
         "Optional illustrative rotator/retarder drift and idealized beacon controller; no mission Jones calibration.",
         "Source and telescope Jones/Mueller calibration, attitude/tracking reference frames, measured residual polarization versus time/elevation, controller latency/bandwidth and insertion loss. A verified negligible-effect bound can justify disabling a mechanism."),
        ("geometry", "Orbit and mission operating envelope",
         "Workbook elements with selectable frame/J2 interpretation and future Earth-orientation assumptions.",
         "Confirmed epoch/time/frame/element convention and reference ephemeris/pass; acquisition mask and mission operating conditions. Compare range, elevation and timing against an independent orbit reference."),
        ("independent_reference", "Independent performance comparison",
         "No independent mission or representative-system dataset has been supplied for fidelity validation.",
         "Hold-out laboratory/representative-link observations of singles, coincidence histograms, accidentals and raw Bell fidelity/tomography across the intended loss/noise range, with calibration IDs, timestamps and measurement uncertainties."),
        ("uncertainty_acceptance", "Uncertainty and acceptance criteria",
         "No mission-calibrated input uncertainty budget or approved accuracy criteria.",
         "Parameter uncertainties and correlations; a documented calibration/validation split; accuracy and coverage requirements agreed before comparison. Propagate input, Monte Carlo and model-discrepancy uncertainties separately."),
    ]
    return dict(schema_version=2, mission_status=STATUS, label=LABEL,
                source_rate_interpretation=source,
                source_model="CW Poisson (not confirmed by the rate clarification)",
                source_fidelity_assumed=float(values["C22"]),
                gate_s=common_gate,
                mean_emitted_pairs_per_gate=float(values["C21"])*1e6*common_gate if common_gate is not None else None,
                detectors=detectors,
                detector_selection=json.loads(EVIDENCE_PATH.read_text())["detector_technology_selection"],
                wavelengths_nm=list(cfg.wavelengths_nm),
                fidelity_definition="Raw modeled fixed-Bell-target overlap including I/4 accidentals; not reconstructed tomography or accidental-subtracted fidelity.",
                independent_validation=dict(status="not_performed", dataset_ids=[], acceptance_criteria=None),
                numerical_verification=dict(
                    status="not_executed_by_this_assessment",
                    command="python -m unittest quasar.test_polarization quasar.test_des quasar.test_fidelity quasar.test_gui quasar.test_mission_validation",
                    scope="Implementation checks are separate from empirical validation. This report does not infer a test pass or mission validity from simulation completion."),
                blocking_items=[dict(id=key, area=area, status="outstanding", current_model=current, evidence_needed=needed)
                                for key, area, current, needed in areas])


def report_markdown(report):
    source = report["source_rate_interpretation"]
    lines = ["# QUASAR mission validation status", "", "**"+report["label"]+"**", "",
             "## Source clarification", "",
             f"Effective source rate: {source['effective_rate_pairs_s']:.9g} emitted pairs/s.",
             f"Interpretation status: {source['status']}.",
             "Recorded user statement: " + source["recorded_evidence"]["statement"] + ".",
             source["recorded_evidence"]["scope"],
             "This record applies only to the original workbook hash and unchanged C21 value; it does not automatically confirm overrides or a replacement workbook.",
             "", "## Current calculation", "",
             f"Assumed source fidelity: {report['source_fidelity_assumed']:g}.",
             report["fidelity_definition"],
             "Source rate has not been reduced to increase fidelity. The CW/pulsed and temporal-statistics assumptions remain unresolved.",
             "", "## Wavelength-specific receivers", "",
             "Presets, manufacturer specifications and remaining assumptions are detailed in DETECTOR_PRESETS.md and configuration.json.", ""]
    for detector in report["detectors"]:
        lines += [f"- {detector['wavelength_nm']} nm / {detector['technology']} / {detector['model'] or 'unspecified model'}: "
                  f"full gate {detector['gate_s']:.9g} s; R times gate {detector['mean_emitted_pairs_per_gate']:.9g}; "
                  f"Gaussian RMS jitter {detector['jitter_s']:.9g} s; PDE/SDE {detector['eta_detector']:g}; "
                  f"{detector['numerical_specifications_status']}."]
    lines += ["", "## Outstanding evidence", ""]
    for item in report["blocking_items"]:
        lines += ["### "+item["area"], "", "Current: "+item["current_model"], "",
                  "Required: "+item["evidence_needed"], ""]
    lines += ["## Verification is not mission validation", "",
              report["numerical_verification"]["scope"],
              "Independent validation: not performed. No reference dataset or approved acceptance limits are attached to this run.",
              "", "See MISSION_VALIDATION.md for the evidence request and validation procedure.", ""]
    return "\n".join(lines)
