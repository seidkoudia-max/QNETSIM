"""Count-weighted fidelity diagnostics, with optional polarization evolution."""
import math

import numpy as np


def background_free_reference(source_rate, source_fidelity, gate_s, jitter_s):
    """CW all-pairs limit when background is negligible and live factors cancel.

    C = R * eta_A * eta_B * q * L_A * L_B
    A = R**2 * eta_A * eta_B * gate * L_A * L_B
    q = erf(gate / (4 * single-detector RMS jitter)).
    """
    q = 1. if jitter_s == 0 else math.erf(gate_s/(4*jitter_s))
    if source_rate <= 0:
        return q, None
    rho = source_rate*gate_s
    return q, (source_fidelity*q+.25*rho)/(q+rho)


def diagnostics(result, source_rate, source_fidelity):
    rows = [r for r in result.metrics if r["joint_visible"]]
    true = sum(r["true_pairs"] for r in rows)
    other = sum(r["other_pair_accidentals"] for r in rows)
    noise = sum(r["noise_accidentals"] for r in rows)
    total = true+other+noise
    genuine_sum = sum(r.get("genuine_fidelity_sum", source_fidelity*r["true_pairs"]) for r in rows)
    f = np.array([r["coincidence_mixture_fidelity"] for r in rows])
    f = f[np.isfinite(f)]
    detector = result.config.detector_for(result.wavelength_nm)
    q, reference = background_free_reference(source_rate, source_fidelity,
                                             detector.gate_s, detector.jitter_s)
    return dict(wavelength_nm=result.wavelength_nm, source_fidelity_assumed=source_fidelity,
                detector_technology=detector.technology, jitter_rms_s=detector.jitter_s,
                polarization_enabled=result.config.polarization[result.wavelength_nm].enabled,
                polarization_compensated=result.config.polarization[result.wavelength_nm].compensate,
                shared_weighted_genuine_fidelity=genuine_sum/true if true else None,
                source_rate_pairs_s=source_rate, full_gate_s=detector.gate_s,
                timing_acceptance=q, mean_emitted_pairs_per_window=source_rate*detector.gate_s,
                background_free_mixture_reference=reference,
                shared_true_pairs=true, shared_other_pair_accidentals=other, shared_noise_accidentals=noise,
                shared_accidentals_over_true=(other+noise)/true if true else None,
                shared_noise_fraction_of_accidentals=noise/(other+noise) if other+noise else None,
                shared_weighted_mixture_fidelity=(genuine_sum+.25*(other+noise))/total if total else None,
                shared_bin_fidelity_min=float(f.min()) if len(f) else None,
                shared_bin_fidelity_max=float(f.max()) if len(f) else None,
                shared_bin_fidelity_std=float(f.std()) if len(f) else None)


def zoom_series(result):
    """Keep gap positions: never connect across unavailable coincidence bins."""
    return np.array([r["coincidence_mixture_fidelity"] if r["joint_visible"] else np.nan
                     for r in result.metrics])


def zoom_limits(results):
    f = np.concatenate([zoom_series(r) for r in results])
    f = f[np.isfinite(f)]
    if not len(f):
        return .25, 1.
    lo, hi = float(f.min()), float(f.max())
    pad = max((hi-lo)*.15, .0001)
    return max(0., lo-pad), min(1., hi+pad)


def explanation(items):
    lines = ["# QUASAR fidelity check", "", "## Interpretation", "",
             "The plotted quantity is the fidelity of a modeled mixture of true and accidental coincidences, not source fidelity or simulated tomography.",
             "F_mix(t) = (F_genuine(t) * true + 0.25 * accidental) / (true + accidental).",
             "F_genuine is the propagated Bell-target overlap when polarization is on, or F_source when it is off.",
             "For a pass, sum F_genuine times true counts before dividing; do not use an unweighted fidelity average.",
             "", "For the CW Poisson all-pairs model with negligible background:",
             "C = R eta_A eta_B q L_A L_B; A_other = R^2 eta_A eta_B gate L_A L_B.",
             "Thus A_other / C = R gate / q and the leading geometry/wavelength factors cancel.",
             "Here q = erf(gate / (4 jitter)), with single-detector RMS jitter.",
             "This is a background-free live-fraction reference, not an exact dead-time theory for correlated photon arrivals.",
             "", "A flat curve is therefore possible despite a varying rate. For R = 1e9/s, gate = 1 ns and q approximately 1, F is approximately (0.99 + 0.25)/2 = 0.62. The Si-APD preset has a smaller q at that gate; use each wavelength's effective timing acceptance below.",
             "The scalar reference uses polarization off. Polarization-enabled runs use two lossless Jones transformations with specified drift and optional sampled inverse compensation. Parameters are illustrative, not derived from spacecraft attitude or telescope mirror coatings. State-dependent loss and calibrated turbulence coupling remain unmodeled.",
             "", "## Sampled results during shared visibility", ""]
    for item in items:
        lines += [f"### {item['wavelength_nm']} nm | {item['detector_technology']}", ""]
        for key in ["source_fidelity_assumed", "polarization_enabled", "polarization_compensated", "shared_weighted_genuine_fidelity", "source_rate_pairs_s", "full_gate_s", "timing_acceptance",
                    "background_free_mixture_reference", "shared_true_pairs", "shared_other_pair_accidentals",
                    "shared_noise_accidentals", "shared_weighted_mixture_fidelity", "shared_bin_fidelity_min",
                    "shared_bin_fidelity_max", "shared_bin_fidelity_std"]:
            lines.append(f"- {key}: {item[key]}")
        lines.append("")
    lines += ["## Before interpreting this as mission fidelity", "",
              "The original 1000 MHz has been confirmed as emitted-pair brightness; see the run-specific mission validation status for applicability. CW/pulsed operation and temporal statistics remain unresolved; the CW model cannot simply be relabeled. Si-APD at 810 nm and SNSPD at 1550 nm specify technologies, not calibrated numerical performance.",
              "A narrower window rejects true pairs when timing jitter is finite. Reducing brightness also changes throughput. Neither parameter was changed to force a higher fidelity.",
              "The zoomed plots expand the vertical scale without adding jitter or changing samples. Noise-only coincidences outside shared visibility are retained in the full plot and CSV, but excluded from the shared-pass zoom.", ""]
    return "\n".join(lines)
