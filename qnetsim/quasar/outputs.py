"""Plots and auditable exports for the isolated QUASAR DES scenario."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import uuid
import zipfile

import numpy as np

from quasar.config import ROOT, values_from_inputs
from quasar.fidelity import diagnostics, explanation, zoom_limits, zoom_series
from quasar.mission_validation import LABEL, assessment, report_markdown

OUTPUT_VERSION = 6


LIMITATIONS = [
    "DES only. No SDN or Ryu controller is imported or started.",
    "Source MHz is used as emitted pairs/s. The user confirmed 1000 MHz = 1e9 pairs/s for the original workbook; see the run-specific mission validation status for applicability. This is not a source-statistics calibration.",
    "CW Poisson source; one photon per station. Each wavelength is a separate two-arm scenario.",
    "Optical throughput, detector efficiency, sky radiance, filter, FOV, timing and dead time are assumptions, not workbook specifications.",
    "Si-APD at 810 nm and SNSPD at 1550 nm, assumed identical at both OGSs. Timing, noise and coincidence window are wavelength-specific. Vendor sensitivity presets distinguish typical values, bounds and assumptions; custom/legacy settings are not datasheets. Effective settings and overrides are exported. Gaussian RMS = FWHM / sqrt(8 ln 2).",
    "The Single Quantum specification-bound preset combines headline bounds that the vendor does not guarantee simultaneously. Its 50 ns effective dead time is an assumption, not a measured recovery time. Gate width and analyzer detector count remain assumptions in both presets.",
    "Both technologies currently use the same configurable nonparalyzable detector abstraction, not device-specific semiconductor/superconductor dynamics. Si-APD afterpulsing and SNSPD efficiency recovery/latching are not modeled.",
    "Equivalent Gaussian beam; pointing accuracy is per-axis RMS. Obscuration is an area fraction.",
    "Cn2 and wind feed HV/Rytov diagnostics only. Turbulence coupling/fade loss is not calibrated; extra per-arm loss is an explicit sensitivity assumption.",
    "Detector channels are uniformly and independently assigned. Polarization analyzer correlations and afterpulsing are not simulated.",
    "Polarization is optional and lossless: independent rotator/retarder Jones matrices propagate an assumed isotropic Bell source. Drift coefficients and OU wander are sensitivity assumptions, not satellite/telescope calibration or atmospheric depolarization.",
    "Compensation uses noisy beacon estimates, an inverse Jones matrix, finite update interval and latency. Ideal beacon acquisition starts at each station's bin-resolved visibility opening; beacon counts, optical insertion loss and actuator dynamics are not modeled.",
    "Coincidence-mixture fidelity combines the propagated genuine Bell overlap and maximally mixed accidentals once. Off recovers fixed source fidelity. This is model-based state overlap, not simulated analyzer tomography or a secure-key result; local unitary drift does not itself destroy entanglement.",
    "Batched polarization uses the bin midpoint; photon mode evaluates genuine pairs at their individual reception times. Reduce batch and wander-grid intervals for convergence checks. Rotation wander is an epoch-anchored, linearly interpolated OU process.",
    "Batched DES uses correlated Poisson count categories, an independent live-fraction dead-time approximation and approximate accidental counts. It does not resolve individual photon times.",
    "Timestamp DES schedules individual candidate detections using Poisson thinning. Lost photons are not scheduled. Dead time persists across bins; all-pairs matching uses delay-corrected noisy tags.",
    "Timestamp DES starts with ready detectors and no events before the selected emission interval. Endpoint transients and noise just outside that interval are not included.",
    "Geometry/efficiency is sampled at bin midpoints. Photon flight delays interpolate boundary geometry with ideal delay compensation. No acquisition or synchronization errors.",
    "Station heights use WGS84; epoch is assumed UTC. Confirm orbital frame and mean vs osculating elements. Future Earth-orientation time data are not mission-certified.",
    "No quantum memory, repeater BSM or 256-pairs/s memory cap is imposed on this direct dual-downlink scenario.",
]

SOURCES = [
    "https://arxiv.org/abs/2012.01725",
    "https://arxiv.org/abs/0903.2160",
    "https://doi.org/10.3390/app8122496",
    "https://doi.org/10.1063/5.0159000",
    "https://rhodesmill.org/skyfield/coordinates.html",
    "https://arxiv.org/abs/1912.06318",
    "https://doi.org/10.1038/s41598-026-55812-2",
    "https://doi.org/10.1002/andp.202100597",
]
from quasar.detector_presets import PRESETS
SOURCES += list(dict.fromkeys(url for spec in PRESETS.values() for url in spec["sources"]))

REVIEWER_QUESTIONS = [
    "The original 1000 MHz rate has been clarified as emitted-pair brightness. Confirm CW versus pulsed operation and supply pair-number/timing statistics, calibration uncertainty and in-band spectral brightness. Changed workbooks/overrides require their own confirmation.",
    "Confirm whether 810 and 1550 nm describe two separate configurations or the two photons of a non-degenerate pair. Current runs assume two equal-wavelength arms per configuration.",
    "Confirm orbital element frame, epoch timezone, mean versus osculating elements, and the intended pass. The semi-major axis is used; 560 km, listed 566.89 km, and WGS84 height are not identical specifications.",
    "Supply transmitted waist/truncation or measured far-field divergence and M2, and clarify whether pointing accuracy is per-axis RMS, radial RMS, or a bound.",
    "Confirm receiver obscuration as blocked area rather than diameter ratio; the Helmos secondary-diameter note supports an area interpretation.",
    "Supply TX/RX optical throughput, wavelength-specific detector PDE, dead time, dark counts, detector allocation, jitter, afterpulsing and saturation behavior.",
    "Supply receiver FOV convention, equivalent filter bandwidth and source fraction passed by the filter, plus coincidence-window and synchronization details.",
    "Supply sky radiance or observing conditions (Sun/Moon angles, weather and aerosol/atmospheric profiles) at each OGS and wavelength. Current radiance and transmission are illustrative, not a weather prediction.",
    "Clarify the altitude reference for the HV surface Cn2 and supply turbulence/coupling/AO information if a calibrated fading penalty is required.",
    "Supply the source Bell state or density matrix, per-wavelength telescope Jones calibration, attitude/tracking reference frames, measured drift spectra, and beacon/compensator update rate, latency and residual errors. The added polarization defaults are an illustrative sensitivity case only.",
]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export(results, provenance, effective_values, out_dir=None, *, make_figures=True, make_archive=True):
    if out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
        out_dir = ROOT/"Output"/"quasar_des_runs"/run_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = results[0].config
    validation = assessment(cfg, effective_values, provenance)
    original_values = values_from_inputs(provenance)
    overrides = {key: {"workbook": original_values.get(key), "effective": value}
                 for key, value in effective_values.items() if original_values.get(key) != value}
    config = dict(backend="des_only", output_version=OUTPUT_VERSION, simulation=asdict(cfg), effective_workbook_values=effective_values,
                  effective_detectors=[r.config.detector_audit(r.wavelength_nm) for r in results],
                  input_provenance=provenance, overrides=overrides, limitations=LIMITATIONS,
                  reviewer_questions=REVIEWER_QUESTIONS, references=SOURCES,
                  mission_validation=validation)
    source_files = [ROOT/"quasar"/name for name in ["config.py", "physics.py", "simulator.py", "outputs.py", "fidelity.py", "detector_presets.py", "DETECTOR_PRESETS.md"]]
    source_files += [ROOT/"studies/quasar_matera_helmos/run_study.py", ROOT/"channels/fso.py", ROOT/"channels/polarization.py",
                     ROOT/"quasar/mission_validation.py", ROOT/"quasar/mission_evidence.json",
                     ROOT/"framework/core/des.py"]
    config["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_files}
    (out_dir/"configuration.json").write_text(json.dumps(clean(config), indent=2, allow_nan=False)+"\n")
    (out_dir/"summary.json").write_text(json.dumps(clean([r.summary for r in results]), indent=2, allow_nan=False)+"\n")
    write_csv(out_dir/"input_audit.csv", provenance["rows"])
    for result in results:
        write_csv(out_dir/f"metrics_{result.wavelength_nm}nm.csv", result.metrics)
        write_csv(out_dir/f"events_{result.wavelength_nm}nm.csv", result.events)
    (out_dir/"ASSUMPTIONS.md").write_text("# QNETSIM QUASAR: DES only\n\n"+
        "\n".join("- "+line for line in LIMITATIONS)+"\n\n## Questions for the reviewer\n\n"+
        "\n".join(f"{i}. {line}" for i,line in enumerate(REVIEWER_QUESTIONS, 1))+
        "\n\n## References\n\n"+"\n".join(SOURCES)+"\n")
    fidelity = [diagnostics(r, float(effective_values["C21"])*1e6, float(effective_values["C22"])) for r in results]
    (out_dir/"fidelity_diagnostics.json").write_text(json.dumps(clean(fidelity), indent=2, allow_nan=False)+"\n")
    (out_dir/"FIDELITY_CHECK.md").write_text(explanation(fidelity))
    (out_dir/"POLARIZATION_MODEL.md").write_text((ROOT/"quasar/POLARIZATION_MODEL.md").read_text())
    (out_dir/"mission_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False)+"\n")
    (out_dir/"MISSION_VALIDATION_STATUS.md").write_text(report_markdown(validation))
    (out_dir/"MISSION_VALIDATION.md").write_text((ROOT/"quasar/MISSION_VALIDATION.md").read_text())
    (out_dir/"DETECTOR_PRESETS.md").write_text((ROOT/"quasar/DETECTOR_PRESETS.md").read_text())
    figures = plot(results, out_dir, fidelity) if make_figures else {}
    archive = out_dir/"quasar_des_results.zip" if make_archive else None
    if archive is not None:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out_dir.iterdir()):
                if path.is_file() and path != archive:
                    z.write(path, path.name)
    return dict(out_dir=str(out_dir), figures={key:str(value) for key,value in figures.items()},
                archive=str(archive) if archive is not None else None, summaries=clean([r.summary for r in results]),
                configuration=clean(config), fidelity_diagnostics=clean(fidelity), output_version=OUTPUT_VERSION,
                mission_validation=validation)


def plot(results, out_dir, fidelity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = {}
    with plt.rc_context({"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.18,"font.size":10}):
        def save(fig, name):
            fig.text(.5, .012, LABEL, ha="center", va="bottom", fontsize=8, color="#6e4f26")
            fig.tight_layout(rect=(0, .05, 1, 1))
            path = out_dir/(name+".png")
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            figures[name] = path
        def times(result):
            return np.array([r["bin_start_epoch_s"]+r["bin_width_s"]/2 for r in result.metrics])
        def series(result, key):
            return np.array([r[key] for r in result.metrics])
        first = results[0]
        t = times(first)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in ["Matera", "Helmos"]:
            axes[0].plot(t, series(first, key+"_elevation_deg"), label=key)
            axes[1].plot(t, series(first, key+"_range_m")/1000, label=key)
        axes[0].set(ylabel="Elevation (deg)", title="Orbital geometry")
        axes[0].axhline(first.metrics[0]["minimum_elevation_deg"], color="gray", ls="--", label="Elevation mask")
        axes[1].set(ylabel="Slant range (km)", title="Matera and Helmos")
        for ax in axes:
            ax.fill_between(t, 0, 1, where=series(first, "joint_visible"),
                            transform=ax.get_xaxis_transform(), color="teal", alpha=.10, label="Shared visibility")
            ax.set_xlabel("Seconds from workbook epoch (UTC assumed)")
            ax.legend()
            ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "geometry")
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False)
        for ax, result in zip(axes[0], results):
            for key in ["Matera", "Helmos"]:
                ax.plot(times(result), series(result, key+"_detection_loss_db"), label=key)
            ax.set(xlabel="Seconds from epoch", ylabel="Loss incl. detection (dB)", title=f"{result.wavelength_nm} nm | {result.config.detector_for(result.wavelength_nm).technology} link budget")
            ax.legend(); ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "link_budget")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for result in results:
            x = times(result)
            line, = axes[0].plot(x, series(result, "true_pairs_per_s"), label=f"{result.wavelength_nm} nm DES")
            axes[0].plot(x, series(result, "expected_true_pairs_per_s"), linestyle="--", alpha=.65,
                         color=line.get_color(), label=f"{result.wavelength_nm} nm reference")
            axes[1].plot(x, series(result, "accidentals_per_s"), label=f"{result.wavelength_nm} nm")
        axes[0].set(ylabel="True detected pairs / s", title="Stochastic DES counts / bin duration")
        axes[1].set(ylabel="Accidental matches / s", title="Other pairs + sky + dark counts")
        for ax in axes:
            ax.set_xlabel("Seconds from epoch"); ax.legend(fontsize=8); ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "rates")
        colors = {810:"#1765a5", 1550:"#cf690c"}
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False, sharey=True)
        for ax, result, diag in zip(axes[0], results, fidelity):
            x = times(result)
            f = series(result, "coincidence_mixture_fidelity")
            ax.plot(x, zoom_series(result), color=colors[result.wavelength_nm], lw=1.1,
                    marker=".", ms=3, label="Coincidence-mixture fidelity")
            outside = ~series(result, "joint_visible").astype(bool)
            ax.scatter(x[outside], f[outside], s=8, color="gray", alpha=.5,
                       label="Outside shared visibility")
            ax.axhline(diag["source_fidelity_assumed"], color="#53645e", ls="--", lw=1,
                       label=f"Source fidelity: {diag['source_fidelity_assumed']:.3f}")
            if diag["polarization_enabled"]:
                ax.plot(x, np.where(series(result, "joint_visible"), series(result, "genuine_pair_fidelity"), np.nan),
                        color="#127d65", lw=1.2, label="Propagated genuine-pair fidelity")
            ax.set(title=f"{result.wavelength_nm} nm | {result.config.detector_for(result.wavelength_nm).technology}", xlabel="Seconds from epoch",
                   ylabel="Conditional mixture fidelity", ylim=(0,1.02))
            ax.set_xlim(result.metrics[0]["bin_start_epoch_s"],
                        result.metrics[-1]["bin_start_epoch_s"]+result.metrics[-1]["bin_width_s"])
            ax.legend(fontsize=8, loc="center right")
            ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "fidelity_noise")
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False, sharey=True)
        limits = zoom_limits(results)
        for ax, result, diag in zip(axes[0], results, fidelity):
            ax.plot(times(result), zoom_series(result), color=colors[result.wavelength_nm], lw=.9,
                    marker=".", ms=3, label="Sampled DES bins")
            if diag["background_free_mixture_reference"] is not None:
                ax.axhline(diag["background_free_mixture_reference"], color="gray", ls="--", lw=1,
                           label="Polarization-off CW reference")
            shared = [r for r in result.metrics if r["joint_visible"]]
            if shared:
                ax.set_xlim(shared[0]["bin_start_epoch_s"], shared[-1]["bin_start_epoch_s"]+shared[-1]["bin_width_s"])
            ax.set(title=f"{result.wavelength_nm} nm - shared-pass zoom", xlabel="Seconds from epoch",
                   ylabel="Mixture fidelity (expanded scale)", ylim=limits)
            ax.ticklabel_format(useOffset=False, axis="both")
            ax.legend(fontsize=8, loc="upper right")
        save(fig, "fidelity_zoom")
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False, sharey=True)
        for ax, result, diag in zip(axes[0], results, fidelity):
            shared = series(result, "joint_visible").astype(bool)
            x = times(result)
            ax.plot(x, np.where(shared, series(result, "genuine_pair_fidelity"), np.nan),
                    color="#127d65", label="Genuine-pair Bell fidelity")
            ax.plot(x, zoom_series(result), color=colors[result.wavelength_nm], label="With accidental mixture")
            ax.axhline(diag["source_fidelity_assumed"], color="gray", ls="--", label="Source fidelity")
            state = "off" if not diag["polarization_enabled"] else ("compensated" if diag["polarization_compensated"] else "uncompensated")
            ax.set(title=f"{result.wavelength_nm} nm | polarization {state}", xlabel="Seconds from epoch", ylabel="Fidelity to fixed Bell target", ylim=(0, 1.02))
            if shared.any():
                ax.set_xlim(x[shared][0], x[shared][-1])
            ax.legend(fontsize=8); ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "polarization")
        fig, ax = plt.subplots(figsize=(12, 4))
        for result in results:
            for key, style in [("Matera", "-"), ("Helmos", "--")]:
                noise = series(result, key+"_noise_counts")/series(result, "bin_width_s")
                ax.plot(times(result), noise, linestyle=style, color=colors[result.wavelength_nm],
                        label=f"{key} {result.wavelength_nm} nm")
        ax.set(ylabel="Accepted sky + dark counts / s", xlabel="Seconds from epoch",
               title="Assumed radiance; detector live-time effects included")
        ax.legend(fontsize=8); ax.ticklabel_format(useOffset=False, axis="x")
        save(fig, "background_noise")
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False)
        for ax, result in zip(axes[0], results):
            if result.config.mode == "photons":
                events = [x for x in result.events if x["event"]=="true_coincidence"]
                x = [e["corrected_tag_ps"]/1e12 for e in events]
                ax.scatter(x, np.arange(1, len(x)+1), s=8)
                ax.set(xlabel="Delay-corrected seconds from simulation start", ylabel="Cumulative true coincidences")
            else:
                events = [x for x in result.events if x["event"]=="coincidence_batch"]
                x = [e["des_time_ps"]/1e12 for e in events]
                ax.plot(x, [e["count"] for e in events], ".-", markersize=3)
                ax.set(xlabel="DES reception seconds from simulation start", ylabel="True pairs / receiver batch")
            ax.set_title(f"{result.wavelength_nm} nm | {result.config.mode} event view")
        save(fig, "events")
    return figures
