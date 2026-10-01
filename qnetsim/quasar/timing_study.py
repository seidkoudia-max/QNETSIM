"""DES coincidence-window/source-brightness sensitivity, separate from the base run."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid
import zipfile

import numpy as np

from quasar.config import ROOT
from quasar.fidelity import diagnostics
from quasar.mission_validation import LABEL
from quasar.outputs import OUTPUT_VERSION, clean, export, write_csv
from quasar.physics import resolve_window
from quasar.simulator import run

STUDY_VERSION = 2
MAX_AXIS_VALUES = 8
MAX_TOTAL_BINS = 250000


def parse_axis(text, name, lower, upper):
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except (ValueError, AttributeError):
        raise ValueError(f"{name}: enter comma-separated numbers.") from None
    if not 1 <= len(values) <= MAX_AXIS_VALUES:
        raise ValueError(f"{name}: enter 1 to {MAX_AXIS_VALUES} values.")
    if any(not math.isfinite(value) or not lower <= value <= upper for value in values):
        raise ValueError(f"{name}: values must be finite and between {lower:g} and {upper:g}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name}: remove duplicate values.")
    return tuple(sorted(values))


@dataclass(frozen=True)
class TimingStudy:
    windows_ps: tuple = (50., 100., 200., 500., 1000.)
    source_rates_mhz: tuple = (10., 100., 250., 500., 1000.)

    def validate(self):
        for name, values, lower, upper in [("Windows (ps)", self.windows_ps, 1, 1e7),
                                           ("Source brightness (MHz)", self.source_rates_mhz, 0, 10000)]:
            parse_axis(",".join(map(str, values)), name, lower, upper)


def study_signature(cfg, values, provenance, spec):
    payload = dict(config=asdict(cfg), values=values, provenance=provenance, study=asdict(spec),
                   study_version=STUDY_VERSION, output_version=OUTPUT_VERSION)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def configurations(cfg, values, spec):
    """No mutation: only emitted brightness and software coincidence width vary."""
    spec.validate()
    cfg.validate(values)
    resolved = resolve_window(cfg, values)
    resolved.validate(values)
    cases = [("baseline", resolved, dict(values))]
    for rate in sorted(spec.source_rates_mhz):
        for window in sorted(spec.windows_ps):
            settings = {nm:replace(resolved.detector_for(nm), gate_s=float(window)*1e-12)
                        for nm in resolved.wavelengths_nm}
            trial = replace(resolved, detector_settings=settings)
            trial_values = {**values, "C21":float(rate)}
            trial.validate(trial_values)
            cases.append((f"case_{len(cases):03d}", trial, trial_values))
    total_bins = math.ceil(resolved.duration_s/resolved.bin_s)*len(cases)*len(cfg.wavelengths_nm)
    if total_bins > MAX_TOTAL_BINS:
        raise ValueError(f"Study would contain {total_bins:,} reporting bins (limit {MAX_TOTAL_BINS:,}). Reduce the grid or interval length; the DES resolution is never changed automatically.")
    return cases


def summarize(result, values, case_id):
    d = diagnostics(result, float(values["C21"])*1e6, float(values["C22"]))
    visible_s = result.summary["shared_visibility_s"]
    return dict(case_id=case_id, is_baseline=case_id == "baseline", wavelength_nm=result.wavelength_nm,
                mode=result.config.mode, seed=result.config.seed,
                source_rate_mhz=float(values["C21"]), full_window_ps=d["full_gate_s"]*1e12,
                shared_visibility_s=visible_s, true_pairs=d["shared_true_pairs"],
                other_pair_accidentals=d["shared_other_pair_accidentals"],
                noise_accidentals=d["shared_noise_accidentals"],
                true_pairs_per_s=d["shared_true_pairs"]/visible_s if visible_s else None,
                raw_mixture_fidelity=d["shared_weighted_mixture_fidelity"],
                genuine_pair_fidelity=d["shared_weighted_genuine_fidelity"],
                timing_acceptance=d["timing_acceptance"],
                jitter_rms_ps=d["jitter_rms_s"]*1e12,
                polarization_enabled=d["polarization_enabled"],
                polarization_compensated=d["polarization_compensated"],
                peak_mean_counts_per_detector_s=result.summary["peak_mean_counts_per_detector_s"],
                detector_operating_warnings=" | ".join(result.summary["detector_operating_warnings"]))


def plot_study(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wavelengths = sorted({r["wavelength_nm"] for r in rows})
    rates = sorted({r["source_rate_mhz"] for r in rows if not r["is_baseline"]})
    colors = dict(zip(rates, plt.get_cmap("viridis")(np.linspace(.08, .86, len(rates)))))
    figures = {}
    with plt.rc_context({"axes.spines.top":False, "axes.spines.right":False,
                         "axes.grid":True, "grid.alpha":.18, "font.size":10}):
        def save(fig, name):
            handles, labels = fig.axes[0].get_legend_handles_labels()
            height = fig.get_size_inches()[1]
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                       bbox_to_anchor=(.5, 1-.6/height), fontsize=9)
            fig.text(.5, .014, LABEL+" | one seeded DES realization per grid point", ha="center", fontsize=8)
            fig.subplots_adjust(left=.09, right=.985, bottom=.75/height,
                                top=1-(.95+.23*math.ceil(len(labels)/3))/height,
                                wspace=.27, hspace=.40)
            path = Path(out_dir)/(name+".png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures[name] = str(path)

        fig, axes = plt.subplots(len(wavelengths), 2, figsize=(12,4.5*len(wavelengths)), squeeze=False)
        for index, nm in enumerate(wavelengths):
            baseline = next(r for r in rows if r["is_baseline"] and r["wavelength_nm"] == nm)
            for rate in rates:
                selected = sorted((r for r in rows if not r["is_baseline"] and r["wavelength_nm"] == nm and r["source_rate_mhz"] == rate), key=lambda r:r["full_window_ps"])
                x = [r["full_window_ps"] for r in selected]
                for ax, key in zip(axes[index], ["raw_mixture_fidelity", "true_pairs_per_s"]):
                    ax.plot(x, [r[key] if r[key] is not None else np.nan for r in selected],
                            marker="o", ms=4, color=colors[rate], label=f"{rate:g} MHz emitted")
            for ax, key, title, ylabel in zip(axes[index], ["raw_mixture_fidelity", "true_pairs_per_s"],
                                            ["Raw mixture fidelity", "Accepted genuine pair rate"],
                                            ["Count-weighted Bell fidelity", "Genuine pairs / shared-visibility second"]):
                if baseline[key] is not None:
                    ax.scatter([baseline["full_window_ps"]], [baseline[key]], marker="*", s=140, color="#b74424", zorder=5,
                               label=f"Current settings ({baseline['source_rate_mhz']:g} MHz)")
                ax.set(xscale="log", xlabel="Full coincidence window (ps)", ylabel=ylabel, title=f"{nm} nm | {title}")
                windows = sorted({float(f"{r['full_window_ps']:.10g}") for r in rows if r["wavelength_nm"] == nm})
                ax.set_xticks(windows, [f"{window:g}" for window in windows])
            axes[index,0].set_ylim(.24, 1.01)
            axes[index,1].set_ylim(bottom=0)
            axes[index,1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        fig.suptitle("QUASAR coincidence-window / brightness sensitivity\nSame geometry, detectors and selected polarization control; actual DES counts", fontsize=13)
        save(fig, "timing_brightness_sweeps")

        fig, axes = plt.subplots(1, len(wavelengths), figsize=(6*len(wavelengths),4.8), squeeze=False)
        for ax, nm in zip(axes[0], wavelengths):
            for rate in rates:
                selected = sorted((r for r in rows if not r["is_baseline"] and r["wavelength_nm"] == nm and r["source_rate_mhz"] == rate), key=lambda r:r["full_window_ps"])
                valid = [r for r in selected if r["raw_mixture_fidelity"] is not None]
                ax.plot([r["true_pairs_per_s"] for r in valid], [r["raw_mixture_fidelity"] for r in valid],
                        "o-", ms=4, color=colors[rate], label=f"{rate:g} MHz")
            b = next(r for r in rows if r["is_baseline"] and r["wavelength_nm"] == nm)
            if b["raw_mixture_fidelity"] is not None:
                ax.scatter([b["true_pairs_per_s"]], [b["raw_mixture_fidelity"]], marker="*", s=140, color="#b74424", label="Current settings", zorder=5)
            ax.set(title=f"{nm} nm | fidelity / rate trade-off", xlabel="Genuine pairs / shared-visibility second", ylabel="Raw mixture fidelity", ylim=(.24,1.01))
            ax.set_xlim(left=0)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        fig.suptitle("Each marker is a simulated coincidence window\nLines connect windows at fixed brightness; they are not fitted optima", fontsize=12)
        save(fig, "timing_brightness_tradeoff")
    return figures


def pass_series(result):
    """Preserve DES bins and empty fidelity; hide non-shared intervals in overlays."""
    rows = result.metrics
    visible = np.array([r["joint_visible"] for r in rows], dtype=bool)
    starts = np.array([r["bin_start_epoch_s"] for r in rows])
    widths = np.array([r["bin_width_s"] for r in rows])
    rate = np.array([r["true_pairs_per_s"] for r in rows], dtype=float)
    fidelity = np.array([r["coincidence_mixture_fidelity"] for r in rows], dtype=float)
    bounds = (float(starts[visible][0]), float((starts+widths)[visible][-1])) if visible.any() else None
    return dict(time=starts+widths/2, rate=np.where(visible, rate, np.nan),
                fidelity=np.where(visible, fidelity, np.nan), bounds=bounds)


def plot_window_pass(cases, source_rate_mhz, out_dir, group_index):
    """Overlays use existing per-case metrics, with no extra simulation or smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first_results = cases[0][1]
    cfg = first_results[0].config
    windows = [results[0].config.detector_for(results[0].wavelength_nm).gate_s*1e12
               for _, results in cases]
    colors = plt.get_cmap("viridis")(np.linspace(.08, .86, len(cases)))
    figures = {}
    with plt.rc_context({"axes.spines.top":False, "axes.spines.right":False,
                         "axes.grid":True, "grid.alpha":.18, "font.size":10}):
        for key, ylabel, title in [("rate", "Accepted genuine pairs / s", "Accepted pair rate"),
                                    ("fidelity", "Raw mixture fidelity", "Raw mixture fidelity")]:
            fig, axes = plt.subplots(1, len(first_results), figsize=(6*len(first_results),4.8), squeeze=False)
            for ax, first in zip(axes[0], first_results):
                nm = first.wavelength_nm
                for (_, results), color, window in zip(cases, colors, windows):
                    result = next(r for r in results if r.wavelength_nm == nm)
                    data = pass_series(result)
                    ax.plot(data["time"], data[key], lw=1.1, marker=".", ms=2,
                            color=color, label=f"{window:g} ps")
                ax.set(title=f"{nm} nm | {cfg.detector_for(nm).technology}",
                       xlabel="Seconds from workbook epoch", ylabel=ylabel)
                bounds = pass_series(first)["bounds"]
                if bounds is not None:
                    ax.set_xlim(*bounds)
                ax.ticklabel_format(useOffset=False, axis="x")
                if key == "rate":
                    ax.set_ylim(bottom=0)
                    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
                else:
                    ax.set_ylim(.24, 1.01)
            handles, labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.82),
                       ncol=4, frameon=False, title="Full coincidence window")
            mode = "Batched DES" if cfg.mode == "batched" else "Timestamp DES"
            fig.suptitle(f"QUASAR | {title} over shared visibility\n"
                         f"Emitted-pair rate: {source_rate_mhz:g} MHz | {mode} | reporting interval: {cfg.bin_s:g} s", fontsize=12)
            fig.subplots_adjust(left=.10, right=.985, bottom=.20,
                                top=.66-.05*(math.ceil(len(labels)/4)-1), wspace=.28)
            fig.text(.5,.025, LABEL+" | same detectors and selected polarization control", ha="center", fontsize=8)
            path = Path(out_dir)/f"window_overlays_{group_index:03d}_{key}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures[key] = str(path)
    return dict(source_rate_mhz=float(source_rate_mhz), windows_ps=windows,
                case_ids=[case_id for case_id, _ in cases], figures=figures)


def run_study(cfg, values, provenance, spec=None, out_dir=None, progress=None):
    spec = spec or TimingStudy()
    cases = configurations(cfg, values, spec)
    # Reject empty visibility before creating a result directory. This also
    # exercises the selected backend's safety checks without a mode fallback.
    baseline = run(cases[0][1], cases[0][2])
    if not all(r.summary["shared_visibility_s"] > 0 for r in baseline):
        raise ValueError("No shared visibility in the selected interval. Select a shared pass or a short interval inside one.")
    if out_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-timing-"+uuid.uuid4().hex[:8]
        out_dir = ROOT/"Output/quasar_des_runs"/name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(study_version=STUDY_VERSION, status="running", label=LABEL,
                    study=asdict(spec), base_configuration=asdict(cfg), base_values=values,
                    resolved_configuration=asdict(cases[0][1]), input_provenance=provenance,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    signature=study_signature(cfg, values, provenance, spec),
                    aggregation="Rate: shared true counts / shared-visibility duration. Fidelity: sum of genuine fidelity-weighted counts plus A/4, divided by all shared coincidence matches. No-coincidence fidelity is undefined.",
                    stochastic_scope="One seeded DES realization per cell; same seed across cells, not independent confidence samples. No uncertainty bands or mission calibration are inferred.",
                    unchanged="Detector efficiency/jitter/dead time, geometry, noise, polarization and DES mode/resolution. Only source brightness and software coincidence window change; no AO is added.",
                    completed_cases=[], completed_wavelength_runs=0, window_overlays=[])
    def save_manifest():
        (out_dir/"study.json").write_text(json.dumps(clean(manifest), indent=2, allow_nan=False)+"\n")
    save_manifest()
    rows = []
    window_cases = []
    try:
        for index, (case_id, trial, trial_values) in enumerate(cases):
            manifest["active_case"] = case_id
            save_manifest()
            results = baseline if index == 0 else run(trial, trial_values)
            export(results, provenance, trial_values, out_dir/case_id, make_figures=False, make_archive=False)
            rows.extend(summarize(r, trial_values, case_id) for r in results)
            if case_id != "baseline":
                window_cases.append((case_id, results))
                last_at_brightness = index+1 == len(cases) or cases[index+1][2]["C21"] != trial_values["C21"]
                if last_at_brightness:
                    manifest["window_overlays"].append(plot_window_pass(
                        window_cases, trial_values["C21"], out_dir, len(manifest["window_overlays"])+1))
                    window_cases = []
            manifest["completed_cases"].append(case_id)
            manifest["completed_wavelength_runs"] += len(results)
            save_manifest()
            if progress is not None:
                progress(index+1, len(cases))
        write_csv(out_dir/"study.csv", rows)
        (out_dir/"study_results.json").write_text(json.dumps(clean(rows), indent=2, allow_nan=False)+"\n")
        figures = plot_study(rows, out_dir)
        (out_dir/"README.md").write_text((ROOT/"quasar/TIMING_STUDY.md").read_text())
        manifest.update(status="complete", active_case=None, figures=figures)
        save_manifest()
        archive = out_dir/"quasar_timing_brightness_study.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file() and path != archive:
                    z.write(path, path.relative_to(out_dir))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        save_manifest()
        raise RuntimeError(f"Timing/brightness study failed in {manifest['active_case']}. Partial audit saved in {out_dir}. {exc}") from exc
    return dict(out_dir=str(out_dir), archive=str(archive), figures=figures, rows=clean(rows),
                csv=str(out_dir/"study.csv"), manifest=clean(manifest),
                window_overlays=clean(manifest["window_overlays"]))
