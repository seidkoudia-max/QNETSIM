"""Paired, same-seed polarization sensitivity runs; never tune source brightness."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np

from quasar.config import ROOT
from quasar.outputs import clean, export, write_csv
from quasar.fidelity import diagnostics
from quasar.simulator import run
from quasar.mission_validation import LABEL


def run_comparison(cfg, values, provenance, out_dir=None):
    if out_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-polarization-"+uuid.uuid4().hex[:8]
        out_dir = ROOT/"Output/quasar_des_runs"/name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cases, rows = {}, []
    for label, enabled, compensate in [("Off", False, False), ("Uncompensated", True, False), ("Compensated", True, True)]:
        pol = {nm:replace(p, enabled=enabled, compensate=compensate) for nm,p in cfg.polarization.items()}
        results = run(replace(cfg, polarization=pol), values)
        cases[label] = results
        export(results, provenance, values, out_dir/label.lower())
        for result in results:
            d = diagnostics(result, float(values["C21"])*1e6, float(values["C22"]))
            rows.append(dict(case=label, wavelength_nm=result.wavelength_nm,
                             genuine_fidelity=d["shared_weighted_genuine_fidelity"],
                             mixture_fidelity=d["shared_weighted_mixture_fidelity"],
                             true_pairs_shared=d["shared_true_pairs"],
                             accidental_matches_shared=d["shared_other_pair_accidentals"]+d["shared_noise_accidentals"],
                             minimum_bin_fidelity=d["shared_bin_fidelity_min"],
                             maximum_bin_fidelity=d["shared_bin_fidelity_max"]))
    # These are unitary transformations, with independent RNGs and no PDL.
    for base, raw, corrected in zip(*cases.values()):
        for b,u,c in zip(base.metrics, raw.metrics, corrected.metrics):
            for key in ["true_pairs", "other_pair_accidentals", "noise_accidentals"]:
                if not b[key] == u[key] == c[key]:
                    raise AssertionError("Lossless polarization must not alter paired count realizations.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Off":"#3c617d", "Uncompensated":"#ca632d", "Compensated":"#147d6e"}
    with plt.rc_context({"axes.grid":True, "grid.alpha":.2, "font.size":10}):
        fig, axes = plt.subplots(2, len(cfg.wavelengths_nm), figsize=(6.5*len(cfg.wavelengths_nm), 8), squeeze=False)
        for label, results in cases.items():
            for column, result in enumerate(results):
                rows_visible = [r for r in result.metrics if r["joint_visible"]]
                t = [r["bin_start_epoch_s"]+r["bin_width_s"]/2 for r in rows_visible]
                for row, key in enumerate(["genuine_pair_fidelity", "coincidence_mixture_fidelity"]):
                    axes[row,column].plot(t, [r[key] for r in rows_visible], label=label, color=colors[label],
                                          ls="--" if label=="Off" else "-", lw=1.3)
                    axes[row,column].set(xlabel="Seconds from workbook epoch", ylabel="Bell-target fidelity",
                                         title=f"{result.wavelength_nm} nm | {'Genuine pairs' if row==0 else 'Including accidentals'}")
                    axes[row,column].ticklabel_format(useOffset=False, axis="both")
                    axes[row,column].legend(fontsize=9)
        for row in range(2):
            lo = min(ax.get_ylim()[0] for ax in axes[row])
            hi = max(ax.get_ylim()[1] for ax in axes[row])
            for ax in axes[row]:
                ax.set_ylim(max(0,lo), min(1.005,hi))
        fig.suptitle("QUASAR polarization sensitivity | illustrative, uncalibrated drift\nSame source, link budget, detector timing and sampled counts in all cases", fontsize=13)
        fig.text(.5, .012, LABEL, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.94))
        figure = out_dir/"polarization_comparison.png"
        fig.savefig(figure, dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(cfg.wavelengths_nm), figsize=(6.5*len(cfg.wavelengths_nm),4), squeeze=False)
        for ax, result in zip(axes[0], cases["Off"]):
            ax.plot([r["bin_start_epoch_s"]+r["bin_width_s"]/2 for r in result.metrics],
                    [r["true_pairs_per_s"] for r in result.metrics], color=colors["Off"])
            ax.set(title=f"{result.wavelength_nm} nm | all three cases identical", xlabel="Seconds from workbook epoch", ylabel="True detected pairs / s")
        fig.text(.5, .012, LABEL, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0,.05,1,1))
        fig.savefig(out_dir/"polarization_comparison_rates.png", dpi=170)
        plt.close(fig)
    write_csv(out_dir/"comparison.csv", rows)
    (out_dir/"comparison.json").write_text(json.dumps(clean(rows), indent=2, allow_nan=False)+"\n")
    (out_dir/"README.md").write_text("# QUASAR polarization sensitivity\n\nThree same-seed DES runs: off, uncompensated and compensated.\n"
        "These are illustrative parameter studies, not measured or calibrated mission results.\n"
        "Source brightness and all scalar losses/noise settings are unchanged. Counts were checked identical in every bin.\n"
        "Fidelity is relative to the fixed selected Bell target. Unitary rotation alone does not destroy entanglement.\n"
        "Turbulence remains diagnostic-only. Each subdirectory contains full settings, events, metrics and model documentation.\n")
    archive = out_dir/"quasar_polarization_comparison.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                z.write(path, path.relative_to(out_dir))
    return dict(out_dir=str(out_dir), figure=str(figure), archive=str(archive), rows=clean(rows))
