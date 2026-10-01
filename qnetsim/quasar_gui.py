"""QNETSIM QUASAR GUI: dedicated DES-only Matera-Helmos dual downlink."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"Output"/"quasar_des_runs"/".mplconfig"))

import pandas as pd
import streamlit as st

from quasar.config import DetectorSettings, DETECTOR_TECHNOLOGIES, Optics, RunConfig, default_inputs, read_workbook, values_from_inputs
from quasar.detector_presets import DEFAULT_PRESET, PRESETS, FWHM_TO_RMS, preset_settings
from quasar.outputs import LIMITATIONS, OUTPUT_VERSION, export
from quasar.simulator import run
from quasar.mission_validation import LABEL, assessment, source_interpretation
from quasar.timing_study import TimingStudy, parse_axis, run_study, study_signature
from channels.polarization import ArmPolarization, PolarizationParams


def number(label, value, lo, hi, key, step=None, fmt=None):
    options = dict(min_value=float(lo), max_value=float(hi), value=float(value), key=key)
    if step is not None:
        options["step"] = float(step)
    if fmt is not None:
        options["format"] = fmt
    return st.number_input(label, **options)


def cell(v, name, key, lo, hi, step=None, fmt=None):
    v[key] = number(name, v[key], lo, hi, "quasar_input_"+key, step, fmt)


@st.cache_data(show_spinner=False)
def file_bytes(path):
    return Path(path).read_bytes()


def scenario_inputs(provenance):
    v = values_from_inputs(provenance)
    st.subheader("Mission inputs")
    st.caption("Prefilled from the reviewer workbook. Edits are recorded as overrides; the original file is never changed.")
    with st.expander("Source and orbit", expanded=True):
        a, b, c = st.columns(3)
        with a:
            cell(v, "Emitted pair rate (MHz)", "C21", 0, 10000, 10)
            cell(v, "Source fidelity", "C22", .25, 1, .01, "%.4f")
            v["C33"] = st.text_input("Element epoch (assumed UTC)", str(v["C33"]), key="quasar_input_C33")
        with b:
            cell(v, "Semi-major axis (km)", "C26", 6578, 42164, 1, "%.6f")
            cell(v, "Inclination (deg)", "C30", 0, 180, .1, "%.6f")
            cell(v, "RAAN (deg)", "C27", 0, 360, .1, "%.6f")
        with c:
            cell(v, "Argument of periapsis (deg)", "C31", 0, 360, .1)
            cell(v, "True anomaly (deg)", "C32", 0, 360, .1, "%.6f")
            cell(v, "Minimum elevation (deg)", "C14", 1, 89, 1)
        st.caption("Circular orbit only. The supplied semi-major axis is authoritative; altitude is not independently forced to 560 km.")
        if source_interpretation(v, provenance)["applies_to_run"]:
            st.info("Source rate confirmed by you: 1000 MHz means 1,000,000,000 emitted pairs/s, not the pump clock. CW/pulsed operation and pair statistics still need confirmation.")
        else:
            st.warning("The source clarification does not cover this workbook or rate override. This run still interprets MHz as emitted pairs/s; confirm the meaning separately.")
    with st.expander("Optical ground stations and transmitters"):
        cols = st.columns(2)
        for col, name, row, tx in zip(cols, ["Matera", "Helmos"], [3, 9], [17, 19]):
            with col:
                st.markdown("**"+name+"**")
                cell(v, name+" latitude (deg)", f"C{row}", -90, 90, .0001, "%.8f")
                cell(v, name+" longitude (deg)", f"C{row+1}", -180, 180, .0001, "%.8f")
                cell(v, name+" altitude (m)", f"C{row+2}", 0, 10000, 1)
                cell(v, name+" receiver diameter (m)", f"C{row+3}", .01, 20, .05)
                cell(v, name+" obscuration area fraction", f"C{row+4}", 0, .99, .01, "%.4f")
                cell(v, name+" transmitter diameter (m)", f"C{tx}", .01, 5, .01)
                cell(v, name+" TX pointing RMS per axis (urad)", f"C{tx+1}", 0, 1000, .1)
        st.caption("Helmos: (0.74 m secondary / 2.3 m primary)^2 is approximately 0.1035. This supports the area-fraction convention.")
    with st.expander("Turbulence diagnostics"):
        a, b = st.columns(2)
        with a:
            v["C36"] = number("HV surface Cn2 (m^-2/3)", float(str(v["C36"]).replace(" ", "")), 0, 1e-10, "quasar_input_C36", 1e-15, "%.3e")
        with b:
            cell(v, "HV RMS wind speed (m/s)", "C37", 0, 200, 1)
        st.info("Cn2 and wind currently produce HV/Rytov diagnostics only. They do not silently impose an uncalibrated throughput or fidelity penalty. Use explicit extra loss below for sensitivity studies.")
    return v


def detector_inputs(nm):
    choices = [DEFAULT_PRESET[nm], "legacy"]
    selected = st.selectbox(f"{nm} nm detector preset", choices,
                            format_func=lambda key: PRESETS[key]["label"] if key != "legacy" else "Legacy assumptions (not datasheet-based)",
                            key=f"quasar_detector_preset_{nm}")
    base = preset_settings(selected) if selected != "legacy" else DetectorSettings(DETECTOR_TECHNOLOGIES[nm])
    efficiency = PRESETS[selected]["efficiency"] if selected != "legacy" else .6
    prefix = f"quasar_detector_{nm}_{selected}_"
    if selected != "legacy":
        for caveat in PRESETS[selected]["caveats"]:
            st.caption(caveat)
        st.markdown("Sources: "+" | ".join(f"[Reference {i}]({url})" for i,url in enumerate(PRESETS[selected]["sources"], 1)))
    a,b,c = st.columns(3)
    with a:
        eta = number("Detector PDE / system detection efficiency", efficiency, .000001, 1, prefix+"eta", .01, "%.4f")
        jitter = number("Single-detector timing jitter FWHM (ps)", base.jitter_s/FWHM_TO_RMS*1e12,
                        0, 1e7, prefix+"jitter_fwhm", 1, "%.3f")
        st.caption(f"Gaussian RMS used by DES: {jitter*FWHM_TO_RMS:.3f} ps. Timing electronics and synchronization uncertainty are not added automatically.")
    with b:
        dark = number("Dark counts/s per detector", base.dark_per_detector_s, 0, 1e8, prefix+"dark", 1)
        dead = number("Effective dead time per detector (ns)", base.deadtime_s*1e9, 0, 1e7, prefix+"dead", 1)
        st.caption("SNSPD recovery: assumed step model, not a brochure-derived recovery curve." if nm == 1550 else "APD dead time: typical catalog value. Afterpulsing is not modeled.")
    with c:
        gate = number("Full coincidence window (ps)", base.gate_s*1e12, 1, 1e7, prefix+"gate", 50)
        count = st.number_input("Detectors per OGS", min_value=1, max_value=64, value=base.detectors, step=1, key=prefix+"count")
        st.caption("Window and four analyzer channels are receiver assumptions, not vendor specifications. Settings apply identically to both OGSs.")
    detector = DetectorSettings(technology=base.technology, model=base.model, preset_id=base.preset_id,
                                jitter_s=jitter*1e-12*FWHM_TO_RMS, gate_s=gate*1e-12,
                                dark_per_detector_s=dark, deadtime_s=dead*1e-9, detectors=int(count))
    return detector, eta


def receiver_inputs(wavelengths):
    optics, detectors = {}, {}
    st.subheader("Receiver and noise assumptions")
    st.caption("These settings are not supplied in the workbook. Values apply to both OGSs at a wavelength, except for the separate sky radiances.")
    for nm in wavelengths:
        with st.expander(f"{nm} nm {DETECTOR_TECHNOLOGIES[nm]} detector and timing", expanded=True):
            detectors[nm], efficiency = detector_inputs(nm)
        with st.expander(f"{nm} nm optical budget and sky background", expanded=True):
            o = Optics()
            prefix = f"quasar_optics_{nm}_"
            values = {"eta_detector": efficiency}
            a, b, c = st.columns(3)
            with a:
                for key, label in [("eta_tx", "TX efficiency"), ("eta_rx", "RX optics / filter efficiency")]:
                    values[key] = number(label, getattr(o,key), .000001, 1, prefix+key, .05, "%.4f")
                st.caption("RX efficiency covers additional optics/coupling upstream of the detector reference plane. Do not count losses already included in SNSPD system efficiency twice.")
                values["m2"] = number("Beam quality M2", o.m2, 1, 20, prefix+"m2", .1)
            with b:
                values["zenith_transmission"] = number("Zenith atmospheric transmission", .9, .000001, 1, prefix+"zenith", .01)
                values["extra_loss_db_per_arm"] = number("Extra loss on each arm (dB)", 0, 0, 100, prefix+"loss", .5)
                values["bandwidth_nm"] = number("Filter equivalent bandwidth (nm)", 1, .001, 100, prefix+"bandwidth", .1, "%.3f")
                values["fov_half_angle_urad"] = number("Receiver FOV half-angle (urad)", 10, .01, 10000, prefix+"fov", 1)
            with c:
                values["sky_matera"] = number("Matera sky radiance (W/m2/sr/nm)", 1e-6, 0, 1, prefix+"sky_m", 1e-6, "%.3e")
                values["sky_helmos"] = number("Helmos sky radiance (W/m2/sr/nm)", 1e-6, 0, 1, prefix+"sky_h", 1e-6, "%.3e")
                st.caption("Radiance is defined at the receiving aperture. No night/day brightness is inferred from the date. Source brightness is assumed to be inside this passband.")
            optics[nm] = Optics(**values)
    return optics, detectors


def polarization_inputs(wavelengths):
    st.subheader("Polarization evolution (optional)")
    enabled = st.checkbox("Enable polarization drift", value=False, key="quasar_pol_enabled")
    compare = st.checkbox("Also generate off / uncompensated / compensated comparison", value=False, key="quasar_pol_compare")
    parameters = {}
    if not enabled and not compare:
        st.caption("Off preserves the original fixed genuine-pair fidelity. Scalar link losses and accidentals still apply.")
        return {nm:PolarizationParams() for nm in wavelengths}, compare
    st.warning("Illustrative sensitivity model, not mission calibration. Lossless polarization rotations change Bell-target overlap, not transmission rate. Turbulence fading is still not enabled by this control.")
    compensate = st.checkbox("Enable polarization compensation", value=True, key="quasar_pol_compensate")
    target = st.selectbox("Source / receiver Bell target", ["phi_plus", "psi_minus"], key="quasar_pol_target")
    for nm in wavelengths:
        default = PolarizationParams()
        prefix = f"quasar_pol_{nm}_"
        with st.expander(f"{nm} nm drift and compensation assumptions", expanded=False):
            reference = number("Angle reference: seconds from workbook epoch", default.reference_epoch_s, 0, 172800, prefix+"reference", 1)
            arms = []
            for col, key, spec in zip(st.columns(2), ["Matera", "Helmos"], [default.matera, default.helmos]):
                with col:
                    st.markdown("**"+key+"**")
                    rotation = number("Rotation at reference (deg)", spec.rotation_deg, -360, 360, prefix+key+"rotation", 1)
                    rate = number("Rotation drift (deg/s)", spec.rotation_rate_deg_s, -10, 10, prefix+key+"rate", .01, "%.4f")
                    phase = number("Retardance at reference (deg)", spec.retardance_deg, -360, 360, prefix+key+"phase", 1)
                    phase_rate = number("Retardance drift (deg/s)", spec.retardance_rate_deg_s, -10, 10, prefix+key+"phase_rate", .01, "%.4f")
                    wander = number("Rotation wander RMS (deg)", spec.wander_rms_deg, 0, 360, prefix+key+"wander", .1)
                    arms.append(ArmPolarization(rotation, rate, phase, phase_rate, wander))
            a,b = st.columns(2)
            with a:
                correlation = number("Wander correlation time (s)", 30, .01, 86400, prefix+"correlation", 1)
                grid = number("Wander sampling interval (s)", 1, .001, 60, prefix+"grid", .1, "%.3f")
                interval = number("Compensation update interval (s)", 10, .001, 3600, prefix+"update", 1, "%.3f")
            with b:
                latency = number("Compensation latency (s)", 1, 0, 3600, prefix+"latency", .1)
                error = number("Estimation RMS per angle (deg)", .2, 0, 180, prefix+"error", .1)
                st.caption("Ideal beacon acquisition at visibility opening; no beacon photon noise model or compensator insertion loss. Use a DES batch no larger than the wander/update interval.")
            parameters[nm] = PolarizationParams(enabled, compensate, target, reference, *arms,
                                                correlation, grid, interval, latency, error)
    return parameters, compare


def render_results(result):
    st.divider()
    st.subheader("DES results")
    st.warning(LABEL+". Completing a run is not evidence of mission accuracy.")
    summaries = result["summaries"]
    for column, summary in zip(st.columns(len(summaries)), summaries):
        with column:
            st.markdown(f"**{summary['wavelength_nm']} nm**")
            st.metric("True detected pairs", f"{summary['true_pairs']:,}")
            st.metric("Mean rate over simulated interval", f"{summary['mean_true_pairs_per_s']:,.1f} pairs/s")
            st.caption(f"{summary['scheduler_events']:,} scheduler events; {summary['shared_visibility_s']:.6g} s of shared visibility")
            st.caption(summary.get("detector_settings", {}).get("model") or summary.get("detector_technology", "Legacy detector assumptions"))
            for warning in summary.get("detector_operating_warnings", []):
                st.warning(warning)
    tabs = st.tabs(["Geometry", "Link budget", "Rates", "Fidelity and noise", "Events", "Audit and downloads"])
    for tab, key in zip(tabs[:5], ["geometry", "link_budget", "rates", "fidelity_noise", "events"]):
        with tab:
            st.image(result["figures"][key], width="stretch")
    with tabs[2]:
        reference = summaries[0]["expectation_reference"]
        st.caption("Dashed curves: "+reference+". Solid curves are sampled DES counts divided by the actual bin duration, not pairs per event.")
    with tabs[3]:
        st.warning("Fidelity is a conditional source/accidental mixture estimate, not state tomography. Empty coincidence bins have no fidelity value.")
        if "fidelity_zoom" in result["figures"]:
            st.image(result["figures"]["fidelity_zoom"], width="stretch")
            st.caption("Separate panels prevent either wavelength being hidden. The zoom uses the same expanded scale for both and shows shared visibility only. No artificial variation is added.")
            rows = result["fidelity_diagnostics"]
            st.dataframe(pd.DataFrame([{
                "Wavelength (nm)":d["wavelength_nm"],
                "True pairs":d["shared_true_pairs"],
                "Wrong-pair matches":d["shared_other_pair_accidentals"],
                "Sky/dark accidental matches":d["shared_noise_accidentals"],
                "Weighted mixture fidelity":d["shared_weighted_mixture_fidelity"],
                "Min bin fidelity":d["shared_bin_fidelity_min"],
                "Max bin fidelity":d["shared_bin_fidelity_max"],
            } for d in rows]), hide_index=True, width="stretch")
            st.markdown("**Why can fidelity be low and nearly flat?**")
            for d in rows:
                st.write(f"{d['wavelength_nm']} nm: {d['source_rate_pairs_s']:,.0f} emitted pairs/s; "
                         f"{d['full_gate_s']*1e9:g} ns full window; "
                         f"R times window = {d['mean_emitted_pairs_per_window']:.4g}; "
                         f"true-pair timing acceptance = {d['timing_acceptance']:.4g}.")
                if d["background_free_mixture_reference"] is not None:
                    st.info(f"{d['wavelength_nm']} nm polarization-off, background-free CW reference: {d['background_free_mixture_reference']:.6f}.")
            st.latex(r"F_{\rm mix}(t)=\frac{F_{\rm genuine}(t) C(t)+\frac14 A(t)}{C(t)+A(t)}")
            st.caption("True coincidences and wrong-pair matches share the leading link-loss factors, so their ratio can stay almost constant while throughput changes substantially.")
            st.caption("F_genuine is propagated through the optional two-arm polarization model, or equals source fidelity when off. Accidentals assume I/4. These are modeled Bell overlaps, not reconstructed tomography. Confirm source statistics and calibrate drift before treating this as a mission prediction.")
            if "polarization" in result["figures"]:
                st.image(result["figures"]["polarization"], width="stretch")
            st.markdown("The count-weighted formula follows from density-matrix mixing and fidelity to a pure Bell target. The white-noise assumption is discussed in [Saggio and Walther, Eq. (6)](https://doi.org/10.1002/andp.202100597); it is not a universal atmospheric-fidelity law.")
            if "polarization_comparison" in result:
                comparison = result["polarization_comparison"]
                st.image(comparison["figure"], width="stretch")
                st.dataframe(pd.DataFrame(comparison["rows"]), hide_index=True, width="stretch")
                st.download_button("Download polarization comparison and all three runs", file_bytes(comparison["archive"]), file_name="quasar_polarization_comparison.zip")
            st.image(result["figures"]["background_noise"], width="stretch")
        else:
            st.info("Run again to generate separate wavelength panels, the fidelity zoom and coincidence diagnostics.")
    with tabs[4]:
        st.caption("Photon mode: arrival timestamps are physical DES times; corrected tags remove modeled flight delay. Coincidence matching is performed after reception. Batched mode logs receiver batches, not individual photons.")
        for summary in summaries:
            path = Path(result["out_dir"])/f"events_{summary['wavelength_nm']}nm.csv"
            with st.expander(f"{summary['wavelength_nm']} nm event log (first 1,000 rows)"):
                st.dataframe(pd.read_csv(path, nrows=1000), hide_index=True, width="stretch")
                st.download_button("Download complete event CSV", file_bytes(str(path)), file_name=path.name,
                                   mime="text/csv", key="log_"+str(summary["wavelength_nm"]))
    with tabs[5]:
        validation = result.get("mission_validation")
        if validation is not None:
            st.markdown("**Mission validation status**")
            st.write("Independent validation: not performed. No reference measurements or approved acceptance criteria are attached to this run.")
            st.dataframe(pd.DataFrame(validation["blocking_items"])[["area", "status", "evidence_needed"]], hide_index=True, width="stretch")
            st.download_button("Download mission validation status", file_bytes(str(Path(result["out_dir"])/"MISSION_VALIDATION_STATUS.md")),
                               file_name="MISSION_VALIDATION_STATUS.md")
        st.download_button("Download all results and assumptions", file_bytes(result["archive"]),
                           file_name="quasar_des_results.zip", mime="application/zip")
        st.caption("Saved locally in "+result["out_dir"])
        for summary in summaries:
            path = Path(result["out_dir"])/f"metrics_{summary['wavelength_nm']}nm.csv"
            st.download_button(f"Download {summary['wavelength_nm']} nm metrics", file_bytes(str(path)),
                               file_name=path.name, mime="text/csv", key="metrics_"+str(summary["wavelength_nm"]))
        st.json(result["configuration"], expanded=False)
        st.json(summaries, expanded=False)


def render_timing_study(cfg, values, provenance, confirmed):
    st.divider()
    st.subheader("Timing and brightness study")
    enabled = st.checkbox("Coincidence-window / source-brightness study", value=False, key="quasar_timing_enabled")
    if not enabled:
        st.caption("Optional additional DES runs. The main simulation and its plots are unchanged.")
        return
    st.caption("Sweep the full coincidence window and emitted-pair brightness. Each grid point uses your selected detector, orbit, background, polarization and DES resolution. This is not an adaptive-optics study.")
    a,b = st.columns(2)
    with a:
        windows_text = st.text_input("Full coincidence windows (ps, comma-separated)", "50, 100, 200, 500, 1000", key="quasar_timing_windows")
    with b:
        rates_text = st.text_input("Emitted-pair brightnesses (MHz, comma-separated)", "10, 100, 250, 500, 1000", key="quasar_timing_rates")
    spec, signature, error = None, None, None
    try:
        spec = TimingStudy(parse_axis(windows_text, "Coincidence windows", 1, 1e7),
                           parse_axis(rates_text, "Emitted-pair brightness", 0, 10000))
        if not cfg.wavelengths_nm:
            raise ValueError("Select at least one wavelength for the study.")
        signature = study_signature(cfg, values, provenance, spec)
        count = len(spec.windows_ps)*len(spec.source_rates_mhz)*len(cfg.wavelengths_nm)
        st.info(f"{count} wavelength/grid simulations plus {len(cfg.wavelengths_nm)} baseline simulations. One seeded realization per point, not confidence intervals. Each axis supports up to 8 values.")
    except ValueError as exc:
        error = str(exc)
        st.error(error)
    if cfg.mode == "photons":
        st.warning("Timestamp DES uses your selected short interval, not the full pass. Photon-event limits remain active at every brightness; reduce the grid or interval if needed.")
    states = []
    for nm in cfg.wavelengths_nm:
        p = cfg.polarization[nm]
        state = "off" if not p.enabled else ("compensated" if p.compensate else "uncompensated")
        states.append(f"{nm} nm: {state}")
    st.caption("Polarization held fixed: "+"; ".join(states)+". Enable compensation above if desired; the study does not switch it on automatically.")
    if st.button("Run timing / brightness study", disabled=not confirmed or error is not None, key="quasar_timing_run"):
        bar = st.progress(0., text="Running the baseline...")
        try:
            result = run_study(cfg, values, provenance, spec,
                               progress=lambda done,total: bar.progress(done/total, text=f"Completed {done} of {total} settings"))
            st.session_state.quasar_timing_result = result
            st.session_state.quasar_timing_signature = signature
            st.success("Timing / brightness study completed. Main simulation settings were not changed.")
        except Exception as exc:
            st.error(f"Study did not complete: {exc}")
            with st.expander("Study technical details"):
                st.exception(exc)
        finally:
            bar.empty()
    if "quasar_timing_result" not in st.session_state:
        return
    result = st.session_state.quasar_timing_result
    if signature is None or st.session_state.get("quasar_timing_signature") != signature:
        st.info("The displayed study belongs to previous settings. Run the study again to apply your changes.")
    st.warning(LABEL+". Narrower windows are sensitivity settings, not guaranteed receiver performance.")
    st.caption("Pair rate = accepted genuine pairs / shared-visibility seconds. Fidelity is coincidence-count weighted. Stars show the study's original receiver window and source brightness; these may differ between wavelengths.")
    for warning in sorted({r["detector_operating_warnings"] for r in result["rows"] if r["detector_operating_warnings"]}):
        st.warning(warning)
    a,b,c = st.tabs(["Fidelity and rate sweeps", "Fidelity / rate trade-off", "Study table and downloads"])
    with a:
        st.image(result["figures"]["timing_brightness_sweeps"], width="stretch")
    with b:
        st.image(result["figures"]["timing_brightness_tradeoff"], width="stretch")
        st.caption("Markers are DES runs, not fitted optima. Lines connect windows at fixed brightness; use the table for exact windows and detector acceptance.")
    with c:
        columns = ["case_id", "wavelength_nm", "source_rate_mhz", "full_window_ps", "raw_mixture_fidelity",
                   "true_pairs_per_s", "timing_acceptance", "genuine_pair_fidelity", "shared_visibility_s",
                   "true_pairs", "other_pair_accidentals", "noise_accidentals", "detector_operating_warnings"]
        st.dataframe(pd.DataFrame(result["rows"])[columns], hide_index=True, width="stretch")
        st.download_button("Download timing / brightness study CSV", file_bytes(result["csv"]), file_name="quasar_timing_brightness_study.csv", mime="text/csv")
        st.download_button("Download complete timing / brightness study", file_bytes(result["archive"]), file_name="quasar_timing_brightness_study.zip", mime="application/zip")
        st.caption("Saved locally in "+result["out_dir"])
        st.json(result["manifest"], expanded=False)


def main():
    st.set_page_config(page_title="QNETSIM QUASAR", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
      .block-container{max-width:1500px;padding-top:1.4rem}
      .quasar-hero{background:linear-gradient(120deg,#10283f,#145a75 65%,#167d85);color:white;
        border-radius:22px;padding:1.6rem 2rem;margin-bottom:1.2rem}
      .quasar-hero h1{color:white;margin:0;font-family:Georgia,serif;font-size:2.5rem}
      .quasar-hero p{margin:.7rem 0 .2rem;max-width:900px}
      .quasar-badge{display:inline-block;padding:.3rem .7rem;border:1px solid #80b9c3;border-radius:20px;margin-top:.8rem}
      @media(max-width:650px){.quasar-hero{padding:1.2rem}.quasar-hero h1{font-size:1.8rem}}
      </style>""", unsafe_allow_html=True)
    logo = ROOT/"assets"/"qnetsim_logo.png"
    if logo.exists():
        st.image(str(logo), width=200)
    st.markdown("""<div class="quasar-hero"><h1>QNETSIM QUASAR</h1>
      <p>Satellite entanglement distribution to Matera and Helmos. Explore the shared overpass,
      optical link budget, sky background and detector timing at 810 nm and 1550 nm.</p>
      <span class="quasar-badge">DES only</span></div>""", unsafe_allow_html=True)
    if "quasar_provenance" not in st.session_state:
        st.session_state.quasar_provenance = default_inputs()
    with st.sidebar:
        st.header("DES simulation")
        st.caption("No SDN controller or Ryu dependency is used.")
        label = st.selectbox("DES resolution", ["Batched DES - full pass", "Timestamp-resolved DES - short interval"])
        mode = "batched" if label.startswith("Batched") else "photons"
        wavelengths = tuple(st.multiselect("Wavelengths (nm)", [810, 1550], default=[810, 1550]))
        seed = st.number_input("Random seed", min_value=0, max_value=2**32-1, value=20260910, step=1)
        if mode == "batched":
            auto = st.checkbox("Find first shared pass automatically", value=True)
            start = number("Start after epoch (s)", 780, 0, 86400, "quasar_batch_start", 1)
            duration = number("Duration (s)", 1200, 1e-8, 86400, "quasar_batch_duration", 10)
            bin_s = number("DES batch / reporting interval (s)", 1, 1e-8, 60, "quasar_batch_bin", .1, "%.6f")
            if auto:
                st.caption("Automatic selection replaces start and duration with a 1,200 s view around the first shared pass.")
            st.info("Stochastic receiver batches. Dead-time and accidental-count approximations are recorded in every export.")
        else:
            auto = False
            start = number("Start after epoch (s)", 1377, 0, 86400, "quasar_photon_start", 1, "%.6f")
            duration = number("Duration (s)", .0001, 1e-8, 86400, "quasar_photon_duration", .0001, "%.8f")
            bin_s = number("Reporting / geometry interval (s)", .00001, 1e-8, 60, "quasar_photon_bin", .00001, "%.8f")
            st.info("Individual arrivals, persistent per-detector dead time, and coincidence matching. Long runs stop at the event limit; no silent fallback.")
        st.caption("Detector presets and wavelength-specific timing controls are in the main receiver section.")
        with st.expander("Orbit interpretation and safety"):
            frame = st.selectbox("Orbital element frame", ["J2000", "MOD"])
            j2 = st.checkbox("Treat elements as osculating and include J2", value=False)
            limit = st.number_input("Maximum photon arrivals per wavelength", min_value=1000, max_value=500000, value=200000, step=1000)
    with st.expander("Workbook and original input audit"):
        upload = st.file_uploader("Optional updated reviewer workbook (same template)", type=["xlsx"])
        if st.button("Load uploaded parameters", disabled=upload is None):
            try:
                provenance = read_workbook(upload.getvalue())
                RunConfig().validate(values_from_inputs(provenance))
                for key in list(st.session_state):
                    if key.startswith("quasar_input_"):
                        del st.session_state[key]
                st.session_state.quasar_provenance = provenance
                st.rerun()
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))
        audit = pd.DataFrame(st.session_state.quasar_provenance["rows"])
        audit["value"] = audit["value"].map(str)
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption("Source SHA-256: "+st.session_state.quasar_provenance["sha256"])
    v = scenario_inputs(st.session_state.quasar_provenance)
    optics, detectors = receiver_inputs(wavelengths)
    polarization, compare_polarization = polarization_inputs(wavelengths)
    cfg = RunConfig(mode=mode, wavelengths_nm=wavelengths, seed=int(seed), auto_pass=auto,
                    start_s=start, duration_s=duration, bin_s=bin_s, frame=frame, include_j2=j2,
                    detector_settings=detectors, max_photon_events=int(limit), optics=optics,
                    polarization=polarization)
    validation = assessment(cfg, v, st.session_state.quasar_provenance)
    signature = hashlib.sha256(json.dumps({"config":asdict(cfg), "values":v, "output_version":OUTPUT_VERSION, "compare_polarization":compare_polarization,
                                          "provenance":st.session_state.quasar_provenance, "mission_validation":validation}, sort_keys=True).encode()).hexdigest()
    st.warning("Preliminary modeling: vendor presets are sensitivity inputs, not calibrated mission receivers. Single Quantum headline bounds are not guaranteed simultaneously. Sky brightness, coupling and coincidence processing still require mission inputs.")
    with st.expander("What is needed for a validated mission prediction?"):
        st.write("The emitted-rate interpretation is recorded separately from calibration. Numerical tests verify implementation; independent reference measurements and an uncertainty budget are needed to assess mission accuracy.")
        st.dataframe(pd.DataFrame(validation["blocking_items"])[["area", "current_model", "evidence_needed"]], hide_index=True, width="stretch")
        st.download_button("Download mission data request and validation procedure", file_bytes(str(ROOT/"quasar/MISSION_VALIDATION.md")),
                           file_name="MISSION_VALIDATION.md")
    confirmed = st.checkbox("I understand the receiver/noise assumptions and the limitations of the selected DES resolution.")
    with st.expander("Model assumptions and missing inputs"):
        for note in LIMITATIONS:
            st.markdown("- "+note)
    if st.button("Run QUASAR - DES only", type="primary", disabled=not confirmed):
        try:
            with st.spinner("Running the DES event queue and preparing plots..."):
                results = run(cfg, v)
                result = export(results, st.session_state.quasar_provenance, v)
                if compare_polarization:
                    from quasar.polarization_study import run_comparison
                    result["polarization_comparison"] = run_comparison(cfg, v, st.session_state.quasar_provenance)
            st.session_state.quasar_result = result
            st.session_state.quasar_result_signature = signature
            st.success("DES run completed. Results and assumptions were saved separately from the other scenarios.")
        except Exception as exc:
            st.error(f"Simulation did not complete: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)
    if "quasar_result" in st.session_state:
        if st.session_state.get("quasar_result_signature") != signature:
            st.info("The displayed results belong to the previous settings. Run again to apply your changes.")
        render_results(st.session_state.quasar_result)
    render_timing_study(cfg, v, st.session_state.quasar_provenance, confirmed)


if __name__ == "__main__":
    main()
