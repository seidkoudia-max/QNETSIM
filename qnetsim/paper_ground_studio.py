from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

from channels.fiber import FiberParams
from channels.fso import FSOParams
from paper_ground_studio_backend import (
    default_ground_out_root,
    load_gallery_manifest,
    run_ground_scenario,
)


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "qnetsim_logo.png"

st.set_page_config(
    page_title="QNETSIM Ground",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 2.2rem;
        }
        .qns-hero {
            background: linear-gradient(135deg, #0f172a 0%, #0f766e 52%, #14b8a6 100%);
            border-radius: 26px;
            padding: 1.4rem 1.7rem;
            color: white;
            box-shadow: 0 18px 40px rgba(9, 30, 66, 0.22);
            margin-bottom: 1.25rem;
        }
        .qns-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }
        .qns-hero p {
            margin: 0.45rem 0 0;
            opacity: 0.92;
            font-size: 0.98rem;
        }
        .qns-chip-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.8rem;
        }
        .qns-chip {
            background: rgba(255,255,255,0.18);
            color: white;
            border: 1px solid rgba(255,255,255,0.22);
            padding: 0.34rem 0.65rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .qns-card-title {
            background: linear-gradient(180deg, #0f766e 0%, #115e59 100%);
            color: white;
            border-radius: 14px 14px 0 0;
            padding: 0.75rem 0.95rem;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.35rem;
            margin-bottom: 0;
        }
        .qns-card-shell {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-top: none;
            border-radius: 0 0 14px 14px;
            padding: 0.8rem 1rem 0.2rem 1rem;
            background: white;
            margin-bottom: 0.9rem;
        }
        .qns-link-button {
            display: inline-block;
            text-decoration: none;
            border-radius: 14px;
            background: linear-gradient(180deg, #173f8a 0%, #1e66f5 100%);
            color: white !important;
            font-weight: 700;
            padding: 0.72rem 1rem;
            box-shadow: 0 10px 20px rgba(30, 102, 245, 0.18);
            margin-right: 0.75rem;
        }
        .qns-link-button:hover {
            color: white !important;
            text-decoration: none;
        }
        .stButton > button {
            border: none;
            border-radius: 14px;
            background: linear-gradient(180deg, #0f766e 0%, #115e59 100%);
            color: white;
            font-weight: 700;
            padding: 0.65rem 1rem;
            box-shadow: 0 10px 20px rgba(15, 118, 110, 0.18);
        }
        .stButton > button:hover {
            color: white;
            border: none;
            background: linear-gradient(180deg, #14b8a6 0%, #0f766e 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sanitize_run_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or datetime.now().strftime("ground-%Y%m%d-%H%M%S")


def parse_float_list(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(int(float(text)))
    return values


def viewer_url_for_out_dir(out_dir: Path) -> str:
    return f"?mode=viewer&out_dir={quote(str(out_dir.resolve()))}"


def is_viewer_mode() -> bool:
    return str(st.query_params.get("mode", "")).strip().lower() == "viewer"


def viewer_out_dir_from_query() -> Path | None:
    raw = str(st.query_params.get("out_dir", "")).strip()
    if not raw:
        return None
    return Path(unquote(raw)).expanduser()


def initialize_state() -> None:
    fso = FSOParams(min_elevation_deg=10.0)
    fiber = FiberParams()
    defaults = {
        "ground_run_name": datetime.now().strftime("ground-%Y%m%d-%H%M%S"),
        "ground_output_root": str(default_ground_out_root()),
        "ground_generated_out_dir": "",
        "ground_seed": 12345,
        "ground_controller_backend": "des_only",
        "ground_duration_s": 1800.0,
        "ground_metrics_dt_s": 10.0,
        "ground_link_update_dt_s": 5.0,
        "ground_gen_delay_s": 0.2,
        "ground_swap_dt_s": 0.2,
        "ground_source_rate_hz": 5_000_000.0,
        "ground_max_modes_per_label": 512,
        "ground_max_bsm_per_cycle": 512,
        "ground_f_min": 0.25,
        "ground_min_elevation_deg": 0.0,
        "ground_visibility_scan_s": 86400.0,
        "ground_visibility_scan_step_s": 10.0,
        "ground_visibility_prebuffer_s": 0.0,
        "ground_pass_selection": "highest_dual",
        "ground_pass_candidate_limit": 64,
        "ground_pass_prefilter_step_s": 300.0,
        "ground_force_visible": False,
        "ground_mem_platform": "REID",
        "ground_mem_t2": 1000.0,
        "ground_mem_d": 7.0,
        "ground_mem_t_mem": 400.0,
        "ground_mem_n": 1.5,
        "ground_mem_m": 1.0,
        "ground_mem_eta_mode0": 0.9,
        "ground_mem_eta_ti": 0.3,
        "ground_bsm_v_hom": 0.97,
        "ground_bsm_eta_det": 0.8,
        "ground_bsm_sigma_det_s": 50e-12,
        "ground_bsm_delta_t_s": 200e-12,
        "ground_bsm_p_dc": 1e-6,
        "ground_bsm_swap_model": "paper",
        "ground_fiber_length_km": 2.0,
        "ground_ogs_name": "Windhof OGS",
        "ground_ogs_lat_deg": 49.65,
        "ground_ogs_lon_deg": 5.95,
        "ground_ogs_alt_m": 320.0,
        "ground_ground_end_name": "Kirchberg End Node",
        "ground_ground_end_lat_deg": 49.6298,
        "ground_ground_end_lon_deg": 6.1720,
        "ground_ground_end_alt_m": 290.0,
        "ground_auto_fiber_length_from_sites": True,
        "ground_enable_t2_sweep": False,
        "ground_t2_sweep_values": "600,1000,2000,3000",
        "ground_enable_rytov_sweep": False,
        "ground_rytov_sweep_values": "0.01,0.05,0.10,0.20",
        "ground_enable_buffer_sweep": False,
        "ground_buffer_sweep_values": "64,128,256,512",
        "ground_enable_multi_pass_visibility": False,
        "ground_multi_pass_horizon_hours": 24.0,
        "ground_multi_pass_sample_dt_s": 60.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for field, value in fso.__dict__.items():
        widget_key = "ground_fso_eta_point_static" if field == "eta_point_static" else f"ground_fso_{field}"
        st.session_state.setdefault(widget_key, value)
    for field, value in fiber.__dict__.items():
        st.session_state.setdefault(f"ground_fiber_{field}", value)
    st.session_state.setdefault("ground_fso_eta_src", 1.0)
    st.session_state.setdefault("ground_fso_eta_det", 1.0)
    st.session_state.setdefault("ground_fso_eta_cpl", 0.95)
    st.session_state.setdefault("ground_fso_d_t", 0.3)
    st.session_state.setdefault("ground_fso_d_r", 2.0)
    st.session_state.setdefault("ground_fso_theta_div", 2e-6)
    st.session_state.setdefault("ground_fso_sigma_r2", 0.01)
    st.session_state.setdefault("ground_fso_p_t", 0.5)
    st.session_state.setdefault("ground_fso_n_dc", 10.0)
    st.session_state.setdefault("ground_fso_enable_stochastic_fading", False)
    st.session_state.setdefault("ground_fso_enable_pointing_jitter", False)


def load_reference_preset() -> None:
    st.session_state.ground_duration_s = 1800.0
    st.session_state.ground_metrics_dt_s = 10.0
    st.session_state.ground_link_update_dt_s = 5.0
    st.session_state.ground_gen_delay_s = 0.2
    st.session_state.ground_swap_dt_s = 0.2
    st.session_state.ground_source_rate_hz = 5_000_000.0
    st.session_state.ground_max_modes_per_label = 512
    st.session_state.ground_max_bsm_per_cycle = 512
    st.session_state.ground_controller_backend = "des_only"
    st.session_state.ground_min_elevation_deg = 0.0
    st.session_state.ground_visibility_prebuffer_s = 0.0
    st.session_state.ground_pass_selection = "highest_dual"
    st.session_state.ground_fso_eta_src = 1.0
    st.session_state.ground_fso_eta_det = 1.0
    st.session_state.ground_fso_eta_cpl = 0.95
    st.session_state.ground_fso_d_t = 0.3
    st.session_state.ground_fso_d_r = 2.0
    st.session_state.ground_fso_theta_div = 2e-6
    st.session_state.ground_fso_sigma_r2 = 0.01
    st.session_state.ground_fso_p_t = 0.5
    st.session_state.ground_fso_n_dc = 10.0
    st.session_state.ground_fso_enable_stochastic_fading = False
    st.session_state.ground_fso_enable_pointing_jitter = False
    st.session_state.ground_mem_platform = "REID"
    st.session_state.ground_mem_t2 = 1000.0
    st.session_state.ground_mem_t_mem = 400.0
    st.session_state.ground_fiber_length_km = 2.0
    st.session_state.ground_ogs_name = "Windhof OGS"
    st.session_state.ground_ogs_lat_deg = 49.65
    st.session_state.ground_ogs_lon_deg = 5.95
    st.session_state.ground_ogs_alt_m = 320.0
    st.session_state.ground_ground_end_name = "Kirchberg End Node"
    st.session_state.ground_ground_end_lat_deg = 49.6298
    st.session_state.ground_ground_end_lon_deg = 6.1720
    st.session_state.ground_ground_end_alt_m = 290.0
    st.session_state.ground_auto_fiber_length_from_sites = True
    st.session_state.ground_enable_t2_sweep = False
    st.session_state.ground_enable_rytov_sweep = False
    st.session_state.ground_enable_buffer_sweep = False
    st.session_state.ground_enable_multi_pass_visibility = False


def build_fso_params() -> FSOParams:
    return FSOParams(
        eta_src=float(st.session_state.ground_fso_eta_src),
        eta_det=float(st.session_state.ground_fso_eta_det),
        eta_cpl=float(st.session_state.ground_fso_eta_cpl),
        eta_point_static=float(st.session_state.ground_fso_eta_point_static),
        d_t=float(st.session_state.ground_fso_d_t),
        d_r=float(st.session_state.ground_fso_d_r),
        theta_div=float(st.session_state.ground_fso_theta_div),
        wavelength_m=float(st.session_state.ground_fso_wavelength_m),
        t_abs=float(st.session_state.ground_fso_t_abs),
        t_sca=float(st.session_state.ground_fso_t_sca),
        c_atm=float(st.session_state.ground_fso_c_atm),
        sigma_r2=float(st.session_state.ground_fso_sigma_r2),
        ao_gain=float(st.session_state.ground_fso_ao_gain),
        turbulence_loss_coeff=float(st.session_state.ground_fso_turbulence_loss_coeff),
        turbulence_quality_coeff=float(st.session_state.ground_fso_turbulence_quality_coeff),
        enable_stochastic_fading=bool(st.session_state.ground_fso_enable_stochastic_fading),
        enable_pointing_jitter=bool(st.session_state.ground_fso_enable_pointing_jitter),
        pointing_jitter_rad=float(st.session_state.ground_fso_pointing_jitter_rad),
        turbulence_floor=float(st.session_state.ground_fso_turbulence_floor),
        polarization_error_rad=float(st.session_state.ground_fso_polarization_error_rad),
        phase_noise_rad=float(st.session_state.ground_fso_phase_noise_rad),
        temporal_mismatch_s=float(st.session_state.ground_fso_temporal_mismatch_s),
        coherence_time_s=float(st.session_state.ground_fso_coherence_time_s),
        p_t=float(st.session_state.ground_fso_p_t),
        n0=float(st.session_state.ground_fso_n0),
        n_dc=float(st.session_state.ground_fso_n_dc),
        n_conv=float(st.session_state.ground_fso_n_conv),
        delta_t=float(st.session_state.ground_fso_delta_t),
        min_elevation_deg=float(st.session_state.ground_min_elevation_deg),
        max_snr_for_mapping=float(st.session_state.ground_fso_max_snr_for_mapping),
    )


def build_fiber_params() -> FiberParams:
    return FiberParams(
        eta_src=float(st.session_state.ground_fiber_eta_src),
        eta_det=float(st.session_state.ground_fiber_eta_det),
        eta_cpl=float(st.session_state.ground_fiber_eta_cpl),
        eta_conn=float(st.session_state.ground_fiber_eta_conn),
        loss_db_per_km=float(st.session_state.ground_fiber_loss_db_per_km),
        insertion_loss_db=float(st.session_state.ground_fiber_insertion_loss_db),
        polarization_error_rad=float(st.session_state.ground_fiber_polarization_error_rad),
        phase_noise_rad=float(st.session_state.ground_fiber_phase_noise_rad),
        temporal_mismatch_s=float(st.session_state.ground_fiber_temporal_mismatch_s),
        coherence_time_s=float(st.session_state.ground_fiber_coherence_time_s),
        p_t=float(st.session_state.ground_fiber_p_t),
        n0=float(st.session_state.ground_fiber_n0),
        n_dc=float(st.session_state.ground_fiber_n_dc),
        n_conv=float(st.session_state.ground_fiber_n_conv),
        delta_t=float(st.session_state.ground_fiber_delta_t),
        enable_length_fluctuations=bool(st.session_state.ground_fiber_enable_length_fluctuations),
        length_jitter_fraction=float(st.session_state.ground_fiber_length_jitter_fraction),
        max_snr_for_mapping=float(st.session_state.ground_fiber_max_snr_for_mapping),
    )


def render_viewer() -> None:
    out_dir = viewer_out_dir_from_query()
    st.title("QNETSIM Ground Plot Viewer")
    if out_dir is None:
        st.error("Missing output directory.")
        return
    manifest = load_gallery_manifest(out_dir)
    if manifest is None:
        st.error(f"No manifest found in {out_dir}")
        return
    st.caption(str(out_dir.resolve()))
    files = manifest.get("files", {})
    cols = st.columns(2)
    with cols[0]:
        summary_path = files.get("summary_txt")
        if summary_path and Path(summary_path).exists():
            st.markdown(f"[summary.txt]({summary_path})")
            st.code(Path(summary_path).read_text(encoding="utf-8")[:1600])
    with cols[1]:
        pass_path = files.get("selected_pass_json")
        if pass_path and Path(pass_path).exists():
            st.markdown(f"[selected_ground_pass.json]({pass_path})")
            st.code(Path(pass_path).read_text(encoding="utf-8")[:1600], language="json")

    for item in manifest.get("images", []):
        st.subheader(str(item.get("title", "Plot")))
        st.image(str(item.get("path", "")), use_container_width=True)


def render_main() -> None:
    profile = st.sidebar.selectbox("Execution profile", ["Historical Ground", "Physical timing S3 (DES only)"],
                                   key="ground_execution_profile")
    if profile == "Physical timing S3 (DES only)":
        from ground_physical_studio import render
        render()
        return
    st.sidebar.warning("Historical mode retains the modes-to-rate pacing rule. Its S3 plateau is not a physical memory-lifetime result.")
    backend = str(st.session_state.get("ground_controller_backend", "des_only")).lower().strip()
    if backend == "sdn_local":
        backend_label = "DES + SDN Local"
    elif backend == "sdn_ryu":
        backend_label = "DES + SDN Ryu"
    else:
        backend_label = "DES only"
    with st.sidebar:
        st.image(str(LOGO_PATH), width=200)
        st.caption("Scenario: Ground")
        st.markdown("### Presets")
        if st.button("Load reference preset", use_container_width=True):
            load_reference_preset()
        st.markdown("### Run Controls")
        st.text_input("Run name", key="ground_run_name")
        st.text_input("Output root", key="ground_output_root")
        st.number_input("Seed", min_value=0, step=1, key="ground_seed")
        st.selectbox(
            "Controller backend",
            options=["des_only", "sdn_local", "sdn_ryu"],
            format_func=lambda value: (
                "DES only"
                if value == "des_only"
                else ("DES + SDN Local" if value == "sdn_local" else "DES + SDN Ryu")
            ),
            key="ground_controller_backend",
        )
        generate = st.button("Generate Ground Results", use_container_width=True)
        out_dir_str = str(st.session_state.get("ground_generated_out_dir", "")).strip()
        if out_dir_str:
            st.markdown(
                f'<a class="qns-link-button" href="{viewer_url_for_out_dir(Path(out_dir_str))}" target="_blank">Open Ground Plot Viewer</a>',
                unsafe_allow_html=True,
            )

    logo_col, copy_col = st.columns([0.18, 0.82], gap="large")
    with logo_col:
        st.image(str(LOGO_PATH), width=200)
    with copy_col:
        st.markdown(
            f"""
            <div class="qns-hero">
              <h1>QNETSIM Ground</h1>
              <p>Hybrid FSO + fiber workflow with the quantum repeater placed on the Windhof OGS, one remote end node on the satellite, and one local ground end node in Kirchberg connected to the OGS through fiber, reusing the Single Repeater orbit and visibility geometry stack.</p>
              <div class="qns-chip-row">
                <span class="qns-chip">Ground scenario</span>
                <span class="qns-chip">{backend_label}</span>
                <span class="qns-chip">FSO + Fiber</span>
                <span class="qns-chip">OGS repeater</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns([1.0, 1.0], gap="large")
    with col_a:
        st.markdown('<div class="qns-card-title">Scenario & Time Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.markdown(
            "The OGS repeater is anchored at Windhof Luxembourg. The satellite end node uses the same orbital propagation and pass-selection stack, while the local ground end node is anchored in Kirchberg Luxembourg and connected to the OGS over fiber."
        )
        st.caption(f"Active backend: {backend_label}")
        loc1, loc2 = st.columns(2)
        with loc1:
            st.text_input("OGS name", key="ground_ogs_name")
            st.number_input("OGS latitude (deg)", format="%.6f", key="ground_ogs_lat_deg")
            st.number_input("OGS longitude (deg)", format="%.6f", key="ground_ogs_lon_deg")
            st.number_input("OGS altitude (m)", min_value=0.0, step=10.0, key="ground_ogs_alt_m")
        with loc2:
            st.text_input("Ground end node name", key="ground_ground_end_name")
            st.number_input("Ground end latitude (deg)", format="%.6f", key="ground_ground_end_lat_deg")
            st.number_input("Ground end longitude (deg)", format="%.6f", key="ground_ground_end_lon_deg")
            st.number_input("Ground end altitude (m)", min_value=0.0, step=10.0, key="ground_ground_end_alt_m")
        st.checkbox("Auto fiber length from site coordinates", key="ground_auto_fiber_length_from_sites")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Duration (s)", min_value=60.0, step=60.0, key="ground_duration_s")
            st.number_input("Metrics dt (s)", min_value=1.0, step=1.0, key="ground_metrics_dt_s")
            st.number_input("Link update dt (s)", min_value=1.0, step=1.0, key="ground_link_update_dt_s")
            st.number_input("Generation delay (s)", min_value=0.001, step=0.05, format="%.3f", key="ground_gen_delay_s")
            st.number_input("Swap interval (s)", min_value=0.001, step=0.05, format="%.3f", key="ground_swap_dt_s")
            st.number_input("Source rate (Hz)", min_value=1.0, step=100000.0, key="ground_source_rate_hz")
        with c2:
            st.number_input("Max modes / label", min_value=1, step=1, key="ground_max_modes_per_label")
            st.number_input("Max BSM / cycle", min_value=1, step=1, key="ground_max_bsm_per_cycle")
            st.number_input("Minimum accepted fidelity", min_value=0.0, max_value=1.0, step=0.01, key="ground_f_min")
            st.number_input("Minimum elevation (deg)", min_value=0.0, step=1.0, key="ground_min_elevation_deg")
            st.checkbox("Force satellite visible", key="ground_force_visible")
        st.selectbox(
            "Pass selection",
            options=["memory_aware", "longest_dual", "highest_dual"],
            key="ground_pass_selection",
            help="Reuses the Single Repeater-style visibility scan but ranks passes for the OGS-side satellite contact only.",
        )
        c3, c4, c5 = st.columns(3)
        with c3:
            st.number_input("Visibility scan (s)", min_value=600.0, step=600.0, key="ground_visibility_scan_s")
        with c4:
            st.number_input("Visibility scan step (s)", min_value=1.0, step=1.0, key="ground_visibility_scan_step_s")
        with c5:
            st.number_input("Visibility prebuffer (s)", min_value=0.0, step=10.0, key="ground_visibility_prebuffer_s")
        st.number_input("Pass candidate limit", min_value=1, step=1, key="ground_pass_candidate_limit")
        st.number_input("Pass prefilter step (s)", min_value=10.0, step=10.0, key="ground_pass_prefilter_step_s")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="qns-card-title">Memory and BSM</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.selectbox("Memory platform", options=["REID", "COLD_ATOM", "TRAPPED_ION"], key="ground_mem_platform")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Memory T2 (s)", min_value=0.001, step=1.0, key="ground_mem_t2")
            st.number_input("Memory d", min_value=0.0, step=0.5, key="ground_mem_d")
            st.number_input("Memory n", min_value=0.1, step=0.1, key="ground_mem_n")
            st.number_input("Memory t_mem (s)", min_value=0.001, step=1.0, key="ground_mem_t_mem")
        with c2:
            st.number_input("Memory m", min_value=0.1, step=0.1, key="ground_mem_m")
            st.number_input("Memory eta_mode0", min_value=0.0, max_value=1.0, step=0.01, key="ground_mem_eta_mode0")
            st.number_input("Memory eta_ti", min_value=0.0, max_value=1.0, step=0.01, key="ground_mem_eta_ti")
        st.divider()
        st.selectbox(
            "BSM swap model",
            options=["paper", "legacy"],
            format_func=lambda value: "Reference" if value == "paper" else "Legacy",
            key="ground_bsm_swap_model",
        )
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("BSM v_hom", min_value=0.0, max_value=1.0, step=0.01, key="ground_bsm_v_hom")
            st.number_input("BSM eta_det", min_value=0.0, max_value=1.0, step=0.01, key="ground_bsm_eta_det")
            st.number_input("BSM sigma_det_s", min_value=0.0, format="%.12f", key="ground_bsm_sigma_det_s")
        with c4:
            st.number_input("BSM delta_t_s", min_value=0.0, format="%.12f", key="ground_bsm_delta_t_s")
            st.number_input("BSM p_dc", min_value=0.0, format="%.8f", key="ground_bsm_p_dc")
        st.markdown("</div>", unsafe_allow_html=True)

    fso_col, fiber_col = st.columns([1.0, 1.0], gap="large")
    with fso_col:
        st.markdown('<div class="qns-card-title">Satellite-to-OGS FSO Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("FSO eta_src", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_eta_src")
            st.number_input("FSO eta_det", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_eta_det")
            st.number_input("FSO eta_cpl", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_eta_cpl")
            st.number_input("FSO eta_point_static", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_eta_point_static")
            st.number_input("FSO d_t (m)", min_value=0.01, step=0.01, key="ground_fso_d_t")
            st.number_input("FSO d_r (m)", min_value=0.01, step=0.01, key="ground_fso_d_r")
        with c2:
            st.number_input("FSO theta_div", min_value=1e-8, format="%.8f", key="ground_fso_theta_div")
            st.number_input("FSO wavelength_m", min_value=1e-9, format="%.9f", key="ground_fso_wavelength_m")
            st.number_input("FSO sigma_r2", min_value=0.0, step=0.01, key="ground_fso_sigma_r2")
            st.number_input("FSO ao_gain", min_value=0.0, step=0.05, key="ground_fso_ao_gain")
            st.number_input("FSO p_t", min_value=0.0, step=0.01, key="ground_fso_p_t")
            st.number_input("FSO n_dc", min_value=0.0, step=1.0, key="ground_fso_n_dc")
        with c3:
            st.checkbox("FSO enable stochastic fading", key="ground_fso_enable_stochastic_fading")
            st.checkbox("FSO enable pointing jitter", key="ground_fso_enable_pointing_jitter")
            st.number_input("FSO pointing_jitter_rad", min_value=0.0, format="%.10f", key="ground_fso_pointing_jitter_rad")
            st.number_input("FSO polarization_error_rad", min_value=0.0, format="%.6f", key="ground_fso_polarization_error_rad")
            st.number_input("FSO phase_noise_rad", min_value=0.0, format="%.6f", key="ground_fso_phase_noise_rad")
            st.number_input("FSO temporal_mismatch_s", min_value=0.0, format="%.12f", key="ground_fso_temporal_mismatch_s")
        st.number_input("FSO t_abs", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_t_abs")
        st.number_input("FSO t_sca", min_value=0.0, max_value=1.0, step=0.01, key="ground_fso_t_sca")
        st.number_input("FSO c_atm", min_value=0.0, step=0.01, key="ground_fso_c_atm")
        st.number_input("FSO turbulence_loss_coeff", min_value=0.0, step=0.01, key="ground_fso_turbulence_loss_coeff")
        st.number_input("FSO turbulence_quality_coeff", min_value=0.0, step=0.01, key="ground_fso_turbulence_quality_coeff")
        st.number_input("FSO turbulence_floor", min_value=0.0, format="%.6f", key="ground_fso_turbulence_floor")
        st.number_input("FSO coherence_time_s", min_value=1e-15, format="%.12f", key="ground_fso_coherence_time_s")
        st.number_input("FSO n0", min_value=0.0, format="%.8f", key="ground_fso_n0")
        st.number_input("FSO n_conv", min_value=0.0, step=0.01, key="ground_fso_n_conv")
        st.number_input("FSO delta_t", min_value=0.0, format="%.12f", key="ground_fso_delta_t")
        st.number_input("FSO max_snr_for_mapping", min_value=1.0, step=1000.0, key="ground_fso_max_snr_for_mapping")
        st.markdown("</div>", unsafe_allow_html=True)

    with fiber_col:
        st.markdown('<div class="qns-card-title">OGS-to-Ground Fiber Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Fiber length (km)", min_value=0.01, step=0.5, key="ground_fiber_length_km")
            st.number_input("Fiber loss_db_per_km", min_value=0.0, step=0.01, key="ground_fiber_loss_db_per_km")
            st.number_input("Fiber insertion_loss_db", min_value=0.0, step=0.05, key="ground_fiber_insertion_loss_db")
            st.number_input("Fiber eta_src", min_value=0.0, max_value=1.0, step=0.01, key="ground_fiber_eta_src")
            st.number_input("Fiber eta_det", min_value=0.0, max_value=1.0, step=0.01, key="ground_fiber_eta_det")
            st.number_input("Fiber eta_cpl", min_value=0.0, max_value=1.0, step=0.01, key="ground_fiber_eta_cpl")
            st.number_input("Fiber eta_conn", min_value=0.0, max_value=1.0, step=0.01, key="ground_fiber_eta_conn")
        with c2:
            st.number_input("Fiber polarization_error_rad", min_value=0.0, format="%.6f", key="ground_fiber_polarization_error_rad")
            st.number_input("Fiber phase_noise_rad", min_value=0.0, format="%.6f", key="ground_fiber_phase_noise_rad")
            st.number_input("Fiber temporal_mismatch_s", min_value=0.0, format="%.12f", key="ground_fiber_temporal_mismatch_s")
            st.number_input("Fiber coherence_time_s", min_value=1e-15, format="%.12f", key="ground_fiber_coherence_time_s")
            st.number_input("Fiber p_t", min_value=0.0, step=0.01, key="ground_fiber_p_t")
            st.number_input("Fiber n0", min_value=0.0, format="%.8f", key="ground_fiber_n0")
            st.number_input("Fiber n_dc", min_value=0.0, step=1.0, key="ground_fiber_n_dc")
            st.number_input("Fiber n_conv", min_value=0.0, step=0.01, key="ground_fiber_n_conv")
            st.number_input("Fiber delta_t", min_value=0.0, format="%.12f", key="ground_fiber_delta_t")
            st.number_input("Fiber max_snr_for_mapping", min_value=1.0, step=1000.0, key="ground_fiber_max_snr_for_mapping")
        st.checkbox("Fiber enable length fluctuations", key="ground_fiber_enable_length_fluctuations")
        st.number_input("Fiber length jitter fraction", min_value=0.0, step=0.01, key="ground_fiber_length_jitter_fraction")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="qns-card-title">Optional Ground Sweep Studies</div>', unsafe_allow_html=True)
    st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
    st.markdown(
        "Tick only the extra studies you want. These stay Ground-only and add optional sweep figures plus composite overlays on top of the base time-series outputs."
    )
    sweep_col_a, sweep_col_b = st.columns([1.0, 1.0], gap="large")
    with sweep_col_a:
        st.checkbox("Run T2 sweep", key="ground_enable_t2_sweep")
        st.text_input("T2 sweep values (s)", key="ground_t2_sweep_values")
        st.checkbox("Run Rytov sweep", key="ground_enable_rytov_sweep")
        st.text_input("Rytov sweep values", key="ground_rytov_sweep_values")
    with sweep_col_b:
        st.checkbox("Run buffer sweep", key="ground_enable_buffer_sweep")
        st.text_input("Buffer sweep values (modes)", key="ground_buffer_sweep_values")
        st.checkbox("Run multi-pass visibility study", key="ground_enable_multi_pass_visibility")
        mp1, mp2 = st.columns(2)
        with mp1:
            st.number_input("Visibility horizon (hours)", min_value=1.0, step=1.0, key="ground_multi_pass_horizon_hours")
        with mp2:
            st.number_input("Visibility sample dt (s)", min_value=10.0, step=10.0, key="ground_multi_pass_sample_dt_s")
    st.info(
        "For the Ground sweeps, the per-cycle end-to-end BSM budget is capped by the configured memory mode count so the buffer capacity remains the active bottleneck."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if generate:
        run_name = sanitize_run_name(str(st.session_state.ground_run_name))
        out_root = Path(str(st.session_state.ground_output_root)).expanduser()
        with st.spinner(f"Running Ground repeater scenario ({backend_label})..."):
            result = run_ground_scenario(
                run_name=run_name,
                output_root=out_root,
                controller_backend=str(st.session_state.ground_controller_backend),
                duration_s=float(st.session_state.ground_duration_s),
                metrics_dt_s=float(st.session_state.ground_metrics_dt_s),
                link_update_dt_s=float(st.session_state.ground_link_update_dt_s),
                gen_delay_s=float(st.session_state.ground_gen_delay_s),
                swap_dt_s=float(st.session_state.ground_swap_dt_s),
                source_rate_hz=float(st.session_state.ground_source_rate_hz),
                max_modes_per_label=int(st.session_state.ground_max_modes_per_label),
                max_bsm_per_cycle=int(st.session_state.ground_max_bsm_per_cycle),
                f_min=float(st.session_state.ground_f_min),
                min_elevation_deg=float(st.session_state.ground_min_elevation_deg),
                visibility_scan_s=float(st.session_state.ground_visibility_scan_s),
                visibility_scan_step_s=float(st.session_state.ground_visibility_scan_step_s),
                visibility_prebuffer_s=float(st.session_state.ground_visibility_prebuffer_s),
                pass_selection=str(st.session_state.ground_pass_selection),
                pass_candidate_limit=int(st.session_state.ground_pass_candidate_limit),
                pass_prefilter_step_s=float(st.session_state.ground_pass_prefilter_step_s),
                force_visible=bool(st.session_state.ground_force_visible),
                fso_params=build_fso_params(),
                fiber_params=build_fiber_params(),
                fiber_length_km=float(st.session_state.ground_fiber_length_km),
                ogs_name=str(st.session_state.ground_ogs_name),
                ogs_lat_deg=float(st.session_state.ground_ogs_lat_deg),
                ogs_lon_deg=float(st.session_state.ground_ogs_lon_deg),
                ogs_alt_m=float(st.session_state.ground_ogs_alt_m),
                ground_end_name=str(st.session_state.ground_ground_end_name),
                ground_end_lat_deg=float(st.session_state.ground_ground_end_lat_deg),
                ground_end_lon_deg=float(st.session_state.ground_ground_end_lon_deg),
                ground_end_alt_m=float(st.session_state.ground_ground_end_alt_m),
                auto_fiber_length_from_sites=bool(st.session_state.ground_auto_fiber_length_from_sites),
                mem_platform=str(st.session_state.ground_mem_platform),
                mem_t2=float(st.session_state.ground_mem_t2),
                mem_d=float(st.session_state.ground_mem_d),
                mem_t_mem=float(st.session_state.ground_mem_t_mem),
                mem_n=float(st.session_state.ground_mem_n),
                mem_m=float(st.session_state.ground_mem_m),
                mem_eta_mode0=float(st.session_state.ground_mem_eta_mode0),
                mem_eta_ti=float(st.session_state.ground_mem_eta_ti),
                bsm_v_hom=float(st.session_state.ground_bsm_v_hom),
                bsm_eta_det=float(st.session_state.ground_bsm_eta_det),
                bsm_sigma_det_s=float(st.session_state.ground_bsm_sigma_det_s),
                bsm_delta_t_s=float(st.session_state.ground_bsm_delta_t_s),
                bsm_p_dc=float(st.session_state.ground_bsm_p_dc),
                bsm_swap_model=str(st.session_state.ground_bsm_swap_model),
                seed=int(st.session_state.ground_seed),
                enable_t2_sweep=bool(st.session_state.ground_enable_t2_sweep),
                t2_sweep_values=parse_float_list(str(st.session_state.ground_t2_sweep_values)),
                enable_rytov_sweep=bool(st.session_state.ground_enable_rytov_sweep),
                rytov_sweep_values=parse_float_list(str(st.session_state.ground_rytov_sweep_values)),
                enable_buffer_sweep=bool(st.session_state.ground_enable_buffer_sweep),
                buffer_sweep_values=parse_int_list(str(st.session_state.ground_buffer_sweep_values)),
                enable_multi_pass_visibility=bool(st.session_state.ground_enable_multi_pass_visibility),
                multi_pass_horizon_hours=float(st.session_state.ground_multi_pass_horizon_hours),
                multi_pass_sample_dt_s=float(st.session_state.ground_multi_pass_sample_dt_s),
            )
        st.session_state.ground_generated_out_dir = str(result.out_dir)
        st.success(f"Generated Ground outputs in {result.out_dir}")

    out_dir_str = str(st.session_state.get("ground_generated_out_dir", "")).strip()
    if out_dir_str:
        out_dir = Path(out_dir_str)
        manifest = load_gallery_manifest(out_dir)
        if manifest:
            st.subheader("Latest Ground Outputs")
            files = manifest.get("files", {})
            col1, col2 = st.columns(2)
            with col1:
                summary_path = files.get("summary_txt")
                if summary_path and Path(summary_path).exists():
                    st.markdown(f"[summary.txt]({summary_path})")
                    st.code(Path(summary_path).read_text(encoding="utf-8")[:1600])
            with col2:
                pass_path = files.get("selected_pass_json")
                if pass_path and Path(pass_path).exists():
                    st.markdown(f"[selected_ground_pass.json]({pass_path})")
                    st.code(Path(pass_path).read_text(encoding="utf-8")[:1600], language="json")
            with st.expander("Inline plot preview", expanded=False):
                for item in manifest.get("images", []):
                    st.subheader(str(item.get("title", "Plot")))
                    st.image(str(item.get("path", "")), use_container_width=True)


def main() -> None:
    inject_theme()
    initialize_state()
    if is_viewer_mode():
        render_viewer()
    else:
        render_main()


if __name__ == "__main__":
    main()
