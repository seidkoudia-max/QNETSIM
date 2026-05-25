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

from channels.fso import FSOParams
from stbd_constellation.config import BSMParameterSet, MemoryParameterSet, default_config
from stbd_figure_studio_backend import (
    default_stbd_out_root,
    load_gallery_manifest,
    run_stbd_constellation_optimization,
)


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "qnetsim_logo.png"

st.set_page_config(
    page_title="QNETSIM STBD",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1520px;
            padding-top: 1.5rem;
            padding-bottom: 2.2rem;
        }
        .qns-hero {
            background: linear-gradient(135deg, #111827 0%, #0b6e4f 55%, #0ea5e9 100%);
            border-radius: 26px;
            padding: 1.4rem 1.8rem;
            color: white;
            box-shadow: 0 18px 40px rgba(9, 30, 66, 0.22);
            margin-bottom: 1.2rem;
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
            background: linear-gradient(180deg, #0b6e4f 0%, #0a5b42 100%);
            color: white;
            border-radius: 14px 14px 0 0;
            padding: 0.75rem 0.95rem;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.35rem;
            margin-bottom: 0;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.16);
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
            background: linear-gradient(180deg, #0b6e4f 0%, #0a5b42 100%);
            color: white;
            font-weight: 700;
            padding: 0.65rem 1rem;
            box-shadow: 0 10px 20px rgba(11, 110, 79, 0.18);
        }
        .stButton > button:hover {
            color: white;
            border: none;
            background: linear-gradient(180deg, #11865f 0%, #0b6e4f 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sanitize_run_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or datetime.now().strftime("stbd-%Y%m%d-%H%M%S")


def viewer_url_for_out_dir(out_dir: Path) -> str:
    return f"?mode=viewer&out_dir={quote(str(out_dir.resolve()))}"


def is_viewer_mode() -> bool:
    mode = str(st.query_params.get("mode", "")).strip().lower()
    return mode == "viewer"


def viewer_out_dir_from_query() -> Path | None:
    raw = str(st.query_params.get("out_dir", "")).strip()
    if not raw:
        return None
    return Path(unquote(raw)).expanduser()


def initialize_state() -> None:
    cfg = default_config()
    defaults = {
        "stbd_run_name": datetime.now().strftime("stbd-basic-%Y%m%d-%H%M%S"),
        "stbd_output_root": str(default_stbd_out_root()),
        "stbd_execution_mode": "basic",
        "stbd_controller_backend": "des_only",
        "stbd_optimizer_method": "random_search",
        "stbd_random_search_trials": 1,
        "stbd_ga_population": 4,
        "stbd_ga_generations": 1,
        "stbd_elite_fraction": 0.5,
        "stbd_mutation_rate": 0.15,
        "stbd_seed": cfg.optimizer.seed,
        "stbd_pilot_horizon_s": 3600.0,
        "stbd_simulation_horizon_s": 7200.0,
        "stbd_geometry_step_s": 60.0,
        "stbd_snapshot_s": 600.0,
        "stbd_launch_step_s": 600.0,
        "stbd_metrics_dt_s": 120.0,
        "stbd_planner_horizon_override_s": 600.0,
        "stbd_gen_delay_s": cfg.time.gen_delay_s,
        "stbd_swap_dt_s": cfg.time.swap_dt_s,
        "stbd_paced_delivery_dt_s": cfg.time.paced_delivery_dt_s,
        "stbd_bsm_round_duration_s": cfg.time.bsm_round_duration_s,
        "stbd_source_rate_hz": 5_000_000.0,
        "stbd_planner_fidelity_threshold": cfg.planner_fidelity_threshold,
        "stbd_stochastic_link_state": cfg.stochastic_link_state,
        "stbd_perfect_isl": cfg.perfect_isl,
        "stbd_support_threshold": cfg.support_threshold,
        "stbd_evaluation_repeats": cfg.evaluation_repeats,
        "stbd_f_min": cfg.f_min,
        "stbd_w1": cfg.objective.w1_path_availability,
        "stbd_w2": cfg.objective.w2_fidelity,
        "stbd_w3": cfg.objective.w3_throughput,
        "stbd_w4": cfg.objective.w4_latency,
        "stbd_w5": cfg.objective.w5_satellite_penalty,
        "stbd_planes_min": 12,
        "stbd_planes_max": 12,
        "stbd_sats_min": 16,
        "stbd_sats_max": 16,
        "stbd_altitude_min_km": 1400.0,
        "stbd_altitude_max_km": 1400.0,
        "stbd_inclination_min_deg": 87.0,
        "stbd_inclination_max_deg": 87.0,
        "stbd_raan_spacing_min_deg": 15.0,
        "stbd_raan_spacing_max_deg": 15.0,
        "stbd_isl_range_min_km": 6000.0,
        "stbd_isl_range_max_km": 6000.0,
        "stbd_min_elevation_min_deg": 5.0,
        "stbd_min_elevation_max_deg": 5.0,
        "stbd_enabled_memory_keys": ["REID"],
        "stbd_generated_out_dir": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    fso = cfg.fso_sets["baseline"]
    for field, value in fso.__dict__.items():
        widget_key = "stbd_fso_eta_point_static" if field == "eta_point_static" else f"stbd_fso_{field}"
        st.session_state.setdefault(widget_key, value)
    isl_fso = cfg.isl_fso_params or fso
    for field, value in isl_fso.__dict__.items():
        widget_key = (
            "stbd_isl_fso_eta_point_static"
            if field == "eta_point_static"
            else f"stbd_isl_fso_{field}"
        )
        st.session_state.setdefault(widget_key, value)

    bsm = cfg.bsm_sets["baseline"]
    for field, value in bsm.__dict__.items():
        st.session_state.setdefault(f"stbd_bsm_{field}", value)

    for memory_key, params in cfg.memory_sets.items():
        prefix = f"stbd_mem_{memory_key.lower()}"
        st.session_state.setdefault(f"{prefix}_t2_s", params.t2_s)
        st.session_state.setdefault(f"{prefix}_d", params.d)
        st.session_state.setdefault(f"{prefix}_n", params.n)
        st.session_state.setdefault(f"{prefix}_t_mem_s", params.t_mem_s)
        st.session_state.setdefault(f"{prefix}_m", params.m)
        st.session_state.setdefault(f"{prefix}_eta_mode0", params.eta_mode0)
        st.session_state.setdefault(f"{prefix}_eta_ti", params.eta_ti)
        st.session_state.setdefault(f"{prefix}_max_modes_per_label", params.max_modes_per_label)


def load_fast_sanity_preset() -> None:
    st.session_state.stbd_execution_mode = "basic"
    st.session_state.stbd_controller_backend = "des_only"
    st.session_state.stbd_optimizer_method = "random_search"
    st.session_state.stbd_random_search_trials = 1
    st.session_state.stbd_ga_population = 4
    st.session_state.stbd_ga_generations = 1
    st.session_state.stbd_elite_fraction = 0.5
    st.session_state.stbd_mutation_rate = 0.15
    st.session_state.stbd_pilot_horizon_s = 3600.0
    st.session_state.stbd_simulation_horizon_s = 7200.0
    st.session_state.stbd_geometry_step_s = 60.0
    st.session_state.stbd_snapshot_s = 600.0
    st.session_state.stbd_launch_step_s = 600.0
    st.session_state.stbd_metrics_dt_s = 120.0
    st.session_state.stbd_planner_horizon_override_s = 600.0
    st.session_state.stbd_paced_delivery_dt_s = 30.0
    st.session_state.stbd_planes_min = 12
    st.session_state.stbd_planes_max = 12
    st.session_state.stbd_sats_min = 16
    st.session_state.stbd_sats_max = 16
    st.session_state.stbd_altitude_min_km = 1400.0
    st.session_state.stbd_altitude_max_km = 1400.0
    st.session_state.stbd_inclination_min_deg = 87.0
    st.session_state.stbd_inclination_max_deg = 87.0
    st.session_state.stbd_raan_spacing_min_deg = 15.0
    st.session_state.stbd_raan_spacing_max_deg = 15.0
    st.session_state.stbd_isl_range_min_km = 6000.0
    st.session_state.stbd_isl_range_max_km = 6000.0
    st.session_state.stbd_min_elevation_min_deg = 5.0
    st.session_state.stbd_min_elevation_max_deg = 5.0
    st.session_state.stbd_enabled_memory_keys = ["REID"]
    st.session_state.stbd_source_rate_hz = 5_000_000.0
    st.session_state.stbd_mem_reid_t2_s = 3600.0
    st.session_state.stbd_mem_reid_t_mem_s = 3600.0
    st.session_state.stbd_mem_reid_eta_mode0 = 1.0
    st.session_state.stbd_mem_reid_eta_ti = 1.0
    st.session_state.stbd_fso_eta_src = 0.90
    st.session_state.stbd_fso_eta_det = 0.82
    st.session_state.stbd_fso_eta_cpl = 0.82
    st.session_state.stbd_fso_eta_point_static = 0.985
    st.session_state.stbd_fso_d_t = 0.22
    st.session_state.stbd_fso_d_r = 0.80
    st.session_state.stbd_fso_theta_div = 8e-6
    st.session_state.stbd_fso_t_abs = 0.96
    st.session_state.stbd_fso_t_sca = 0.96
    st.session_state.stbd_fso_c_atm = 0.04
    st.session_state.stbd_fso_sigma_r2 = 0.06
    st.session_state.stbd_fso_ao_gain = 1.2
    st.session_state.stbd_fso_turbulence_loss_coeff = 0.05
    st.session_state.stbd_fso_turbulence_quality_coeff = 0.02
    st.session_state.stbd_fso_enable_stochastic_fading = False
    st.session_state.stbd_fso_enable_pointing_jitter = False
    st.session_state.stbd_fso_polarization_error_rad = 0.015
    st.session_state.stbd_fso_phase_noise_rad = 0.05
    st.session_state.stbd_fso_temporal_mismatch_s = 10e-12
    st.session_state.stbd_fso_p_t = 0.5
    st.session_state.stbd_fso_n_dc = 80.0
    st.session_state.stbd_isl_fso_eta_src = 0.92
    st.session_state.stbd_isl_fso_eta_det = 0.85
    st.session_state.stbd_isl_fso_eta_cpl = 0.90
    st.session_state.stbd_isl_fso_eta_point_static = 0.99
    st.session_state.stbd_isl_fso_d_t = 0.18
    st.session_state.stbd_isl_fso_d_r = 0.60
    st.session_state.stbd_isl_fso_theta_div = 5e-6
    st.session_state.stbd_isl_fso_t_abs = 1.0
    st.session_state.stbd_isl_fso_t_sca = 1.0
    st.session_state.stbd_isl_fso_c_atm = 0.0
    st.session_state.stbd_isl_fso_sigma_r2 = 0.0
    st.session_state.stbd_isl_fso_enable_stochastic_fading = False
    st.session_state.stbd_isl_fso_enable_pointing_jitter = False
    st.session_state.stbd_isl_fso_polarization_error_rad = 0.0
    st.session_state.stbd_isl_fso_phase_noise_rad = 0.02
    st.session_state.stbd_isl_fso_temporal_mismatch_s = 5e-12
    st.session_state.stbd_isl_fso_p_t = 0.5
    st.session_state.stbd_isl_fso_n_dc = 10.0


def load_full_search_preset() -> None:
    cfg = default_config()
    st.session_state.stbd_execution_mode = "basic"
    st.session_state.stbd_controller_backend = "des_only"
    st.session_state.stbd_optimizer_method = "random_search"
    st.session_state.stbd_random_search_trials = cfg.optimizer.random_search_trials
    st.session_state.stbd_ga_population = cfg.optimizer.ga_population
    st.session_state.stbd_ga_generations = cfg.optimizer.ga_generations
    st.session_state.stbd_elite_fraction = cfg.optimizer.elite_fraction
    st.session_state.stbd_mutation_rate = cfg.optimizer.mutation_rate
    st.session_state.stbd_pilot_horizon_s = cfg.optimizer.pilot_horizon_s
    st.session_state.stbd_simulation_horizon_s = cfg.time.simulation_horizon_s
    st.session_state.stbd_geometry_step_s = cfg.time.geometry_step_s
    st.session_state.stbd_snapshot_s = cfg.time.snapshot_s
    st.session_state.stbd_launch_step_s = cfg.time.launch_step_s
    st.session_state.stbd_metrics_dt_s = cfg.time.metrics_dt_s
    st.session_state.stbd_planner_horizon_override_s = 600.0
    st.session_state.stbd_paced_delivery_dt_s = cfg.time.paced_delivery_dt_s
    st.session_state.stbd_planes_min = cfg.search.planes_min
    st.session_state.stbd_planes_max = cfg.search.planes_max
    st.session_state.stbd_sats_min = cfg.search.sats_per_plane_min
    st.session_state.stbd_sats_max = cfg.search.sats_per_plane_max
    st.session_state.stbd_altitude_min_km = cfg.search.altitude_km_min
    st.session_state.stbd_altitude_max_km = cfg.search.altitude_km_max
    st.session_state.stbd_inclination_min_deg = cfg.search.inclination_deg_min
    st.session_state.stbd_inclination_max_deg = cfg.search.inclination_deg_max
    st.session_state.stbd_raan_spacing_min_deg = cfg.search.raan_spacing_deg_min
    st.session_state.stbd_raan_spacing_max_deg = cfg.search.raan_spacing_deg_max
    st.session_state.stbd_isl_range_min_km = cfg.search.max_isl_range_km_min
    st.session_state.stbd_isl_range_max_km = cfg.search.max_isl_range_km_max
    st.session_state.stbd_min_elevation_min_deg = cfg.search.min_elevation_deg_min
    st.session_state.stbd_min_elevation_max_deg = cfg.search.min_elevation_deg_max
    st.session_state.stbd_enabled_memory_keys = list(cfg.memory_sets.keys())
    st.session_state.stbd_source_rate_hz = cfg.source_rate_hz


def load_segmented_snapshot_preset() -> None:
    st.session_state.stbd_execution_mode = "segmented"
    st.session_state.stbd_controller_backend = "des_only"
    st.session_state.stbd_optimizer_method = "random_search"
    st.session_state.stbd_random_search_trials = 1
    st.session_state.stbd_ga_population = 4
    st.session_state.stbd_ga_generations = 1
    st.session_state.stbd_elite_fraction = 0.5
    st.session_state.stbd_mutation_rate = 0.15
    st.session_state.stbd_pilot_horizon_s = 3600.0
    st.session_state.stbd_simulation_horizon_s = 7200.0
    st.session_state.stbd_geometry_step_s = 10.0
    st.session_state.stbd_snapshot_s = 150.0
    st.session_state.stbd_launch_step_s = 150.0
    st.session_state.stbd_metrics_dt_s = 30.0
    st.session_state.stbd_planner_horizon_override_s = 600.0
    st.session_state.stbd_paced_delivery_dt_s = 30.0
    st.session_state.stbd_planes_min = 12
    st.session_state.stbd_planes_max = 12
    st.session_state.stbd_sats_min = 16
    st.session_state.stbd_sats_max = 16
    st.session_state.stbd_altitude_min_km = 1400.0
    st.session_state.stbd_altitude_max_km = 1400.0
    st.session_state.stbd_inclination_min_deg = 87.0
    st.session_state.stbd_inclination_max_deg = 87.0
    st.session_state.stbd_raan_spacing_min_deg = 15.0
    st.session_state.stbd_raan_spacing_max_deg = 15.0
    st.session_state.stbd_isl_range_min_km = 6000.0
    st.session_state.stbd_isl_range_max_km = 6000.0
    st.session_state.stbd_min_elevation_min_deg = 5.0
    st.session_state.stbd_min_elevation_max_deg = 5.0
    st.session_state.stbd_enabled_memory_keys = ["REID"]
    st.session_state.stbd_source_rate_hz = 5_000_000.0
    st.session_state.stbd_mem_reid_t2_s = 3600.0
    st.session_state.stbd_mem_reid_t_mem_s = 3600.0
    st.session_state.stbd_mem_reid_eta_mode0 = 1.0
    st.session_state.stbd_mem_reid_eta_ti = 1.0
    st.session_state.stbd_fso_eta_src = 0.90
    st.session_state.stbd_fso_eta_det = 0.82
    st.session_state.stbd_fso_eta_cpl = 0.82
    st.session_state.stbd_fso_eta_point_static = 0.985
    st.session_state.stbd_fso_d_t = 0.22
    st.session_state.stbd_fso_d_r = 0.80
    st.session_state.stbd_fso_theta_div = 8e-6
    st.session_state.stbd_fso_t_abs = 0.96
    st.session_state.stbd_fso_t_sca = 0.96
    st.session_state.stbd_fso_c_atm = 0.04
    st.session_state.stbd_fso_sigma_r2 = 0.06
    st.session_state.stbd_fso_ao_gain = 1.2
    st.session_state.stbd_fso_turbulence_loss_coeff = 0.05
    st.session_state.stbd_fso_turbulence_quality_coeff = 0.02
    st.session_state.stbd_fso_enable_stochastic_fading = False
    st.session_state.stbd_fso_enable_pointing_jitter = False
    st.session_state.stbd_fso_polarization_error_rad = 0.015
    st.session_state.stbd_fso_phase_noise_rad = 0.05
    st.session_state.stbd_fso_temporal_mismatch_s = 10e-12
    st.session_state.stbd_fso_p_t = 0.5
    st.session_state.stbd_fso_n_dc = 80.0
    st.session_state.stbd_isl_fso_eta_src = 0.92
    st.session_state.stbd_isl_fso_eta_det = 0.85
    st.session_state.stbd_isl_fso_eta_cpl = 0.90
    st.session_state.stbd_isl_fso_eta_point_static = 0.99
    st.session_state.stbd_isl_fso_d_t = 0.18
    st.session_state.stbd_isl_fso_d_r = 0.60
    st.session_state.stbd_isl_fso_theta_div = 5e-6
    st.session_state.stbd_isl_fso_t_abs = 1.0
    st.session_state.stbd_isl_fso_t_sca = 1.0
    st.session_state.stbd_isl_fso_c_atm = 0.0
    st.session_state.stbd_isl_fso_sigma_r2 = 0.0
    st.session_state.stbd_isl_fso_enable_stochastic_fading = False
    st.session_state.stbd_isl_fso_enable_pointing_jitter = False
    st.session_state.stbd_isl_fso_polarization_error_rad = 0.0
    st.session_state.stbd_isl_fso_phase_noise_rad = 0.02
    st.session_state.stbd_isl_fso_temporal_mismatch_s = 5e-12
    st.session_state.stbd_isl_fso_p_t = 0.5
    st.session_state.stbd_isl_fso_n_dc = 10.0


def load_paced_snapshot_preset() -> None:
    load_segmented_snapshot_preset()
    st.session_state.stbd_execution_mode = "paced"
    st.session_state.stbd_controller_backend = "des_only"
    st.session_state.stbd_launch_step_s = 600.0
    st.session_state.stbd_paced_delivery_dt_s = 30.0


def load_paced_snapshot_local_preset() -> None:
    load_paced_snapshot_preset()
    st.session_state.stbd_controller_backend = "sdn_local"


def load_paced_snapshot_ryu_preset() -> None:
    load_paced_snapshot_preset()
    st.session_state.stbd_controller_backend = "sdn_ryu"


def _stbd_mode_label(mode: str) -> str:
    normalized = str(mode).lower().strip()
    if normalized == "segmented":
        return "STBD segmented"
    if normalized == "paced":
        return "STBD paced"
    return "STBD basic"


def _stbd_backend_label(backend: str) -> str:
    normalized = str(backend).lower().strip()
    if normalized == "sdn_local":
        return "DES + SDN Local"
    if normalized == "sdn_ryu":
        return "DES + SDN Ryu"
    return "DES only"


def build_fso_params_from_prefix(prefix: str) -> FSOParams:
    return FSOParams(
        eta_src=float(st.session_state[f"{prefix}_eta_src"]),
        eta_det=float(st.session_state[f"{prefix}_eta_det"]),
        eta_cpl=float(st.session_state[f"{prefix}_eta_cpl"]),
        eta_point_static=float(st.session_state[f"{prefix}_eta_point_static"]),
        d_t=float(st.session_state[f"{prefix}_d_t"]),
        d_r=float(st.session_state[f"{prefix}_d_r"]),
        theta_div=float(st.session_state[f"{prefix}_theta_div"]),
        wavelength_m=float(st.session_state[f"{prefix}_wavelength_m"]),
        t_abs=float(st.session_state[f"{prefix}_t_abs"]),
        t_sca=float(st.session_state[f"{prefix}_t_sca"]),
        c_atm=float(st.session_state[f"{prefix}_c_atm"]),
        sigma_r2=float(st.session_state[f"{prefix}_sigma_r2"]),
        ao_gain=float(st.session_state[f"{prefix}_ao_gain"]),
        turbulence_loss_coeff=float(st.session_state[f"{prefix}_turbulence_loss_coeff"]),
        turbulence_quality_coeff=float(st.session_state[f"{prefix}_turbulence_quality_coeff"]),
        enable_stochastic_fading=bool(st.session_state[f"{prefix}_enable_stochastic_fading"]),
        enable_pointing_jitter=bool(st.session_state[f"{prefix}_enable_pointing_jitter"]),
        pointing_jitter_rad=float(st.session_state[f"{prefix}_pointing_jitter_rad"]),
        turbulence_floor=float(st.session_state[f"{prefix}_turbulence_floor"]),
        polarization_error_rad=float(st.session_state[f"{prefix}_polarization_error_rad"]),
        phase_noise_rad=float(st.session_state[f"{prefix}_phase_noise_rad"]),
        temporal_mismatch_s=float(st.session_state[f"{prefix}_temporal_mismatch_s"]),
        coherence_time_s=float(st.session_state[f"{prefix}_coherence_time_s"]),
        p_t=float(st.session_state[f"{prefix}_p_t"]),
        n0=float(st.session_state[f"{prefix}_n0"]),
        n_dc=float(st.session_state[f"{prefix}_n_dc"]),
        n_conv=float(st.session_state[f"{prefix}_n_conv"]),
        delta_t=float(st.session_state[f"{prefix}_delta_t"]),
        min_elevation_deg=float(st.session_state[f"{prefix}_min_elevation_deg"]),
        max_snr_for_mapping=float(st.session_state[f"{prefix}_max_snr_for_mapping"]),
    )


def build_fso_params() -> FSOParams:
    return build_fso_params_from_prefix("stbd_fso")


def build_isl_fso_params() -> FSOParams:
    return build_fso_params_from_prefix("stbd_isl_fso")


def build_bsm_params() -> BSMParameterSet:
    return BSMParameterSet(
        v_hom=float(st.session_state.stbd_bsm_v_hom),
        eta_det=float(st.session_state.stbd_bsm_eta_det),
        sigma_det_s=float(st.session_state.stbd_bsm_sigma_det_s),
        delta_t_s=float(st.session_state.stbd_bsm_delta_t_s),
        p_dc=float(st.session_state.stbd_bsm_p_dc),
        swap_model=str(st.session_state.stbd_bsm_swap_model),
    )


def build_memory_sets() -> dict:
    memory_sets = {}
    for key in ("REID", "COLD_ATOM", "TRAPPED_ION"):
        prefix = f"stbd_mem_{key.lower()}"
        memory_sets[key] = {
            "platform": key,
            "t2_s": float(st.session_state[f"{prefix}_t2_s"]),
            "d": float(st.session_state[f"{prefix}_d"]),
            "n": float(st.session_state[f"{prefix}_n"]),
            "t_mem_s": float(st.session_state[f"{prefix}_t_mem_s"]),
            "m": float(st.session_state[f"{prefix}_m"]),
            "eta_mode0": float(st.session_state[f"{prefix}_eta_mode0"]),
            "eta_ti": float(st.session_state[f"{prefix}_eta_ti"]),
            "max_modes_per_label": int(st.session_state[f"{prefix}_max_modes_per_label"]),
        }
    return memory_sets


def render_viewer() -> None:
    out_dir = viewer_out_dir_from_query()
    st.title("QNETSIM STBD Plot Viewer")
    if out_dir is None:
        st.error("No output directory was provided.")
        return
    manifest = load_gallery_manifest(out_dir)
    if not manifest:
        st.error(f"No gallery manifest found in {out_dir}.")
        return
    variant = str(manifest.get("variant", "")).strip()
    if variant:
        st.caption(f"Variant: {variant}")
    st.caption(str(out_dir))
    files = manifest.get("files", {})
    if files:
        st.subheader("Files")
        for label, path in files.items():
            st.markdown(f"- `{label}`: [{Path(path).name}]({path})")
    for item in manifest.get("images", []):
        st.subheader(str(item.get("title", "Plot")))
        st.image(str(item.get("path", "")), use_container_width=True)


def render_main() -> None:
    mode = str(st.session_state.get("stbd_execution_mode", "basic")).lower().strip()
    mode_label = _stbd_mode_label(mode)
    backend = str(st.session_state.get("stbd_controller_backend", "des_only")).lower().strip()
    backend_label = _stbd_backend_label(backend)
    if LOGO_PATH.exists():
        left, right = st.columns([1, 7], vertical_alignment="center")
        with left:
            st.image(str(LOGO_PATH), width=200)
        with right:
            st.markdown(
                f"""
                <div class="qns-hero">
                  <h1>QNETSIM STBD</h1>
                  <p>Separate STBD workflow for Luxembourg–Singapore constellation optimization with repository physical layers and DES as the source of truth, focused on space-time entanglement routing beyond single-hop quantum repeater chains, with an optional Ryu-backed SDN controller path.</p>
                  <div class="qns-chip-row">
                    <span class="qns-chip">{mode_label}</span>
                    <span class="qns-chip">Luxembourg–Singapore</span>
                    <span class="qns-chip">{backend_label}</span>
                    <span class="qns-chip">STBD space-time path</span>
                    <span class="qns-chip">Outer-loop constellation optimizer</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.title("QNETSIM STBD")

    with st.sidebar:
        st.header("Run Controls")
        st.caption(f"Scenario: {mode_label} | {backend_label}")
        st.markdown("**Presets**")
        preset_fast_col, preset_seg_col, preset_paced_col, preset_local_col, preset_ryu_col, preset_full_col = st.columns(6)
        with preset_fast_col:
            if st.button("STBD basic / Fast sanity", use_container_width=True):
                load_fast_sanity_preset()
                st.rerun()
        with preset_seg_col:
            if st.button("STBD segmented / 600-150", use_container_width=True):
                load_segmented_snapshot_preset()
                st.rerun()
        with preset_paced_col:
            if st.button("STBD paced / 600-150-30", use_container_width=True):
                load_paced_snapshot_preset()
                st.rerun()
        with preset_local_col:
            if st.button("STBD paced + Local", use_container_width=True):
                load_paced_snapshot_local_preset()
                st.rerun()
        with preset_ryu_col:
            if st.button("STBD paced + Ryu", use_container_width=True):
                load_paced_snapshot_ryu_preset()
                st.rerun()
        with preset_full_col:
            if st.button("STBD basic / Full search", use_container_width=True):
                load_full_search_preset()
                st.rerun()
        st.caption("Use `STBD basic / Fast sanity` for the old baseline, `STBD segmented / 600-150` for 600 s horizon with 150 s snapshots, `STBD paced / 600-150-30` for the same segmented route logic with paced 30 s final deliveries, `STBD paced + Local` for the in-process SDN controller path, `STBD paced + Ryu` for the Ryu-backed SDN controller path, and `STBD basic / Full search` for the heavier Luxembourg-Singapore search space.")
        st.selectbox(
            "Execution mode",
            options=["basic", "segmented", "paced"],
            format_func=lambda value: (
                "STBD basic"
                if value == "basic"
                else ("STBD segmented" if value == "segmented" else "STBD paced")
            ),
            key="stbd_execution_mode",
        )
        st.selectbox(
            "Controller backend",
            options=["des_only", "sdn_local", "sdn_ryu"],
            format_func=_stbd_backend_label,
            key="stbd_controller_backend",
            help="DES only runs the STBD path entirely in-process. DES + SDN Local keeps the same STBD route planning but gates segment activation through an in-process controller core. DES + SDN Ryu uses the same controller logic through a local Ryu app.",
        )
        st.text_input("Run name", key="stbd_run_name")
        st.text_input("Output root", key="stbd_output_root")
        st.selectbox(
            "Optimizer method",
            options=["random_search", "genetic_algorithm"],
            key="stbd_optimizer_method",
        )
        st.number_input("Random search trials", min_value=1, step=1, key="stbd_random_search_trials")
        st.number_input("GA population", min_value=2, step=1, key="stbd_ga_population")
        st.number_input("GA generations", min_value=1, step=1, key="stbd_ga_generations")
        st.number_input("Elite fraction", min_value=0.05, max_value=1.0, step=0.05, key="stbd_elite_fraction")
        st.number_input("Mutation rate", min_value=0.0, max_value=1.0, step=0.05, key="stbd_mutation_rate")
        st.number_input("Seed", min_value=0, step=1, key="stbd_seed")
        st.number_input("Evaluation repeats", min_value=1, step=1, key="stbd_evaluation_repeats")
        st.number_input("Pilot horizon (s)", min_value=600.0, step=600.0, key="stbd_pilot_horizon_s")
        st.divider()
        generate = st.button("Generate STBD Results", use_container_width=True)
        out_dir_str = str(st.session_state.get("stbd_generated_out_dir", "")).strip()
        if out_dir_str:
            viewer_url = viewer_url_for_out_dir(Path(out_dir_str))
            st.markdown(
                f'<a class="qns-link-button" href="{viewer_url}" target="_blank">Open STBD Plot Viewer</a>',
                unsafe_allow_html=True,
            )

    col_a, col_b = st.columns([1.0, 1.0], gap="large")
    with col_a:
        st.markdown('<div class="qns-card-title">Scenario & Objective</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="qns-card-shell">
            <b>Ground stations</b><br/>
            Luxembourg OGS / Windhof: 49.65°, 5.95°<br/>
            Singapore OGS: 1.2903°, 103.8520°<br/><br/>
            <b>Objective</b><br/>
            Maximize path availability, fidelity, and throughput while penalizing latency and satellite count, using a space-time path horizon tied to the selected memory coherence time unless a fixed STBD horizon override is supplied below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "If you want a quick check that the STBD workflow is alive, start with the `STBD basic / Fast sanity` preset in the sidebar. "
            "The full search can take much longer."
        )
    with col_b:
        st.markdown('<div class="qns-card-title">Time & DES Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Simulation horizon (s)", min_value=600.0, step=600.0, key="stbd_simulation_horizon_s")
            st.number_input("Geometry timestep (s)", min_value=1.0, step=1.0, key="stbd_geometry_step_s")
            st.number_input("STBD snapshot (s)", min_value=10.0, step=10.0, key="stbd_snapshot_s")
            st.number_input("Launch step (s)", min_value=10.0, step=10.0, key="stbd_launch_step_s")
            st.number_input("Metrics dt (s)", min_value=1.0, step=1.0, key="stbd_metrics_dt_s")
            st.number_input("Fixed STBD horizon (s, 0 = use memory T2)", min_value=0.0, step=300.0, key="stbd_planner_horizon_override_s")
        with c2:
            st.number_input("Generation delay (s)", min_value=0.001, step=0.05, format="%.3f", key="stbd_gen_delay_s")
            st.number_input("Swap interval (s)", min_value=0.001, step=0.05, format="%.3f", key="stbd_swap_dt_s")
            st.number_input("Paced delivery cadence (s)", min_value=1.0, step=1.0, key="stbd_paced_delivery_dt_s")
            st.number_input("BSM round duration (s)", min_value=1e-6, step=1e-4, format="%.6f", key="stbd_bsm_round_duration_s")
            st.number_input("Source rate (Hz)", min_value=1.0, step=100000.0, key="stbd_source_rate_hz")
            st.number_input("Support threshold", min_value=1, step=1, key="stbd_support_threshold")
        st.checkbox("Stochastic link state", key="stbd_stochastic_link_state")
        st.checkbox("Perfect ISL", key="stbd_perfect_isl")
        st.number_input("Fidelity threshold", min_value=0.0, max_value=1.0, step=0.01, key="stbd_planner_fidelity_threshold")
        st.number_input("Minimum accepted fidelity", min_value=0.0, max_value=1.0, step=0.01, key="stbd_f_min")
        st.caption("Set `Fixed STBD horizon` to a positive value if you want to keep the space-time graph horizon constant while sweeping physical memory T2 values such as 1500, 2000, and 3600 s.")
        st.markdown("</div>", unsafe_allow_html=True)

    exp1, exp2, exp3 = st.columns(3, gap="large")
    with exp1:
        st.markdown('<div class="qns-card-title">Objective Weights</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.number_input("w1: availability", min_value=0.0, max_value=1.0, step=0.05, key="stbd_w1")
        st.number_input("w2: fidelity", min_value=0.0, max_value=1.0, step=0.05, key="stbd_w2")
        st.number_input("w3: throughput", min_value=0.0, max_value=1.0, step=0.05, key="stbd_w3")
        st.number_input("w4: latency penalty", min_value=0.0, max_value=1.0, step=0.05, key="stbd_w4")
        st.number_input("w5: satellite penalty", min_value=0.0, max_value=1.0, step=0.05, key="stbd_w5")
        st.markdown("</div>", unsafe_allow_html=True)
    with exp2:
        st.markdown('<div class="qns-card-title">Constellation Search Ranges</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.number_input("Planes min", min_value=1, step=1, key="stbd_planes_min")
            st.number_input("Satellites/plane min", min_value=1, step=1, key="stbd_sats_min")
            st.number_input("Altitude min (km)", min_value=100.0, step=50.0, key="stbd_altitude_min_km")
            st.number_input("Inclination min (deg)", min_value=0.0, step=1.0, key="stbd_inclination_min_deg")
            st.number_input("RAAN spacing min (deg)", min_value=0.0, step=1.0, key="stbd_raan_spacing_min_deg")
            st.number_input("ISL range min (km)", min_value=100.0, step=100.0, key="stbd_isl_range_min_km")
            st.number_input("Min elevation min (deg)", min_value=0.0, step=1.0, key="stbd_min_elevation_min_deg")
        with b:
            st.number_input("Planes max", min_value=1, step=1, key="stbd_planes_max")
            st.number_input("Satellites/plane max", min_value=1, step=1, key="stbd_sats_max")
            st.number_input("Altitude max (km)", min_value=100.0, step=50.0, key="stbd_altitude_max_km")
            st.number_input("Inclination max (deg)", min_value=0.0, step=1.0, key="stbd_inclination_max_deg")
            st.number_input("RAAN spacing max (deg)", min_value=0.0, step=1.0, key="stbd_raan_spacing_max_deg")
            st.number_input("ISL range max (km)", min_value=100.0, step=100.0, key="stbd_isl_range_max_km")
            st.number_input("Min elevation max (deg)", min_value=0.0, step=1.0, key="stbd_min_elevation_max_deg")
        st.markdown("</div>", unsafe_allow_html=True)
    with exp3:
        st.markdown('<div class="qns-card-title">Physical Set Selection</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.multiselect(
            "Memory platforms available to optimizer",
            options=["REID", "COLD_ATOM", "TRAPPED_ION"],
            key="stbd_enabled_memory_keys",
        )
        st.caption("The optimizer samples the enabled memory platforms while using the shared FSO and BSM parameter sets below.")
        st.markdown("</div>", unsafe_allow_html=True)

    fso_col, isl_col, bsm_col = st.columns([1.0, 1.0, 0.9], gap="large")
    with fso_col:
        st.markdown('<div class="qns-card-title">Ground-Space FSO Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("eta_src", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_eta_src")
            st.number_input("eta_det", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_eta_det")
            st.number_input("eta_cpl", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_eta_cpl")
            st.number_input("eta_point_static", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_eta_point_static")
            st.number_input("d_t (m)", min_value=0.01, step=0.01, key="stbd_fso_d_t")
            st.number_input("d_r (m)", min_value=0.01, step=0.01, key="stbd_fso_d_r")
            st.number_input("theta_div (rad)", min_value=1e-8, format="%.8f", key="stbd_fso_theta_div")
            st.number_input("wavelength_m", min_value=1e-9, format="%.9f", key="stbd_fso_wavelength_m")
        with c2:
            st.number_input("t_abs", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_t_abs")
            st.number_input("t_sca", min_value=0.0, max_value=1.0, step=0.01, key="stbd_fso_t_sca")
            st.number_input("c_atm", min_value=0.0, step=0.01, key="stbd_fso_c_atm")
            st.number_input("sigma_r2", min_value=0.0, step=0.01, key="stbd_fso_sigma_r2")
            st.number_input("ao_gain", min_value=0.0, step=0.05, key="stbd_fso_ao_gain")
            st.number_input("turbulence_loss_coeff", min_value=0.0, step=0.01, key="stbd_fso_turbulence_loss_coeff")
            st.number_input("turbulence_quality_coeff", min_value=0.0, step=0.01, key="stbd_fso_turbulence_quality_coeff")
            st.number_input("min_elevation_deg", min_value=0.0, step=1.0, key="stbd_fso_min_elevation_deg")
        with c3:
            st.checkbox("enable_stochastic_fading", key="stbd_fso_enable_stochastic_fading")
            st.checkbox("enable_pointing_jitter", key="stbd_fso_enable_pointing_jitter")
            st.number_input("pointing_jitter_rad", min_value=0.0, format="%.10f", key="stbd_fso_pointing_jitter_rad")
            st.number_input("turbulence_floor", min_value=0.0, format="%.6f", key="stbd_fso_turbulence_floor")
            st.number_input("polarization_error_rad", min_value=0.0, format="%.6f", key="stbd_fso_polarization_error_rad")
            st.number_input("phase_noise_rad", min_value=0.0, format="%.6f", key="stbd_fso_phase_noise_rad")
            st.number_input("temporal_mismatch_s", min_value=0.0, format="%.12f", key="stbd_fso_temporal_mismatch_s")
            st.number_input("coherence_time_s", min_value=1e-15, format="%.12f", key="stbd_fso_coherence_time_s")
            st.number_input("p_t", min_value=0.0, step=0.01, key="stbd_fso_p_t")
            st.number_input("n0", min_value=0.0, format="%.8f", key="stbd_fso_n0")
            st.number_input("n_dc", min_value=0.0, step=1.0, key="stbd_fso_n_dc")
            st.number_input("n_conv", min_value=0.0, step=0.01, key="stbd_fso_n_conv")
            st.number_input("delta_t", min_value=0.0, format="%.12f", key="stbd_fso_delta_t")
            st.number_input("max_snr_for_mapping", min_value=1.0, step=1000.0, key="stbd_fso_max_snr_for_mapping")
        st.markdown("</div>", unsafe_allow_html=True)

    with isl_col:
        st.markdown('<div class="qns-card-title">Inter-Satellite Link (ISL) Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.caption("Uses the same repo FSO model as ground links, but with a separate optical budget and no atmospheric/turbulence impact in the intersatellite path.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("ISL eta_src", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_eta_src")
            st.number_input("ISL eta_det", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_eta_det")
            st.number_input("ISL eta_cpl", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_eta_cpl")
            st.number_input("ISL eta_point_static", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_eta_point_static")
            st.number_input("ISL d_t (m)", min_value=0.01, step=0.01, key="stbd_isl_fso_d_t")
            st.number_input("ISL d_r (m)", min_value=0.01, step=0.01, key="stbd_isl_fso_d_r")
            st.number_input("ISL theta_div (rad)", min_value=1e-8, format="%.8f", key="stbd_isl_fso_theta_div")
            st.number_input("ISL wavelength_m", min_value=1e-9, format="%.9f", key="stbd_isl_fso_wavelength_m")
        with c2:
            st.number_input("ISL t_abs", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_t_abs")
            st.number_input("ISL t_sca", min_value=0.0, max_value=1.0, step=0.01, key="stbd_isl_fso_t_sca")
            st.number_input("ISL c_atm", min_value=0.0, step=0.01, key="stbd_isl_fso_c_atm")
            st.number_input("ISL sigma_r2", min_value=0.0, step=0.01, key="stbd_isl_fso_sigma_r2")
            st.number_input("ISL ao_gain", min_value=0.0, step=0.05, key="stbd_isl_fso_ao_gain")
            st.number_input("ISL turbulence_loss_coeff", min_value=0.0, step=0.01, key="stbd_isl_fso_turbulence_loss_coeff")
            st.number_input("ISL turbulence_quality_coeff", min_value=0.0, step=0.01, key="stbd_isl_fso_turbulence_quality_coeff")
            st.number_input("ISL min_elevation_deg", min_value=0.0, step=1.0, key="stbd_isl_fso_min_elevation_deg")
        with c3:
            st.checkbox("ISL enable_stochastic_fading", key="stbd_isl_fso_enable_stochastic_fading")
            st.checkbox("ISL enable_pointing_jitter", key="stbd_isl_fso_enable_pointing_jitter")
            st.number_input("ISL pointing_jitter_rad", min_value=0.0, format="%.10f", key="stbd_isl_fso_pointing_jitter_rad")
            st.number_input("ISL turbulence_floor", min_value=0.0, format="%.6f", key="stbd_isl_fso_turbulence_floor")
            st.number_input("ISL polarization_error_rad", min_value=0.0, format="%.6f", key="stbd_isl_fso_polarization_error_rad")
            st.number_input("ISL phase_noise_rad", min_value=0.0, format="%.6f", key="stbd_isl_fso_phase_noise_rad")
            st.number_input("ISL temporal_mismatch_s", min_value=0.0, format="%.12f", key="stbd_isl_fso_temporal_mismatch_s")
            st.number_input("ISL coherence_time_s", min_value=1e-15, format="%.12f", key="stbd_isl_fso_coherence_time_s")
            st.number_input("ISL p_t", min_value=0.0, step=0.01, key="stbd_isl_fso_p_t")
            st.number_input("ISL n0", min_value=0.0, format="%.8f", key="stbd_isl_fso_n0")
            st.number_input("ISL n_dc", min_value=0.0, step=1.0, key="stbd_isl_fso_n_dc")
            st.number_input("ISL n_conv", min_value=0.0, step=0.01, key="stbd_isl_fso_n_conv")
            st.number_input("ISL delta_t", min_value=0.0, format="%.12f", key="stbd_isl_fso_delta_t")
            st.number_input("ISL max_snr_for_mapping", min_value=1.0, step=1000.0, key="stbd_isl_fso_max_snr_for_mapping")
        st.markdown("</div>", unsafe_allow_html=True)

    with bsm_col:
        st.markdown('<div class="qns-card-title">BSM / HOM Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
        st.selectbox(
            "swap fidelity model",
            options=["paper", "legacy"],
            format_func=lambda value: "Reference" if value == "paper" else "Legacy",
            key="stbd_bsm_swap_model",
            help="Use the reference swapped-fidelity law or the historical compact proxy.",
        )
        st.number_input("v_hom", min_value=0.0, max_value=1.0, step=0.01, key="stbd_bsm_v_hom")
        st.number_input("eta_det", min_value=0.0, max_value=1.0, step=0.01, key="stbd_bsm_eta_det")
        st.number_input("sigma_det_s", min_value=0.0, format="%.12f", key="stbd_bsm_sigma_det_s")
        st.number_input("delta_t_s", min_value=0.0, format="%.12f", key="stbd_bsm_delta_t_s")
        st.number_input("p_dc", min_value=0.0, format="%.8f", key="stbd_bsm_p_dc")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="qns-card-title">Quantum Memory Parameter Sets</div>', unsafe_allow_html=True)
    st.markdown('<div class="qns-card-shell">', unsafe_allow_html=True)
    tabs = st.tabs(["REID", "COLD_ATOM", "TRAPPED_ION"])
    for tab, key in zip(tabs, ("REID", "COLD_ATOM", "TRAPPED_ION")):
        prefix = f"stbd_mem_{key.lower()}"
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("T2 (s)", min_value=0.001, step=1.0, key=f"{prefix}_t2_s")
                st.number_input("d", min_value=0.0, step=0.5, key=f"{prefix}_d")
                st.number_input("n", min_value=0.1, step=0.1, key=f"{prefix}_n")
                st.number_input("t_mem (s)", min_value=0.001, step=1.0, key=f"{prefix}_t_mem_s")
            with c2:
                st.number_input("m", min_value=0.1, step=0.1, key=f"{prefix}_m")
                st.number_input("eta_mode0", min_value=0.0, max_value=1.0, step=0.01, key=f"{prefix}_eta_mode0")
                st.number_input("eta_ti", min_value=0.0, max_value=1.0, step=0.01, key=f"{prefix}_eta_ti")
                st.number_input("max modes / label", min_value=1, step=1, key=f"{prefix}_max_modes_per_label")
    st.markdown("</div>", unsafe_allow_html=True)

    if generate:
        fso_params = build_fso_params()
        isl_fso_params = build_isl_fso_params()
        bsm_params = build_bsm_params()
        memory_dicts = build_memory_sets()
        memory_sets = {
            key: MemoryParameterSet(**params)
            for key, params in memory_dicts.items()
        }
        run_name = sanitize_run_name(str(st.session_state.stbd_run_name))
        out_root = Path(str(st.session_state.stbd_output_root)).expanduser()
        with st.spinner("Running Luxembourg–Singapore STBD constellation optimization with DES..."):
            result = run_stbd_constellation_optimization(
                run_name=run_name,
                output_root=out_root,
                controller_backend=str(st.session_state.stbd_controller_backend),
                optimizer_method=str(st.session_state.stbd_optimizer_method),
                random_search_trials=int(st.session_state.stbd_random_search_trials),
                ga_population=int(st.session_state.stbd_ga_population),
                ga_generations=int(st.session_state.stbd_ga_generations),
                elite_fraction=float(st.session_state.stbd_elite_fraction),
                mutation_rate=float(st.session_state.stbd_mutation_rate),
                seed=int(st.session_state.stbd_seed),
                pilot_horizon_s=float(st.session_state.stbd_pilot_horizon_s),
                simulation_horizon_s=float(st.session_state.stbd_simulation_horizon_s),
                geometry_step_s=float(st.session_state.stbd_geometry_step_s),
                snapshot_s=float(st.session_state.stbd_snapshot_s),
                launch_step_s=float(st.session_state.stbd_launch_step_s),
                metrics_dt_s=float(st.session_state.stbd_metrics_dt_s),
                gen_delay_s=float(st.session_state.stbd_gen_delay_s),
                swap_dt_s=float(st.session_state.stbd_swap_dt_s),
                paced_delivery_dt_s=float(st.session_state.stbd_paced_delivery_dt_s),
                bsm_round_duration_s=float(st.session_state.stbd_bsm_round_duration_s),
                source_rate_hz=float(st.session_state.stbd_source_rate_hz),
                planner_fidelity_threshold=float(st.session_state.stbd_planner_fidelity_threshold),
                stochastic_link_state=bool(st.session_state.stbd_stochastic_link_state),
                perfect_isl=bool(st.session_state.stbd_perfect_isl),
                support_threshold=int(st.session_state.stbd_support_threshold),
                evaluation_repeats=int(st.session_state.stbd_evaluation_repeats),
                f_min=float(st.session_state.stbd_f_min),
                planner_horizon_override_s=float(st.session_state.stbd_planner_horizon_override_s),
                execution_mode=str(st.session_state.stbd_execution_mode),
                objective_w1=float(st.session_state.stbd_w1),
                objective_w2=float(st.session_state.stbd_w2),
                objective_w3=float(st.session_state.stbd_w3),
                objective_w4=float(st.session_state.stbd_w4),
                objective_w5=float(st.session_state.stbd_w5),
                planes_min=int(st.session_state.stbd_planes_min),
                planes_max=int(st.session_state.stbd_planes_max),
                sats_min=int(st.session_state.stbd_sats_min),
                sats_max=int(st.session_state.stbd_sats_max),
                altitude_min_km=float(st.session_state.stbd_altitude_min_km),
                altitude_max_km=float(st.session_state.stbd_altitude_max_km),
                inclination_min_deg=float(st.session_state.stbd_inclination_min_deg),
                inclination_max_deg=float(st.session_state.stbd_inclination_max_deg),
                raan_spacing_min_deg=float(st.session_state.stbd_raan_spacing_min_deg),
                raan_spacing_max_deg=float(st.session_state.stbd_raan_spacing_max_deg),
                max_isl_range_min_km=float(st.session_state.stbd_isl_range_min_km),
                max_isl_range_max_km=float(st.session_state.stbd_isl_range_max_km),
                min_elevation_min_deg=float(st.session_state.stbd_min_elevation_min_deg),
                min_elevation_max_deg=float(st.session_state.stbd_min_elevation_max_deg),
                enabled_memory_keys=list(st.session_state.stbd_enabled_memory_keys),
                fso_params=fso_params,
                isl_fso_params=isl_fso_params,
                bsm_params=bsm_params,
                memory_sets=memory_sets,
            )
        st.session_state.stbd_generated_out_dir = str(result.out_dir)
        st.success(f"Generated STBD outputs in {result.out_dir}")

    out_dir_str = str(st.session_state.get("stbd_generated_out_dir", "")).strip()
    if out_dir_str:
        out_dir = Path(out_dir_str)
        manifest = load_gallery_manifest(out_dir)
        if manifest:
            st.subheader("Latest STBD Outputs")
            files = manifest.get("files", {})
            if files:
                summary_path = files.get("summary_txt")
                best_config_path = files.get("best_config_json")
                col1, col2 = st.columns(2)
                with col1:
                    if summary_path and Path(summary_path).exists():
                        st.markdown(f"[summary.txt]({summary_path})")
                        st.code(Path(summary_path).read_text(encoding='utf-8')[:1600])
                with col2:
                    if best_config_path and Path(best_config_path).exists():
                        st.markdown(f"[best_config.json]({best_config_path})")
                        st.code(Path(best_config_path).read_text(encoding='utf-8')[:1600], language="json")
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
