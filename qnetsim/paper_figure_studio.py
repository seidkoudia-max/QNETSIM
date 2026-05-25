from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import quote, unquote

import streamlit as st

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channels.fso import FSOAnalyticalChannelModel, FSOParams, efficiency_to_loss_db, werner_to_fidelity
from des_no_sdn_paper_figures import (
    BSMStudyConfig,
    MEMORY_PLATFORM_LABELS,
    MemoryStudyConfig,
    memory_platform_label,
    study_fso_params,
)
from paper_figure_studio_backend import (
    StudioPlotSelection,
    backend_label,
    backend_options,
    default_studio_out_root,
    generate_backend_plots,
)


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "qnetsim_logo.png"
DEFAULT_FSO = study_fso_params(10.0)
DEFAULT_BSM = BSMStudyConfig()
SINGAPORE_LAT_LON = (1.3521, 103.8198)
LUXEMBOURG_LAT_LON = (49.6116, 6.1319)
DEFAULT_MEMORY_PRESETS = {
    "REID": MemoryStudyConfig(
        platform="REID",
        t2_s=100.0,
        d=7.0,
        t_mem_s=10.0,
        n=1.5,
        m=1.0,
        eta_mode0=0.9,
        eta_ti=0.3,
        max_modes_per_label=256,
    ),
    "COLD_ATOM": MemoryStudyConfig(
        platform="COLD_ATOM",
        t2_s=0.05,
        d=20.0,
        t_mem_s=1.0,
        n=1.3,
        m=1.0,
        eta_mode0=0.9,
        eta_ti=0.3,
        max_modes_per_label=256,
    ),
    "TRAPPED_ION": MemoryStudyConfig(
        platform="TRAPPED_ION",
        t2_s=10.0,
        d=7.0,
        t_mem_s=10.0,
        n=1.0,
        m=1.0,
        eta_mode0=0.9,
        eta_ti=0.3,
        max_modes_per_label=256,
    ),
}


st.set_page_config(
    page_title="QNETSIM Single Repeater",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GALLERY_MANIFEST = "gallery_manifest.json"


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
            background: linear-gradient(135deg, #0b1220 0%, #173f8a 52%, #1e66f5 100%);
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
            opacity: 0.9;
            font-size: 0.98rem;
        }
        .qns-card-title {
            background: linear-gradient(180deg, #e53935 0%, #c62828 100%);
            color: white;
            border-radius: 14px 14px 0 0;
            padding: 0.75rem 0.95rem;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.35rem;
            margin-bottom: 0;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.16);
        }
        .qns-card-copy {
            color: #475569;
            font-size: 0.92rem;
            margin: 0.55rem 0 0.9rem;
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
        .stButton > button {
            border: none;
            border-radius: 14px;
            background: linear-gradient(180deg, #e53935 0%, #c62828 100%);
            color: white;
            font-weight: 700;
            padding: 0.65rem 1rem;
            box-shadow: 0 10px 20px rgba(198, 40, 40, 0.18);
        }
        .stButton > button:hover {
            color: white;
            border: none;
            background: linear-gradient(180deg, #ef5350 0%, #d32f2f 100%);
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
            background: linear-gradient(180deg, #2450a6 0%, #2b76ff 100%);
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
        }
        [data-testid="stImage"] img {
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for piece in raw.split(","):
        token = piece.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def sanitize_run_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or datetime.now().strftime("run-%Y%m%d-%H%M%S")


def preset_for_platform(platform: str) -> MemoryStudyConfig:
    return DEFAULT_MEMORY_PRESETS[str(platform).upper()]


def is_viewer_mode() -> bool:
    mode = str(st.query_params.get("mode", "")).strip().lower()
    viewer = str(st.query_params.get("viewer", "")).strip().lower()
    return mode == "viewer" or viewer in {"1", "true", "yes"}


def viewer_out_dir_from_query() -> Path | None:
    raw = str(st.query_params.get("out_dir", "")).strip()
    if not raw:
        return None
    return Path(unquote(raw)).expanduser()


def viewer_url_for_out_dir(out_dir: Path) -> str:
    return f"?mode=viewer&out_dir={quote(str(out_dir.resolve()))}"


def save_gallery_manifest(out_dir: Path, generated: Dict[str, Path]) -> None:
    payload = {
        "out_dir": str(out_dir.resolve()),
        "items": [
            {"title": title, "path": str(Path(path).resolve())}
            for title, path in generated.items()
        ],
    }
    (out_dir / GALLERY_MANIFEST).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def discover_gallery_images(out_dir: Path) -> List[tuple[str, str]]:
    items: List[tuple[str, str]] = []
    for path in sorted(out_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        items.append((path.stem.replace("_", " ").title(), str(path.resolve())))
    return items


def load_gallery_items(out_dir: Path) -> List[tuple[str, str]]:
    manifest = out_dir / GALLERY_MANIFEST
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        items = []
        for item in data.get("items", []):
            title = str(item.get("title", "")).strip()
            path = str(item.get("path", "")).strip()
            if title and path and Path(path).exists():
                items.append((title, path))
        if items:
            return items

    return discover_gallery_images(out_dir)


def apply_sg_lux_reference_preset(repeater_count: int | None = None) -> None:
    count = int(
        repeater_count
        if repeater_count is not None
        else st.session_state.get("studio_sg_lux_repeater_count", 5)
    )
    if count <= 3:
        isl_distance_km = 2800.0
        mem_t2 = 1000.0
        mem_t_mem = 200.0
        altitude_km = 1500.0
    elif count == 5:
        isl_distance_km = 1800.0
        mem_t2 = 600.0
        mem_t_mem = 140.0
        altitude_km = 1200.0
    else:
        isl_distance_km = 1500.0
        mem_t2 = 450.0
        mem_t_mem = 120.0
        altitude_km = 1200.0

    st.session_state.studio_sg_lux_repeater_count = count
    st.session_state.studio_mem_platform = "REID"
    st.session_state.studio_last_mem_platform = "REID"
    st.session_state.studio_mem_t2 = mem_t2
    st.session_state.studio_mem_d = 20.0
    st.session_state.studio_mem_t_mem = mem_t_mem
    st.session_state.studio_mem_n = 1.5
    st.session_state.studio_mem_m = 1.0
    st.session_state.studio_mem_max_modes = 256
    st.session_state.studio_mem_eta_mode0 = 0.9
    st.session_state.studio_mem_eta_ti = 0.3

    st.session_state.studio_des_source_rate_hz = 2e5
    st.session_state.studio_min_elevation_deg = 10.0

    st.session_state.studio_fso_eta_src = 0.95
    st.session_state.studio_fso_eta_det = 0.90
    st.session_state.studio_fso_eta_cpl = 0.90
    st.session_state.studio_fso_eta_point_static = 0.99
    st.session_state.studio_fso_d_t = 0.25
    st.session_state.studio_fso_d_r = 0.80
    st.session_state.studio_fso_theta_div = 6e-6
    st.session_state.studio_fso_t_abs = 0.98
    st.session_state.studio_fso_t_sca = 0.98
    st.session_state.studio_fso_c_atm = 0.02
    st.session_state.studio_fso_sigma_r2 = 0.02
    st.session_state.studio_fso_turbulence_loss_coeff = 0.01
    st.session_state.studio_fso_turbulence_quality_coeff = 0.005
    st.session_state.studio_fso_p_t = 0.95
    st.session_state.studio_fso_n0 = 0.0
    st.session_state.studio_fso_n_dc = 10.0
    st.session_state.studio_fso_n_conv = 0.0
    st.session_state.studio_fso_delta_t = 1e-9
    st.session_state.studio_fso_enable_pointing_jitter = True
    st.session_state.studio_fso_pointing_jitter_rad = 3e-8
    st.session_state.studio_fso_enable_stochastic_fading = True

    st.session_state.studio_isl_eta_src = 0.95
    st.session_state.studio_isl_eta_det = 0.90
    st.session_state.studio_isl_eta_cpl = 0.90
    st.session_state.studio_isl_eta_point_static = 0.99
    st.session_state.studio_isl_d_t = 0.25
    st.session_state.studio_isl_d_r = 0.80
    st.session_state.studio_isl_theta_div = 6e-6
    st.session_state.studio_isl_p_t = 0.95
    st.session_state.studio_isl_n0 = 0.0
    st.session_state.studio_isl_n_dc = 0.0
    st.session_state.studio_isl_n_conv = 0.0
    st.session_state.studio_isl_delta_t = 1e-9
    st.session_state.studio_isl_enable_pointing_jitter = True
    st.session_state.studio_isl_pointing_jitter_rad = 8e-9

    st.session_state.studio_bsm_v_hom = 0.995
    st.session_state.studio_bsm_eta_det = 0.98
    st.session_state.studio_bsm_sigma_det_s = 5e-12
    st.session_state.studio_bsm_delta_t_s = 1e-9
    st.session_state.studio_bsm_p_dc = 0.0

    st.session_state.studio_sg_lux_altitude_km = altitude_km
    st.session_state.studio_sg_lux_half_window_s = 180.0
    st.session_state.studio_sg_lux_dt_s = 5.0
    st.session_state.studio_sg_lux_snapshot_s = 15.0
    st.session_state.studio_sg_lux_velocity_km_s = 7.2
    st.session_state.studio_sg_lux_isl_distance_km = isl_distance_km
    st.session_state.studio_sg_lux_ground_offset_km = 0.0
    st.session_state.studio_sg_lux_isl_variation_fraction = 0.03


def apply_sg_lux_long_window_preset() -> None:
    st.session_state.studio_sg_lux_repeater_count = 6
    st.session_state.studio_mem_platform = "REID"
    st.session_state.studio_last_mem_platform = "REID"
    st.session_state.studio_mem_t2 = 1500.0
    st.session_state.studio_mem_d = 20.0
    st.session_state.studio_mem_t_mem = 1500.0
    st.session_state.studio_mem_n = 1.5
    st.session_state.studio_mem_m = 1.0
    st.session_state.studio_mem_max_modes = 256
    st.session_state.studio_mem_eta_mode0 = 0.9
    st.session_state.studio_mem_eta_ti = 0.3

    st.session_state.studio_des_source_rate_hz = 4e5
    st.session_state.studio_min_elevation_deg = 10.0
    st.session_state.studio_t2_values = "30, 300, 600, 1500"

    st.session_state.studio_fso_eta_src = 0.95
    st.session_state.studio_fso_eta_det = 0.90
    st.session_state.studio_fso_eta_cpl = 0.90
    st.session_state.studio_fso_eta_point_static = 0.99
    st.session_state.studio_fso_d_t = 0.25
    st.session_state.studio_fso_d_r = 0.80
    st.session_state.studio_fso_theta_div = 6e-6
    st.session_state.studio_fso_t_abs = 0.98
    st.session_state.studio_fso_t_sca = 0.98
    st.session_state.studio_fso_c_atm = 0.02
    st.session_state.studio_fso_sigma_r2 = 0.02
    st.session_state.studio_fso_turbulence_loss_coeff = 0.01
    st.session_state.studio_fso_turbulence_quality_coeff = 0.005
    st.session_state.studio_fso_p_t = 0.95
    st.session_state.studio_fso_n0 = 0.0
    st.session_state.studio_fso_n_dc = 10.0
    st.session_state.studio_fso_n_conv = 0.0
    st.session_state.studio_fso_delta_t = 1e-9
    st.session_state.studio_fso_enable_pointing_jitter = True
    st.session_state.studio_fso_pointing_jitter_rad = 3e-8
    st.session_state.studio_fso_enable_stochastic_fading = True

    st.session_state.studio_isl_eta_src = 0.95
    st.session_state.studio_isl_eta_det = 0.90
    st.session_state.studio_isl_eta_cpl = 0.90
    st.session_state.studio_isl_eta_point_static = 0.99
    st.session_state.studio_isl_d_t = 0.25
    st.session_state.studio_isl_d_r = 0.80
    st.session_state.studio_isl_theta_div = 6e-6
    st.session_state.studio_isl_p_t = 0.95
    st.session_state.studio_isl_n0 = 0.0
    st.session_state.studio_isl_n_dc = 0.0
    st.session_state.studio_isl_n_conv = 0.0
    st.session_state.studio_isl_delta_t = 1e-9
    st.session_state.studio_isl_enable_pointing_jitter = True
    st.session_state.studio_isl_pointing_jitter_rad = 8e-9

    st.session_state.studio_bsm_v_hom = 0.995
    st.session_state.studio_bsm_eta_det = 0.98
    st.session_state.studio_bsm_sigma_det_s = 5e-12
    st.session_state.studio_bsm_delta_t_s = 1e-9
    st.session_state.studio_bsm_p_dc = 0.0

    st.session_state.studio_sg_lux_altitude_km = 1600.0
    st.session_state.studio_sg_lux_half_window_s = 360.0
    st.session_state.studio_sg_lux_dt_s = 5.0
    st.session_state.studio_sg_lux_snapshot_s = 60.0
    st.session_state.studio_sg_lux_velocity_km_s = 7.2
    st.session_state.studio_sg_lux_isl_distance_km = 1500.0
    st.session_state.studio_sg_lux_ground_offset_km = 0.0
    st.session_state.studio_sg_lux_isl_variation_fraction = 0.03


def initialize_state() -> None:
    if "studio_mem_platform" not in st.session_state:
        st.session_state.studio_mem_platform = "REID"
    if "studio_last_mem_platform" not in st.session_state:
        st.session_state.studio_last_mem_platform = None

    defaults = {
        "studio_min_elevation_deg": 10.0,
        "studio_source_rate_hz": 5.9e6,
        "studio_des_source_rate_hz": 5e5,
        "studio_ssqr_efficiency": 0.003,
        "studio_annual_contact_fraction": 0.04,
        "studio_des_repeats": 4,
        "studio_sweep_repeats": 3,
        "studio_seed": 12345,
        "studio_backend": "des_only",
        "studio_run_name": datetime.now().strftime("studio-%Y%m%d-%H%M%S"),
        "studio_output_root": str(default_studio_out_root()),
        "studio_t2_values": "0.3, 3, 30, 300",
        "studio_rytov_values": "0.02, 0.08, 0.2, 0.5",
        "studio_paris_vienna_altitudes": "240, 360, 520, 840",
        "studio_plot_representative": True,
        "studio_plot_sweeps": True,
        "studio_plot_annual": True,
        "studio_plot_buffered_annual": True,
        "studio_plot_buffer_altitudes": True,
        "studio_plot_buffer_t2": True,
        "studio_plot_buffer_examples": False,
        "studio_plot_buffer_consumption": False,
        "studio_plot_composite": False,
        "studio_plot_sg_lux_chain": False,
        "studio_sg_lux_repeater_count": 5,
        "studio_sg_lux_altitude_km": 1200.0,
        "studio_sg_lux_half_window_s": 180.0,
        "studio_sg_lux_dt_s": 5.0,
        "studio_sg_lux_snapshot_s": 15.0,
        "studio_sg_lux_velocity_km_s": 7.2,
        "studio_sg_lux_isl_distance_km": 1500.0,
        "studio_sg_lux_ground_offset_km": 0.0,
        "studio_sg_lux_isl_variation_fraction": 0.03,
        "studio_sg_lux_reference_loaded": False,
        "studio_generated_files": {},
        "studio_generated_out_dir": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for key, value in DEFAULT_FSO.__dict__.items():
        st.session_state.setdefault(f"studio_fso_{key}", value)
    for key, value in DEFAULT_BSM.__dict__.items():
        st.session_state.setdefault(f"studio_bsm_{key}", value)

    selected_platform = str(st.session_state.studio_mem_platform).upper()
    if st.session_state.studio_last_mem_platform != selected_platform:
        preset = preset_for_platform(selected_platform)
        st.session_state.studio_mem_t2 = preset.t2_s
        st.session_state.studio_mem_d = preset.d
        st.session_state.studio_mem_t_mem = preset.t_mem_s
        st.session_state.studio_mem_n = preset.n
        st.session_state.studio_mem_m = preset.m
        st.session_state.studio_mem_eta_mode0 = preset.eta_mode0
        st.session_state.studio_mem_eta_ti = preset.eta_ti
        st.session_state.studio_mem_max_modes = preset.max_modes_per_label
        st.session_state.studio_last_mem_platform = selected_platform


def fso_params_from_state() -> FSOParams:
    return FSOParams(
        eta_src=float(st.session_state.studio_fso_eta_src),
        eta_det=float(st.session_state.studio_fso_eta_det),
        eta_cpl=float(st.session_state.studio_fso_eta_cpl),
        eta_point_static=float(st.session_state.studio_fso_eta_point_static),
        d_t=float(st.session_state.studio_fso_d_t),
        d_r=float(st.session_state.studio_fso_d_r),
        theta_div=float(st.session_state.studio_fso_theta_div),
        t_abs=float(st.session_state.studio_fso_t_abs),
        t_sca=float(st.session_state.studio_fso_t_sca),
        c_atm=float(st.session_state.studio_fso_c_atm),
        sigma_r2=float(st.session_state.studio_fso_sigma_r2),
        turbulence_loss_coeff=float(st.session_state.studio_fso_turbulence_loss_coeff),
        turbulence_quality_coeff=float(
            st.session_state.studio_fso_turbulence_quality_coeff
        ),
        p_t=float(st.session_state.studio_fso_p_t),
        n0=float(st.session_state.studio_fso_n0),
        n_dc=float(st.session_state.studio_fso_n_dc),
        n_conv=float(st.session_state.studio_fso_n_conv),
        delta_t=float(st.session_state.studio_fso_delta_t),
        min_elevation_deg=float(st.session_state.studio_min_elevation_deg),
        pointing_jitter_rad=float(st.session_state.studio_fso_pointing_jitter_rad),
        enable_pointing_jitter=bool(st.session_state.studio_fso_enable_pointing_jitter),
        enable_stochastic_fading=bool(
            st.session_state.studio_fso_enable_stochastic_fading
        ),
    )


def memory_config_from_state() -> MemoryStudyConfig:
    return MemoryStudyConfig(
        platform=str(st.session_state.studio_mem_platform).upper(),
        t2_s=float(st.session_state.studio_mem_t2),
        d=float(st.session_state.studio_mem_d),
        t_mem_s=float(st.session_state.studio_mem_t_mem),
        n=float(st.session_state.studio_mem_n),
        m=float(st.session_state.studio_mem_m),
        eta_mode0=float(st.session_state.studio_mem_eta_mode0),
        eta_ti=float(st.session_state.studio_mem_eta_ti),
        max_modes_per_label=int(st.session_state.studio_mem_max_modes),
    )


def bsm_config_from_state() -> BSMStudyConfig:
    return BSMStudyConfig(
        v_hom=float(st.session_state.studio_bsm_v_hom),
        eta_det=float(st.session_state.studio_bsm_eta_det),
        sigma_det_s=float(st.session_state.studio_bsm_sigma_det_s),
        delta_t_s=float(st.session_state.studio_bsm_delta_t_s),
        p_dc=float(st.session_state.studio_bsm_p_dc),
    )


def isl_params_from_state() -> FSOParams:
    return FSOParams(
        eta_src=float(st.session_state.studio_isl_eta_src),
        eta_det=float(st.session_state.studio_isl_eta_det),
        eta_cpl=float(st.session_state.studio_isl_eta_cpl),
        eta_point_static=float(st.session_state.studio_isl_eta_point_static),
        d_t=float(st.session_state.studio_isl_d_t),
        d_r=float(st.session_state.studio_isl_d_r),
        theta_div=float(st.session_state.studio_isl_theta_div),
        t_abs=1.0,
        t_sca=1.0,
        c_atm=0.0,
        sigma_r2=0.0,
        turbulence_loss_coeff=0.0,
        turbulence_quality_coeff=0.0,
        p_t=float(st.session_state.studio_isl_p_t),
        n0=float(st.session_state.studio_isl_n0),
        n_dc=float(st.session_state.studio_isl_n_dc),
        n_conv=float(st.session_state.studio_isl_n_conv),
        delta_t=float(st.session_state.studio_isl_delta_t),
        min_elevation_deg=0.0,
        pointing_jitter_rad=float(st.session_state.studio_isl_pointing_jitter_rad),
        enable_pointing_jitter=bool(st.session_state.studio_isl_enable_pointing_jitter),
        enable_stochastic_fading=False,
    )


def plot_selection_from_state() -> StudioPlotSelection:
    return StudioPlotSelection(
        representative_overpasses=bool(st.session_state.studio_plot_representative),
        archetype_sweeps=bool(st.session_state.studio_plot_sweeps),
        annual_throughput=bool(st.session_state.studio_plot_annual),
        buffered_annual_throughput=bool(st.session_state.studio_plot_buffered_annual),
        paris_vienna_buffer_altitudes=bool(st.session_state.studio_plot_buffer_altitudes),
        paris_vienna_buffer_t2=bool(st.session_state.studio_plot_buffer_t2),
        buffer_examples=bool(st.session_state.studio_plot_buffer_examples),
        buffer_consumption=bool(st.session_state.studio_plot_buffer_consumption),
        overview_composite=bool(st.session_state.studio_plot_composite),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r_km * math.asin(math.sqrt(max(0.0, a)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_singapore_luxembourg_three_repeater_plots(
    out_dir: Path,
    *,
    ground_fso: FSOParams,
    isl_fso: FSOParams,
    memory_config: MemoryStudyConfig,
    bsm_config: BSMStudyConfig,
    des_source_rate_hz: float,
    t2_values_s: List[float],
    rytov_values: List[float],
    altitude_km: float,
    half_window_s: float,
    dt_s: float,
    velocity_km_s: float,
    isl_distance_km: float,
    ground_offset_km: float,
    isl_variation_fraction: float,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}
    total_baseline_km = haversine_km(
        SINGAPORE_LAT_LON[0],
        SINGAPORE_LAT_LON[1],
        LUXEMBOURG_LAT_LON[0],
        LUXEMBOURG_LAT_LON[1],
    )
    times = [
        -float(half_window_s) + i * float(dt_s)
        for i in range(int(round((2.0 * float(half_window_s)) / float(dt_s))) + 1)
    ]

    def build_ground_model(sigma_r2: float) -> FSOAnalyticalChannelModel:
        params = FSOParams(**{**ground_fso.__dict__, "sigma_r2": float(sigma_r2)})
        return FSOAnalyticalChannelModel(params)

    isl_model = FSOAnalyticalChannelModel(isl_fso)
    swap_visibility = _clip01(
        bsm_config.v_hom
        * math.exp(
            -(
                float(bsm_config.sigma_det_s)
                / max(float(bsm_config.delta_t_s), 1e-15)
            )
            ** 2
        )
        * (1.0 - float(bsm_config.p_dc))
    )
    swap_success = _clip01((float(bsm_config.eta_det) ** 2) * swap_visibility)
    ground_eta_ref = max(
        1e-12,
        ground_fso.eta_src
        * ground_fso.eta_det
        * ground_fso.eta_cpl
        * ground_fso.eta_point_static,
    )
    isl_eta_ref = max(
        1e-12,
        isl_fso.eta_src * isl_fso.eta_det * isl_fso.eta_cpl * isl_fso.eta_point_static,
    )

    def hop_series(*, sigma_r2: float, t2_s: float) -> Dict[str, List[float]]:
        ground_model = build_ground_model(sigma_r2)
        losses_gs: List[float] = []
        losses_isl12: List[float] = []
        losses_isl23: List[float] = []
        losses_gl: List[float] = []
        rates: List[float] = []
        fidelities: List[float] = []
        rep1: List[float] = []
        rep2: List[float] = []
        rep3: List[float] = []
        b1 = b2 = b3 = 0.0
        cap = float(memory_config.max_modes_per_label)
        decay = math.exp(-float(dt_s) / max(float(t2_s), 1e-9))
        for t in times:
            ground_horiz = math.sqrt((float(velocity_km_s) * t) ** 2 + float(ground_offset_km) ** 2)
            slant_ground_m = 1e3 * math.sqrt(float(altitude_km) ** 2 + ground_horiz**2)
            elev_deg = math.degrees(
                math.atan2(float(altitude_km), max(1e-9, ground_horiz))
            )
            isl12_km = float(isl_distance_km) * (
                1.0 + float(isl_variation_fraction) * math.sin(t / 90.0)
            )
            isl23_km = float(isl_distance_km) * (
                1.0 + float(isl_variation_fraction) * math.cos(t / 110.0)
            )

            eta_gs = ground_model.mean_efficiency(
                distance_m=slant_ground_m,
                elevation_deg=elev_deg,
                intersatellite=False,
            )
            eta_gl = ground_model.mean_efficiency(
                distance_m=slant_ground_m,
                elevation_deg=elev_deg,
                intersatellite=False,
            )
            eta_isl12 = isl_model.mean_efficiency(
                distance_m=1e3 * isl12_km,
                elevation_deg=90.0,
                intersatellite=True,
            )
            eta_isl23 = isl_model.mean_efficiency(
                distance_m=1e3 * isl23_km,
                elevation_deg=90.0,
                intersatellite=True,
            )

            losses_gs.append(
                ground_model.mean_loss_db(
                    distance_m=slant_ground_m,
                    elevation_deg=elev_deg,
                    intersatellite=False,
                )
            )
            losses_gl.append(
                ground_model.mean_loss_db(
                    distance_m=slant_ground_m,
                    elevation_deg=elev_deg,
                    intersatellite=False,
                )
            )
            losses_isl12.append(
                isl_model.mean_loss_db(
                    distance_m=1e3 * isl12_km,
                    elevation_deg=90.0,
                    intersatellite=True,
                )
            )
            losses_isl23.append(
                isl_model.mean_loss_db(
                    distance_m=1e3 * isl23_km,
                    elevation_deg=90.0,
                    intersatellite=True,
                )
            )

            hop_etas = [eta_gs, eta_isl12, eta_isl23, eta_gl]
            hop_rates = [float(des_source_rate_hz) * eta for eta in hop_etas]
            wait_s = 0.5 + 0.06 * abs(t)
            memory_factor = math.exp(-wait_s / max(float(t2_s), 1e-9))
            chain_rate = min(hop_rates) * (swap_success**3) * memory_factor
            rates.append(chain_rate)

            hop_werners = [
                math.sqrt(_clip01(eta_gs / ground_eta_ref)),
                math.sqrt(_clip01(eta_isl12 / isl_eta_ref)),
                math.sqrt(_clip01(eta_isl23 / isl_eta_ref)),
                math.sqrt(_clip01(eta_gl / ground_eta_ref)),
            ]
            chain_werner = _clip01(
                hop_werners[0]
                * hop_werners[1]
                * hop_werners[2]
                * hop_werners[3]
                * (swap_visibility**3)
                * memory_factor
            )
            fidelities.append(werner_to_fidelity(chain_werner))

            scaled = [cap * 0.18 * _clip01(eta / max(1e-12, max(hop_etas))) for eta in hop_etas]
            service1 = 0.28 * min(b1 + scaled[0], scaled[1] + b1)
            service2 = 0.28 * min(b2 + scaled[1], scaled[2] + b2)
            service3 = 0.28 * min(b3 + scaled[2], scaled[3] + b3)
            b1 = min(cap, max(0.0, b1 * decay + abs(scaled[0] - scaled[1]) - service1))
            b2 = min(cap, max(0.0, b2 * decay + abs(scaled[1] - scaled[2]) - service2))
            b3 = min(cap, max(0.0, b3 * decay + abs(scaled[2] - scaled[3]) - service3))
            rep1.append(b1)
            rep2.append(b2)
            rep3.append(b3)

        return {
            "loss_ground_left_db": losses_gs,
            "loss_isl12_db": losses_isl12,
            "loss_isl23_db": losses_isl23,
            "loss_ground_right_db": losses_gl,
            "rate": rates,
            "fidelity": fidelities,
            "buffer_r1": rep1,
            "buffer_r2": rep2,
            "buffer_r3": rep3,
        }

    base = hop_series(sigma_r2=ground_fso.sigma_r2, t2_s=memory_config.t2_s)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.8), sharex=True)
    axes[0].plot(times, base["loss_ground_left_db"], color="#c62828", label="Singapore–R1")
    axes[0].plot(times, base["loss_isl12_db"], color="#1565c0", linestyle="--", label="R1–R2 ISL")
    axes[0].plot(times, base["loss_isl23_db"], color="#00897b", linestyle="-.", label="R2–R3 ISL")
    axes[0].plot(times, base["loss_ground_right_db"], color="#6a1b9a", label="R3–Luxembourg")
    axes[0].set_ylabel("Link loss (dB)")
    axes[0].set_title(
        "Singapore–Luxembourg 3-Repeater Corridor: Hop Losses\n"
        f"3 satellite repeaters, altitude = {altitude_km:.0f} km, total ground baseline = {total_baseline_km:.0f} km"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, base["rate"], color="#f57c00", label="End-to-end mean rate")
    axes[1].plot(times, [100.0 * f for f in base["fidelity"]], color="#263238", label="Median fidelity × 100")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Chain performance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    loss_path = out_dir / "singapore_luxembourg_three_repeater_link_budget.png"
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)
    generated["Singapore–Luxembourg Link Budget"] = loss_path

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharex=True)
    t2_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    rytov_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    for color, t2_s in zip(t2_palette, t2_values_s):
        series = hop_series(sigma_r2=ground_fso.sigma_r2, t2_s=float(t2_s))
        axes[0, 0].plot(times, series["rate"], color=color, label=rf"$T_2$ = {float(t2_s):g} s")
        axes[1, 0].plot(times, series["fidelity"], color=color, label=rf"$T_2$ = {float(t2_s):g} s")
    for color, sigma_r2 in zip(rytov_palette, rytov_values):
        series = hop_series(sigma_r2=float(sigma_r2), t2_s=memory_config.t2_s)
        axes[0, 1].plot(times, series["rate"], color=color, label=rf"$\sigma_R^2$ = {float(sigma_r2):g}")
        axes[1, 1].plot(times, series["fidelity"], color=color, label=rf"$\sigma_R^2$ = {float(sigma_r2):g}")
    axes[0, 0].set_title("Mean rate with memory $T_2$ overlays")
    axes[0, 1].set_title("Mean rate with ground-turbulence overlays")
    axes[1, 0].set_title("Median fidelity with memory $T_2$ overlays")
    axes[1, 1].set_title("Median fidelity with ground-turbulence overlays")
    for ax in axes[0]:
        ax.set_ylabel("Mean rate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    for ax in axes[1]:
        ax.set_ylabel("Median fidelity")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.25, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        "Singapore–Luxembourg 3-Repeater Corridor Sweeps\n"
        "Ground hops use the GUI FSO model; ISLs reuse optical pointing and detection terms with turbulence disabled"
    )
    fig.tight_layout()
    sweep_path = out_dir / "singapore_luxembourg_three_repeater_rate_fidelity_sweeps.png"
    fig.savefig(sweep_path, dpi=180)
    plt.close(fig)
    generated["Singapore–Luxembourg Rate/Fidelity Sweeps"] = sweep_path

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.plot(times, base["buffer_r1"], color="#c62828", label="Repeater R1 buffer")
    ax.plot(times, base["buffer_r2"], color="#1565c0", label="Repeater R2 buffer")
    ax.plot(times, base["buffer_r3"], color="#00897b", label="Repeater R3 buffer")
    ax.axhline(memory_config.max_modes_per_label, color="black", linestyle=":", label="Per-repeater capacity")
    ax.set_title(
        "Singapore–Luxembourg 3-Repeater Corridor: Buffer Stress Proxy\n"
        f"Memory platform = {memory_platform_label(memory_config.platform)}, capacity = {memory_config.max_modes_per_label} modes"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Buffered pairs (proxy)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    buffer_path = out_dir / "singapore_luxembourg_three_repeater_buffers.png"
    fig.savefig(buffer_path, dpi=180)
    plt.close(fig)
    generated["Singapore–Luxembourg Buffer Stress"] = buffer_path
    return generated


def render_hero() -> None:
    left, right = st.columns([1, 7], vertical_alignment="center")
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=200)
    with right:
        st.markdown(
            """
            <div class="qns-hero">
              <h1>QNETSIM Single Repeater</h1>
              <p>
                Explore the single-satellite quantum repeater scenario connecting two
                optical ground stations through one spaceborne repeater with repository
                FSO, memory, and BSM models.
              </p>
              <div class="qns-chip-row">
                <span class="qns-chip">Single Repeater</span>
                <span class="qns-chip">DES backend</span>
                <span class="qns-chip">Overpass geometry</span>
                <span class="qns-chip">Archetype + annual + buffer views</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_card_title(title: str, body: str) -> None:
    st.markdown(f'<div class="qns-card-title">{title}</div>', unsafe_allow_html=True)
    st.caption(body)


def render_sidebar() -> None:
    with st.sidebar:
        st.caption("Scenario: Single Repeater")
        st.subheader("Run Controls")
        st.radio(
            "Simulation backend",
            options=backend_options(),
            key="studio_backend",
            horizontal=False,
            format_func=backend_label,
            help="Choose whether the figure pipeline is controller-free DES, SDN with the in-process repeater controller, or SDN through the standalone Ryu controller.",
        )
        st.text_input("Run name", key="studio_run_name")
        st.text_input("Output root", key="studio_output_root")
        st.number_input("Random seed", key="studio_seed", min_value=1, step=1)
        st.number_input("DES repeats", key="studio_des_repeats", min_value=1, max_value=24, step=1)
        st.number_input(
            "Sweep repeats",
            key="studio_sweep_repeats",
            min_value=1,
            max_value=16,
            step=1,
            help="Higher values smooth the archetype sweep plots.",
        )
        st.number_input(
            "Source rate for analytical traces (pairs/s)",
            key="studio_source_rate_hz",
            min_value=1.0,
            format="%.6g",
        )
        st.number_input(
            "Source rate for DES runs (pairs/s)",
            key="studio_des_source_rate_hz",
            min_value=1.0,
            format="%.6g",
        )
        st.number_input(
            "SSQR efficiency",
            key="studio_ssqr_efficiency",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
        )
        st.number_input(
            "Annual contact fraction",
            key="studio_annual_contact_fraction",
            min_value=0.0,
            max_value=1.0,
            step=0.005,
            format="%.4f",
        )
        st.divider()
        st.subheader("Plot Selection")
        st.checkbox("Representative overpass figures", key="studio_plot_representative")
        st.checkbox("Archetype T2 / Rytov sweeps", key="studio_plot_sweeps")
        st.checkbox("Optimized annual throughput", key="studio_plot_annual")
        st.checkbox("Buffered annual throughput", key="studio_plot_buffered_annual")
        st.checkbox("Paris-Vienna buffer vs altitude", key="studio_plot_buffer_altitudes")
        st.checkbox("Paris-Vienna buffer vs T2", key="studio_plot_buffer_t2")
        st.checkbox("Buffer pass examples", key="studio_plot_buffer_examples")
        st.checkbox("Buffer consumption summary", key="studio_plot_buffer_consumption")
        st.checkbox("Representative composite", key="studio_plot_composite")
        st.divider()
        st.subheader("Sweep Lists")
        st.text_input("T2 sweep values (s)", key="studio_t2_values")
        st.text_input("Rytov sweep values", key="studio_rytov_values")
        st.text_input("Paris-Vienna panel altitudes (km)", key="studio_paris_vienna_altitudes")


def render_fso_card() -> None:
    render_card_title(
        "Quantum Channel (FSO) Parameters",
        "Repository FSO knobs used by the analytical link model and the DES-backed sweeps.",
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.slider(r"$\eta_{src}$", 0.0, 1.0, key="studio_fso_eta_src")
            st.slider(r"$\eta_{det}$", 0.0, 1.0, key="studio_fso_eta_det")
            st.slider(r"$\eta_{cpl}$", 0.0, 1.0, key="studio_fso_eta_cpl")
            st.slider(r"$\eta_{point}$", 0.0, 1.0, key="studio_fso_eta_point_static")
            st.number_input(r"$d_t$ (m)", key="studio_fso_d_t", min_value=0.01, format="%.6g")
            st.number_input(r"$d_r$ (m)", key="studio_fso_d_r", min_value=0.01, format="%.6g")
            st.number_input(
                r"$\theta_{div}$ (rad)",
                key="studio_fso_theta_div",
                min_value=1e-9,
                format="%.6g",
            )
            st.number_input(r"$\theta_{min}$ (deg)", key="studio_min_elevation_deg", min_value=0.0, max_value=89.0, format="%.3f")
        with c2:
            st.number_input(r"$t_{abs}$", key="studio_fso_t_abs", min_value=0.0, max_value=1.0, format="%.6g")
            st.number_input(r"$t_{sca}$", key="studio_fso_t_sca", min_value=0.0, max_value=1.0, format="%.6g")
            st.number_input(r"$c_{atm}$", key="studio_fso_c_atm", min_value=0.0, format="%.6g")
            st.number_input(r"$\sigma_R^2$", key="studio_fso_sigma_r2", min_value=0.0, format="%.6g")
            st.number_input("Turbulence loss coefficient", key="studio_fso_turbulence_loss_coeff", min_value=0.0, format="%.6g")
            st.number_input("Turbulence quality coefficient", key="studio_fso_turbulence_quality_coeff", min_value=0.0, format="%.6g")
            st.number_input(r"$p_t$", key="studio_fso_p_t", min_value=0.0, format="%.6g")
            st.number_input(r"$n_0$", key="studio_fso_n0", min_value=0.0, format="%.6g")
            st.number_input(r"$n_{dc}$ (Hz)", key="studio_fso_n_dc", min_value=0.0, format="%.6g")
            st.number_input(r"$n_{conv}$", key="studio_fso_n_conv", min_value=0.0, format="%.6g")
            st.number_input(r"$\Delta t$ (s)", key="studio_fso_delta_t", min_value=1e-12, format="%.6g")
        with st.expander("Advanced channel knobs", expanded=False):
            a1, a2 = st.columns(2)
            with a1:
                st.checkbox("Enable pointing jitter", key="studio_fso_enable_pointing_jitter")
                st.number_input(
                    "Pointing jitter (rad)",
                    key="studio_fso_pointing_jitter_rad",
                    min_value=0.0,
                    format="%.6g",
                )
            with a2:
                st.checkbox("Enable stochastic fading", key="studio_fso_enable_stochastic_fading")


def render_memory_card() -> None:
    render_card_title(
        "Quantum Memory Parameters",
        "Choose the platform and tune the corresponding repository memory model.",
    )
    with st.container(border=True):
        st.radio(
            "Memory platform",
            options=list(MEMORY_PLATFORM_LABELS.keys()),
            key="studio_mem_platform",
            horizontal=True,
            format_func=memory_platform_label,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(r"$T_2$ (s)", key="studio_mem_t2", min_value=1e-6, format="%.6g")
            st.number_input("Capacity per link", key="studio_mem_max_modes", min_value=1, step=1)
            platform = str(st.session_state.studio_mem_platform).upper()
            if platform != "TRAPPED_ION":
                st.number_input(r"$d$", key="studio_mem_d", min_value=0.0, format="%.6g")
                st.number_input(r"$n$", key="studio_mem_n", min_value=0.1, format="%.6g")
                st.number_input(r"$t_{mem}$ (s)", key="studio_mem_t_mem", min_value=1e-6, format="%.6g")
                st.number_input(r"$m$", key="studio_mem_m", min_value=0.1, format="%.6g")
            else:
                st.number_input(r"$\eta_{ti}$", key="studio_mem_eta_ti", min_value=0.0, max_value=1.0, format="%.6g")
        with c2:
            platform = str(st.session_state.studio_mem_platform).upper()
            if platform == "COLD_ATOM":
                st.number_input(r"$\eta_{mode0}$", key="studio_mem_eta_mode0", min_value=0.0, max_value=1.0, format="%.6g")
                st.info("Cold-atom runs use the EIT + mode-envelope memory model from the repository.")
            elif platform == "REID":
                st.info("REID runs use the CRIB + dynamical-decoupling memory model with the time-dependent retrieval envelope.")
            else:
                st.info("Trapped-ion runs use the exponential coherence / recall model with ion-photon efficiency.")


def render_bsm_card() -> None:
    render_card_title(
        "BSM (HOM) Parameters",
        "Bell-state measurement controls that shape swap success and swapped fidelity.",
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(r"$v_{hom}$", key="studio_bsm_v_hom", min_value=0.0, max_value=1.0, format="%.6g")
            st.number_input(r"$\eta^{BSM}_{det}$", key="studio_bsm_eta_det", min_value=0.0, max_value=1.0, format="%.6g")
            st.number_input(r"$\sigma_{det}$ (s)", key="studio_bsm_sigma_det_s", min_value=1e-15, format="%.6g")
        with c2:
            st.number_input(r"$\Delta t_{BSM}$ (s)", key="studio_bsm_delta_t_s", min_value=1e-15, format="%.6g")
            st.number_input(r"$p_{dc}$", key="studio_bsm_p_dc", min_value=0.0, max_value=1.0, format="%.6g")


def render_isl_card() -> None:
    render_card_title(
        "Inter-Satellite Link (ISL) Parameters",
        "Optical ISL module for the Singapore–Luxembourg repeater corridor. It reuses pointing and optics terms while disabling atmospheric turbulence.",
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.slider(r"$\eta_{src}^{ISL}$", 0.0, 1.0, key="studio_isl_eta_src")
            st.slider(r"$\eta_{det}^{ISL}$", 0.0, 1.0, key="studio_isl_eta_det")
            st.slider(r"$\eta_{cpl}^{ISL}$", 0.0, 1.0, key="studio_isl_eta_cpl")
            st.slider(r"$\eta_{point}^{ISL}$", 0.0, 1.0, key="studio_isl_eta_point_static")
            st.number_input(r"$d_t^{ISL}$ (m)", key="studio_isl_d_t", min_value=0.01, format="%.6g")
            st.number_input(r"$d_r^{ISL}$ (m)", key="studio_isl_d_r", min_value=0.01, format="%.6g")
            st.number_input(
                r"$\theta_{div}^{ISL}$ (rad)",
                key="studio_isl_theta_div",
                min_value=1e-9,
                format="%.6g",
            )
        with c2:
            st.number_input(r"$p_t^{ISL}$", key="studio_isl_p_t", min_value=0.0, format="%.6g")
            st.number_input(r"$n_0^{ISL}$", key="studio_isl_n0", min_value=0.0, format="%.6g")
            st.number_input(r"$n_{dc}^{ISL}$ (Hz)", key="studio_isl_n_dc", min_value=0.0, format="%.6g")
            st.number_input(r"$n_{conv}^{ISL}$", key="studio_isl_n_conv", min_value=0.0, format="%.6g")
            st.number_input(r"$\Delta t^{ISL}$ (s)", key="studio_isl_delta_t", min_value=1e-12, format="%.6g")
            st.checkbox("Enable ISL pointing jitter", key="studio_isl_enable_pointing_jitter")
            st.number_input(
                "ISL pointing jitter (rad)",
                key="studio_isl_pointing_jitter_rad",
                min_value=0.0,
                format="%.6g",
            )
        st.info(
            "The ISL module keeps optical efficiencies, apertures, divergence, pointing, and noise terms, but sets atmospheric absorption, scattering, and turbulence effects to zero. The corridor geometry now uses a spherical-Earth satellite train with Earth-occulted ISLs rather than always-visible sinusoidal links."
        )


def validate_inputs(
    t2_values: List[float],
    rytov_values: List[float],
    altitude_values: List[float],
    selection: StudioPlotSelection,
) -> None:
    if selection.paris_vienna_buffer_t2 and len(t2_values) != 4:
        raise ValueError("Paris-Vienna buffer-vs-T2 uses a 2x2 panel and expects exactly 4 T2 values.")
    if selection.paris_vienna_buffer_altitudes and len(altitude_values) != 4:
        raise ValueError(
            "Paris-Vienna buffer-vs-altitude uses a 2x2 panel and expects exactly 4 altitude values."
        )
    if selection.archetype_sweeps and len(rytov_values) == 0:
        raise ValueError("Please provide at least one Rytov value for the archetype sweeps.")
    if selection.archetype_sweeps and len(t2_values) == 0:
        raise ValueError("Please provide at least one T2 value for the archetype sweeps.")


def render_viewer_launch() -> None:
    files = st.session_state.get("studio_generated_files", {})
    out_dir = st.session_state.get("studio_generated_out_dir", "")
    if not files:
        st.info("Generate a study, then open the separate viewer window for the plots.")
        return

    st.success(f"Plots written to {out_dir}")
    viewer_url = viewer_url_for_out_dir(Path(out_dir))
    st.markdown(
        f'<a class="qns-link-button" href="{viewer_url}" target="_blank" rel="noopener noreferrer">Open Plot Viewer</a>',
        unsafe_allow_html=True,
    )
    st.caption("The viewer opens in a separate browser window or tab and shows the generated Single Repeater plots without the parameter controls.")


def render_gallery_items(items: List[tuple[str, str]]) -> None:
    if not items:
        st.info("No images were found in this output directory yet.")
        return

    st.success(f"Loaded {len(items)} plot files.")
    for index in range(0, len(items), 2):
        columns = st.columns(2)
        for col, (title, path) in zip(columns, items[index : index + 2]):
            with col:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)


def render_viewer_page() -> None:
    inject_theme()
    out_dir = viewer_out_dir_from_query()
    st.markdown("## QNETSIM Single Repeater Plot Viewer")
    if out_dir is None:
        st.warning("No output directory was provided to the viewer.")
        return
    if not out_dir.exists():
        st.error(f"Output directory not found: {out_dir}")
        return

    st.caption(f"Showing plots from `{out_dir}`")
    items = load_gallery_items(out_dir)
    render_gallery_items(items)


def render_inline_gallery() -> None:
    files = st.session_state.get("studio_generated_files", {})
    if not files:
        return
    items = list(files.items())
    for index in range(0, len(items), 2):
        columns = st.columns(2)
        for col, (title, path) in zip(columns, items[index : index + 2]):
            with col:
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)


def main() -> None:
    if is_viewer_mode():
        render_viewer_page()
        return

    inject_theme()
    initialize_state()
    render_hero()
    render_sidebar()

    left, right = st.columns([1.45, 1.0], vertical_alignment="top")
    with left:
        render_fso_card()
    with right:
        render_memory_card()
        render_bsm_card()

    st.divider()
    st.subheader("Generate Single Repeater Plot Families")
    st.caption(
        "The app writes the generated figures and CSV companions into a timestamped output folder under the selected root."
    )

    if st.button("Generate Plots", use_container_width=True):
        try:
            t2_values = parse_float_list(str(st.session_state.studio_t2_values))
            rytov_values = parse_float_list(str(st.session_state.studio_rytov_values))
            altitude_values = parse_float_list(
                str(st.session_state.studio_paris_vienna_altitudes)
            )
            selection = plot_selection_from_state()
            validate_inputs(
                t2_values,
                rytov_values,
                altitude_values,
                selection,
            )

            out_root = Path(str(st.session_state.studio_output_root)).expanduser().resolve()
            run_name = sanitize_run_name(str(st.session_state.studio_run_name))
            out_dir = out_root / run_name / str(st.session_state.studio_backend)

            progress = st.progress(10, text="Preparing the DES figure study…")
            generated = generate_backend_plots(
                out_dir,
                backend_key=str(st.session_state.studio_backend),
                selection=selection,
                fso_params=fso_params_from_state(),
                memory_config=memory_config_from_state(),
                bsm_config=bsm_config_from_state(),
                source_rate_hz=float(st.session_state.studio_source_rate_hz),
                des_source_rate_hz=float(st.session_state.studio_des_source_rate_hz),
                ssqr_efficiency=float(st.session_state.studio_ssqr_efficiency),
                min_elevation_deg=float(st.session_state.studio_min_elevation_deg),
                des_repeats=int(st.session_state.studio_des_repeats),
                sweep_repeats=int(st.session_state.studio_sweep_repeats),
                seed=int(st.session_state.studio_seed),
                annual_contact_fraction=float(
                    st.session_state.studio_annual_contact_fraction
                ),
                t2_values_s=t2_values,
                rytov_values=rytov_values,
                paris_vienna_altitudes_km=altitude_values,
            )
            progress.progress(100, text="Plots ready.")
            st.session_state.studio_generated_files = {
                title: str(path) for title, path in generated.items()
            }
            st.session_state.studio_generated_out_dir = str(out_dir)
            save_gallery_manifest(out_dir, {title: Path(path) for title, path in generated.items()})
            st.rerun()
        except Exception as exc:
            st.error(f"Plot generation failed: {exc}")

    render_viewer_launch()
    with st.expander("Inline preview", expanded=False):
        render_inline_gallery()


if __name__ == "__main__":
    main()
