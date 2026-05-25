from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List

from des_stbd_physics_simulator import build_parser as build_des_parser
from mobility.geometry import GeometryEngine
from mobility.propagator import GroundStation, SkyfieldPropagator

from .config import BSMParameterSet, MemoryParameterSet, StbdConstellationConfig
from .constellation_geometry import ConstellationProposal, build_constellation_ephemerides


DEFAULT_EPOCH_UNIX_S = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp()


@dataclass(frozen=True)
class ScenarioBundle:
    proposal: ConstellationProposal
    args: argparse.Namespace
    satellites: List
    ground_nodes: List[GroundStation]
    geometry: GeometryEngine
    src_name: str
    dst_name: str
    node_names: List[str]
    out_dir: Path


def _base_args() -> argparse.Namespace:
    parser = build_des_parser()
    return parser.parse_args([])


def _build_ground_nodes(
    cfg: StbdConstellationConfig, *, min_elevation_deg: float
) -> List[GroundStation]:
    return [
        GroundStation(
            cfg.ground_luxembourg.name,
            lat_deg=cfg.ground_luxembourg.latitude_deg,
            lon_deg=cfg.ground_luxembourg.longitude_deg,
            alt_m=cfg.ground_luxembourg.altitude_m,
            min_elevation_deg=float(min_elevation_deg),
        ),
        GroundStation(
            cfg.ground_singapore.name,
            lat_deg=cfg.ground_singapore.latitude_deg,
            lon_deg=cfg.ground_singapore.longitude_deg,
            alt_m=cfg.ground_singapore.altitude_m,
            min_elevation_deg=float(min_elevation_deg),
        ),
    ]


def _apply_physical_sets(
    args: argparse.Namespace,
    *,
    fso_params,
    isl_fso_params,
    memory_params: MemoryParameterSet,
    bsm_params: BSMParameterSet,
) -> None:
    args.fso_eta_src = float(fso_params.eta_src)
    args.fso_eta_det = float(fso_params.eta_det)
    args.fso_eta_cpl = float(fso_params.eta_cpl)
    args.fso_eta_point = float(fso_params.eta_point_static)
    args.fso_d_t = float(fso_params.d_t)
    args.fso_d_r = float(fso_params.d_r)
    args.fso_theta_div = float(fso_params.theta_div)
    args.fso_wavelength_m = float(fso_params.wavelength_m)
    args.fso_t_abs = float(fso_params.t_abs)
    args.fso_t_sca = float(fso_params.t_sca)
    args.fso_c_atm = float(fso_params.c_atm)
    args.fso_sigma_r2 = float(fso_params.sigma_r2)
    args.fso_ao_gain = float(fso_params.ao_gain)
    args.fso_turbulence_loss_coeff = float(fso_params.turbulence_loss_coeff)
    args.fso_turbulence_quality_coeff = float(fso_params.turbulence_quality_coeff)
    args.fso_delta_t = float(fso_params.delta_t)
    args.fso_n0 = float(fso_params.n0)
    args.fso_n_dc = float(fso_params.n_dc)
    args.fso_n_conv = float(fso_params.n_conv)
    args.fso_p_t = float(fso_params.p_t)
    args.fso_pointing_jitter_rad = float(fso_params.pointing_jitter_rad)
    args.disable_fso_pointing_jitter = not bool(fso_params.enable_pointing_jitter)
    args.disable_fso_stochastic_fading = not bool(fso_params.enable_stochastic_fading)
    args.min_elevation_deg = float(fso_params.min_elevation_deg)
    args.fso_min_elevation_deg = float(fso_params.min_elevation_deg)
    args.fso_turbulence_floor = float(fso_params.turbulence_floor)
    args.fso_polarization_error_rad = float(fso_params.polarization_error_rad)
    args.fso_phase_noise_rad = float(fso_params.phase_noise_rad)
    args.fso_temporal_mismatch_s = float(fso_params.temporal_mismatch_s)
    args.fso_coherence_time_s = float(fso_params.coherence_time_s)
    args.fso_max_snr_for_mapping = float(fso_params.max_snr_for_mapping)
    if isl_fso_params is not None:
        args.isl_fso_eta_src = float(isl_fso_params.eta_src)
        args.isl_fso_eta_det = float(isl_fso_params.eta_det)
        args.isl_fso_eta_cpl = float(isl_fso_params.eta_cpl)
        args.isl_fso_eta_point = float(isl_fso_params.eta_point_static)
        args.isl_fso_d_t = float(isl_fso_params.d_t)
        args.isl_fso_d_r = float(isl_fso_params.d_r)
        args.isl_fso_theta_div = float(isl_fso_params.theta_div)
        args.isl_fso_wavelength_m = float(isl_fso_params.wavelength_m)
        args.isl_fso_t_abs = float(isl_fso_params.t_abs)
        args.isl_fso_t_sca = float(isl_fso_params.t_sca)
        args.isl_fso_c_atm = float(isl_fso_params.c_atm)
        args.isl_fso_sigma_r2 = float(isl_fso_params.sigma_r2)
        args.isl_fso_ao_gain = float(isl_fso_params.ao_gain)
        args.isl_fso_turbulence_loss_coeff = float(
            isl_fso_params.turbulence_loss_coeff
        )
        args.isl_fso_turbulence_quality_coeff = float(
            isl_fso_params.turbulence_quality_coeff
        )
        args.isl_fso_delta_t = float(isl_fso_params.delta_t)
        args.isl_fso_n0 = float(isl_fso_params.n0)
        args.isl_fso_n_dc = float(isl_fso_params.n_dc)
        args.isl_fso_n_conv = float(isl_fso_params.n_conv)
        args.isl_fso_p_t = float(isl_fso_params.p_t)
        args.isl_fso_pointing_jitter_rad = float(isl_fso_params.pointing_jitter_rad)
        args.disable_isl_fso_pointing_jitter = not bool(
            isl_fso_params.enable_pointing_jitter
        )
        args.disable_isl_fso_stochastic_fading = not bool(
            isl_fso_params.enable_stochastic_fading
        )
        args.isl_fso_min_elevation_deg = float(isl_fso_params.min_elevation_deg)
        args.isl_fso_turbulence_floor = float(isl_fso_params.turbulence_floor)
        args.isl_fso_polarization_error_rad = float(
            isl_fso_params.polarization_error_rad
        )
        args.isl_fso_phase_noise_rad = float(isl_fso_params.phase_noise_rad)
        args.isl_fso_temporal_mismatch_s = float(isl_fso_params.temporal_mismatch_s)
        args.isl_fso_coherence_time_s = float(isl_fso_params.coherence_time_s)
        args.isl_fso_max_snr_for_mapping = float(isl_fso_params.max_snr_for_mapping)

    args.mem_platform = str(memory_params.platform).upper()
    args.mem_t2 = float(memory_params.t2_s)
    args.mem_d = float(memory_params.d)
    args.mem_t_mem = float(memory_params.t_mem_s)
    args.mem_n = float(memory_params.n)
    args.mem_m = float(memory_params.m)
    args.mem_eta_mode0 = float(memory_params.eta_mode0)
    args.mem_eta_ti = float(memory_params.eta_ti)
    args.max_modes_per_label = int(memory_params.max_modes_per_label)

    args.bsm_v_hom = float(bsm_params.v_hom)
    args.bsm_eta_det = float(bsm_params.eta_det)
    args.bsm_sigma_det_s = float(bsm_params.sigma_det_s)
    args.bsm_delta_t_s = float(bsm_params.delta_t_s)
    args.bsm_p_dc = float(bsm_params.p_dc)
    args.bsm_swap_model = str(getattr(bsm_params, "swap_model", "paper"))


def build_scenario(
    cfg: StbdConstellationConfig,
    proposal: ConstellationProposal,
    *,
    out_dir: Path,
    horizon_s: float | None = None,
    epoch_unix_s: float = DEFAULT_EPOCH_UNIX_S,
) -> ScenarioBundle:
    planner_coherence_s = float(cfg.memory_sets[proposal.memory_key].t2_s)
    planner_horizon_s = (
        float(cfg.planner_horizon_override_s)
        if cfg.planner_horizon_override_s is not None
        else float(planner_coherence_s)
    )
    args = _base_args()
    args.scenario = "ground-ground"
    args.constellation_source = "tle"
    args.tle = None
    args.omm_json = None
    args.num_sats = int(proposal.total_satellites)
    args.satellite_selection = str(cfg.satellite_selection)
    args.walker_planes = int(proposal.planes)
    args.walker_sats_per_plane = int(proposal.sats_per_plane)
    args.walker_altitude_km = float(proposal.altitude_km)
    args.walker_inclination_deg = float(proposal.inclination_deg)
    args.walker_phase = int(proposal.phasing_factor)
    args.duration_s = float(
        cfg.time.simulation_horizon_s if horizon_s is None else horizon_s
    )
    args.snapshot_s = float(cfg.time.snapshot_s)
    args.metrics_dt_s = float(cfg.time.metrics_dt_s)
    args.link_update_dt_s = float(cfg.time.geometry_step_s)
    args.gen_delay_s = float(cfg.time.gen_delay_s)
    args.swap_dt_s = float(cfg.time.swap_dt_s)
    args.paced_delivery_dt_s = float(cfg.time.paced_delivery_dt_s)
    args.bsm_round_duration_s = float(cfg.time.bsm_round_duration_s)
    args.source_rate_hz = float(cfg.source_rate_hz)
    args.planner_coherence_s = float(planner_coherence_s)
    args.planner_horizon_s = float(planner_horizon_s)
    args.planner_fidelity_threshold = float(cfg.planner_fidelity_threshold)
    args.max_link_distance_km = float(proposal.max_isl_range_km)
    args.w_fid = float(cfg.objective.w2_fidelity)
    args.w_mem = float(cfg.objective.w1_path_availability)
    args.fid_utility_mode = str(cfg.fid_utility_mode)
    args.no_event_log = True
    args.no_plots = True
    args.perfect_isl = bool(cfg.perfect_isl)
    args.stochastic_link_state = bool(cfg.stochastic_link_state)
    args.force_visible = bool(cfg.force_visible)
    args.f_min = float(cfg.f_min)
    args.seed = int(cfg.optimizer.seed)
    args.memory_effective_wait_s = 0.0
    args.stbd_execution_mode = str(getattr(cfg, "execution_mode", "basic"))
    args.stbd_controller_backend = str(getattr(cfg, "controller_backend", "des_only"))

    _apply_physical_sets(
        args,
        fso_params=cfg.fso_sets[proposal.fso_key],
        isl_fso_params=cfg.isl_fso_params,
        memory_params=cfg.memory_sets[proposal.memory_key],
        bsm_params=cfg.bsm_sets[proposal.bsm_key],
    )
    args.min_elevation_deg = float(proposal.min_elevation_deg)

    satellites = build_constellation_ephemerides(
        proposal,
        epoch_unix_s=float(epoch_unix_s),
    )
    ground_nodes = _build_ground_nodes(
        cfg,
        min_elevation_deg=float(proposal.min_elevation_deg),
    )
    geometry = GeometryEngine(
        SkyfieldPropagator(
            satellites=satellites,
            ground_nodes=ground_nodes,
            epoch_unix_s=float(epoch_unix_s),
        )
    )
    src_name = ground_nodes[0].name
    dst_name = ground_nodes[1].name
    node_names = [sat.name for sat in satellites] + [src_name, dst_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    return ScenarioBundle(
        proposal=proposal,
        args=args,
        satellites=satellites,
        ground_nodes=ground_nodes,
        geometry=geometry,
        src_name=src_name,
        dst_name=dst_name,
        node_names=node_names,
        out_dir=out_dir,
    )
