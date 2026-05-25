from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from channels.fso import FSOParams


@dataclass(frozen=True)
class GroundEndpoint:
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass(frozen=True)
class MemoryParameterSet:
    platform: str
    t2_s: float
    d: float
    n: float
    t_mem_s: float
    m: float
    eta_mode0: float
    eta_ti: float
    max_modes_per_label: int


@dataclass(frozen=True)
class BSMParameterSet:
    v_hom: float
    eta_det: float
    sigma_det_s: float
    delta_t_s: float
    p_dc: float
    swap_model: str = "paper"


@dataclass(frozen=True)
class ObjectiveWeights:
    w1_path_availability: float = 0.35
    w2_fidelity: float = 0.30
    w3_throughput: float = 0.20
    w4_latency: float = 0.10
    w5_satellite_penalty: float = 0.05


@dataclass(frozen=True)
class SearchRanges:
    planes_min: int = 4
    planes_max: int = 24
    sats_per_plane_min: int = 4
    sats_per_plane_max: int = 40
    altitude_km_min: float = 500.0
    altitude_km_max: float = 1500.0
    inclination_deg_min: float = 20.0
    inclination_deg_max: float = 98.0
    raan_spacing_deg_min: float = 360.0 / 24.0
    raan_spacing_deg_max: float = 90.0
    max_isl_range_km_min: float = 1500.0
    max_isl_range_km_max: float = 6000.0
    min_elevation_deg_min: float = 10.0
    min_elevation_deg_max: float = 30.0

    @property
    def max_satellites(self) -> int:
        return int(self.planes_max) * int(self.sats_per_plane_max)


@dataclass(frozen=True)
class TimeSettings:
    simulation_horizon_s: float = 24.0 * 3600.0
    geometry_step_s: float = 10.0
    snapshot_s: float = 300.0
    launch_step_s: float = 300.0
    metrics_dt_s: float = 10.0
    gen_delay_s: float = 0.2
    swap_dt_s: float = 0.2
    paced_delivery_dt_s: float = 30.0
    bsm_round_duration_s: float = 1e-3


@dataclass(frozen=True)
class OptimizerSettings:
    method: str = "random_search"
    random_search_trials: int = 8
    ga_population: int = 6
    ga_generations: int = 3
    elite_fraction: float = 0.34
    mutation_rate: float = 0.20
    seed: int = 12345
    pilot_horizon_s: float = 2.0 * 3600.0


@dataclass
class StbdConstellationConfig:
    ground_luxembourg: GroundEndpoint = field(
        default_factory=lambda: GroundEndpoint(
            name="Luxembourg-Windhof",
            latitude_deg=49.65,
            longitude_deg=5.95,
            altitude_m=300.0,
        )
    )
    ground_singapore: GroundEndpoint = field(
        default_factory=lambda: GroundEndpoint(
            name="Singapore",
            latitude_deg=1.2903,
            longitude_deg=103.8520,
            altitude_m=15.0,
        )
    )
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    search: SearchRanges = field(default_factory=SearchRanges)
    time: TimeSettings = field(default_factory=TimeSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    fso_sets: Dict[str, FSOParams] = field(default_factory=dict)
    isl_fso_params: FSOParams | None = None
    memory_sets: Dict[str, MemoryParameterSet] = field(default_factory=dict)
    bsm_sets: Dict[str, BSMParameterSet] = field(default_factory=dict)
    results_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )
    planner_fidelity_threshold: float = 0.25
    fid_utility_mode: str = "product"
    force_visible: bool = False
    stochastic_link_state: bool = True
    perfect_isl: bool = False
    support_threshold: int = 1
    evaluation_repeats: int = 1
    source_rate_hz: float = 1_000_000.0
    satellite_selection: str = "ranked_endpoint_visibility"
    f_min: float = 0.25
    planner_horizon_override_s: float | None = None
    execution_mode: str = "basic"
    controller_backend: str = "des_only"


def default_config() -> StbdConstellationConfig:
    fso_default = FSOParams(
        eta_src=0.85,
        eta_det=0.75,
        eta_cpl=0.70,
        eta_point_static=0.97,
        d_t=0.15,
        d_r=0.50,
        theta_div=1.2e-5,
        t_abs=0.94,
        t_sca=0.94,
        c_atm=0.06,
        sigma_r2=0.08,
        turbulence_loss_coeff=0.08,
        turbulence_quality_coeff=0.03,
        p_t=0.5,
        n0=1e-5,
        n_dc=120.0,
        n_conv=0.0,
        delta_t=1e-9,
        min_elevation_deg=10.0,
        pointing_jitter_rad=0.0,
        enable_pointing_jitter=False,
        enable_stochastic_fading=True,
    )
    isl_fso_default = FSOParams(
        eta_src=0.92,
        eta_det=0.85,
        eta_cpl=0.90,
        eta_point_static=0.99,
        d_t=0.18,
        d_r=0.60,
        theta_div=5e-6,
        wavelength_m=810e-9,
        t_abs=1.0,
        t_sca=1.0,
        c_atm=0.0,
        sigma_r2=0.0,
        ao_gain=1.0,
        turbulence_loss_coeff=0.0,
        turbulence_quality_coeff=0.0,
        enable_stochastic_fading=False,
        enable_pointing_jitter=False,
        pointing_jitter_rad=0.0,
        turbulence_floor=1e-4,
        polarization_error_rad=0.0,
        phase_noise_rad=0.02,
        temporal_mismatch_s=5e-12,
        coherence_time_s=100e-12,
        p_t=0.5,
        n0=1e-5,
        n_dc=10.0,
        n_conv=0.0,
        delta_t=1e-9,
        min_elevation_deg=0.0,
        max_snr_for_mapping=1e12,
    )
    memory_sets = {
        "REID": MemoryParameterSet(
            platform="REID",
            t2_s=600.0,
            d=7.0,
            n=1.5,
            t_mem_s=600.0,
            m=1.0,
            eta_mode0=0.90,
            eta_ti=0.30,
            max_modes_per_label=256,
        ),
        "COLD_ATOM": MemoryParameterSet(
            platform="COLD_ATOM",
            t2_s=300.0,
            d=20.0,
            n=1.3,
            t_mem_s=300.0,
            m=1.0,
            eta_mode0=0.90,
            eta_ti=0.30,
            max_modes_per_label=256,
        ),
        "TRAPPED_ION": MemoryParameterSet(
            platform="TRAPPED_ION",
            t2_s=600.0,
            d=7.0,
            n=1.0,
            t_mem_s=600.0,
            m=1.0,
            eta_mode0=0.90,
            eta_ti=0.30,
            max_modes_per_label=256,
        ),
    }
    bsm_sets = {
        "baseline": BSMParameterSet(
            v_hom=0.97,
            eta_det=0.82,
            sigma_det_s=50e-12,
            delta_t_s=200e-12,
            p_dc=1e-6,
            swap_model="paper",
        )
    }
    return StbdConstellationConfig(
        fso_sets={"baseline": fso_default},
        isl_fso_params=isl_fso_default,
        memory_sets=memory_sets,
        bsm_sets=bsm_sets,
    )
