from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channels.fso import (
    FSOAnalyticalChannelModel,
    FSOParams,
    efficiency_to_loss_db,
)
from framework.core.des import DESEnv
from des_physics_no_sdn_simulator import (
    PS_PER_S,
    DesPhysicsNoSdn,
    build_bsm_model,
    build_fso_model,
    build_memory_model,
    build_parser as build_des_parser,
    require_qnet_des,
)
from mobility.archetype_geometry import (
    OverpassScenario,
    ParametricOverpassGeometry,
    SyntheticOverpassGeometry,
)
from mobility.geometry import ContactGeometry


MU_EARTH_M3_S2 = 3.986004418e14
R_EARTH_M = 6_378_137.0


MEMORY_PLATFORM_LABELS = {
    "REID": "REID",
    "COLD_ATOM": "Cold Atom",
    "TRAPPED_ION": "Trapped Ion",
}


@dataclass(frozen=True)
class MemoryStudyConfig:
    platform: str = "COLD_ATOM"
    t2_s: float = 100.0
    d: float = 20.0
    t_mem_s: float = 1000.0
    n: float = 1.3
    m: float = 1.0
    eta_mode0: float = 0.9
    eta_ti: float = 0.3
    max_modes_per_label: int = 8192


@dataclass(frozen=True)
class BSMStudyConfig:
    v_hom: float = 0.97
    eta_det: float = 0.82
    sigma_det_s: float = 50e-12
    delta_t_s: float = 200e-12
    p_dc: float = 1e-6


@dataclass(frozen=True)
class CityPair:
    key: str
    title: str
    city_a: str
    lat_a: float
    lon_a: float
    city_b: str
    lat_b: float
    lon_b: float
    track_angle_deg: float

    @property
    def baseline_km(self) -> float:
        return haversine_km(self.lat_a, self.lon_a, self.lat_b, self.lon_b)


REPRESENTATIVE_OVERPASSES = [
    OverpassScenario(
        key="zenith_zenith",
        title="(a) Zenith-zenith (Delta = 0 km, phi = 0°)",
        baseline_km=1000.0,
        track_angle_deg=0.0,
        delta_km=0.0,
        half_window_s=150.0,
    ),
    OverpassScenario(
        key="symmetric",
        title="(b) Symmetric (Delta = 0 km, phi = 90°)",
        baseline_km=1000.0,
        track_angle_deg=90.0,
        delta_km=0.0,
        half_window_s=220.0,
    ),
    OverpassScenario(
        key="zenith_a_90",
        title="(c) Zenith A 90° (Delta = 500 km, phi = 90°)",
        baseline_km=1000.0,
        track_angle_deg=90.0,
        delta_km=500.0,
        half_window_s=170.0,
    ),
    OverpassScenario(
        key="zenith_a_45",
        title="(d) Zenith A 45° (Delta = 500 km, phi = 45°)",
        baseline_km=1000.0,
        track_angle_deg=45.0,
        delta_km=500.0,
        half_window_s=220.0,
    ),
]


CITY_PAIRS = [
    CityPair(
        key="paris_nice",
        title="(a) Paris-Nice",
        city_a="Paris",
        lat_a=48.8566,
        lon_a=2.3522,
        city_b="Nice",
        lat_b=43.7102,
        lon_b=7.2620,
        track_angle_deg=33.3,
    ),
    CityPair(
        key="london_berlin",
        title="(b) London-Berlin",
        city_a="London",
        lat_a=51.5074,
        lon_a=-0.1278,
        city_b="Berlin",
        lat_b=52.5200,
        lon_b=13.4050,
        track_angle_deg=97.0,
    ),
    CityPair(
        key="seoul_tokyo",
        title="(c) Seoul-Tokyo",
        city_a="Seoul",
        lat_a=37.5665,
        lon_a=126.9780,
        city_b="Tokyo",
        lat_b=35.6762,
        lon_b=139.6503,
        track_angle_deg=79.6,
    ),
    CityPair(
        key="madrid_brussels",
        title="(d) Madrid-Brussels",
        city_a="Madrid",
        lat_a=40.4168,
        lon_a=-3.7038,
        city_b="Brussels",
        lat_b=50.8503,
        lon_b=4.3517,
        track_angle_deg=152.0,
    ),
]

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
    return 2.0 * r_km * math.asin(math.sqrt(a))


def baseline_bearing_proxy_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing if bearing <= 180.0 else 360.0 - bearing


REQUESTED_ANNUAL_CITY_PAIRS = [
    CityPair(
        key="paris_berlin",
        title="(a) Paris-Berlin",
        city_a="Paris",
        lat_a=48.8566,
        lon_a=2.3522,
        city_b="Berlin",
        lat_b=52.5200,
        lon_b=13.4050,
        track_angle_deg=baseline_bearing_proxy_deg(48.8566, 2.3522, 52.5200, 13.4050),
    ),
    CityPair(
        key="lisbon_warsaw",
        title="(b) Lisbon-Warsaw",
        city_a="Lisbon",
        lat_a=38.7223,
        lon_a=-9.1393,
        city_b="Warsaw",
        lat_b=52.2297,
        lon_b=21.0122,
        track_angle_deg=baseline_bearing_proxy_deg(38.7223, -9.1393, 52.2297, 21.0122),
    ),
]

OPTIMIZED_ANNUAL_CITY_PAIRS = [
    CityPair(
        key="paris_vienna",
        title="(a) Paris-Vienna",
        city_a="Paris",
        lat_a=48.8566,
        lon_a=2.3522,
        city_b="Vienna",
        lat_b=48.2082,
        lon_b=16.3738,
        track_angle_deg=0.0,
    ),
    CityPair(
        key="warsaw_lisbon",
        title="(b) Warsaw-Lisbon",
        city_a="Warsaw",
        lat_a=52.2297,
        lon_a=21.0122,
        city_b="Lisbon",
        lat_b=38.7223,
        lon_b=-9.1393,
        track_angle_deg=0.0,
    ),
]


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(v for v in values if not math.isnan(v))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def weighted_quantile(
    values: Iterable[float],
    weights: Iterable[float],
    q: float,
) -> float:
    pairs = sorted(
        (float(v), float(w))
        for v, w in zip(values, weights)
        if not math.isnan(float(v)) and float(w) > 0.0
    )
    if not pairs:
        return float("nan")
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return float("nan")
    threshold = float(q) * total_weight
    accum = 0.0
    for value, weight in pairs:
        accum += weight
        if accum >= threshold:
            return value
    return pairs[-1][0]


def orbital_period_s(altitude_km: float) -> float:
    radius_m = R_EARTH_M + float(altitude_km) * 1e3
    return 2.0 * math.pi * math.sqrt((radius_m**3) / MU_EARTH_M3_S2)


def study_fso_params(min_elevation_deg: float) -> FSOParams:
    return FSOParams(
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
        delta_t=1e-9,
        min_elevation_deg=float(min_elevation_deg),
    )


def memory_platform_label(platform: str) -> str:
    key = str(platform).upper().strip()
    return MEMORY_PLATFORM_LABELS.get(key, key.replace("_", " ").title())


def build_des_args(
    *,
    duration_s: float,
    min_elevation_deg: float,
    seed: int,
    source_rate_hz: float,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> argparse.Namespace:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    args = build_des_parser().parse_args([])
    args.seed = int(seed)
    args.duration_s = float(duration_s)
    args.metrics_dt_s = 2.0
    args.link_update_dt_s = 1.0
    args.gen_delay_s = 0.2
    args.swap_dt_s = 0.2
    args.source_rate_hz = float(source_rate_hz)
    args.max_modes_per_label = int(memory_config.max_modes_per_label)
    args.max_bsm_per_cycle = 4096
    args.f_min = 0.25
    args.memory_effective_wait_s = 0.0
    args.min_elevation_deg = float(min_elevation_deg)
    args.auto_start_at_visibility = False
    args.force_visible = False
    args.no_plots = True
    args.no_event_log = True
    args.mem_platform = str(memory_config.platform)
    args.mem_t2 = float(memory_config.t2_s)
    args.mem_d = float(memory_config.d)
    args.mem_t_mem = float(memory_config.t_mem_s)
    args.mem_n = float(memory_config.n)
    args.mem_m = float(memory_config.m)
    args.mem_eta_mode0 = float(memory_config.eta_mode0)
    args.mem_eta_ti = float(memory_config.eta_ti)
    args.bsm_v_hom = float(bsm_config.v_hom)
    args.bsm_eta_det = float(bsm_config.eta_det)
    args.bsm_sigma_det_s = float(bsm_config.sigma_det_s)
    args.bsm_delta_t_s = float(bsm_config.delta_t_s)
    args.bsm_p_dc = float(bsm_config.p_dc)
    args.fso_eta_src = fso_params.eta_src
    args.fso_eta_det = fso_params.eta_det
    args.fso_eta_cpl = fso_params.eta_cpl
    args.fso_eta_point = fso_params.eta_point_static
    args.fso_d_t = fso_params.d_t
    args.fso_d_r = fso_params.d_r
    args.fso_theta_div = fso_params.theta_div
    args.fso_t_abs = fso_params.t_abs
    args.fso_t_sca = fso_params.t_sca
    args.fso_c_atm = fso_params.c_atm
    args.fso_sigma_r2 = fso_params.sigma_r2
    args.fso_turbulence_loss_coeff = fso_params.turbulence_loss_coeff
    args.fso_turbulence_quality_coeff = fso_params.turbulence_quality_coeff
    args.fso_delta_t = fso_params.delta_t
    args.fso_n0 = fso_params.n0
    args.fso_n_dc = fso_params.n_dc
    args.fso_n_conv = fso_params.n_conv
    args.fso_p_t = fso_params.p_t
    args.fso_pointing_jitter_rad = fso_params.pointing_jitter_rad
    args.disable_fso_pointing_jitter = not bool(fso_params.enable_pointing_jitter)
    args.disable_fso_stochastic_fading = not bool(
        fso_params.enable_stochastic_fading
    )
    return args


def representative_trace(
    scenario: OverpassScenario,
    *,
    fso_model: FSOAnalyticalChannelModel,
    min_elevation_deg: float,
    source_rate_hz: float,
    ssqr_efficiency: float,
) -> List[Dict[str, float]]:
    geom = ParametricOverpassGeometry(scenario, min_elevation_deg)
    rows: List[Dict[str, float]] = []
    for t_s in range(
        -int(scenario.half_window_s), int(scenario.half_window_s) + 1
    ):
        t_ps = int((t_s + scenario.half_window_s) * PS_PER_S)
        alice = geom.contact_geometry("Alice", "R1", t_ps)
        bob = geom.contact_geometry("Bob", "R1", t_ps)
        eta_a_raw = fso_model.mean_efficiency(
            distance_m=alice.distance_m,
            elevation_deg=alice.elevation_deg,
            intersatellite=False,
            apply_min_elevation=False,
        )
        eta_b_raw = fso_model.mean_efficiency(
            distance_m=bob.distance_m,
            elevation_deg=bob.elevation_deg,
            intersatellite=False,
            apply_min_elevation=False,
        )
        eta_a = eta_a_raw if alice.visible else 0.0
        eta_b = eta_b_raw if bob.visible else 0.0
        rate_a = source_rate_hz * eta_a
        rate_b = source_rate_hz * eta_b
        rows.append(
            {
                "time_s": float(t_s),
                "alice_loss_db": efficiency_to_loss_db(eta_a_raw),
                "bob_loss_db": efficiency_to_loss_db(eta_b_raw),
                "alice_visible": 1.0 if alice.visible else 0.0,
                "bob_visible": 1.0 if bob.visible else 0.0,
                "dddl_rate": source_rate_hz * eta_a * eta_b,
                "ssqr_rate": ssqr_efficiency * 0.5 * min(rate_a, rate_b),
                "ssqr_equal_rate": ssqr_efficiency * 0.5 * min(rate_a, rate_b),
                "ssqr_opt_rate": (
                    ssqr_efficiency * (rate_a * rate_b) / (rate_a + rate_b)
                    if (rate_a + rate_b) > 0.0
                    else 0.0
                ),
            }
        )
    return rows


def run_des_fidelity_stats(
    scenario: OverpassScenario,
    *,
    repeats: int,
    seed: int,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> Dict[str, List[float]]:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    per_bin_fidelity: List[List[float]] = []
    per_bin_successes: List[List[int]] = []
    per_bin_rate: List[List[float]] = []
    times: List[float] = []
    duration_s = 2.0 * scenario.half_window_s

    for repeat in range(max(1, repeats)):
        args = build_des_args(
            duration_s=duration_s,
            min_elevation_deg=min_elevation_deg,
            seed=seed + repeat,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )
        env = DESEnv(name=f"des_{scenario.key}_{repeat}", default=True)
        sim = DesPhysicsNoSdn(
            name="physics",
            env=env,
            args=args,
            geometry=ParametricOverpassGeometry(scenario, min_elevation_deg),
            fso=build_fso_model(args),
            memory=build_memory_model(args),
            bsm=build_bsm_model(args),
        )
        env.init()
        env.run(end_time=int(duration_s * PS_PER_S), logging=False, summary=False)

        if not times:
            times = [row.t_s - scenario.half_window_s for row in sim.metrics]
            per_bin_fidelity = [[] for _ in times]
            per_bin_successes = [[] for _ in times]
            per_bin_rate = [[] for _ in times]

        for idx, row in enumerate(sim.metrics):
            per_bin_rate[idx].append(float(row.throughput_pairs_per_s))
            per_bin_successes[idx].append(int(row.successes))
            per_bin_fidelity[idx].append(
                float(row.mean_fidelity_success) if row.successes > 0 else float("nan")
            )

    min_success_support = max(3, int(repeats))
    fidelity_success = []
    fidelity_q1 = []
    fidelity_q3 = []
    support_successes = []
    for values, weights in zip(per_bin_fidelity, per_bin_successes):
        valid = [
            (float(value), int(weight))
            for value, weight in zip(values, weights)
            if not math.isnan(float(value)) and int(weight) > 0
        ]
        total_successes = sum(weight for _, weight in valid)
        support_successes.append(total_successes)
        if total_successes < min_success_support:
            fidelity_success.append(float("nan"))
            fidelity_q1.append(float("nan"))
            fidelity_q3.append(float("nan"))
            continue
        fidelity_success.append(
            sum(value * weight for value, weight in valid) / total_successes
        )
        fidelity_q1.append(
            weighted_quantile(
                [value for value, _ in valid],
                [weight for _, weight in valid],
                0.25,
            )
        )
        fidelity_q3.append(
            weighted_quantile(
                [value for value, _ in valid],
                [weight for _, weight in valid],
                0.75,
            )
        )
    rate_mean = [
        sum(values) / len(values) if values else float("nan")
        for values in per_bin_rate
    ]
    rate_std = []
    for values in per_bin_rate:
        if len(values) <= 1:
            rate_std.append(0.0 if values else float("nan"))
            continue
        mu = sum(values) / len(values)
        rate_std.append(
            math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))
        )

    return {
        "time_s": times,
        "fidelity_median": fidelity_success,
        "fidelity_q1": fidelity_q1,
        "fidelity_q3": fidelity_q3,
        "rate_mean": rate_mean,
        "rate_std": rate_std,
        "support_successes": support_successes,
    }


def plot_des_parameter_sweeps(
    scenarios: List[OverpassScenario],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    t2_values_s: List[float],
    rytov_values: List[float],
    sweep_repeats: int,
    base_fso_params: FSOParams | None = None,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> None:
    base_fso_params = base_fso_params or study_fso_params(min_elevation_deg)
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    t2_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    rytov_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    sweep_rows: List[Dict[str, float | str]] = []

    fidelity_fig, fidelity_axes = plt.subplots(4, 2, figsize=(14.5, 18), sharex=False)
    rate_fig, rate_axes = plt.subplots(4, 2, figsize=(14.5, 18), sharex=False)

    for row_idx, scenario in enumerate(scenarios):
        for color, mem_t2_s in zip(t2_palette, t2_values_s):
            stats = run_des_fidelity_stats(
                scenario,
                repeats=sweep_repeats,
                seed=seed + 1000 * row_idx + int(mem_t2_s * 10),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                fso_params=base_fso_params,
                memory_config=replace(memory_config, t2_s=float(mem_t2_s)),
                bsm_config=bsm_config,
            )
            fidelity_axes[row_idx, 0].plot(
                stats["time_s"],
                stats["fidelity_median"],
                color=color,
                label=rf"$T_2$ = {mem_t2_s:g} s",
            )
            rate_axes[row_idx, 0].plot(
                stats["time_s"],
                stats["rate_mean"],
                color=color,
                label=rf"$T_2$ = {mem_t2_s:g} s",
            )
            for idx, t_s in enumerate(stats["time_s"]):
                sweep_rows.append(
                    {
                        "scenario": scenario.key,
                        "sweep_kind": "t2",
                        "sweep_value": mem_t2_s,
                        "time_s": t_s,
                        "fidelity_median": stats["fidelity_median"][idx],
                        "rate_mean": stats["rate_mean"][idx],
                    }
                )

        for color, sigma_r2 in zip(rytov_palette, rytov_values):
            fso_params = replace(base_fso_params, sigma_r2=float(sigma_r2))
            stats = run_des_fidelity_stats(
                scenario,
                repeats=sweep_repeats,
                seed=seed + 10000 + 1000 * row_idx + int(100 * sigma_r2),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
            )
            fidelity_axes[row_idx, 1].plot(
                stats["time_s"],
                stats["fidelity_median"],
                color=color,
                label=rf"$\sigma_R^2$ = {sigma_r2:g}",
            )
            rate_axes[row_idx, 1].plot(
                stats["time_s"],
                stats["rate_mean"],
                color=color,
                label=rf"$\sigma_R^2$ = {sigma_r2:g}",
            )
            for idx, t_s in enumerate(stats["time_s"]):
                sweep_rows.append(
                    {
                        "scenario": scenario.key,
                        "sweep_kind": "rytov",
                        "sweep_value": sigma_r2,
                        "time_s": t_s,
                        "fidelity_median": stats["fidelity_median"][idx],
                        "rate_mean": stats["rate_mean"][idx],
                    }
                )

        for col, ax in enumerate(fidelity_axes[row_idx]):
            add_label = "(a) $T_2$ overlays" if col == 0 else "(b) Rytov overlays"
            ax.set_title(f"{scenario.title} {add_label}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Successful-Pair Fidelity")
            ax.set_ylim(0.25, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        for col, ax in enumerate(rate_axes[row_idx]):
            add_label = "(a) $T_2$ overlays" if col == 0 else "(b) Rytov overlays"
            ax.set_title(f"{scenario.title} {add_label}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Rate")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

    fidelity_fig.suptitle(
        "Archetype DES Successful-Pair Fidelity Sweeps\n"
        "Paper-style spherical overpass geometry with repository FSO, memory, and BSM models through QNET DES"
    )
    fidelity_fig.tight_layout()
    fidelity_fig.savefig(out_dir / "des_fidelity_t2_and_rytov_sweeps.png", dpi=180)
    fidelity_fig.savefig(out_dir / "des_fidelity_t2_and_rytov_sweeps.pdf")
    fidelity_fig.savefig(out_dir / "archetype_des_fidelity_t2_and_rytov_sweeps.png", dpi=180)
    fidelity_fig.savefig(out_dir / "archetype_des_fidelity_t2_and_rytov_sweeps.pdf")
    plt.close(fidelity_fig)

    rate_fig.tight_layout()
    rate_fig.savefig(out_dir / "des_ssqr_rate_t2_and_rytov_sweeps.png", dpi=180)
    rate_fig.savefig(out_dir / "des_ssqr_rate_t2_and_rytov_sweeps.pdf")
    rate_fig.savefig(out_dir / "archetype_des_ssqr_rate_t2_and_rytov_sweeps.png", dpi=180)
    rate_fig.savefig(out_dir / "archetype_des_ssqr_rate_t2_and_rytov_sweeps.pdf")
    plt.close(rate_fig)

    write_csv_rows(out_dir / "des_parameter_sweep_stats.csv", sweep_rows)
    write_csv_rows(out_dir / "archetype_des_parameter_sweep_stats.csv", sweep_rows)


def annualised_pdv_curve(
    pair: CityPair,
    *,
    altitudes_km: Iterable[float],
    fso_model_factory,
    min_elevation_deg: float,
    source_rate_hz: float,
    ssqr_efficiency: float,
    annual_contact_fraction: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for altitude_km in altitudes_km:
        scenario = OverpassScenario(
            key=f"{pair.key}_{int(altitude_km)}",
            title=pair.title,
            baseline_km=pair.baseline_km,
            track_angle_deg=pair.track_angle_deg,
            delta_km=0.0,
            altitude_km=float(altitude_km),
            half_window_s=max(
                180.0,
                1.5 * float(altitude_km) / 7.2 + pair.baseline_km / 7.2 + 60.0,
            ),
        )
        trace = representative_trace(
            scenario,
            fso_model=fso_model_factory(min_elevation_deg),
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            ssqr_efficiency=ssqr_efficiency,
        )
        dt_s = 1.0
        dddl_per_pass = sum(row["dddl_rate"] * dt_s for row in trace)
        equal_per_pass = sum(row["ssqr_equal_rate"] * dt_s for row in trace)
        opt_per_pass = sum(row["ssqr_opt_rate"] * dt_s for row in trace)
        annual_factor = (
            float(annual_contact_fraction) * (365.25 * 24.0 * 3600.0)
            / orbital_period_s(float(altitude_km))
        )
        rows.append(
            {
                "altitude_km": float(altitude_km),
                "dddl_annual_pairs": dddl_per_pass * annual_factor,
                "equal_split_annual_pairs": equal_per_pass * annual_factor,
                "optimal_split_annual_pairs": opt_per_pass * annual_factor,
            }
        )
    return rows


def annualised_pdv_curve_optimized_geometry(
    pair: CityPair,
    *,
    altitudes_km: Iterable[float],
    fso_model_factory,
    min_elevation_deg: float,
    source_rate_hz: float,
    ssqr_efficiency: float,
    annual_contact_fraction: float,
    phi_grid_deg: Iterable[float] = tuple(range(0, 91, 5)),
    delta_fractions: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for altitude_km in altitudes_km:
        annual_factor = (
            float(annual_contact_fraction) * (365.25 * 24.0 * 3600.0)
            / orbital_period_s(float(altitude_km))
        )
        best_geometry = None
        best_opt_annual = -1.0
        best_curves = None

        delta_grid_km = [
            float(fraction) * float(pair.baseline_km) for fraction in delta_fractions
        ]
        for delta_km in delta_grid_km:
            for phi_deg in phi_grid_deg:
                scenario = OverpassScenario(
                    key=(
                        f"{pair.key}_{int(round(delta_km))}_"
                        f"{int(round(phi_deg))}_{int(altitude_km)}"
                    ),
                    title=pair.title,
                    baseline_km=pair.baseline_km,
                    track_angle_deg=float(phi_deg),
                    delta_km=float(delta_km),
                    altitude_km=float(altitude_km),
                    half_window_s=max(
                        180.0,
                        1.5 * float(altitude_km) / 7.2 + pair.baseline_km / 7.2 + 60.0,
                    ),
                )
                trace = representative_trace(
                    scenario,
                    fso_model=fso_model_factory(min_elevation_deg),
                    min_elevation_deg=min_elevation_deg,
                    source_rate_hz=source_rate_hz,
                    ssqr_efficiency=ssqr_efficiency,
                )
                dt_s = 1.0
                dddl_per_pass = sum(row["dddl_rate"] * dt_s for row in trace)
                equal_per_pass = sum(row["ssqr_equal_rate"] * dt_s for row in trace)
                opt_per_pass = sum(row["ssqr_opt_rate"] * dt_s for row in trace)
                curves = (
                    dddl_per_pass * annual_factor,
                    equal_per_pass * annual_factor,
                    opt_per_pass * annual_factor,
                )
                if curves[2] > best_opt_annual:
                    best_opt_annual = curves[2]
                    best_geometry = (float(delta_km), float(phi_deg))
                    best_curves = curves

        if best_geometry is None or best_curves is None:
            continue

        delta_km, phi_deg = best_geometry
        rows.append(
            {
                "altitude_km": float(altitude_km),
                "dddl_annual_pairs": best_curves[0],
                "equal_split_annual_pairs": best_curves[1],
                "optimal_split_annual_pairs": best_curves[2],
                "best_delta_km": delta_km,
                "best_track_angle_deg": phi_deg,
            }
        )
    return rows


def run_des_pass_summary(
    scenario: OverpassScenario,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> Dict[str, float]:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    duration_s = 2.0 * scenario.half_window_s
    args = build_des_args(
        duration_s=duration_s,
        min_elevation_deg=min_elevation_deg,
        seed=seed,
        source_rate_hz=source_rate_hz,
        fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
    )
    env = DESEnv(name=f"des_buffered_{scenario.key}_{seed}", default=True)
    sim = DesPhysicsNoSdn(
        name="physics",
        env=env,
        args=args,
        geometry=ParametricOverpassGeometry(scenario, min_elevation_deg),
        fso=build_fso_model(args),
        memory=build_memory_model(args),
        bsm=build_bsm_model(args),
    )
    env.init()
    env.run(end_time=int(duration_s * PS_PER_S), logging=False, summary=False)

    success_count = int(sim.cumulative_successes)
    fail_link = sum(row.fail_link for row in sim.metrics)
    fail_recall = sum(row.fail_recall for row in sim.metrics)
    fail_bsm = sum(row.fail_bsm for row in sim.metrics)
    fail_storage = sum(row.fail_storage_full for row in sim.metrics)
    total_trials = success_count + fail_link + fail_recall + fail_bsm + fail_storage
    occupancies_total = [int(row.memory_ar1 + row.memory_r1b) for row in sim.metrics]
    occupancies_per_label = [
        max(int(row.memory_ar1), int(row.memory_r1b)) for row in sim.metrics
    ]
    both_visible_fraction = (
        sum(int(row.both_visible) for row in sim.metrics) / len(sim.metrics)
        if sim.metrics
        else 0.0
    )
    return {
        "pass_pairs": float(success_count),
        "annual_rate_pairs_per_s": success_count / max(1e-12, duration_s),
        "peak_total_buffer_tokens": float(max(occupancies_total) if occupancies_total else 0.0),
        "mean_total_buffer_tokens": (
            float(sum(occupancies_total) / len(occupancies_total)) if occupancies_total else 0.0
        ),
        "peak_per_label_buffer_tokens": float(
            max(occupancies_per_label) if occupancies_per_label else 0.0
        ),
        "mean_per_label_buffer_tokens": (
            float(sum(occupancies_per_label) / len(occupancies_per_label))
            if occupancies_per_label
            else 0.0
        ),
        "storage_full_fraction": (
            fail_storage / total_trials if total_trials > 0 else 0.0
        ),
        "mean_wait_success_s": (
            sum(float(row.mean_wait_success_s) * int(row.successes) for row in sim.metrics)
            / max(1, success_count)
            if success_count > 0
            else 0.0
        ),
        "both_visible_fraction": float(both_visible_fraction),
    }


def annualised_des_buffered_curve_optimized_geometry(
    pair: CityPair,
    *,
    altitudes_km: Iterable[float],
    min_elevation_deg: float,
    source_rate_hz: float,
    annual_contact_fraction: float,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    phi_grid_deg: Iterable[float] = tuple(range(0, 91, 15)),
    delta_fractions: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> List[Dict[str, float]]:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    rows: List[Dict[str, float]] = []
    for altitude_index, altitude_km in enumerate(altitudes_km):
        annual_factor = (
            float(annual_contact_fraction) * (365.25 * 24.0 * 3600.0)
            / orbital_period_s(float(altitude_km))
        )
        best_row = None
        best_annual_pairs = -1.0
        delta_grid_km = [
            float(fraction) * float(pair.baseline_km) for fraction in delta_fractions
        ]
        for delta_index, delta_km in enumerate(delta_grid_km):
            for phi_index, phi_deg in enumerate(phi_grid_deg):
                scenario = OverpassScenario(
                    key=(
                        f"{pair.key}_{int(round(delta_km))}_"
                        f"{int(round(phi_deg))}_{int(altitude_km)}"
                    ),
                    title=pair.title,
                    baseline_km=pair.baseline_km,
                    track_angle_deg=float(phi_deg),
                    delta_km=float(delta_km),
                    altitude_km=float(altitude_km),
                    half_window_s=max(
                        180.0,
                        1.5 * float(altitude_km) / 7.2 + pair.baseline_km / 7.2 + 60.0,
                    ),
                )
                summary = run_des_pass_summary(
                    scenario,
                    min_elevation_deg=min_elevation_deg,
                    source_rate_hz=source_rate_hz,
                    fso_params=study_fso_params(min_elevation_deg),
                    seed=seed + 10000 * altitude_index + 1000 * delta_index + 97 * phi_index,
                    memory_config=memory_config,
                    bsm_config=bsm_config,
                )
                annual_pairs = summary["pass_pairs"] * annual_factor
                if annual_pairs <= best_annual_pairs:
                    continue
                best_annual_pairs = annual_pairs
                best_row = {
                    "altitude_km": float(altitude_km),
                    "annual_pairs": float(annual_pairs),
                    "best_delta_km": float(delta_km),
                    "best_track_angle_deg": float(phi_deg),
                    "peak_total_buffer_tokens": summary["peak_total_buffer_tokens"],
                    "mean_total_buffer_tokens": summary["mean_total_buffer_tokens"],
                    "peak_per_label_buffer_tokens": summary["peak_per_label_buffer_tokens"],
                    "mean_per_label_buffer_tokens": summary["mean_per_label_buffer_tokens"],
                    "storage_full_fraction": summary["storage_full_fraction"],
                    "mean_wait_success_s": summary["mean_wait_success_s"],
                    "both_visible_fraction": summary["both_visible_fraction"],
                }
        if best_row is not None:
            rows.append(best_row)
    return rows


def write_csv_rows(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def default_paper_out_dir() -> str:
    return str(Path(__file__).resolve().parent / "Output" / "des_no_sdn_paper_cases")


def plot_representative_losses(
    scenarios: List[OverpassScenario],
    traces: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    fso_model: FSOAnalyticalChannelModel,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    floor_db = fso_model.mean_loss_db(
        distance_m=500e3,
        elevation_deg=90.0,
        intersatellite=False,
        apply_min_elevation=False,
    )
    for ax, scenario in zip(axes.flat, scenarios):
        rows = traces[scenario.key]
        xs = [row["time_s"] for row in rows]
        ax.plot(xs, [row["alice_loss_db"] for row in rows], color="red", label="Alice")
        ax.plot(
            xs,
            [row["bob_loss_db"] for row in rows],
            color="blue",
            linestyle="--",
            label="Bob",
        )
        ax.axhline(floor_db, color="black", linestyle=":", linewidth=1.5)
        ax.set_title(scenario.title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Downlink Loss (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / "representative_downlink_loss.png", dpi=180)
    fig.savefig(out_dir / "representative_downlink_loss.pdf")
    plt.close(fig)


def plot_representative_rates(
    scenarios: List[OverpassScenario],
    traces: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for ax, scenario in zip(axes.flat, scenarios):
        rows = traces[scenario.key]
        xs = [row["time_s"] for row in rows]
        ax.plot(
            xs,
            [row["dddl_rate"] for row in rows],
            color="tab:blue",
            linestyle="-.",
            label="DDDL",
        )
        ax.plot(
            xs,
            [row["ssqr_rate"] for row in rows],
            color="tab:orange",
            label="SSQR",
        )
        ax.set_title(scenario.title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distribution Rate (pairs s$^{-1}$)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "representative_distribution_rates.png", dpi=180)
    fig.savefig(out_dir / "representative_distribution_rates.pdf")
    plt.close(fig)


def plot_altitude_dependence(
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, pair in zip(axes.flat, CITY_PAIRS):
        rows = rows_by_pair[pair.key]
        xs = [row["altitude_km"] for row in rows]
        ax.axvspan(200.0, 400.0, color="0.8", alpha=0.6)
        ax.axvline(400.0, color="red", linestyle="--", linewidth=1.0)
        ax.axvline(500.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            xs,
            [row["equal_split_annual_pairs"] / 1e3 for row in rows],
            label="Equal Split",
        )
        ax.plot(
            xs,
            [row["optimal_split_annual_pairs"] / 1e3 for row in rows],
            label="Optimal Split",
        )
        ax.plot(
            xs,
            [row["dddl_annual_pairs"] / 1e3 for row in rows],
            color="black",
            linestyle="-.",
            label="DDDL",
        )
        ax.set_title(f"{pair.title} (d = {pair.baseline_km:.0f} km)")
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Average Annual PDV × 10$^3$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "altitudinal_pdv_dependence.png", dpi=180)
    fig.savefig(out_dir / "altitudinal_pdv_dependence.pdf")
    plt.close(fig)


def plot_requested_annual_pair_distribution(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(14.5, 5.6), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        rows = rows_by_pair[pair.key]
        xs = [row["altitude_km"] for row in rows]
        ax.axvspan(200.0, 400.0, color="0.8", alpha=0.6)
        ax.axvline(400.0, color="red", linestyle="--", linewidth=1.0)
        ax.axvline(500.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            xs,
            [row["equal_split_annual_pairs"] / 1e3 for row in rows],
            label="Equal Split",
        )
        ax.plot(
            xs,
            [row["optimal_split_annual_pairs"] / 1e3 for row in rows],
            label="Optimal Split",
        )
        ax.plot(
            xs,
            [row["dddl_annual_pairs"] / 1e3 for row in rows],
            color="black",
            linestyle="-.",
            label="DDDL",
        )
        ax.set_title(
            f"{pair.title} (d = {pair.baseline_km:.0f} km, "
            f"$\\phi_{{proxy}}$ = {pair.track_angle_deg:.1f}°)"
        )
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Average Annual Pairs × 10$^3$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        "Annual Pair-Distribution Simulation for Requested OGS Links"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "annual_pair_distribution_paris_berlin_lisbon_warsaw.png", dpi=180)
    fig.savefig(out_dir / "annual_pair_distribution_paris_berlin_lisbon_warsaw.pdf")
    plt.close(fig)


def plot_optimized_annual_pair_distribution(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(14.8, 5.8), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        rows = rows_by_pair[pair.key]
        xs = [row["altitude_km"] for row in rows]
        ax.axvspan(200.0, 400.0, color="0.85", alpha=0.55)
        ax.axvline(400.0, color="red", linestyle="--", linewidth=1.0)
        ax.axvline(500.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            xs,
            [row["equal_split_annual_pairs"] / 1e3 for row in rows],
            label="Equal Split",
        )
        ax.plot(
            xs,
            [row["optimal_split_annual_pairs"] / 1e3 for row in rows],
            label="Optimal Split",
        )
        ax.plot(
            xs,
            [row["dddl_annual_pairs"] / 1e3 for row in rows],
            color="black",
            linestyle="-.",
            label="DDDL",
        )
        ax.set_title(f"{pair.title} (d = {pair.baseline_km:.0f} km)")
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Annual Throughput × 10$^3$ pairs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        "Optimized-Geometry Annual Throughput\n"
        "Geometry chosen per altitude to maximize optimal-split annual throughput"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.png", dpi=180)
    fig.savefig(out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.pdf")
    plt.close(fig)


def plot_buffered_annual_throughput(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    reference_rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    memory_config: MemoryStudyConfig | None = None,
) -> None:
    memory_config = memory_config or MemoryStudyConfig()
    fig, axes = plt.subplots(1, len(pairs), figsize=(14.8, 5.8), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        ref_rows = reference_rows_by_pair[pair.key]
        buf_rows = rows_by_pair[pair.key]
        xs_ref = [row["altitude_km"] for row in ref_rows]
        xs_buf = [row["altitude_km"] for row in buf_rows]
        ax.axvspan(200.0, 400.0, color="0.85", alpha=0.55)
        ax.axvline(400.0, color="red", linestyle="--", linewidth=1.0)
        ax.axvline(500.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            xs_ref,
            [row["optimal_split_annual_pairs"] / 1e3 for row in ref_rows],
            color="#f57c00",
            label="Optimized no-buffer",
        )
        ax.plot(
            xs_buf,
            [row["annual_pairs"] / 1e3 for row in buf_rows],
            color="#7b2cbf",
            linewidth=2.4,
            label="Buffered DES",
        )
        ax.set_title(f"{pair.title} (d = {pair.baseline_km:.0f} km)")
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Annual Throughput × 10$^3$ pairs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        "Buffered Annual Throughput With Optimized Geometry\n"
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"$T_2$ = {memory_config.t2_s:g} s, "
        f"capacity = {memory_config.max_modes_per_label} modes per label"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.png", dpi=180)
    fig.savefig(out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.pdf")
    plt.close(fig)


def plot_buffer_consumption(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    max_modes_per_label: int,
) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(14.8, 5.8), sharey=False)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        rows = rows_by_pair[pair.key]
        xs = [row["altitude_km"] for row in rows]
        peak = [row["peak_per_label_buffer_tokens"] for row in rows]
        mean_occ = [row["mean_per_label_buffer_tokens"] for row in rows]
        total_peak = [row["peak_total_buffer_tokens"] for row in rows]
        storage = [row["storage_full_fraction"] for row in rows]
        ax.plot(xs, peak, color="#7b2cbf", label="Peak buffer")
        ax.plot(xs, mean_occ, color="#3c91e6", linestyle="--", label="Mean buffer")
        ax.plot(xs, total_peak, color="#6c757d", linestyle=":", label="Peak total buffer")
        ax.axhline(max_modes_per_label, color="black", linestyle=":", label="256-mode per-link limit")
        ax.axhline(2 * max_modes_per_label, color="black", linestyle="--", alpha=0.6, label="512-mode total limit")
        ax.set_title(f"{pair.title} (d = {pair.baseline_km:.0f} km)")
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Buffered pairs")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, storage, color="#c44536", linestyle="-.", label="Storage-full fraction")
        ax2.set_ylabel("Storage-full fraction")
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left")
    fig.suptitle(
        "Buffered Memory Consumption Under Optimized Geometry\n"
        f"Per-link occupancy and overflow for {max_modes_per_label}-mode memories"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "buffer_consumption_paris_vienna_warsaw_lisbon.png", dpi=180)
    fig.savefig(out_dir / "buffer_consumption_paris_vienna_warsaw_lisbon.pdf")
    plt.close(fig)


def _buffer_scenario_from_row(pair: CityPair, row: Dict[str, float | str]) -> OverpassScenario:
    altitude_km = float(row["altitude_km"])
    return OverpassScenario(
        key=(
            f"{pair.key}_{int(round(float(row['best_delta_km'])))}_"
            f"{int(round(float(row['best_track_angle_deg'])))}_{int(round(altitude_km))}"
        ),
        title=pair.title,
        baseline_km=pair.baseline_km,
        track_angle_deg=float(row["best_track_angle_deg"]),
        delta_km=float(row["best_delta_km"]),
        altitude_km=altitude_km,
        half_window_s=max(180.0, 1.5 * altitude_km / 7.2 + pair.baseline_km / 7.2 + 60.0),
    )


def run_des_pass_sim(
    scenario: OverpassScenario,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> DesPhysicsNoSdn:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    duration_s = 2.0 * scenario.half_window_s
    args = build_des_args(
        duration_s=duration_s,
        min_elevation_deg=min_elevation_deg,
        seed=seed,
        source_rate_hz=source_rate_hz,
        fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
    )
    env = DESEnv(name=f"des_buffer_trace_{scenario.key}_{seed}", default=True)
    sim = DesPhysicsNoSdn(
        name="physics",
        env=env,
        args=args,
        geometry=ParametricOverpassGeometry(scenario, min_elevation_deg),
        fso=build_fso_model(args),
        memory=build_memory_model(args),
        bsm=build_bsm_model(args),
    )
    env.init()
    env.run(end_time=int(duration_s * PS_PER_S), logging=False, summary=False)
    return sim


def _contiguous_true_spans(xs: List[float], mask: List[int]) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    start = None
    prev_x = None
    for x, visible in zip(xs, mask):
        if visible and start is None:
            start = x
        if (not visible) and start is not None and prev_x is not None:
            spans.append((start, prev_x))
            start = None
        prev_x = x
    if start is not None and prev_x is not None:
        spans.append((start, prev_x))
    return spans


def plot_buffer_over_pass_examples(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    base_fso_params: FSOParams | None = None,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
) -> None:
    base_fso_params = base_fso_params or study_fso_params(min_elevation_deg)
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    pair_by_key = {pair.key: pair for pair in pairs}
    examples: List[Tuple[str, Dict[str, float]]] = []

    pv_rows = rows_by_pair.get("paris_vienna", [])
    if pv_rows:
        examples.append(
            (
                "Paris-Vienna best-throughput pass",
                max(pv_rows, key=lambda row: float(row["annual_pairs"])),
            )
        )

    wl_rows = rows_by_pair.get("warsaw_lisbon", [])
    if wl_rows:
        zero_rows = [row for row in wl_rows if float(row["annual_pairs"]) <= 0.0]
        if zero_rows:
            examples.append(
                (
                    "Warsaw-Lisbon blocked pass",
                    max(zero_rows, key=lambda row: float(row["altitude_km"])),
                )
            )
        examples.append(
            (
                "Warsaw-Lisbon active pass",
                max(wl_rows, key=lambda row: float(row["annual_pairs"])),
            )
        )

    if not examples:
        return

    fig, axes = plt.subplots(1, len(examples), figsize=(6.2 * len(examples), 5.6), sharey=False)
    if len(examples) == 1:
        axes = [axes]
    csv_rows: List[Dict[str, float | str]] = []

    for panel_idx, (ax, (label, row)) in enumerate(zip(axes, examples)):
        pair_key = "paris_vienna" if "Paris-Vienna" in label else "warsaw_lisbon"
        pair = pair_by_key[pair_key]
        scenario = _buffer_scenario_from_row(pair, row)
        sim = run_des_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=base_fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )

        xs = [float(metric.t_s) - scenario.half_window_s for metric in sim.metrics]
        occ_a = [int(metric.memory_ar1) for metric in sim.metrics]
        occ_b = [int(metric.memory_r1b) for metric in sim.metrics]
        occ_total = [a + b for a, b in zip(occ_a, occ_b)]

        spans_a = _contiguous_true_spans(xs, [int(metric.ar1_visible) for metric in sim.metrics])
        spans_b = _contiguous_true_spans(xs, [int(metric.r1b_visible) for metric in sim.metrics])
        for idx, (start, end) in enumerate(spans_a):
            ax.axvspan(start, end, color="#f8d7da", alpha=0.45, label="A-R1 visible" if idx == 0 else None, zorder=0)
        for idx, (start, end) in enumerate(spans_b):
            ax.axvspan(start, end, color="#dbeafe", alpha=0.45, label="R1-B visible" if idx == 0 else None, zorder=0)

        ax.plot(xs, occ_a, color="#c44536", label="A-R1 buffer")
        ax.plot(xs, occ_b, color="#3c91e6", label="R1-B buffer")
        ax.plot(xs, occ_total, color="#6c757d", linestyle=":", label="Total buffer")
        ax.axhline(
            memory_config.max_modes_per_label,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"{memory_config.max_modes_per_label} per-link",
        )
        ax.axhline(
            2 * memory_config.max_modes_per_label,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"{2 * memory_config.max_modes_per_label} total",
        )
        ax.set_title(
            f"({chr(97 + panel_idx)}) {label}\n"
            f"h = {float(row['altitude_km']):.0f} km, phi = {float(row['best_track_angle_deg']):.0f}°, "
            f"Delta = {float(row['best_delta_km']):.0f} km"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Buffered pairs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        for x_s, a_val, b_val, total_val, metric in zip(xs, occ_a, occ_b, occ_total, sim.metrics):
            csv_rows.append(
                {
                    "example": label,
                    "pair": pair.title,
                    "altitude_km": float(row["altitude_km"]),
                    "best_delta_km": float(row["best_delta_km"]),
                    "best_track_angle_deg": float(row["best_track_angle_deg"]),
                    "time_s": x_s,
                    "buffer_ar1": a_val,
                    "buffer_r1b": b_val,
                    "buffer_total": total_val,
                    "ar1_visible": int(metric.ar1_visible),
                    "r1b_visible": int(metric.r1b_visible),
                    "throughput_pairs_per_s": float(metric.throughput_pairs_per_s),
                }
            )

    fig.suptitle(
        "Buffer Occupancy Over a Pass\n"
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"$T_2$ = {memory_config.t2_s:g} s, "
        f"capacity = {memory_config.max_modes_per_label} per link"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "buffer_over_pass_examples.png", dpi=180)
    fig.savefig(out_dir / "buffer_over_pass_examples.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "buffer_over_pass_examples.csv", csv_rows)


def _plot_buffer_panel(
    ax,
    *,
    panel_idx: int,
    title: str,
    subtitle: str,
    sim: DesPhysicsNoSdn,
    scenario: OverpassScenario,
    max_modes_per_label: int,
    csv_rows: List[Dict[str, float | str]],
    csv_label: str,
) -> None:
    xs = [float(metric.t_s) - scenario.half_window_s for metric in sim.metrics]
    occ_a = [int(metric.memory_ar1) for metric in sim.metrics]
    occ_b = [int(metric.memory_r1b) for metric in sim.metrics]
    occ_total = [a + b for a, b in zip(occ_a, occ_b)]

    spans_a = _contiguous_true_spans(xs, [int(metric.ar1_visible) for metric in sim.metrics])
    spans_b = _contiguous_true_spans(xs, [int(metric.r1b_visible) for metric in sim.metrics])
    for idx, (start, end) in enumerate(spans_a):
        ax.axvspan(
            start,
            end,
            color="#f8d7da",
            alpha=0.45,
            label="A-R1 visible" if idx == 0 else None,
            zorder=0,
        )
    for idx, (start, end) in enumerate(spans_b):
        ax.axvspan(
            start,
            end,
            color="#dbeafe",
            alpha=0.45,
            label="R1-B visible" if idx == 0 else None,
            zorder=0,
        )

    ax.plot(xs, occ_a, color="#c44536", label="A-R1 buffer")
    ax.plot(xs, occ_b, color="#3c91e6", label="R1-B buffer")
    ax.plot(xs, occ_total, color="#6c757d", linestyle=":", label="Total buffer")
    ax.axhline(
        max_modes_per_label,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"{max_modes_per_label:.0f} per-link",
    )
    ax.axhline(
        2 * max_modes_per_label,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"{2 * max_modes_per_label:.0f} total",
    )
    ax.set_title(f"({chr(97 + panel_idx)}) {title}\n{subtitle}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Buffered pairs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    for x_s, a_val, b_val, total_val, metric in zip(xs, occ_a, occ_b, occ_total, sim.metrics):
        csv_rows.append(
            {
                "example": csv_label,
                "time_s": x_s,
                "buffer_ar1": a_val,
                "buffer_r1b": b_val,
                "buffer_total": total_val,
                "ar1_visible": int(metric.ar1_visible),
                "r1b_visible": int(metric.r1b_visible),
                "throughput_pairs_per_s": float(metric.throughput_pairs_per_s),
                "mean_fidelity_success": float(metric.mean_fidelity_success),
                "successes": int(metric.successes),
            }
        )


def plot_paris_vienna_buffer_over_altitudes(
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    base_fso_params: FSOParams | None = None,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    altitudes_km: Iterable[float] = (240.0, 360.0, 520.0, 840.0),
) -> None:
    base_fso_params = base_fso_params or study_fso_params(min_elevation_deg)
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    pair = next(pair for pair in OPTIMIZED_ANNUAL_CITY_PAIRS if pair.key == "paris_vienna")
    rows = rows_by_pair.get(pair.key, [])
    if not rows:
        return

    selected_rows: List[Dict[str, float]] = []
    for altitude_km in altitudes_km:
        selected_rows.append(
            min(rows, key=lambda row: abs(float(row["altitude_km"]) - float(altitude_km)))
        )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharey=True)
    csv_rows: List[Dict[str, float | str]] = []
    for panel_idx, (ax, row) in enumerate(zip(axes.flat, selected_rows)):
        scenario = _buffer_scenario_from_row(pair, row)
        sim = run_des_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=base_fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )
        _plot_buffer_panel(
            ax,
            panel_idx=panel_idx,
            title="Paris-Vienna",
            subtitle=(
                f"h = {float(row['altitude_km']):.0f} km, "
                f"phi = {float(row['best_track_angle_deg']):.0f}°, "
                f"Delta = {float(row['best_delta_km']):.0f} km"
            ),
            sim=sim,
            scenario=scenario,
            max_modes_per_label=memory_config.max_modes_per_label,
            csv_rows=csv_rows,
            csv_label=(
                f"Paris-Vienna altitude sweep @ {float(row['altitude_km']):.0f} km"
            ),
        )

    fig.tight_layout()
    fig.savefig(out_dir / "paris_vienna_buffer_over_pass_altitudes.png", dpi=180)
    fig.savefig(out_dir / "paris_vienna_buffer_over_pass_altitudes.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "paris_vienna_buffer_over_pass_altitudes.csv", csv_rows)


def plot_paris_vienna_buffer_over_t2(
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    base_fso_params: FSOParams | None = None,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    t2_values_s: Iterable[float] = (0.3, 3.0, 30.0, 300.0),
) -> None:
    base_fso_params = base_fso_params or study_fso_params(min_elevation_deg)
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    pair = next(pair for pair in OPTIMIZED_ANNUAL_CITY_PAIRS if pair.key == "paris_vienna")
    rows = rows_by_pair.get(pair.key, [])
    if not rows:
        return

    best_row = max(rows, key=lambda row: float(row["annual_pairs"]))
    scenario = _buffer_scenario_from_row(pair, best_row)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharey=True)
    csv_rows: List[Dict[str, float | str]] = []
    for panel_idx, (ax, mem_t2_s) in enumerate(zip(axes.flat, t2_values_s)):
        panel_memory = replace(memory_config, t2_s=float(mem_t2_s))
        sim = run_des_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=base_fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=panel_memory,
            bsm_config=bsm_config,
        )
        _plot_buffer_panel(
            ax,
            panel_idx=panel_idx,
            title="Paris-Vienna best-throughput pass",
            subtitle=(
                f"h = {scenario.altitude_km:.0f} km, phi = {scenario.track_angle_deg:.0f}°, "
                f"Delta = {scenario.delta_km:.0f} km, $T_2$ = {float(mem_t2_s):g} s"
            ),
            sim=sim,
            scenario=scenario,
            max_modes_per_label=panel_memory.max_modes_per_label,
            csv_rows=csv_rows,
            csv_label=f"Paris-Vienna T2 sweep @ {float(mem_t2_s):g} s",
        )

    fig.suptitle(
        "Paris-Vienna Buffer Occupancy Over a Pass for Several Memory $T_2$\n"
        f"Best-throughput geometry: h = {scenario.altitude_km:.0f} km, "
        f"phi = {scenario.track_angle_deg:.0f}°, Delta = {scenario.delta_km:.0f} km, "
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"capacity = {memory_config.max_modes_per_label} per link"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "paris_vienna_buffer_over_pass_t2_sweep.png", dpi=180)
    fig.savefig(out_dir / "paris_vienna_buffer_over_pass_t2_sweep.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "paris_vienna_buffer_over_pass_t2_sweep.csv", csv_rows)


def plot_des_fidelity(
    scenarios: List[OverpassScenario],
    stats_by_scenario: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, scenario in zip(axes.flat, scenarios):
        stats = stats_by_scenario[scenario.key]
        xs = stats["time_s"]
        fidelity_median = stats["fidelity_median"]
        fidelity_q1 = stats["fidelity_q1"]
        fidelity_q3 = stats["fidelity_q3"]
        rate_mean = stats["rate_mean"]
        rate_std = stats["rate_std"]

        ax.fill_between(xs, fidelity_q1, fidelity_q3, color="0.85", alpha=0.7)
        line1 = ax.plot(xs, fidelity_median, color="black", label="Median Fidelity")[0]
        ax.set_ylim(0.25, 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Median Fidelity")
        ax.set_title(scenario.title)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        lower = [max(0.0, m - s) for m, s in zip(rate_mean, rate_std)]
        upper = [m + s for m, s in zip(rate_mean, rate_std)]
        ax2.fill_between(xs, lower, upper, color="orange", alpha=0.15)
        line2 = ax2.plot(xs, rate_mean, color="orange", label="Mean SSQR Rate")[0]
        ax2.set_ylabel("Mean SSQR Rate")

        ax.legend([line1, line2], ["Median Fidelity", "Mean SSQR Rate"], loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "des_fidelity_and_ssqr_rate.png", dpi=180)
    fig.savefig(out_dir / "des_fidelity_and_ssqr_rate.pdf")
    plt.close(fig)


def plot_case_overview_composite(out_dir: Path) -> None:
    panels = [
        ("Representative Downlink Loss", out_dir / "representative_downlink_loss.png"),
        ("Representative Distribution Rates", out_dir / "representative_distribution_rates.png"),
        ("Altitudinal PDV Dependence", out_dir / "altitudinal_pdv_dependence.png"),
        ("DES Fidelity And SSQR Rate", out_dir / "des_fidelity_and_ssqr_rate.png"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (title, path) in zip(axes.flat, panels):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        "Paper-Style Composite: Representative Overpasses, Altitude Dependence, and DES Statistics\n"
        "Controller-free Q-NETSIM figures grounded in the repository FSO, memory, and BSM models"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "paper_representative_case_overview_composite.png", dpi=220)
    fig.savefig(out_dir / "paper_representative_case_overview_composite.pdf")
    plt.close(fig)


def main() -> None:
    require_qnet_des("the controller-free DES paper figures")
    parser = argparse.ArgumentParser(
        description="Generate controller-free representative overpass and altitude-study figures."
    )
    parser.add_argument("--out", default=default_paper_out_dir())
    parser.add_argument("--source-rate-hz", type=float, default=5.9e6)
    parser.add_argument("--des-source-rate-hz", type=float, default=5e5)
    parser.add_argument("--ssqr-efficiency", type=float, default=0.003)
    parser.add_argument("--min-elevation-deg", type=float, default=10.0)
    parser.add_argument("--des-repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--annual-contact-fraction", type=float, default=0.04)
    parser.add_argument("--buffer-mem-t2-s", type=float, default=100.0)
    parser.add_argument("--buffer-max-modes", type=int, default=256)
    parser.add_argument("--no-des-fidelity", action="store_true")
    parser.add_argument("--no-parameter-sweeps", action="store_true")
    parser.add_argument("--sweep-repeats", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fso_params = study_fso_params(args.min_elevation_deg)
    fso_model = FSOAnalyticalChannelModel(fso_params)
    memory_config = MemoryStudyConfig(
        platform="COLD_ATOM",
        t2_s=args.buffer_mem_t2_s,
        max_modes_per_label=args.buffer_max_modes,
    )
    bsm_config = BSMStudyConfig()

    traces: Dict[str, List[Dict[str, float]]] = {}
    combined_trace_rows: List[Dict[str, float]] = []
    for scenario in REPRESENTATIVE_OVERPASSES:
        rows = representative_trace(
            scenario,
            fso_model=fso_model,
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.source_rate_hz,
            ssqr_efficiency=args.ssqr_efficiency,
        )
        traces[scenario.key] = rows
        for row in rows:
            combined_trace_rows.append({"scenario": scenario.key, **row})
    write_csv_rows(out_dir / "representative_overpasses.csv", combined_trace_rows)

    altitude_rows: Dict[str, List[Dict[str, float]]] = {}
    combined_altitude_rows: List[Dict[str, float]] = []
    altitudes = list(range(200, 1001, 20))
    for pair in CITY_PAIRS:
        rows = annualised_pdv_curve(
            pair,
            altitudes_km=altitudes,
            fso_model_factory=lambda min_el: FSOAnalyticalChannelModel(
                study_fso_params(min_el)
            ),
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.source_rate_hz,
            ssqr_efficiency=args.ssqr_efficiency,
            annual_contact_fraction=args.annual_contact_fraction,
        )
        altitude_rows[pair.key] = rows
        for row in rows:
            combined_altitude_rows.append(
                {
                    "pair": pair.key,
                    "pair_distance_km": pair.baseline_km,
                    "track_angle_deg": pair.track_angle_deg,
                    **row,
                }
            )
    write_csv_rows(out_dir / "altitude_dependence.csv", combined_altitude_rows)

    requested_rows: Dict[str, List[Dict[str, float]]] = {}
    requested_combined_rows: List[Dict[str, float]] = []
    for pair in REQUESTED_ANNUAL_CITY_PAIRS:
        rows = annualised_pdv_curve(
            pair,
            altitudes_km=altitudes,
            fso_model_factory=lambda min_el: FSOAnalyticalChannelModel(
                study_fso_params(min_el)
            ),
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.source_rate_hz,
            ssqr_efficiency=args.ssqr_efficiency,
            annual_contact_fraction=args.annual_contact_fraction,
        )
        requested_rows[pair.key] = rows
        for row in rows:
            requested_combined_rows.append(
                {
                    "pair": pair.key,
                    "pair_distance_km": pair.baseline_km,
                    "track_angle_proxy_deg": pair.track_angle_deg,
                    **row,
                }
            )
    write_csv_rows(
        out_dir / "annual_pair_distribution_paris_berlin_lisbon_warsaw.csv",
        requested_combined_rows,
    )

    optimized_rows: Dict[str, List[Dict[str, float]]] = {}
    optimized_combined_rows: List[Dict[str, float]] = []
    for pair in OPTIMIZED_ANNUAL_CITY_PAIRS:
        rows = annualised_pdv_curve_optimized_geometry(
            pair,
            altitudes_km=altitudes,
            fso_model_factory=lambda min_el: FSOAnalyticalChannelModel(
                study_fso_params(min_el)
            ),
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.source_rate_hz,
            ssqr_efficiency=args.ssqr_efficiency,
            annual_contact_fraction=args.annual_contact_fraction,
        )
        optimized_rows[pair.key] = rows
        for row in rows:
            optimized_combined_rows.append(
                {
                    "pair": pair.key,
                    "pair_distance_km": pair.baseline_km,
                    **row,
                }
            )
    write_csv_rows(
        out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.csv",
        optimized_combined_rows,
    )

    buffered_rows: Dict[str, List[Dict[str, float]]] = {}
    buffered_combined_rows: List[Dict[str, float]] = []
    buffered_altitudes = list(range(200, 1001, 40))
    for index, pair in enumerate(OPTIMIZED_ANNUAL_CITY_PAIRS):
        rows = annualised_des_buffered_curve_optimized_geometry(
            pair,
            altitudes_km=buffered_altitudes,
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.des_source_rate_hz,
            annual_contact_fraction=args.annual_contact_fraction,
            seed=args.seed + 70000 + 1000 * index,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )
        buffered_rows[pair.key] = rows
        for row in rows:
            buffered_combined_rows.append(
                {
                    "pair": pair.key,
                    "pair_distance_km": pair.baseline_km,
                    **row,
                }
            )
    write_csv_rows(
        out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.csv",
        buffered_combined_rows,
    )

    plot_representative_losses(
        REPRESENTATIVE_OVERPASSES,
        traces,
        out_dir,
        fso_model=fso_model,
    )
    plot_representative_rates(REPRESENTATIVE_OVERPASSES, traces, out_dir)
    plot_altitude_dependence(altitude_rows, out_dir)
    plot_requested_annual_pair_distribution(
        REQUESTED_ANNUAL_CITY_PAIRS,
        requested_rows,
        out_dir,
    )
    plot_optimized_annual_pair_distribution(
        OPTIMIZED_ANNUAL_CITY_PAIRS,
        optimized_rows,
        out_dir,
    )
    plot_buffered_annual_throughput(
        OPTIMIZED_ANNUAL_CITY_PAIRS,
        buffered_rows,
        optimized_rows,
        out_dir,
        memory_config=memory_config,
    )
    plot_buffer_consumption(
        OPTIMIZED_ANNUAL_CITY_PAIRS,
        buffered_rows,
        out_dir,
        max_modes_per_label=args.buffer_max_modes,
    )
    plot_buffer_over_pass_examples(
        OPTIMIZED_ANNUAL_CITY_PAIRS,
        buffered_rows,
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 90000,
        base_fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
    )
    plot_paris_vienna_buffer_over_altitudes(
        buffered_rows,
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 91000,
        base_fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
    )
    plot_paris_vienna_buffer_over_t2(
        buffered_rows,
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 92000,
        base_fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
    )

    if not bool(args.no_des_fidelity):
        fidelity_rows: List[Dict[str, float]] = []
        stats_by_scenario: Dict[str, Dict[str, List[float]]] = {}
        for index, scenario in enumerate(REPRESENTATIVE_OVERPASSES):
            stats = run_des_fidelity_stats(
                scenario,
                repeats=args.des_repeats,
                seed=args.seed + 1000 * index,
                min_elevation_deg=args.min_elevation_deg,
                source_rate_hz=args.des_source_rate_hz,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
            )
            stats_by_scenario[scenario.key] = stats
            for idx, t_s in enumerate(stats["time_s"]):
                fidelity_rows.append(
                    {
                        "scenario": scenario.key,
                        "time_s": t_s,
                        "fidelity_median": stats["fidelity_median"][idx],
                        "fidelity_q1": stats["fidelity_q1"][idx],
                        "fidelity_q3": stats["fidelity_q3"][idx],
                        "rate_mean": stats["rate_mean"][idx],
                        "rate_std": stats["rate_std"][idx],
                    }
                )
        write_csv_rows(out_dir / "des_fidelity_stats.csv", fidelity_rows)
        plot_des_fidelity(REPRESENTATIVE_OVERPASSES, stats_by_scenario, out_dir)

    if not bool(args.no_parameter_sweeps):
        plot_des_parameter_sweeps(
            REPRESENTATIVE_OVERPASSES,
            out_dir,
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.des_source_rate_hz,
            seed=args.seed + 50000,
            t2_values_s=[0.3, 3.0, 30.0, 300.0],
            rytov_values=[0.02, 0.08, 0.2, 0.5],
            sweep_repeats=args.sweep_repeats,
            base_fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )

    plot_case_overview_composite(out_dir)
    print(f"[des-no-sdn] wrote figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
