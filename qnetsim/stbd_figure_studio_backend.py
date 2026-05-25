from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from channels.fso import FSOParams
from stbd_constellation.config import (
    BSMParameterSet,
    MemoryParameterSet,
    ObjectiveWeights,
    OptimizerSettings,
    SearchRanges,
    StbdConstellationConfig,
    TimeSettings,
    default_config,
)
from stbd_constellation.main import RunOptimizationResult, run_optimization


GALLERY_MANIFEST = "gallery_manifest.json"


def default_stbd_out_root() -> Path:
    return Path(__file__).resolve().parent / "Output" / "gui_stbd_constellation_runs"


def save_gallery_manifest(out_dir: Path, result: RunOptimizationResult, *, variant: str) -> None:
    image_items = []
    for key, path in result.artifacts.items():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image_items.append(
            {
                "title": key.replace("_", " ").title(),
                "path": str(path.resolve()),
            }
        )
    payload = {
        "out_dir": str(out_dir.resolve()),
        "variant": str(variant),
        "images": image_items,
        "files": {
            "best_config_json": str(result.best_config_path.resolve()),
            "summary_txt": str(result.summary_path.resolve()),
            "timeseries_csv": str(result.timeseries_path.resolve()),
            "launch_results_json": str(result.launch_results_path.resolve()),
            **{
                key: str(path.resolve())
                for key, path in result.artifacts.items()
            },
        },
    }
    (out_dir / GALLERY_MANIFEST).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_gallery_manifest(out_dir: Path) -> dict | None:
    manifest = out_dir / GALLERY_MANIFEST
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))


def _memory_set(
    *,
    platform: str,
    t2_s: float,
    d: float,
    n: float,
    t_mem_s: float,
    m: float,
    eta_mode0: float,
    eta_ti: float,
    max_modes_per_label: int,
) -> MemoryParameterSet:
    return MemoryParameterSet(
        platform=str(platform).upper(),
        t2_s=float(t2_s),
        d=float(d),
        n=float(n),
        t_mem_s=float(t_mem_s),
        m=float(m),
        eta_mode0=float(eta_mode0),
        eta_ti=float(eta_ti),
        max_modes_per_label=int(max_modes_per_label),
    )


def run_stbd_constellation_optimization(
    *,
    run_name: str,
    output_root: Path,
    controller_backend: str,
    optimizer_method: str,
    random_search_trials: int,
    ga_population: int,
    ga_generations: int,
    elite_fraction: float,
    mutation_rate: float,
    seed: int,
    pilot_horizon_s: float,
    simulation_horizon_s: float,
    geometry_step_s: float,
    snapshot_s: float,
    launch_step_s: float,
    metrics_dt_s: float,
    gen_delay_s: float,
    swap_dt_s: float,
    paced_delivery_dt_s: float,
    bsm_round_duration_s: float,
    source_rate_hz: float,
    planner_fidelity_threshold: float,
    stochastic_link_state: bool,
    perfect_isl: bool,
    support_threshold: int,
    evaluation_repeats: int,
    f_min: float,
    planner_horizon_override_s: float | None,
    execution_mode: str,
    objective_w1: float,
    objective_w2: float,
    objective_w3: float,
    objective_w4: float,
    objective_w5: float,
    planes_min: int,
    planes_max: int,
    sats_min: int,
    sats_max: int,
    altitude_min_km: float,
    altitude_max_km: float,
    inclination_min_deg: float,
    inclination_max_deg: float,
    raan_spacing_min_deg: float,
    raan_spacing_max_deg: float,
    max_isl_range_min_km: float,
    max_isl_range_max_km: float,
    min_elevation_min_deg: float,
    min_elevation_max_deg: float,
    enabled_memory_keys: Sequence[str],
    fso_params: FSOParams,
    isl_fso_params: FSOParams,
    bsm_params: BSMParameterSet,
    memory_sets: Dict[str, MemoryParameterSet],
) -> RunOptimizationResult:
    cfg = default_config()
    cfg.results_root = Path(output_root).expanduser().resolve() / run_name
    cfg.optimizer = OptimizerSettings(
        method=str(optimizer_method),
        random_search_trials=int(random_search_trials),
        ga_population=int(ga_population),
        ga_generations=int(ga_generations),
        elite_fraction=float(elite_fraction),
        mutation_rate=float(mutation_rate),
        seed=int(seed),
        pilot_horizon_s=float(pilot_horizon_s),
    )

    cfg.time = replace(
        cfg.time,
        simulation_horizon_s=float(simulation_horizon_s),
        geometry_step_s=float(geometry_step_s),
        snapshot_s=float(snapshot_s),
        launch_step_s=float(launch_step_s),
        metrics_dt_s=float(metrics_dt_s),
        gen_delay_s=float(gen_delay_s),
        swap_dt_s=float(swap_dt_s),
        paced_delivery_dt_s=float(paced_delivery_dt_s),
        bsm_round_duration_s=float(bsm_round_duration_s),
    )
    cfg.source_rate_hz = float(source_rate_hz)
    cfg.planner_fidelity_threshold = float(planner_fidelity_threshold)
    cfg.stochastic_link_state = bool(stochastic_link_state)
    cfg.perfect_isl = bool(perfect_isl)
    cfg.support_threshold = int(support_threshold)
    cfg.evaluation_repeats = int(evaluation_repeats)
    cfg.f_min = float(f_min)
    cfg.planner_horizon_override_s = (
        None
        if planner_horizon_override_s is None or float(planner_horizon_override_s) <= 0.0
        else float(planner_horizon_override_s)
    )
    cfg.execution_mode = str(execution_mode)
    cfg.controller_backend = str(controller_backend).lower().strip()

    cfg.objective = ObjectiveWeights(
        w1_path_availability=float(objective_w1),
        w2_fidelity=float(objective_w2),
        w3_throughput=float(objective_w3),
        w4_latency=float(objective_w4),
        w5_satellite_penalty=float(objective_w5),
    )

    cfg.search = replace(
        cfg.search,
        planes_min=int(planes_min),
        planes_max=int(planes_max),
        sats_per_plane_min=int(sats_min),
        sats_per_plane_max=int(sats_max),
        altitude_km_min=float(altitude_min_km),
        altitude_km_max=float(altitude_max_km),
        inclination_deg_min=float(inclination_min_deg),
        inclination_deg_max=float(inclination_max_deg),
        raan_spacing_deg_min=float(raan_spacing_min_deg),
        raan_spacing_deg_max=float(raan_spacing_max_deg),
        max_isl_range_km_min=float(max_isl_range_min_km),
        max_isl_range_km_max=float(max_isl_range_max_km),
        min_elevation_deg_min=float(min_elevation_min_deg),
        min_elevation_deg_max=float(min_elevation_max_deg),
    )
    cfg.fso_sets = {"baseline": fso_params}
    cfg.isl_fso_params = isl_fso_params
    cfg.bsm_sets = {"baseline": bsm_params}
    filtered_memory = {
        str(key).upper(): value
        for key, value in memory_sets.items()
        if str(key).upper() in {str(item).upper() for item in enabled_memory_keys}
    }
    cfg.memory_sets = filtered_memory or {
        str(key).upper(): value for key, value in memory_sets.items()
    }

    result = run_optimization(cfg, out_dir=cfg.results_root)
    mode = str(execution_mode).lower().strip()
    backend = str(controller_backend).lower().strip()
    if backend == "sdn_local":
        backend_suffix = " + SDN Local"
    elif backend == "sdn_ryu":
        backend_suffix = " + SDN Ryu"
    else:
        backend_suffix = ""
    if mode == "segmented":
        variant = f"STBD segmented{backend_suffix}"
    elif mode == "paced":
        variant = f"STBD paced{backend_suffix}"
    else:
        variant = f"STBD basic{backend_suffix}"
    save_gallery_manifest(result.out_dir, result, variant=variant)
    return result
