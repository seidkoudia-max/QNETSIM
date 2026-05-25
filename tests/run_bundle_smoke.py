from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BUNDLE_ROOT / "qnetsim"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import paper_figure_studio  # noqa: F401
import paper_ground_studio  # noqa: F401
import paper_stbd_studio  # noqa: F401
from channels.fiber import FiberParams
from channels.fso import FSOParams
from des_no_sdn_paper_figures import BSMStudyConfig, MemoryStudyConfig
from des_stbd_physics_simulator import (
    build_parser as build_stbd_des_parser,
    run_once as run_stbd_des_once,
    summarize_run as summarize_stbd_des_run,
    write_outputs as write_stbd_des_outputs,
)
from paper_figure_studio_backend import StudioPlotSelection, generate_backend_plots
from paper_ground_studio_backend import run_ground_scenario


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_single_repeater_smoke(root: Path) -> dict[str, str]:
    out_dir = _reset_dir(root / "single_repeater_des")
    selection = StudioPlotSelection(
        representative_overpasses=True,
        archetype_sweeps=False,
        annual_throughput=False,
        buffered_annual_throughput=False,
        paris_vienna_buffer_altitudes=False,
        paris_vienna_buffer_t2=False,
        buffer_examples=False,
        buffer_consumption=False,
        overview_composite=False,
    )
    artifacts = generate_backend_plots(
        out_dir,
        backend_key="des_only",
        selection=selection,
        fso_params=FSOParams(),
        memory_config=MemoryStudyConfig(),
        bsm_config=BSMStudyConfig(),
        source_rate_hz=100_000.0,
        des_source_rate_hz=50_000.0,
        ssqr_efficiency=0.5,
        min_elevation_deg=10.0,
        des_repeats=1,
        sweep_repeats=1,
        seed=123,
        annual_contact_fraction=0.05,
        t2_values_s=[3.0],
        rytov_values=[0.05],
        paris_vienna_altitudes_km=[360.0],
    )
    return {key: str(path) for key, path in artifacts.items()}


def run_stbd_smoke(root: Path) -> dict[str, str]:
    out_dir = _reset_dir(root / "stbd_des")
    args = build_stbd_des_parser().parse_args([])
    args.out = str(out_dir)
    args.duration_s = 600.0
    args.metrics_dt_s = 30.0
    args.link_update_dt_s = 30.0
    args.snapshot_s = 60.0
    args.source_rate_hz = 200_000.0
    args.max_modes_per_label = 128
    args.max_bsm_per_cycle = 128
    args.no_event_log = True
    args.no_plots = False
    args.stochastic_link_state = False
    args.perfect_isl = True
    sim, route_result, route_edges = run_stbd_des_once(args)
    write_stbd_des_outputs(sim, stbd_result=route_result, route_edges=route_edges, out_dir=out_dir)
    summary = summarize_stbd_des_run(sim, stbd_result=route_result, route_edges=route_edges)
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return {
        "throughput_fidelity": str((out_dir / "stbd_throughput_fidelity.png").resolve()),
        "link_budget": str((out_dir / "stbd_link_budget.png").resolve()),
        "summary": str((out_dir / "smoke_summary.json").resolve()),
    }


def run_ground_smoke(root: Path, *, controller_backend: str, run_name: str) -> dict[str, str]:
    out_root = _reset_dir(root / run_name)
    result = run_ground_scenario(
        run_name=run_name,
        output_root=out_root,
        controller_backend=controller_backend,
        duration_s=900.0,
        metrics_dt_s=30.0,
        link_update_dt_s=10.0,
        gen_delay_s=0.2,
        swap_dt_s=0.2,
        source_rate_hz=200_000.0,
        max_modes_per_label=32,
        max_bsm_per_cycle=32,
        f_min=0.25,
        min_elevation_deg=10.0,
        visibility_scan_s=6 * 3600.0,
        visibility_scan_step_s=30.0,
        visibility_prebuffer_s=120.0,
        pass_selection="memory_aware",
        pass_candidate_limit=8,
        pass_prefilter_step_s=60.0,
        force_visible=True,
        fso_params=FSOParams(enable_stochastic_fading=False, enable_pointing_jitter=False),
        fiber_params=FiberParams(),
        fiber_length_km=10.0,
        mem_platform="REID",
        mem_t2=600.0,
        mem_d=7.0,
        mem_t_mem=600.0,
        mem_n=1.5,
        mem_m=1.0,
        mem_eta_mode0=0.9,
        mem_eta_ti=0.3,
        bsm_v_hom=0.97,
        bsm_eta_det=0.82,
        bsm_sigma_det_s=50e-12,
        bsm_delta_t_s=200e-12,
        bsm_p_dc=1e-6,
        bsm_swap_model="paper",
        seed=456,
        enable_t2_sweep=False,
        enable_rytov_sweep=False,
        enable_buffer_sweep=False,
        enable_multi_pass_visibility=False,
    )
    return {key: str(path) for key, path in result.artifacts.items()}


def main() -> None:
    root = _reset_dir(Path("/tmp/qnetsim_standalone_smoke"))
    results = {
        "imports": "ok",
        "single_repeater_des": run_single_repeater_smoke(root),
        "stbd_des": run_stbd_smoke(root),
        "ground_des": run_ground_smoke(root, controller_backend="des_only", run_name="ground_des"),
        "ground_ryu": run_ground_smoke(root, controller_backend="sdn_ryu", run_name="ground_ryu"),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
