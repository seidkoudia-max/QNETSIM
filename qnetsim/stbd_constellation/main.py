from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import StbdConstellationConfig, default_config
from .constellation_optimizer import OptimizationResult, evaluate_proposal, optimize_constellation
from .metrics_collector import (
    launch_results_to_rows,
    write_best_config_json,
    write_summary,
)
from .plotting import generate_result_artifacts


@dataclass(frozen=True)
class RunOptimizationResult:
    out_dir: Path
    best_config_path: Path
    summary_path: Path
    timeseries_path: Path
    launch_results_path: Path
    artifacts: Dict[str, Path]
    optimization: OptimizationResult


def run_optimization(
    cfg: StbdConstellationConfig | None = None,
    *,
    out_dir: Path | None = None,
) -> RunOptimizationResult:
    cfg = default_config() if cfg is None else cfg
    root = (out_dir or cfg.results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    search_cfg = deepcopy(cfg)
    controller_backend = str(getattr(cfg, "controller_backend", "des_only")).lower().strip()
    if controller_backend != "des_only":
        search_cfg.controller_backend = "des_only"

    optimization = optimize_constellation(search_cfg, root_out_dir=root)
    best = optimization.best
    if controller_backend != "des_only":
        best = evaluate_proposal(
            cfg,
            best.proposal,
            root_out_dir=root,
            candidate_id=f"best_full_horizon_{controller_backend}",
            search_stage=f"final_full_horizon_{controller_backend}",
            horizon_s=float(cfg.time.simulation_horizon_s),
            repeats=int(cfg.evaluation_repeats),
            seed=int(cfg.optimizer.seed) + 999983,
        )
        optimization = OptimizationResult(
            best=best,
            candidates=[*optimization.candidates, best.to_record()],
        )
    pair_label = f"{cfg.ground_luxembourg.name}–{cfg.ground_singapore.name}"

    best_config_path = root / "best_config.json"
    summary_path = root / "summary.txt"
    launch_results_path = root / "launch_results.json"
    write_best_config_json(
        best_config_path,
        proposal=best.proposal,
        metrics=best.metrics,
    )
    write_summary(
        summary_path,
        proposal=best.proposal,
        metrics=best.metrics,
    )
    launch_rows = launch_results_to_rows(best.run_result.launch_results)
    launch_results_path.write_text(json.dumps(launch_rows, indent=2), encoding="utf-8")

    artifacts = generate_result_artifacts(
        root,
        cfg=cfg,
        pair_label=pair_label,
        metrics=best.metrics,
        timeseries=best.run_result.timeseries,
        launch_results=best.run_result.launch_results,
        raw_event_rows=best.run_result.raw_event_rows,
        candidates=optimization.candidates,
    )
    return RunOptimizationResult(
        out_dir=root,
        best_config_path=best_config_path,
        summary_path=summary_path,
        timeseries_path=artifacts["timeseries_csv"],
        launch_results_path=launch_results_path,
        artifacts=artifacts,
        optimization=optimization,
    )


if __name__ == "__main__":
    result = run_optimization()
    print(result.out_dir)
