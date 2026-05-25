from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .config import StbdConstellationConfig
from .constellation_geometry import (
    ConstellationProposal,
    crossover_proposals,
    mutate_proposal,
    sample_random_proposal,
)
from .metrics_collector import CandidateRecord, MetricsSummary, collect_metrics
from .qnetsim_adapter import AdapterRunResult, run_constellation_des
from .scenario_builder import ScenarioBundle, build_scenario


@dataclass(frozen=True)
class CandidateEvaluation:
    proposal: ConstellationProposal
    metrics: MetricsSummary
    run_result: AdapterRunResult
    candidate_id: str
    search_stage: str
    out_dir: Path

    def to_record(self) -> CandidateRecord:
        return CandidateRecord(
            proposal=self.proposal,
            metrics=self.metrics,
            candidate_id=self.candidate_id,
            search_stage=self.search_stage,
        )


@dataclass(frozen=True)
class OptimizationResult:
    best: CandidateEvaluation
    candidates: List[CandidateRecord]


def _candidate_out_dir(root: Path, candidate_id: str) -> Path:
    path = root / "candidates" / candidate_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_proposal(
    cfg: StbdConstellationConfig,
    proposal: ConstellationProposal,
    *,
    root_out_dir: Path,
    candidate_id: str,
    search_stage: str,
    horizon_s: float,
    repeats: int,
    seed: int,
) -> CandidateEvaluation:
    out_dir = _candidate_out_dir(root_out_dir, candidate_id)
    bundle: ScenarioBundle = build_scenario(
        cfg,
        proposal,
        out_dir=out_dir,
        horizon_s=float(horizon_s),
    )
    run_result = run_constellation_des(
        bundle,
        launch_step_s=float(cfg.time.launch_step_s),
        repeats=int(repeats),
        seed=int(seed),
        series_bin_s=float(cfg.time.metrics_dt_s),
    )
    metrics = collect_metrics(
        proposal,
        run_result,
        weights=cfg.objective,
        search=cfg.search,
    )
    return CandidateEvaluation(
        proposal=proposal,
        metrics=metrics,
        run_result=run_result,
        candidate_id=candidate_id,
        search_stage=search_stage,
        out_dir=out_dir,
    )


def _random_search(
    cfg: StbdConstellationConfig,
    *,
    root_out_dir: Path,
    rng: random.Random,
) -> List[CandidateEvaluation]:
    memory_keys = tuple(cfg.memory_sets.keys())
    bsm_keys = tuple(cfg.bsm_sets.keys())
    fso_keys = tuple(cfg.fso_sets.keys())
    evaluations: List[CandidateEvaluation] = []
    for idx in range(int(cfg.optimizer.random_search_trials)):
        proposal = sample_random_proposal(
            rng=rng,
            search=cfg.search,
            memory_keys=memory_keys,
            bsm_keys=bsm_keys,
            fso_keys=fso_keys,
        )
        evaluation = evaluate_proposal(
            cfg,
            proposal,
            root_out_dir=root_out_dir,
            candidate_id=f"random_{idx:03d}",
            search_stage="random_search",
            horizon_s=float(cfg.optimizer.pilot_horizon_s),
            repeats=int(cfg.evaluation_repeats),
            seed=int(cfg.optimizer.seed) + idx * 7919,
        )
        evaluations.append(evaluation)
    return evaluations


def _genetic_search(
    cfg: StbdConstellationConfig,
    *,
    root_out_dir: Path,
    rng: random.Random,
) -> List[CandidateEvaluation]:
    memory_keys = tuple(cfg.memory_sets.keys())
    bsm_keys = tuple(cfg.bsm_sets.keys())
    fso_keys = tuple(cfg.fso_sets.keys())
    population_size = max(2, int(cfg.optimizer.ga_population))
    generations = max(1, int(cfg.optimizer.ga_generations))
    elite_count = max(1, int(round(population_size * float(cfg.optimizer.elite_fraction))))

    population = [
        sample_random_proposal(
            rng=rng,
            search=cfg.search,
            memory_keys=memory_keys,
            bsm_keys=bsm_keys,
            fso_keys=fso_keys,
        )
        for _ in range(population_size)
    ]

    evaluations: List[CandidateEvaluation] = []
    candidate_counter = 0
    for generation in range(generations):
        generation_evals: List[CandidateEvaluation] = []
        for proposal in population:
            evaluation = evaluate_proposal(
                cfg,
                proposal,
                root_out_dir=root_out_dir,
                candidate_id=f"ga_g{generation:02d}_{candidate_counter:03d}",
                search_stage=f"ga_generation_{generation}",
                horizon_s=float(cfg.optimizer.pilot_horizon_s),
                repeats=int(cfg.evaluation_repeats),
                seed=int(cfg.optimizer.seed) + generation * 100003 + candidate_counter,
            )
            candidate_counter += 1
            generation_evals.append(evaluation)
        generation_evals.sort(key=lambda item: float(item.metrics.score), reverse=True)
        evaluations.extend(generation_evals)

        elites = [item.proposal for item in generation_evals[:elite_count]]
        next_population: List[ConstellationProposal] = list(elites)
        while len(next_population) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover_proposals(parent_a, parent_b, rng=rng)
            child = mutate_proposal(
                child,
                rng=rng,
                search=cfg.search,
                memory_keys=memory_keys,
                bsm_keys=bsm_keys,
                fso_keys=fso_keys,
                mutation_rate=float(cfg.optimizer.mutation_rate),
            )
            next_population.append(child)
        population = next_population[:population_size]
    return evaluations


def optimize_constellation(
    cfg: StbdConstellationConfig,
    *,
    root_out_dir: Path,
) -> OptimizationResult:
    rng = random.Random(int(cfg.optimizer.seed))
    method = str(cfg.optimizer.method).strip().lower()
    if method == "genetic_algorithm":
        candidate_evals = _genetic_search(cfg, root_out_dir=root_out_dir, rng=rng)
    else:
        candidate_evals = _random_search(cfg, root_out_dir=root_out_dir, rng=rng)

    if not candidate_evals:
        raise RuntimeError("optimizer produced no candidate evaluations")

    candidate_evals.sort(key=lambda item: float(item.metrics.score), reverse=True)
    search_best = candidate_evals[0]
    final_best = evaluate_proposal(
        cfg,
        search_best.proposal,
        root_out_dir=root_out_dir,
        candidate_id="best_full_horizon",
        search_stage="final_full_horizon",
        horizon_s=float(cfg.time.simulation_horizon_s),
        repeats=int(cfg.evaluation_repeats),
        seed=int(cfg.optimizer.seed) + 999983,
    )

    records = [item.to_record() for item in candidate_evals]
    records.append(final_best.to_record())
    return OptimizationResult(best=final_best, candidates=records)
