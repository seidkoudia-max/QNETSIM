from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import ObjectiveWeights, SearchRanges
from .constellation_geometry import ConstellationProposal
from .qnetsim_adapter import AdapterRunResult, LaunchResult


@dataclass(frozen=True)
class MetricsSummary:
    average_end_to_end_fidelity: float
    average_delayed_fidelity: float
    average_entanglement_throughput: float
    average_latency_s: float
    peak_memory_occupancy: int
    average_memory_occupancy: float
    average_swap_success_probability: float
    average_token_lifetime_s: float
    request_completion_ratio: float
    path_availability_fraction: float
    average_hops: float
    average_service_duration_s: float
    nonzero_service_bins: int
    total_service_bins: int
    number_of_satellites_penalty: float
    score: float


@dataclass(frozen=True)
class CandidateRecord:
    proposal: ConstellationProposal
    metrics: MetricsSummary
    candidate_id: str
    search_stage: str

    def to_flat_dict(self) -> dict:
        payload = {
            "candidate_id": self.candidate_id,
            "search_stage": self.search_stage,
            **self.proposal.to_dict(),
        }
        payload.update(asdict(self.metrics))
        return payload


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if not math.isnan(float(value)) and float(weight) > 0.0
    ]
    if not pairs:
        return 0.0
    total_weight = sum(weight for _value, weight in pairs)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _mean(values: Iterable[float]) -> float:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    if not cleaned:
        return 0.0
    return sum(cleaned) / len(cleaned)


def collect_metrics(
    proposal: ConstellationProposal,
    run_result: AdapterRunResult,
    *,
    weights: ObjectiveWeights,
    search: SearchRanges,
) -> MetricsSummary:
    timeseries = list(run_result.timeseries)
    launches = list(run_result.launch_results)

    throughput_weights = [
        max(0.0, float(row.throughput_pairs_per_s)) for row in timeseries
    ]
    fidelity = _weighted_mean(
        (float(row.fidelity) for row in timeseries),
        throughput_weights,
    )
    delayed_fidelity = _weighted_mean(
        (float(row.delayed_fidelity) for row in timeseries),
        throughput_weights,
    )
    average_throughput = _mean(float(row.throughput_pairs_per_s) for row in timeseries)
    path_availability_fraction = _mean(float(row.availability) for row in timeseries)
    average_hops = _weighted_mean(
        (float(row.hops) for row in timeseries),
        throughput_weights,
    )
    nonzero_service_bins = sum(
        1 for row in timeseries if float(row.availability) > 0.0
    )
    total_service_bins = len(timeseries)

    successful_launches = [row for row in launches if int(row.success_count) > 0]
    launch_weights = [
        max(1.0, float(row.success_count)) for row in successful_launches
    ]
    average_latency_s = _weighted_mean(
        (float(row.average_latency_s) for row in successful_launches),
        launch_weights,
    )
    average_swap_success_probability = _weighted_mean(
        (float(row.swap_success_probability) for row in successful_launches),
        launch_weights,
    )
    average_token_lifetime_s = _weighted_mean(
        (float(row.token_lifetime_s) for row in successful_launches),
        launch_weights,
    )
    average_service_duration_s = _mean(
        float(row.route_end_s) - float(row.route_start_s)
        for row in launches
        if row.route_found
        and not math.isnan(float(row.route_start_s))
        and not math.isnan(float(row.route_end_s))
    )
    average_memory_occupancy = _mean(
        float(row.max_memory_occupancy) for row in launches
    )
    peak_memory_occupancy = max(
        (int(row.max_memory_occupancy) for row in launches),
        default=0,
    )
    request_completion_ratio = (
        float(len(successful_launches)) / float(max(1, len(launches)))
    )
    satellite_penalty = float(proposal.total_satellites) / float(
        max(1, search.max_satellites)
    )

    score = (
        float(weights.w1_path_availability) * float(path_availability_fraction)
        + float(weights.w2_fidelity) * float(fidelity)
        + float(weights.w3_throughput) * float(average_throughput)
        - float(weights.w4_latency) * float(average_latency_s)
        - float(weights.w5_satellite_penalty) * float(satellite_penalty)
    )

    return MetricsSummary(
        average_end_to_end_fidelity=float(fidelity),
        average_delayed_fidelity=float(delayed_fidelity),
        average_entanglement_throughput=float(average_throughput),
        average_latency_s=float(average_latency_s),
        peak_memory_occupancy=int(peak_memory_occupancy),
        average_memory_occupancy=float(average_memory_occupancy),
        average_swap_success_probability=float(average_swap_success_probability),
        average_token_lifetime_s=float(average_token_lifetime_s),
        request_completion_ratio=float(request_completion_ratio),
        path_availability_fraction=float(path_availability_fraction),
        average_hops=float(average_hops),
        average_service_duration_s=float(average_service_duration_s),
        nonzero_service_bins=int(nonzero_service_bins),
        total_service_bins=int(total_service_bins),
        number_of_satellites_penalty=float(satellite_penalty),
        score=float(score),
    )


def write_summary(
    out_path: Path,
    *,
    proposal: ConstellationProposal,
    metrics: MetricsSummary,
) -> None:
    lines = [
        "STBD Luxembourg-Singapore Constellation Optimization Summary",
        "",
        "Best constellation proposal:",
        f"  planes: {proposal.planes}",
        f"  satellites_per_plane: {proposal.sats_per_plane}",
        f"  total_satellites: {proposal.total_satellites}",
        f"  altitude_km: {proposal.altitude_km:.3f}",
        f"  inclination_deg: {proposal.inclination_deg:.3f}",
        f"  raan_spacing_deg: {proposal.raan_spacing_deg:.3f}",
        f"  phasing_factor: {proposal.phasing_factor}",
        f"  max_isl_range_km: {proposal.max_isl_range_km:.3f}",
        f"  min_elevation_deg: {proposal.min_elevation_deg:.3f}",
        f"  memory_key: {proposal.memory_key}",
        f"  bsm_key: {proposal.bsm_key}",
        f"  fso_key: {proposal.fso_key}",
        "",
        "Metrics:",
        f"  score: {metrics.score:.6f}",
        f"  path_availability_fraction: {metrics.path_availability_fraction:.6f}",
        f"  average_end_to_end_fidelity: {metrics.average_end_to_end_fidelity:.6f}",
        f"  average_delayed_fidelity: {metrics.average_delayed_fidelity:.6f}",
        f"  average_entanglement_throughput: {metrics.average_entanglement_throughput:.6f}",
        f"  average_latency_s: {metrics.average_latency_s:.6f}",
        f"  average_memory_occupancy: {metrics.average_memory_occupancy:.6f}",
        f"  peak_memory_occupancy: {metrics.peak_memory_occupancy}",
        f"  average_swap_success_probability: {metrics.average_swap_success_probability:.6f}",
        f"  average_token_lifetime_s: {metrics.average_token_lifetime_s:.6f}",
        f"  request_completion_ratio: {metrics.request_completion_ratio:.6f}",
        f"  average_hops: {metrics.average_hops:.6f}",
        f"  average_service_duration_s: {metrics.average_service_duration_s:.6f}",
        f"  nonzero_service_bins: {metrics.nonzero_service_bins}",
        f"  total_service_bins: {metrics.total_service_bins}",
        f"  number_of_satellites_penalty: {metrics.number_of_satellites_penalty:.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_config_json(
    out_path: Path,
    *,
    proposal: ConstellationProposal,
    metrics: MetricsSummary,
) -> None:
    payload = {
        "proposal": proposal.to_dict(),
        "metrics": asdict(metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def launch_results_to_rows(launches: Sequence[LaunchResult]) -> List[dict]:
    rows: List[dict] = []
    for row in launches:
        rows.append(
            {
                "launch_s": float(row.launch_s),
                "route_found": int(bool(row.route_found)),
                "route_start_s": float(row.route_start_s),
                "route_end_s": float(row.route_end_s),
                "segment_count": int(row.segment_count),
                "hops": int(row.hops),
                "segmented_fidelity": float(row.segmented_fidelity),
                "delayed_fidelity": float(row.delayed_fidelity),
                "success_count": int(row.success_count),
                "throughput_pairs_per_s": float(row.throughput_pairs_per_s),
                "mean_fidelity_success": float(row.mean_fidelity_success),
                "average_latency_s": float(row.average_latency_s),
                "max_memory_occupancy": int(row.max_memory_occupancy),
                "swap_success_probability": float(row.swap_success_probability),
                "token_lifetime_s": float(row.token_lifetime_s),
                "path_availability": float(row.path_availability),
                "physical_nodes": list(row.physical_nodes),
                "segment_snapshot_indices": list(row.segment_snapshot_indices),
                "segment_routes": list(row.segment_routes),
                "segment_start_s": list(row.segment_start_s),
                "segment_end_s": list(row.segment_end_s),
            }
        )
    return rows
