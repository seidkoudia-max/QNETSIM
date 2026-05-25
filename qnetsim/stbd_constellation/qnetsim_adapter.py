from __future__ import annotations

import math
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import json
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from algorithms.stbd import plan_stbd_route
from framework.core.des import DESEnv
from des_stbd_physics_simulator import (
    PS_PER_S,
    STBDPhysicsDES,
    build_bsm_model,
    build_fso_model,
    build_memory_model,
    build_route_edges,
    summarize_run,
)
from sdn_controller.stbd_segment_controller_core import STBDSegmentControllerCore

from .scenario_builder import ScenarioBundle


@dataclass(frozen=True)
class LaunchResult:
    launch_s: float
    route_found: bool
    route_start_s: float
    route_end_s: float
    segment_count: int
    hops: int
    segmented_fidelity: float
    delayed_fidelity: float
    success_count: int
    throughput_pairs_per_s: float
    mean_fidelity_success: float
    average_latency_s: float
    max_memory_occupancy: int
    swap_success_probability: float
    token_lifetime_s: float
    path_availability: float
    physical_nodes: tuple[str, ...] = field(default_factory=tuple)
    segment_snapshot_indices: tuple[int, ...] = field(default_factory=tuple)
    segment_routes: tuple[str, ...] = field(default_factory=tuple)
    segment_start_s: tuple[float, ...] = field(default_factory=tuple)
    segment_end_s: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimeSeriesRow:
    time_s: float
    throughput_pairs_per_s: float
    fidelity: float
    delayed_fidelity: float
    hops: float
    availability: float


@dataclass(frozen=True)
class AdapterRunResult:
    launch_results: List[LaunchResult]
    timeseries: List[TimeSeriesRow]
    raw_event_rows: List[dict]


def _clone_args(base: Namespace, **updates) -> Namespace:
    cloned = Namespace(**vars(base))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights:
        return 0.0
    total = sum(float(w) for w in weights)
    if total <= 0.0:
        return 0.0
    return sum(float(v) * float(w) for v, w in zip(values, weights)) / total


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_local_port(port: int, *, timeout_s: float = 12.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            try:
                sock.connect(("127.0.0.1", int(port)))
                return
            except OSError:
                time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for the local STBD Ryu controller on port {port}.")


def _resolve_ryu_python(root_dir: Path) -> str:
    candidates = [
        root_dir.parent / ".venv39" / "bin" / "python",
        root_dir.parent / ".venv39" / "bin" / "python3",
        Path(sys.executable),
    ]
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return str(candidate)
    return str(sys.executable)


@contextmanager
def _ryu_controller_context(root_dir: Path, out_dir: Path) -> Iterator[str]:
    port = _pick_free_local_port()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ryu_stbd_constellation_controller.log"
    python_bin = _resolve_ryu_python(root_dir)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            [
                python_bin,
                "-m",
                "ryu.cmd.manager",
                str(root_dir / "sdn_controller" / "ryu_stbd_constellation_app.py"),
                "--wsapi-port",
                str(port),
            ],
            cwd=str(root_dir),
            env=dict(os.environ),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_local_port(port)
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _post_json(url: str, payload: dict, timeout_s: float = 20.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        try:
            msg = exc.read().decode("utf-8")
        except Exception:
            msg = str(exc)
        raise RuntimeError(f"HTTP {exc.code} on {url}: {msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"POST failed for {url}: {exc}") from exc


class RyuSTBDControllerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = str(base_url).rstrip("/")

    def reset(self) -> None:
        _post_json(f"{self.base_url}/sdn/event", {"type": "RESET"})

    def install_request(
        self,
        *,
        req_id: int,
        execution_mode: str,
        final_label: str,
        route_start_s: float,
        route_end_s: float,
        final_segment_start_s: float,
        final_segment_end_s: float,
        segments: List[dict],
    ) -> None:
        _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "INSTALL_REQUEST",
                "req_id": int(req_id),
                "execution_mode": str(execution_mode),
                "final_label": str(final_label),
                "route_start_s": float(route_start_s),
                "route_end_s": float(route_end_s),
                "final_segment_start_s": float(final_segment_start_s),
                "final_segment_end_s": float(final_segment_end_s),
                "segments": list(segments),
            },
        )

    def poll(self, *, req_id: int, t_s: float) -> Dict[str, Any]:
        response = _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "POLL",
                "req_id": int(req_id),
                "t_s": float(t_s),
            },
        )
        return response if isinstance(response, dict) else {}


class LocalSTBDControllerClient:
    def __init__(self) -> None:
        self.core = STBDSegmentControllerCore()

    def reset(self) -> None:
        self.core.reset()

    def install_request(
        self,
        *,
        req_id: int,
        execution_mode: str,
        final_label: str,
        route_start_s: float,
        route_end_s: float,
        final_segment_start_s: float,
        final_segment_end_s: float,
        segments: List[dict],
    ) -> None:
        self.core.install_request(
            req_id=int(req_id),
            execution_mode=str(execution_mode),
            final_label=str(final_label),
            route_start_s=float(route_start_s),
            route_end_s=float(route_end_s),
            final_segment_start_s=float(final_segment_start_s),
            final_segment_end_s=float(final_segment_end_s),
            segments=list(segments),
        )

    def poll(self, *, req_id: int, t_s: float) -> Dict[str, Any]:
        return self.core.poll(req_id=int(req_id), t_s=float(t_s))


def _run_single_launch(
    bundle: ScenarioBundle,
    *,
    launch_s: float,
    seed: int,
    controller_url: str | None = None,
) -> tuple[LaunchResult, List[dict]]:
    remaining_s = max(0.0, float(bundle.args.duration_s) - float(launch_s))
    planning_horizon_s = min(float(bundle.args.planner_horizon_s), remaining_s)
    if planning_horizon_s < float(bundle.args.snapshot_s):
        return (
            LaunchResult(
                launch_s=float(launch_s),
                route_found=False,
                route_start_s=float("nan"),
                route_end_s=float("nan"),
                segment_count=0,
                hops=0,
                segmented_fidelity=float("nan"),
                delayed_fidelity=float("nan"),
                success_count=0,
                throughput_pairs_per_s=0.0,
                mean_fidelity_success=float("nan"),
                average_latency_s=float("nan"),
                max_memory_occupancy=0,
                swap_success_probability=float("nan"),
                token_lifetime_s=float("nan"),
                path_availability=0.0,
            ),
            [],
        )

    plan_args = _clone_args(
        bundle.args,
        planner_start_s=float(launch_s),
        planner_coherence_s=float(bundle.args.planner_coherence_s),
        planner_horizon_s=float(bundle.args.planner_horizon_s),
        duration_s=float(planning_horizon_s),
        seed=int(seed),
        time_offset_s=float(launch_s),
    )
    try:
        route_result = plan_stbd_route(
            geometry=bundle.geometry,
            channel_model=build_fso_model(plan_args),
            node_names=bundle.node_names,
            src=bundle.src_name,
            dst=bundle.dst_name,
            start_s=float(launch_s),
            duration_s=float(planning_horizon_s),
            snapshot_s=float(plan_args.snapshot_s),
            coherence_time_s=float(plan_args.planner_coherence_s),
            max_link_distance_m=float(plan_args.max_link_distance_km) * 1e3,
            fidelity_threshold=float(plan_args.planner_fidelity_threshold),
            w_fid=float(plan_args.w_fid),
            w_mem=float(plan_args.w_mem),
            fid_utility_mode=str(plan_args.fid_utility_mode),
            bsm_round_duration_s=float(plan_args.bsm_round_duration_s),
            destination_horizon_s=float(plan_args.planner_horizon_s),
            perfect_intersatellite=bool(getattr(plan_args, "perfect_isl", False)),
        )
    except Exception:
        return (
            LaunchResult(
                launch_s=float(launch_s),
                route_found=False,
                route_start_s=float("nan"),
                route_end_s=float("nan"),
                segment_count=0,
                hops=0,
                segmented_fidelity=float("nan"),
                delayed_fidelity=float("nan"),
                success_count=0,
                throughput_pairs_per_s=0.0,
                mean_fidelity_success=float("nan"),
                average_latency_s=float("nan"),
                max_memory_occupancy=0,
                swap_success_probability=float("nan"),
                token_lifetime_s=float("nan"),
                path_availability=0.0,
            ),
            [],
        )

    route_edges = build_route_edges(route_result.physical_nodes)
    route_start_s = min(float(segment.start_s) for segment in route_result.segments)
    route_end_s = max(float(segment.end_s) for segment in route_result.segments)
    sim_duration_s = min(
        remaining_s,
        max(float(bundle.args.snapshot_s), route_end_s - float(launch_s) + float(bundle.args.snapshot_s)),
    )
    requested_backend = str(getattr(bundle.args, "stbd_controller_backend", "des_only")).lower().strip()
    active_backend = requested_backend if requested_backend in {"sdn_local", "sdn_ryu"} else "des_only"
    sim_args = _clone_args(
        bundle.args,
        duration_s=float(sim_duration_s),
        planner_start_s=float(launch_s),
        planner_coherence_s=float(bundle.args.planner_coherence_s),
        planner_horizon_s=float(bundle.args.planner_horizon_s),
        seed=int(seed),
        time_offset_s=float(launch_s),
        stbd_controller_backend=active_backend,
    )
    if active_backend != "des_only":
        sim_args.direct_swap_on_full_visibility = False

    env = DESEnv(name=f"stbd_constellation_{int(launch_s)}", default=True)
    if active_backend == "sdn_ryu" and controller_url:
        controller_client = RyuSTBDControllerClient(controller_url)
    elif active_backend == "sdn_local":
        controller_client = LocalSTBDControllerClient()
    else:
        controller_client = None
    sim = STBDPhysicsDES(
        name="stbd_constellation_des",
        env=env,
        args=sim_args,
        geometry=bundle.geometry,
        fso=build_fso_model(sim_args),
        memory=build_memory_model(sim_args),
        bsm=build_bsm_model(sim_args),
        stbd_result=route_result,
        route_edges=route_edges,
        controller_client=controller_client,
    )
    env.init()
    env.run(end_time=int(float(sim_args.duration_s) * PS_PER_S), logging=False, summary=False)
    summary = summarize_run(sim, stbd_result=route_result, route_edges=route_edges)
    execution = summary["execution"]

    total_event_count = sum(int(event.count) for event in sim.end_to_end_events)
    avg_latency_s = (
        sum(float(event.t_s) * int(event.count) for event in sim.end_to_end_events)
        / total_event_count
        if total_event_count > 0
        else float("nan")
    )
    max_memory = max((int(metric.total_memory_occupancy) for metric in sim.metrics), default=0)
    swap_attempts = sum(int(event.attempts) for event in sim.swap_events)
    swap_successes = sum(int(event.bsm_successes) for event in sim.swap_events)
    swap_success_probability = (
        float(swap_successes) / float(swap_attempts) if swap_attempts > 0 else float("nan")
    )

    raw_rows = [
        {
            "launch_s": float(launch_s),
            "abs_time_s": float(launch_s) + float(event.t_s),
            "count": int(event.count),
            "fidelity": float(event.fidelity),
            "kind": str(event.kind),
        }
        for event in sim.end_to_end_events
    ]

    result = LaunchResult(
        launch_s=float(launch_s),
        route_found=True,
        route_start_s=float(route_start_s),
        route_end_s=float(route_end_s),
        segment_count=int(len(route_result.segments)),
        hops=max(1, len(route_result.physical_nodes) - 1),
        segmented_fidelity=float(route_result.segmented_fidelity),
        delayed_fidelity=float(route_result.delayed_fidelity),
        success_count=int(execution["cumulative_successes"]),
        throughput_pairs_per_s=float(execution["throughput_pairs_per_s"]),
        mean_fidelity_success=float(execution["mean_fidelity_success"]),
        average_latency_s=float(avg_latency_s),
        max_memory_occupancy=int(max_memory),
        swap_success_probability=float(swap_success_probability),
        token_lifetime_s=max(0.0, float(route_end_s) - float(launch_s)),
        path_availability=1.0,
        physical_nodes=tuple(str(node) for node in route_result.physical_nodes),
        segment_snapshot_indices=tuple(int(segment.snapshot_index) for segment in route_result.segments),
        segment_routes=tuple(" -> ".join(segment.nodes) for segment in route_result.segments),
        segment_start_s=tuple(float(segment.start_s) for segment in route_result.segments),
        segment_end_s=tuple(float(segment.end_s) for segment in route_result.segments),
    )
    return result, raw_rows


def _rasterize_event_service(
    *,
    launch_results: Sequence[LaunchResult],
    raw_event_rows: Sequence[dict],
    total_duration_s: float,
    bin_s: float,
) -> List[TimeSeriesRow]:
    launch_by_time = {float(row.launch_s): row for row in launch_results}
    rows: List[TimeSeriesRow] = []
    t = 0.0
    while t < total_duration_s - 1e-12:
        t_mid = t + 0.5 * bin_s
        bin_events = [
            event
            for event in raw_event_rows
            if float(t) - 1e-12 <= float(event["abs_time_s"]) < float(t + bin_s) + 1e-12
        ]
        total_count = sum(int(event.get("count", 0)) for event in bin_events)
        throughput = float(total_count) / float(bin_s) if total_count > 0 else 0.0
        if total_count > 0:
            weights = [float(int(event.get("count", 0))) for event in bin_events]
            fidelity = _weighted_mean(
                [float(event["fidelity"]) for event in bin_events],
                weights,
            )
            delayed_fidelity = _weighted_mean(
                [
                    float(launch_by_time[float(event["launch_s"])].delayed_fidelity)
                    for event in bin_events
                ],
                weights,
            )
            hops = _weighted_mean(
                [
                    float(launch_by_time[float(event["launch_s"])].hops)
                    for event in bin_events
                ],
                weights,
            )
        else:
            fidelity = 0.0
            delayed_fidelity = 0.0
            hops = 0.0
        availability = 1.0 if total_count > 0 else 0.0
        rows.append(
            TimeSeriesRow(
                time_s=float(t_mid),
                throughput_pairs_per_s=float(throughput),
                fidelity=float(fidelity),
                delayed_fidelity=float(delayed_fidelity),
                hops=float(hops),
                availability=float(availability),
            )
        )
        t += float(bin_s)
    return rows


def run_constellation_des(
    bundle: ScenarioBundle,
    *,
    launch_step_s: float,
    repeats: int,
    seed: int,
    series_bin_s: float,
) -> AdapterRunResult:
    backend = str(getattr(bundle.args, "stbd_controller_backend", "des_only")).lower().strip()
    if backend == "sdn_local":
        return _run_constellation_des_with_controller(
            bundle,
            launch_step_s=float(launch_step_s),
            repeats=int(repeats),
            seed=int(seed),
            series_bin_s=float(series_bin_s),
            controller_url=None,
        )
    if backend == "sdn_ryu":
        root_dir = Path(__file__).resolve().parent.parent
        with _ryu_controller_context(root_dir, bundle.out_dir) as controller_url:
            return _run_constellation_des_with_controller(
                bundle,
                launch_step_s=float(launch_step_s),
                repeats=int(repeats),
                seed=int(seed),
                series_bin_s=float(series_bin_s),
                controller_url=str(controller_url),
            )
    return _run_constellation_des_with_controller(
        bundle,
        launch_step_s=float(launch_step_s),
        repeats=int(repeats),
        seed=int(seed),
        series_bin_s=float(series_bin_s),
        controller_url=None,
    )


def _run_constellation_des_with_controller(
    bundle: ScenarioBundle,
    *,
    launch_step_s: float,
    repeats: int,
    seed: int,
    series_bin_s: float,
    controller_url: str | None,
) -> AdapterRunResult:
    launch_results: List[LaunchResult] = []
    raw_rows: List[dict] = []
    launches: List[float] = []
    t = 0.0
    while t < float(bundle.args.duration_s) - 1e-12:
        launches.append(float(t))
        t += float(launch_step_s)

    for launch_index, launch_s in enumerate(launches):
        trial_rows: List[LaunchResult] = []
        trial_events: List[dict] = []
        for repeat in range(max(1, int(repeats))):
            launch_seed = int(seed) + 100003 * launch_index + repeat
            result, events = _run_single_launch(
                bundle,
                launch_s=float(launch_s),
                seed=int(launch_seed),
                controller_url=controller_url,
            )
            trial_rows.append(result)
            trial_events.extend(events)

        if len(trial_rows) == 1:
            launch_results.append(trial_rows[0])
        else:
            supported = [row for row in trial_rows if row.route_found]
            if supported:
                weights = [max(1.0, float(row.success_count)) for row in supported]
                representative = max(
                    supported,
                    key=lambda row: (
                        float(row.success_count),
                        float(row.throughput_pairs_per_s),
                        -float(row.launch_s),
                    ),
                )
                launch_results.append(
                    LaunchResult(
                        launch_s=float(launch_s),
                        route_found=True,
                        route_start_s=float(representative.route_start_s),
                        route_end_s=float(representative.route_end_s),
                        segment_count=int(round(_weighted_mean([float(row.segment_count) for row in supported], weights))),
                        hops=int(round(_weighted_mean([float(row.hops) for row in supported], weights))),
                        segmented_fidelity=_weighted_mean([float(row.segmented_fidelity) for row in supported], weights),
                        delayed_fidelity=_weighted_mean([float(row.delayed_fidelity) for row in supported], weights),
                        success_count=int(round(sum(float(row.success_count) for row in supported) / len(supported))),
                        throughput_pairs_per_s=_weighted_mean([float(row.throughput_pairs_per_s) for row in supported], weights),
                        mean_fidelity_success=_weighted_mean(
                            [float(row.mean_fidelity_success) for row in supported if not math.isnan(float(row.mean_fidelity_success))],
                            [
                                max(1.0, float(row.success_count))
                                for row in supported
                                if not math.isnan(float(row.mean_fidelity_success))
                            ] or [1.0],
                        ),
                        average_latency_s=_weighted_mean(
                            [float(row.average_latency_s) for row in supported if not math.isnan(float(row.average_latency_s))],
                            [
                                max(1.0, float(row.success_count))
                                for row in supported
                                if not math.isnan(float(row.average_latency_s))
                            ] or [1.0],
                        ),
                        max_memory_occupancy=int(max(row.max_memory_occupancy for row in supported)),
                        swap_success_probability=_weighted_mean(
                            [float(row.swap_success_probability) for row in supported if not math.isnan(float(row.swap_success_probability))],
                            [
                                max(1.0, float(row.success_count))
                                for row in supported
                                if not math.isnan(float(row.swap_success_probability))
                            ] or [1.0],
                        ),
                        token_lifetime_s=float(representative.token_lifetime_s),
                        path_availability=1.0,
                        physical_nodes=tuple(representative.physical_nodes),
                        segment_snapshot_indices=tuple(representative.segment_snapshot_indices),
                        segment_routes=tuple(representative.segment_routes),
                        segment_start_s=tuple(representative.segment_start_s),
                        segment_end_s=tuple(representative.segment_end_s),
                    )
                )
            else:
                launch_results.append(trial_rows[0])
        raw_rows.extend(trial_events)

    timeseries = _rasterize_event_service(
        launch_results=launch_results,
        raw_event_rows=raw_rows,
        total_duration_s=float(bundle.args.duration_s),
        bin_s=float(series_bin_s),
    )
    return AdapterRunResult(
        launch_results=launch_results,
        timeseries=timeseries,
        raw_event_rows=raw_rows,
    )
