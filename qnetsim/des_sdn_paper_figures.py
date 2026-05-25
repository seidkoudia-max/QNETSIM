from __future__ import annotations

import argparse
import math
import os
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channels.fso import FSOParams, werner_to_fidelity
from des_no_sdn_paper_figures import (
    BSMStudyConfig,
    CITY_PAIRS,
    MemoryStudyConfig,
    OPTIMIZED_ANNUAL_CITY_PAIRS,
    REPRESENTATIVE_OVERPASSES,
    CityPair,
    FSOAnalyticalChannelModel,
    OverpassScenario,
    ParametricOverpassGeometry,
    _buffer_scenario_from_row,
    annualised_pdv_curve,
    annualised_pdv_curve_optimized_geometry,
    build_des_args,
    memory_platform_label,
    orbital_period_s,
    plot_altitude_dependence,
    plot_buffer_consumption,
    plot_buffer_over_pass_examples,
    plot_case_overview_composite,
    plot_optimized_annual_pair_distribution,
    plot_representative_losses,
    plot_representative_rates,
    quantile,
    representative_trace,
    study_fso_params,
    weighted_quantile,
    write_csv_rows,
)
from des_physics_no_sdn_simulator import (
    DESEnv,
    DesPhysicsNoSdn,
    EventHandler,
    PS_PER_S,
    TokenBatch,
    _sample_binomial,
    build_bsm_model,
    build_fso_model,
    build_memory_model,
    require_qnet_des,
)
from quantum.bsm import BellStateMeasurementModel
from quantum.memory import MemoryModelBase
from sdn_controller.repeater_controller_core import RepeaterSdnControllerCore


def post_json(url: str, payload: dict, timeout_s: float = 20.0) -> Any:
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
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8")
        except Exception:
            msg = str(e)
        raise RuntimeError(f"HTTP {e.code} on {url}: {msg}") from e
    except Exception as e:
        raise RuntimeError(f"POST failed for {url}: {e}") from e


class ControllerClient(Protocol):
    def reset(self) -> None: ...
    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None: ...
    def update_link_state(self, payload: Dict[str, Any]) -> None: ...
    def poll(self, node: str, t_ps: int) -> List[Dict[str, Any]]: ...
    def report_storage(
        self,
        *,
        req_id: int,
        label: str,
        accepted_count: int,
    ) -> None: ...
    def report_swap_result(
        self,
        *,
        req_id: int,
        success_count: int,
    ) -> None: ...


class InprocControllerClient:
    def __init__(self, core: RepeaterSdnControllerCore) -> None:
        self.core = core

    def reset(self) -> None:
        self.core.reset()

    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None:
        self.core.submit_request(
            req_id=req_id,
            src=src,
            dst=dst,
            f_min=f_min,
            timeout_ps=timeout_ps,
            t_ps=t_ps,
            target_pairs=target_pairs,
        )

    def update_link_state(self, payload: Dict[str, Any]) -> None:
        self.core.update_link_state(payload)

    def poll(self, node: str, t_ps: int) -> List[Dict[str, Any]]:
        return self.core.poll(node, t_ps)

    def report_storage(
        self,
        *,
        req_id: int,
        label: str,
        accepted_count: int,
    ) -> None:
        self.core.report_storage(
            req_id=req_id,
            label=label,
            accepted_count=accepted_count,
        )

    def report_swap_result(
        self,
        *,
        req_id: int,
        success_count: int,
    ) -> None:
        self.core.report_swap_result(
            req_id=req_id,
            success_count=success_count,
        )


class RyuControllerClient:
    def __init__(self, controller_url: str) -> None:
        self.controller_url = str(controller_url).rstrip("/")

    def reset(self) -> None:
        post_json(f"{self.controller_url}/sdn/event", {"type": "RESET"})

    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None:
        post_json(
            f"{self.controller_url}/sdn/event",
            {
                "type": "ENT_REQUEST",
                "req_id": int(req_id),
                "src": str(src),
                "dst": str(dst),
                "f_min": float(f_min),
                "timeout_ps": int(timeout_ps),
                "t_ps": int(t_ps),
                "target_pairs": int(target_pairs),
            },
        )

    def update_link_state(self, payload: Dict[str, Any]) -> None:
        post_json(f"{self.controller_url}/sdn/fso/update", payload)

    def poll(self, node: str, t_ps: int) -> List[Dict[str, Any]]:
        out = post_json(
            f"{self.controller_url}/sdn/event",
            {"type": "POLL", "node": str(node), "t_ps": int(t_ps)},
        )
        return list(out) if isinstance(out, list) else []

    def report_storage(
        self,
        *,
        req_id: int,
        label: str,
        accepted_count: int,
    ) -> None:
        post_json(
            f"{self.controller_url}/sdn/event",
            {
                "type": "REPORT_STORAGE",
                "req_id": int(req_id),
                "label": str(label),
                "accepted_count": int(accepted_count),
            },
        )

    def report_swap_result(
        self,
        *,
        req_id: int,
        success_count: int,
    ) -> None:
        post_json(
            f"{self.controller_url}/sdn/event",
            {
                "type": "SWAP_RESULT",
                "req_id": int(req_id),
                "success_count": int(success_count),
            },
        )


def _mirror_artifact_stem(out_dir: Path, source_stem: str, *target_stems: str) -> None:
    for suffix in (".png", ".pdf", ".csv"):
        source = out_dir / f"{source_stem}{suffix}"
        if not source.exists():
            continue
        for target_stem in target_stems:
            shutil.copy2(source, out_dir / f"{target_stem}{suffix}")


class DesPhysicsWithRepeaterSdn(DesPhysicsNoSdn):
    def __init__(
        self,
        *,
        name: str,
        env: DESEnv,
        args: argparse.Namespace,
        geometry: ParametricOverpassGeometry,
        controller: ControllerClient,
    ) -> None:
        super().__init__(
            name=name,
            env=env,
            args=args,
            geometry=geometry,
            fso=build_fso_model(args),
            memory=build_memory_model(args),
            bsm=build_bsm_model(args),
        )
        self.controller = controller
        self.req_id = int(getattr(args, "req_id", 1))

    @property
    def memory_model(self) -> MemoryModelBase:
        return self.memory

    @property
    def bsm_model(self) -> BellStateMeasurementModel:
        return self.bsm

    def init(self) -> None:
        self.controller.reset()
        self.scheduler.schedule_now(EventHandler(self, "orbit_update"))
        self.scheduler.schedule_after(1, EventHandler(self, "submit_request"))
        self.scheduler.schedule_after(
            int(float(self.args.gen_delay_s) * PS_PER_S),
            EventHandler(self, "generation_tick"),
        )
        self.scheduler.schedule_after(
            int(float(self.args.swap_dt_s) * PS_PER_S),
            EventHandler(self, "swap_tick"),
        )
        self.scheduler.schedule_after(
            int(float(self.args.metrics_dt_s) * PS_PER_S),
            EventHandler(self, "metrics_tick"),
        )

    def submit_request(self) -> None:
        self.controller.submit_request(
            req_id=self.req_id,
            src="Alice",
            dst="Bob",
            f_min=float(self.args.f_min),
            timeout_ps=0,
            t_ps=int(self.env.now),
            target_pairs=0,
        )
        self.log("CTRL_REQUEST_SUBMIT", req_id=self.req_id)

    def orbit_update(self) -> None:
        for label in ("A-R1", "R1-B"):
            state = self.compute_link(label)
            self.link_states[label] = state
            self.controller.update_link_state(
                {
                    "type": "LINK_STATE_UPDATE",
                    "t_ps": int(self.env.now),
                    "u": "Alice" if label == "A-R1" else "R1",
                    "v": "R1" if label == "A-R1" else "Bob",
                    "label": label,
                    "distance_m": float(state.distance_m),
                    "visible": bool(state.visible),
                    "elevation_deg": float(state.elevation_deg),
                    "p_link": float(state.p_link),
                    "fidelity": float(state.fidelity),
                    "u_fid": float(state.p_link * state.fidelity),
                    "source_node": "R1",
                }
            )
            self.log(
                "CTRL_LINK_UPDATE",
                label=label,
                visible=state.visible,
                elevation_deg=state.elevation_deg,
                p_link=state.p_link,
            )
        self._schedule_next("orbit_update", float(self.args.link_update_dt_s))

    def generation_tick(self) -> None:
        actions = self.controller.poll("R1", int(self.env.now))
        seen_labels: set[str] = set()
        for action in actions:
            if str(action.get("msg_type", "")) != "StartSchedule":
                continue
            label = str(action["label"])
            if label in seen_labels:
                continue
            seen_labels.add(label)
            self._execute_scheduled_generation(
                label=label,
                req_id=int(action["req_id"]),
            )
        self._schedule_next("generation_tick", float(self.args.gen_delay_s))

    def _execute_scheduled_generation(self, *, label: str, req_id: int) -> None:
        dt_s = max(0.0, float(self.args.gen_delay_s))
        attempts = max(1, int(round(max(0.0, float(self.args.source_rate_hz)) * dt_s)))
        link = self.link_states.get(label) or self.compute_link(label)
        capacity = max(0, int(self.args.max_modes_per_label) - self.count_tokens(label))
        if capacity <= 0:
            self.fail_storage_full_bin += attempts
            self.failures_bin += attempts
            self.log("CTRL_GEN_SKIP", label=label, reason="capacity")
            return
        if not link.visible:
            self.fail_link_bin += attempts
            self.failures_bin += attempts
            self.log("CTRL_GEN_SKIP", label=label, reason="not_visible")
            return

        link_success = _sample_binomial(self.rng, attempts, link.p_link)
        stored = _sample_binomial(self.rng, link_success, self.memory_model.write_success_prob())
        accepted = min(stored, capacity)
        dropped = max(0, stored - accepted)
        self.fail_link_bin += attempts - link_success
        self.fail_recall_bin += link_success - stored
        self.fail_storage_full_bin += dropped
        self.failures_bin += (attempts - link_success) + (link_success - stored) + dropped
        if accepted > 0:
            self.tokens[label].append(
                TokenBatch(
                    label=label,
                    count=accepted,
                    ready_ps=int(self.env.now),
                    werner_w=float(link.werner_w),
                    fidelity=float(link.fidelity),
                )
            )
            self.controller.report_storage(
                req_id=req_id,
                label=label,
                accepted_count=accepted,
            )
        self.log(
            "CTRL_GEN_BATCH",
            label=label,
            attempts=attempts,
            link_success=link_success,
            stored=accepted,
            dropped=dropped,
        )

    def swap_tick(self) -> None:
        actions = self.controller.poll("R1", int(self.env.now))
        for action in actions:
            if str(action.get("msg_type", "")) != "InstructSwap":
                continue
            self._execute_controller_swap(
                req_id=int(action["req_id"]),
                count=int(action["count"]),
            )
        self._schedule_next("swap_tick", float(self.args.swap_dt_s))

    def _execute_controller_swap(self, *, req_id: int, count: int) -> None:
        available = min(
            self.count_tokens("A-R1"),
            self.count_tokens("R1-B"),
            int(count),
        )
        if available <= 0:
            return
        left_batches = self.pop_tokens("A-R1", available)
        right_batches = self.pop_tokens("R1-B", available)
        delivered = 0
        for n, left, right in self.pair_batches(left_batches, right_batches):
            delivered += self._execute_swap_batch_for_controller(n, left, right)
        self.controller.report_swap_result(req_id=req_id, success_count=delivered)

    def _execute_swap_batch_for_controller(
        self,
        n: int,
        left: TokenBatch,
        right: TokenBatch,
    ) -> int:
        wait_left_s = (int(self.env.now) - left.ready_ps) / PS_PER_S
        wait_right_s = (int(self.env.now) - right.ready_ps) / PS_PER_S
        recall = (
            self.memory_model.conditional_recall_prob(wait_left_s)
            * self.memory_model.conditional_recall_prob(wait_right_s)
        )
        recalled = _sample_binomial(self.rng, n, recall)
        self.fail_recall_bin += n - recalled
        self.failures_bin += n - recalled
        if recalled <= 0:
            self.log("CTRL_SWAP_BATCH", attempts=n, recalled=0, successes=0, fidelity=0.25)
            return 0

        f_left = werner_to_fidelity(
            left.werner_w * self.memory_model.coherence_factor(wait_left_s)
        )
        f_right = werner_to_fidelity(
            right.werner_w * self.memory_model.coherence_factor(wait_right_s)
        )

        f_out = self.bsm_model.swapped_fidelity(f_left, f_right)
        if f_out < float(self.args.f_min):
            self.failures_bin += recalled
            self.log(
                "CTRL_SWAP_BATCH",
                attempts=n,
                recalled=recalled,
                successes=0,
                fidelity=f_out,
                reason="below_f_min",
            )
            return 0

        successes = _sample_binomial(
            self.rng,
            recalled,
            self.bsm_model.p_succ_bsm(1.0, 1.0),
        )
        self.fail_bsm_bin += recalled - successes
        self.failures_bin += recalled - successes
        self.successes_bin += successes
        self.cumulative_successes += successes
        self.fidelity_sum_bin += successes * f_out
        self.success_wait_sum_bin += successes * 0.5 * (wait_left_s + wait_right_s)
        self.log(
            "CTRL_SWAP_BATCH",
            attempts=n,
            recalled=recalled,
            successes=successes,
            fidelity=f_out,
        )
        return successes


def default_sdn_paper_out_dir() -> str:
    return str(Path(__file__).resolve().parent / "Output" / "des_sdn_paper_cases")


def default_ryu_paper_out_dir() -> str:
    return str(Path(__file__).resolve().parent / "Output" / "des_ryu_paper_cases")


def make_controller_client(
    *,
    controller_backend: str,
    controller_url: str,
    max_modes_per_label: int,
    max_bsm_per_cycle: int,
) -> ControllerClient:
    backend = str(controller_backend).strip().lower()
    if backend == "inproc":
        return InprocControllerClient(
            RepeaterSdnControllerCore(
                max_modes_per_label=max_modes_per_label,
                max_bsm_per_cycle=max_bsm_per_cycle,
            )
        )
    if backend == "ryu":
        client = RyuControllerClient(controller_url)
        client.reset()
        return client
    raise ValueError(f"unknown controller backend: {controller_backend}")


def run_sdn_pass_sim(
    scenario: OverpassScenario,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str = "inproc",
    controller_url: str = "http://127.0.0.1:8080",
) -> DesPhysicsWithRepeaterSdn:
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
    args.req_id = 1
    env = DESEnv(name=f"des_sdn_{scenario.key}_{seed}", default=True)
    controller = make_controller_client(
        controller_backend=controller_backend,
        controller_url=controller_url,
        max_modes_per_label=int(args.max_modes_per_label),
        max_bsm_per_cycle=int(args.max_bsm_per_cycle),
    )
    sim = DesPhysicsWithRepeaterSdn(
        name="sdn_physics",
        env=env,
        args=args,
        geometry=ParametricOverpassGeometry(scenario, min_elevation_deg),
        controller=controller,
    )
    env.init()
    env.run(end_time=int(duration_s * PS_PER_S), logging=False, summary=False)
    return sim


def run_sdn_fidelity_stats(
    scenario: OverpassScenario,
    *,
    repeats: int,
    seed: int,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str = "inproc",
    controller_url: str = "http://127.0.0.1:8080",
) -> Dict[str, List[float]]:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    per_bin_fidelity: List[List[float]] = []
    per_bin_successes: List[List[int]] = []
    per_bin_rate: List[List[float]] = []
    times: List[float] = []
    duration_s = 2.0 * scenario.half_window_s

    for repeat in range(max(1, repeats)):
        sim = run_sdn_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            seed=seed + repeat,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
        )
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
    rate_std: List[float] = []
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


def run_sdn_pass_summary(
    scenario: OverpassScenario,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str = "inproc",
    controller_url: str = "http://127.0.0.1:8080",
) -> Dict[str, float]:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    sim = run_sdn_pass_sim(
        scenario,
        min_elevation_deg=min_elevation_deg,
        source_rate_hz=source_rate_hz,
        fso_params=fso_params,
        seed=seed,
        memory_config=memory_config,
        bsm_config=bsm_config,
        controller_backend=controller_backend,
        controller_url=controller_url,
    )
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
    duration_s = 2.0 * scenario.half_window_s
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


def annualised_sdn_buffered_curve_optimized_geometry(
    pair: CityPair,
    *,
    altitudes_km: Iterable[float],
    min_elevation_deg: float,
    source_rate_hz: float,
    fso_params: FSOParams,
    annual_contact_fraction: float,
    seed: int,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    phi_grid_deg: Iterable[float] = (0.0, 30.0, 60.0, 90.0),
    delta_fractions: Iterable[float] = (0.0, 0.5, 1.0),
    controller_backend: str = "inproc",
    controller_url: str = "http://127.0.0.1:8080",
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
                summary = run_sdn_pass_summary(
                    scenario,
                    min_elevation_deg=min_elevation_deg,
                    source_rate_hz=source_rate_hz,
                    fso_params=fso_params,
                    seed=seed + 10000 * altitude_index + 1000 * delta_index + 97 * phi_index,
                    memory_config=memory_config,
                    bsm_config=bsm_config,
                    controller_backend=controller_backend,
                    controller_url=controller_url,
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


def plot_sdn_parameter_sweeps(
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    t2_values_s: List[float],
    rytov_values: List[float],
    sweep_repeats: int,
    base_fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str,
    controller_url: str,
) -> None:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    t2_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    rytov_palette = ["#0b6e4f", "#3c91e6", "#f57c00", "#c44536"]
    sweep_rows: List[Dict[str, float | str]] = []

    fidelity_fig, fidelity_axes = plt.subplots(4, 2, figsize=(14.5, 18), sharex=False)
    rate_fig, rate_axes = plt.subplots(4, 2, figsize=(14.5, 18), sharex=False)

    for row_idx, scenario in enumerate(REPRESENTATIVE_OVERPASSES):
        for color, mem_t2_s in zip(t2_palette, t2_values_s):
            stats = run_sdn_fidelity_stats(
                scenario,
                repeats=sweep_repeats,
                seed=seed + 1000 * row_idx + int(mem_t2_s * 10),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                fso_params=base_fso_params,
                memory_config=replace(memory_config, t2_s=float(mem_t2_s)),
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
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
            stats = run_sdn_fidelity_stats(
                scenario,
                repeats=sweep_repeats,
                seed=seed + 10000 + 1000 * row_idx + int(100 * sigma_r2),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
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
        "Archetype SDN-Controlled DES Successful-Pair Fidelity Sweeps\n"
        "Paper-style spherical overpass geometry with repeater-side SDN control and repository FSO, memory, and BSM models"
    )
    fidelity_fig.tight_layout()
    fidelity_fig.savefig(out_dir / "sdn_des_fidelity_t2_and_rytov_sweeps.png", dpi=180)
    fidelity_fig.savefig(out_dir / "sdn_des_fidelity_t2_and_rytov_sweeps.pdf")
    plt.close(fidelity_fig)

    rate_fig.suptitle(
        "Archetype SDN-Controlled DES Mean Rate Sweeps\n"
        "Paper-style spherical overpass geometry with repeater-side SDN control and repository FSO, memory, and BSM models"
    )
    rate_fig.tight_layout()
    rate_fig.savefig(out_dir / "sdn_des_rate_t2_and_rytov_sweeps.png", dpi=180)
    rate_fig.savefig(out_dir / "sdn_des_rate_t2_and_rytov_sweeps.pdf")
    plt.close(rate_fig)

    write_csv_rows(out_dir / "sdn_des_parameter_sweep_stats.csv", sweep_rows)


def plot_sdn_fidelity_and_rate(
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    repeats: int,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str,
    controller_url: str,
) -> None:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    stats_by_scenario: Dict[str, Dict[str, List[float]]] = {}
    rows: List[Dict[str, float | str]] = []
    for index, scenario in enumerate(REPRESENTATIVE_OVERPASSES):
        stats = run_sdn_fidelity_stats(
            scenario,
            repeats=repeats,
            seed=seed + 1000 * index,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
        )
        stats_by_scenario[scenario.key] = stats
        for idx, t_s in enumerate(stats["time_s"]):
            rows.append(
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
    write_csv_rows(out_dir / "sdn_des_fidelity_stats.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, scenario in zip(axes.flat, REPRESENTATIVE_OVERPASSES):
        stats = stats_by_scenario[scenario.key]
        xs = stats["time_s"]
        ax.fill_between(xs, stats["fidelity_q1"], stats["fidelity_q3"], color="0.85", alpha=0.7)
        line1 = ax.plot(xs, stats["fidelity_median"], color="black", label="Median Fidelity")[0]
        ax.set_ylim(0.25, 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Median Fidelity")
        ax.set_title(scenario.title)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        lower = [max(0.0, m - s) for m, s in zip(stats["rate_mean"], stats["rate_std"])]
        upper = [m + s for m, s in zip(stats["rate_mean"], stats["rate_std"])]
        ax2.fill_between(xs, lower, upper, color="orange", alpha=0.15)
        line2 = ax2.plot(xs, stats["rate_mean"], color="orange", label="Mean Rate")[0]
        ax2.set_ylabel("Mean Rate")
        ax.legend([line1, line2], ["Median Fidelity", "Mean Rate"], loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_des_fidelity_and_rate.png", dpi=180)
    fig.savefig(out_dir / "sdn_des_fidelity_and_rate.pdf")
    plt.close(fig)


def _contiguous_true_spans(xs: List[float], mask: List[int]) -> List[tuple[float, float]]:
    spans: List[tuple[float, float]] = []
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


def _plot_buffer_panel(
    ax,
    *,
    panel_idx: int,
    title: str,
    subtitle: str,
    sim: DesPhysicsWithRepeaterSdn,
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


def plot_sdn_buffer_over_pass_examples(
    pairs: List[CityPair],
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str,
    controller_url: str,
) -> None:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    pair_by_key = {pair.key: pair for pair in pairs}
    examples: List[tuple[str, Dict[str, float]]] = []

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
        sim = run_sdn_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
        )
        _plot_buffer_panel(
            ax,
            panel_idx=panel_idx,
            title=label,
            subtitle=(
                f"h = {float(row['altitude_km']):.0f} km, phi = {float(row['best_track_angle_deg']):.0f}°, "
                f"Delta = {float(row['best_delta_km']):.0f} km"
            ),
            sim=sim,
            scenario=scenario,
            max_modes_per_label=memory_config.max_modes_per_label,
            csv_rows=csv_rows,
            csv_label=label,
        )

    fig.suptitle(
        "Buffer Occupancy Over a Pass Under SDN Repeater Control\n"
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"$T_2$ = {memory_config.t2_s:g} s, capacity = {memory_config.max_modes_per_label} per link"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_buffer_over_pass_examples.png", dpi=180)
    fig.savefig(out_dir / "sdn_buffer_over_pass_examples.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "sdn_buffer_over_pass_examples.csv", csv_rows)


def plot_sdn_paris_vienna_buffer_over_altitudes(
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str,
    controller_url: str,
    altitudes_km: Iterable[float] = (240.0, 360.0, 520.0, 840.0),
) -> None:
    memory_config = memory_config or MemoryStudyConfig()
    bsm_config = bsm_config or BSMStudyConfig()
    pair = next(pair for pair in OPTIMIZED_ANNUAL_CITY_PAIRS if pair.key == "paris_vienna")
    rows = rows_by_pair.get(pair.key, [])
    if not rows:
        return
    selected_rows = [
        min(rows, key=lambda row: abs(float(row["altitude_km"]) - float(altitude_km)))
        for altitude_km in altitudes_km
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharey=True)
    csv_rows: List[Dict[str, float | str]] = []
    for panel_idx, (ax, row) in enumerate(zip(axes.flat, selected_rows)):
        scenario = _buffer_scenario_from_row(pair, row)
        sim = run_sdn_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
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
            csv_label=f"Paris-Vienna altitude sweep @ {float(row['altitude_km']):.0f} km",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_paris_vienna_buffer_over_pass_altitudes.png", dpi=180)
    fig.savefig(out_dir / "sdn_paris_vienna_buffer_over_pass_altitudes.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "sdn_paris_vienna_buffer_over_pass_altitudes.csv", csv_rows)


def plot_sdn_paris_vienna_buffer_over_t2(
    rows_by_pair: Dict[str, List[Dict[str, float]]],
    out_dir: Path,
    *,
    min_elevation_deg: float,
    source_rate_hz: float,
    seed: int,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig | None = None,
    bsm_config: BSMStudyConfig | None = None,
    controller_backend: str,
    controller_url: str,
    t2_values_s: Iterable[float] = (0.3, 3.0, 30.0, 300.0),
) -> None:
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
        sim = run_sdn_pass_sim(
            scenario,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=source_rate_hz,
            fso_params=fso_params,
            seed=seed + 100 * panel_idx,
            memory_config=panel_memory,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
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
        "Paris-Vienna SDN-Controlled Buffer Occupancy Over a Pass for Several Memory $T_2$\n"
        f"Repeater-side SDN control, h = {scenario.altitude_km:.0f} km, "
        f"phi = {scenario.track_angle_deg:.0f}°, Delta = {scenario.delta_km:.0f} km, "
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"capacity = {memory_config.max_modes_per_label} per link"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_paris_vienna_buffer_over_pass_t2_sweep.png", dpi=180)
    fig.savefig(out_dir / "sdn_paris_vienna_buffer_over_pass_t2_sweep.pdf")
    plt.close(fig)
    write_csv_rows(out_dir / "sdn_paris_vienna_buffer_over_pass_t2_sweep.csv", csv_rows)


def plot_sdn_buffered_annual_throughput(
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
            label="Buffered SDN DES",
        )
        ax.set_title(f"{pair.title} (d = {pair.baseline_km:.0f} km)")
        ax.set_xlabel("h (km)")
        ax.set_ylabel("Annual Throughput × 10$^3$ pairs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        "Buffered Annual Throughput With Optimized Geometry Under SDN Repeater Control\n"
        f"{memory_platform_label(memory_config.platform)} memory, "
        f"$T_2$ = {memory_config.t2_s:g} s, "
        f"capacity = {memory_config.max_modes_per_label} modes per label"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_buffered_annual_throughput.png", dpi=180)
    fig.savefig(out_dir / "sdn_buffered_annual_throughput.pdf")
    plt.close(fig)


def plot_sdn_overview_composite(out_dir: Path) -> None:
    panels = [
        ("SDN DES Mean Rate Sweeps", out_dir / "sdn_des_rate_t2_and_rytov_sweeps.png"),
        ("SDN DES Median Fidelity Sweeps", out_dir / "sdn_des_fidelity_t2_and_rytov_sweeps.png"),
        ("Paris-Vienna Buffer Over Altitudes", out_dir / "sdn_paris_vienna_buffer_over_pass_altitudes.png"),
        ("Paris-Vienna Buffer Over T2", out_dir / "sdn_paris_vienna_buffer_over_pass_t2_sweep.png"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (title, path) in zip(axes.flat, panels):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        "Paper-Style Composite: SDN-Controlled Repeater Studies\n"
        "Repeater-side SDN decisions layered over the repository FSO, memory, and BSM models through QNET DES"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sdn_repeater_case_overview_composite.png", dpi=220)
    fig.savefig(out_dir / "sdn_repeater_case_overview_composite.pdf")
    plt.close(fig)


def generate_sdn_studio_plots(
    out_dir: Path,
    *,
    selection: Dict[str, bool],
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig,
    bsm_config: BSMStudyConfig,
    source_rate_hz: float,
    des_source_rate_hz: float,
    ssqr_efficiency: float,
    min_elevation_deg: float,
    des_repeats: int,
    sweep_repeats: int,
    seed: int,
    annual_contact_fraction: float,
    t2_values_s: Iterable[float],
    rytov_values: Iterable[float],
    paris_vienna_altitudes_km: Iterable[float],
    controller_backend: str,
    controller_url: str,
) -> Dict[str, Path]:
    require_qnet_des("the QNETSIM figure studio SDN backends")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Path] = {}
    fso_model = FSOAnalyticalChannelModel(fso_params)
    t2_values = [float(value) for value in t2_values_s]
    rytov_values_list = [float(value) for value in rytov_values]
    altitude_values = [float(value) for value in paris_vienna_altitudes_km]

    def selected(name: str) -> bool:
        return bool(selection.get(name, False))

    needs_representative = selected("representative_overpasses") or selected("overview_composite")
    needs_reference_altitude = selected("overview_composite")
    needs_optimized = (
        selected("annual_throughput")
        or selected("buffered_annual_throughput")
        or selected("paris_vienna_buffer_altitudes")
        or selected("paris_vienna_buffer_t2")
        or selected("buffer_examples")
        or selected("buffer_consumption")
    )
    needs_buffered = (
        selected("buffered_annual_throughput")
        or selected("paris_vienna_buffer_altitudes")
        or selected("paris_vienna_buffer_t2")
        or selected("buffer_examples")
        or selected("buffer_consumption")
    )

    traces: Dict[str, List[Dict[str, float]]] = {}
    altitude_rows: Dict[str, List[Dict[str, float]]] = {}
    optimized_rows: Dict[str, List[Dict[str, float]]] = {}
    buffered_rows: Dict[str, List[Dict[str, float]]] = {}

    if needs_representative:
        combined_trace_rows: List[Dict[str, float]] = []
        for scenario in REPRESENTATIVE_OVERPASSES:
            rows = representative_trace(
                scenario,
                fso_model=fso_model,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
            )
            traces[scenario.key] = rows
            for row in rows:
                combined_trace_rows.append({"scenario": scenario.key, **row})
        write_csv_rows(out_dir / "representative_overpasses.csv", combined_trace_rows)
        plot_representative_losses(
            REPRESENTATIVE_OVERPASSES,
            traces,
            out_dir,
            fso_model=fso_model,
        )
        plot_representative_rates(REPRESENTATIVE_OVERPASSES, traces, out_dir)
        generated["Representative Downlink Loss"] = out_dir / "representative_downlink_loss.png"
        generated["Representative Distribution Rates"] = (
            out_dir / "representative_distribution_rates.png"
        )

        plot_sdn_fidelity_and_rate(
            out_dir,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=des_source_rate_hz,
            seed=seed + 60000,
            repeats=des_repeats,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
        )
        _mirror_artifact_stem(
            out_dir,
            "sdn_des_fidelity_and_rate",
            "des_fidelity_and_ssqr_rate",
        )
        generated["DES Fidelity and Rate"] = out_dir / "des_fidelity_and_ssqr_rate.png"

    if selected("archetype_sweeps"):
        plot_sdn_parameter_sweeps(
            out_dir,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=des_source_rate_hz,
            seed=seed + 50000,
            t2_values_s=t2_values,
            rytov_values=rytov_values_list,
            sweep_repeats=sweep_repeats,
            base_fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=controller_backend,
            controller_url=controller_url,
        )
        _mirror_artifact_stem(
            out_dir,
            "sdn_des_rate_t2_and_rytov_sweeps",
            "des_ssqr_rate_t2_and_rytov_sweeps",
            "archetype_des_ssqr_rate_t2_and_rytov_sweeps",
        )
        _mirror_artifact_stem(
            out_dir,
            "sdn_des_fidelity_t2_and_rytov_sweeps",
            "des_fidelity_t2_and_rytov_sweeps",
            "archetype_des_fidelity_t2_and_rytov_sweeps",
        )
        generated["Archetype Rate Sweeps"] = (
            out_dir / "archetype_des_ssqr_rate_t2_and_rytov_sweeps.png"
        )
        generated["Archetype Fidelity Sweeps"] = (
            out_dir / "archetype_des_fidelity_t2_and_rytov_sweeps.png"
        )

    if needs_reference_altitude:
        altitudes = list(range(200, 1001, 20))
        combined_altitude_rows: List[Dict[str, float]] = []
        for pair in CITY_PAIRS:
            rows = annualised_pdv_curve(
                pair,
                altitudes_km=altitudes,
                fso_model_factory=lambda _min_el: FSOAnalyticalChannelModel(fso_params),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
                annual_contact_fraction=annual_contact_fraction,
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
        plot_altitude_dependence(altitude_rows, out_dir)
        generated["Altitudinal PDV Dependence"] = out_dir / "altitudinal_pdv_dependence.png"

    if needs_optimized:
        optimized_combined_rows: List[Dict[str, float]] = []
        for pair in OPTIMIZED_ANNUAL_CITY_PAIRS:
            rows = annualised_pdv_curve_optimized_geometry(
                pair,
                altitudes_km=list(range(200, 1001, 20)),
                fso_model_factory=lambda _min_el: FSOAnalyticalChannelModel(fso_params),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
                annual_contact_fraction=annual_contact_fraction,
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
        if selected("annual_throughput"):
            plot_optimized_annual_pair_distribution(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                optimized_rows,
                out_dir,
            )
            generated["Optimized Annual Throughput"] = (
                out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.png"
            )

    if needs_buffered:
        buffered_combined_rows: List[Dict[str, float]] = []
        for index, pair in enumerate(OPTIMIZED_ANNUAL_CITY_PAIRS):
            rows = annualised_sdn_buffered_curve_optimized_geometry(
                pair,
                altitudes_km=list(range(200, 1001, 40)),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                fso_params=fso_params,
                annual_contact_fraction=annual_contact_fraction,
                seed=seed + 70000 + 1000 * index,
                memory_config=memory_config,
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
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

        if selected("buffered_annual_throughput"):
            plot_sdn_buffered_annual_throughput(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                optimized_rows,
                out_dir,
                memory_config=memory_config,
            )
            _mirror_artifact_stem(
                out_dir,
                "sdn_buffered_annual_throughput",
                "buffered_annual_throughput_paris_vienna_warsaw_lisbon",
            )
            generated["Buffered Annual Throughput"] = (
                out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.png"
            )

        if selected("buffer_consumption"):
            plot_buffer_consumption(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                out_dir,
                max_modes_per_label=memory_config.max_modes_per_label,
            )
            generated["Buffer Consumption"] = (
                out_dir / "buffer_consumption_paris_vienna_warsaw_lisbon.png"
            )

        if selected("buffer_examples"):
            plot_sdn_buffer_over_pass_examples(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 90000,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
            )
            _mirror_artifact_stem(
                out_dir,
                "sdn_buffer_over_pass_examples",
                "buffer_over_pass_examples",
            )
            generated["Buffer Over Pass Examples"] = out_dir / "buffer_over_pass_examples.png"

        if selected("paris_vienna_buffer_altitudes"):
            plot_sdn_paris_vienna_buffer_over_altitudes(
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 91000,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
                altitudes_km=altitude_values,
            )
            _mirror_artifact_stem(
                out_dir,
                "sdn_paris_vienna_buffer_over_pass_altitudes",
                "paris_vienna_buffer_over_pass_altitudes",
            )
            generated["Paris-Vienna Buffer vs Altitude"] = (
                out_dir / "paris_vienna_buffer_over_pass_altitudes.png"
            )

        if selected("paris_vienna_buffer_t2"):
            plot_sdn_paris_vienna_buffer_over_t2(
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 92000,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                controller_backend=controller_backend,
                controller_url=controller_url,
                t2_values_s=t2_values,
            )
            _mirror_artifact_stem(
                out_dir,
                "sdn_paris_vienna_buffer_over_pass_t2_sweep",
                "paris_vienna_buffer_over_pass_t2_sweep",
            )
            generated["Paris-Vienna Buffer vs T2"] = (
                out_dir / "paris_vienna_buffer_over_pass_t2_sweep.png"
            )

    if selected("overview_composite"):
        if (out_dir / "des_fidelity_and_ssqr_rate.png").exists():
            plot_case_overview_composite(out_dir)
            generated["Representative Composite"] = (
                out_dir / "paper_representative_case_overview_composite.png"
            )

    return generated


def main() -> None:
    require_qnet_des("the SDN-controlled DES paper figures")
    parser = argparse.ArgumentParser(
        description="Generate SDN-controlled repeater paper-style figures."
    )
    parser.add_argument("--out", default=default_sdn_paper_out_dir())
    parser.add_argument(
        "--controller-backend",
        choices=["inproc", "ryu"],
        default="inproc",
        help="Use the in-process repeater controller or a standalone Ryu REST controller.",
    )
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    parser.add_argument("--des-source-rate-hz", type=float, default=5e5)
    parser.add_argument("--min-elevation-deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--des-repeats", type=int, default=8)
    parser.add_argument("--sweep-repeats", type=int, default=4)
    parser.add_argument("--annual-contact-fraction", type=float, default=0.04)
    parser.add_argument("--buffer-mem-t2-s", type=float, default=100.0)
    parser.add_argument("--buffer-max-modes", type=int, default=256)
    parser.add_argument(
        "--mem-platform",
        choices=["REID", "COLD_ATOM", "TRAPPED_ION"],
        default="REID",
    )
    args = parser.parse_args()

    out_value = str(args.out)
    if out_value == default_sdn_paper_out_dir() and str(args.controller_backend) == "ryu":
        out_value = default_ryu_paper_out_dir()
    out_dir = Path(out_value)
    out_dir.mkdir(parents=True, exist_ok=True)
    fso_params = study_fso_params(args.min_elevation_deg)
    memory_config = MemoryStudyConfig(
        platform=str(args.mem_platform).upper(),
        t2_s=float(args.buffer_mem_t2_s),
        max_modes_per_label=int(args.buffer_max_modes),
    )
    bsm_config = BSMStudyConfig()

    plot_sdn_parameter_sweeps(
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
        controller_backend=args.controller_backend,
        controller_url=args.controller_url,
    )
    plot_sdn_fidelity_and_rate(
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 60000,
        repeats=args.des_repeats,
        fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
        controller_backend=args.controller_backend,
        controller_url=args.controller_url,
    )

    optimized_rows: Dict[str, List[Dict[str, float]]] = {}
    optimized_combined_rows: List[Dict[str, float]] = []
    altitudes = list(range(200, 1001, 20))
    for pair in OPTIMIZED_ANNUAL_CITY_PAIRS:
        rows = annualised_pdv_curve_optimized_geometry(
            pair,
            altitudes_km=altitudes,
            fso_model_factory=lambda min_el: build_fso_model(
                build_des_args(
                    duration_s=1.0,
                    min_elevation_deg=min_el,
                    seed=args.seed,
                    source_rate_hz=args.des_source_rate_hz,
                    fso_params=study_fso_params(min_el),
                )
            ),
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=5.9e6,
            ssqr_efficiency=0.003,
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
    write_csv_rows(out_dir / "sdn_reference_optimized_annual_throughput.csv", optimized_combined_rows)

    buffered_rows: Dict[str, List[Dict[str, float]]] = {}
    buffered_combined_rows: List[Dict[str, float]] = []
    buffered_altitudes = list(range(200, 1001, 80))
    for index, pair in enumerate(OPTIMIZED_ANNUAL_CITY_PAIRS):
        rows = annualised_sdn_buffered_curve_optimized_geometry(
            pair,
            altitudes_km=buffered_altitudes,
            min_elevation_deg=args.min_elevation_deg,
            source_rate_hz=args.des_source_rate_hz,
            fso_params=fso_params,
            annual_contact_fraction=args.annual_contact_fraction,
            seed=args.seed + 70000 + 1000 * index,
            memory_config=memory_config,
            bsm_config=bsm_config,
            controller_backend=args.controller_backend,
            controller_url=args.controller_url,
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
    write_csv_rows(out_dir / "sdn_buffered_annual_throughput.csv", buffered_combined_rows)

    plot_sdn_buffered_annual_throughput(
        OPTIMIZED_ANNUAL_CITY_PAIRS,
        buffered_rows,
        optimized_rows,
        out_dir,
        memory_config=memory_config,
    )
    plot_sdn_paris_vienna_buffer_over_altitudes(
        buffered_rows,
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 91000,
        fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
        controller_backend=args.controller_backend,
        controller_url=args.controller_url,
    )
    plot_sdn_paris_vienna_buffer_over_t2(
        buffered_rows,
        out_dir,
        min_elevation_deg=args.min_elevation_deg,
        source_rate_hz=args.des_source_rate_hz,
        seed=args.seed + 92000,
        fso_params=fso_params,
        memory_config=memory_config,
        bsm_config=bsm_config,
        controller_backend=args.controller_backend,
        controller_url=args.controller_url,
    )
    plot_sdn_overview_composite(out_dir)
    print(f"[des-sdn] wrote figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
