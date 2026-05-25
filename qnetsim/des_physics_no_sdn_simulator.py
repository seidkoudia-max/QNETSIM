from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from framework.core.des import DESEnv, Entity, EventHandler

HAVE_QNET_DES = True
QNET_DES_IMPORT_ERROR: Exception | None = None


def require_qnet_des(context: str = "this workflow") -> None:
    return None

from channels.fso import FSOAnalyticalChannelModel, FSOParams, werner_to_fidelity
from mobility.geometry import GeometryEngine
from mobility.pass_planner import PassPlannerConfig, SatellitePassSelection, select_best_satellite_pass
from mobility.propagator import GroundStation, SkyfieldPropagator
from mobility.tle_loader import (
    SatelliteEphemeris,
    load_ephemerides,
    representative_epoch_unix_s,
)
from quantum.bsm import BSMParams, BellStateMeasurementModel
from quantum.memory import (
    ColdAtomMemoryModel,
    ColdAtomParams,
    MemoryModelBase,
    REIDMemoryModel,
    REIDParams,
    TrappedIonMemoryModel,
    TrappedIonParams,
)


PS_PER_S = 1_000_000_000_000


def _sample_poisson(rng: random.Random, lam: float) -> int:
    lam = max(0.0, float(lam))
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        threshold = math.exp(-lam)
        k = 0
        p = 1.0
        while p > threshold:
            k += 1
            p *= rng.random()
        return k - 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def _sample_binomial(rng: random.Random, n: int, p: float) -> int:
    n = max(0, int(n))
    p = max(0.0, min(1.0, float(p)))
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n <= 128:
        return sum(1 for _ in range(n) if rng.random() < p)
    mean = n * p
    variance = n * p * (1.0 - p)
    if mean < 30.0 and p < 0.05:
        return min(n, _sample_poisson(rng, mean))
    return max(0, min(n, int(round(rng.gauss(mean, math.sqrt(variance))))))


def default_ground_stations(min_elevation_deg: float) -> Tuple[GroundStation, GroundStation]:
    return (
        GroundStation("Alice", lat_deg=48.8566, lon_deg=2.3522, alt_m=35.0, min_elevation_deg=min_elevation_deg),
        GroundStation("Bob", lat_deg=48.2082, lon_deg=16.3738, alt_m=171.0, min_elevation_deg=min_elevation_deg),
    )


def default_satellites(tle_path: str | None, omm_json_path: str | None) -> List[SatelliteEphemeris]:
    return load_ephemerides(
        tle_path=tle_path,
        omm_json_path=omm_json_path,
        use_demo_if_missing=(tle_path is None and omm_json_path is None),
    )


def select_satellite_for_run(
    *,
    sats: List[SatelliteEphemeris],
    alice: GroundStation,
    bob: GroundStation,
    epoch_unix_s: float,
    args: argparse.Namespace,
) -> SatellitePassSelection:
    if not sats:
        raise ValueError("need at least one satellite")
    cfg = PassPlannerConfig(
        scan_start_ps=0,
        scan_end_ps=int(float(args.visibility_scan_s) * PS_PER_S),
        fine_step_ps=int(max(1.0, float(args.visibility_scan_step_s)) * PS_PER_S),
        coarse_step_ps=int(max(30.0, float(args.pass_prefilter_step_s)) * PS_PER_S),
        candidate_limit=int(args.pass_candidate_limit),
        objective=str(args.pass_selection),
        memory_coherence_s=max(1e-6, float(args.mem_t2)),
    )
    return select_best_satellite_pass(
        satellites=sats,
        ground_a=alice,
        ground_b=bob,
        epoch_unix_s=epoch_unix_s,
        cfg=cfg,
        sat_name=(str(args.sat_name).strip() if args.sat_name else None),
    )


def build_memory_model(args: argparse.Namespace) -> MemoryModelBase:
    platform = str(args.mem_platform).upper()
    if platform == "REID":
        return REIDMemoryModel(
            REIDParams(
                d=float(args.mem_d),
                t2_spin_dd_s=float(args.mem_t2),
                n=float(args.mem_n),
                t_mem_s=float(args.mem_t_mem),
                m=float(args.mem_m),
            )
        )
    if platform == "COLD_ATOM":
        return ColdAtomMemoryModel(
            ColdAtomParams(
                d=float(args.mem_d),
                t2_sw_s=float(args.mem_t2),
                n=float(args.mem_n),
                eta_mode0=float(args.mem_eta_mode0),
                t_mode_s=float(args.mem_t_mem),
                m_mode=float(args.mem_m),
            )
        )
    if platform == "TRAPPED_ION":
        return TrappedIonMemoryModel(
            TrappedIonParams(t2_ti_s=float(args.mem_t2), eta_ti=float(args.mem_eta_ti))
        )
    raise ValueError(f"unknown memory platform: {args.mem_platform}")


def build_fso_model(args: argparse.Namespace) -> FSOAnalyticalChannelModel:
    def _get(name: str, default):
        return getattr(args, name, default)

    def _build_params(prefix: str, disable_prefix: str) -> FSOParams:
        return FSOParams(
            eta_src=float(_get(f"{prefix}eta_src", 0.90)),
            eta_det=float(_get(f"{prefix}eta_det", 0.80)),
            eta_cpl=float(_get(f"{prefix}eta_cpl", 0.80)),
            eta_point_static=float(_get(f"{prefix}eta_point", 0.98)),
            d_t=float(_get(f"{prefix}d_t", 0.20)),
            d_r=float(_get(f"{prefix}d_r", 1.0)),
            theta_div=float(_get(f"{prefix}theta_div", 3e-6)),
            wavelength_m=float(_get(f"{prefix}wavelength_m", 810e-9)),
            t_abs=float(_get(f"{prefix}t_abs", 0.95)),
            t_sca=float(_get(f"{prefix}t_sca", 0.95)),
            c_atm=float(_get(f"{prefix}c_atm", 0.05)),
            sigma_r2=float(_get(f"{prefix}sigma_r2", 0.05)),
            ao_gain=float(_get(f"{prefix}ao_gain", 1.0)),
            turbulence_loss_coeff=float(
                _get(f"{prefix}turbulence_loss_coeff", 0.06)
            ),
            turbulence_quality_coeff=float(
                _get(f"{prefix}turbulence_quality_coeff", 0.025)
            ),
            delta_t=float(_get(f"{prefix}delta_t", 1e-9)),
            n0=float(_get(f"{prefix}n0", 1e-5)),
            n_dc=float(_get(f"{prefix}n_dc", 100.0)),
            n_conv=float(_get(f"{prefix}n_conv", 0.0)),
            p_t=float(_get(f"{prefix}p_t", 0.05)),
            pointing_jitter_rad=float(_get(f"{prefix}pointing_jitter_rad", 1e-8)),
            enable_pointing_jitter=not bool(_get(f"{disable_prefix}pointing_jitter", False)),
            enable_stochastic_fading=not bool(_get(f"{disable_prefix}stochastic_fading", False)),
            min_elevation_deg=float(
                _get(f"{prefix}min_elevation_deg", _get("min_elevation_deg", 5.0))
            ),
            turbulence_floor=float(_get(f"{prefix}turbulence_floor", 1e-4)),
            polarization_error_rad=float(
                _get(f"{prefix}polarization_error_rad", 0.02)
            ),
            phase_noise_rad=float(_get(f"{prefix}phase_noise_rad", 0.10)),
            temporal_mismatch_s=float(
                _get(f"{prefix}temporal_mismatch_s", 20e-12)
            ),
            coherence_time_s=float(
                _get(f"{prefix}coherence_time_s", 100e-12)
            ),
            max_snr_for_mapping=float(
                _get(f"{prefix}max_snr_for_mapping", 1e12)
            ),
        )

    ground_model = FSOAnalyticalChannelModel(
        _build_params(prefix="fso_", disable_prefix="disable_fso_")
    )
    if not hasattr(args, "isl_fso_eta_src"):
        return ground_model

    isl_model = FSOAnalyticalChannelModel(
        _build_params(prefix="isl_fso_", disable_prefix="disable_isl_fso_")
    )

    class DualFSOChannelModel:
        def __init__(
            self,
            ground: FSOAnalyticalChannelModel,
            isl: FSOAnalyticalChannelModel,
        ) -> None:
            self.ground = ground
            self.isl = isl
            self.p = ground.p

        def _pick(self, intersatellite: bool) -> FSOAnalyticalChannelModel:
            return self.isl if intersatellite else self.ground

        def link_metrics(self, *, intersatellite: bool, **kwargs):
            return self._pick(intersatellite).link_metrics(
                intersatellite=intersatellite,
                **kwargs,
            )

        def mean_link_metrics(self, *, intersatellite: bool, **kwargs):
            return self._pick(intersatellite).mean_link_metrics(
                intersatellite=intersatellite,
                **kwargs,
            )

        def mean_efficiency(self, *, intersatellite: bool, **kwargs):
            return self._pick(intersatellite).mean_efficiency(
                intersatellite=intersatellite,
                **kwargs,
            )

        def mean_loss_db(self, *, intersatellite: bool, **kwargs):
            return self._pick(intersatellite).mean_loss_db(
                intersatellite=intersatellite,
                **kwargs,
            )

    return DualFSOChannelModel(ground_model, isl_model)


def build_bsm_model(args: argparse.Namespace) -> BellStateMeasurementModel:
    return BellStateMeasurementModel(
        BSMParams(
            v_hom=float(args.bsm_v_hom),
            eta_det=float(args.bsm_eta_det),
            sigma_det_s=float(args.bsm_sigma_det_s),
            delta_t_s=float(args.bsm_delta_t_s),
            p_dc=float(args.bsm_p_dc),
            swap_model=str(getattr(args, "bsm_swap_model", "legacy")),
        )
    )


@dataclass
class LinkState:
    label: str
    t_ps: int
    visible: bool
    distance_m: float
    elevation_deg: float
    p_link: float
    werner_w: float
    fidelity: float


@dataclass
class TokenBatch:
    label: str
    count: int
    ready_ps: int
    werner_w: float
    fidelity: float


@dataclass
class MetricRow:
    t_s: float
    ar1_visible: int
    r1b_visible: int
    both_visible: int
    ar1_distance_km: float
    r1b_distance_km: float
    ar1_p_link: float
    r1b_p_link: float
    ar1_elevation_deg: float
    r1b_elevation_deg: float
    ar1_loss_db: float
    r1b_loss_db: float
    ar1_fidelity: float
    r1b_fidelity: float
    successes: int
    failures: int
    mean_fidelity_success: float
    mean_wait_success_s: float
    throughput_pairs_per_s: float
    cumulative_successes: int
    memory_ar1: int
    memory_r1b: int
    fail_link: int
    fail_recall: int
    fail_bsm: int
    fail_storage_full: int


@dataclass
class SweepPoint:
    x: float
    y: float
    success_rate: float
    mean_fidelity_success: float
    throughput_pairs_per_s: float
    success_count: int
    total_attempts: int


class DesPhysicsNoSdn(Entity):
    def __init__(
        self,
        *,
        name: str,
        env: DESEnv,
        args: argparse.Namespace,
        geometry: GeometryEngine,
        fso: FSOAnalyticalChannelModel,
        memory: MemoryModelBase,
        bsm: BellStateMeasurementModel,
    ):
        super().__init__(name=name, env=env)
        self.args = args
        self.geometry = geometry
        self.fso = fso
        self.memory = memory
        self.bsm = bsm
        self.rng = random.Random(int(args.seed))
        self.link_states: Dict[str, LinkState] = {}
        self.tokens: Dict[str, List[TokenBatch]] = defaultdict(list)
        self.metrics: List[MetricRow] = []
        self.event_log: List[List[object]] = []

        self.successes_bin = 0
        self.failures_bin = 0
        self.fidelity_sum_bin = 0.0
        self.success_wait_sum_bin = 0.0
        self.fail_link_bin = 0
        self.fail_recall_bin = 0
        self.fail_bsm_bin = 0
        self.fail_storage_full_bin = 0
        self.cumulative_successes = 0

    def log(self, kind: str, **payload: object) -> None:
        if not bool(self.args.no_event_log):
            self.event_log.append([int(self.env.now), kind, payload])

    def init(self) -> None:
        self.scheduler.schedule_now(EventHandler(self, "orbit_update"))
        self.scheduler.schedule_after(int(float(self.args.gen_delay_s) * PS_PER_S), EventHandler(self, "generation_tick"))
        self.scheduler.schedule_after(int(float(self.args.swap_dt_s) * PS_PER_S), EventHandler(self, "swap_tick"))
        self.scheduler.schedule_after(int(float(self.args.metrics_dt_s) * PS_PER_S), EventHandler(self, "metrics_tick"))

    def compute_link(self, label: str) -> LinkState:
        if label == "A-R1":
            u, v = "Alice", "R1"
        elif label == "R1-B":
            u, v = "R1", "Bob"
        else:
            raise KeyError(label)
        geom = self.geometry.contact_geometry(u, v, int(self.env.now))
        visible = geom.elevation_deg >= float(self.args.min_elevation_deg)
        if bool(self.args.force_visible):
            visible = True
        if visible:
            p_link, w, fid = self.fso.mean_link_metrics(
                distance_m=geom.distance_m,
                elevation_deg=geom.elevation_deg,
                intersatellite=False,
            )
        else:
            p_link, w, fid = 0.0, 0.0, 0.25
        return LinkState(label, int(self.env.now), visible, geom.distance_m, geom.elevation_deg, p_link, w, fid)

    def orbit_update(self) -> None:
        for label in ("A-R1", "R1-B"):
            state = self.compute_link(label)
            self.link_states[label] = state
            self.log("LINK_UPDATE", label=label, visible=state.visible, elevation_deg=state.elevation_deg, p_link=state.p_link)
        self._schedule_next("orbit_update", float(self.args.link_update_dt_s))

    def generation_tick(self) -> None:
        dt_s = max(0.0, float(self.args.gen_delay_s))
        attempts = max(1, int(round(max(0.0, float(self.args.source_rate_hz)) * dt_s)))
        for label in ("A-R1", "R1-B"):
            link = self.link_states.get(label) or self.compute_link(label)
            capacity = max(0, int(self.args.max_modes_per_label) - self.count_tokens(label))
            if capacity <= 0:
                continue
            if not link.visible:
                continue
            link_success = _sample_binomial(self.rng, attempts, link.p_link)
            stored = _sample_binomial(self.rng, link_success, self.memory.write_success_prob())
            accepted = min(stored, capacity)
            dropped = max(0, stored - accepted)
            self.fail_link_bin += attempts - link_success
            self.fail_recall_bin += link_success - stored
            self.fail_storage_full_bin += dropped
            self.failures_bin += (attempts - link_success) + (link_success - stored) + dropped
            if accepted > 0:
                self.tokens[label].append(TokenBatch(label, accepted, int(self.env.now), link.werner_w, link.fidelity))
            self.log("GEN_BATCH", label=label, attempts=attempts, link_success=link_success, stored=accepted, dropped=dropped)
        self._schedule_next("generation_tick", float(self.args.gen_delay_s))

    def swap_tick(self) -> None:
        count = min(
            self.count_tokens("A-R1"),
            self.count_tokens("R1-B"),
            int(self.args.max_bsm_per_cycle),
        )
        if count > 0:
            left_batches = self.pop_tokens("A-R1", count)
            right_batches = self.pop_tokens("R1-B", count)
            for n, left, right in self.pair_batches(left_batches, right_batches):
                self.execute_swap_batch(n, left, right)
        self._schedule_next("swap_tick", float(self.args.swap_dt_s))

    def execute_swap_batch(self, n: int, left: TokenBatch, right: TokenBatch) -> None:
        wait_left_s = (int(self.env.now) - left.ready_ps) / PS_PER_S + float(self.args.memory_effective_wait_s)
        wait_right_s = (int(self.env.now) - right.ready_ps) / PS_PER_S + float(self.args.memory_effective_wait_s)
        recall = self.memory.conditional_recall_prob(wait_left_s) * self.memory.conditional_recall_prob(wait_right_s)
        recalled = _sample_binomial(self.rng, n, recall)
        self.fail_recall_bin += n - recalled
        self.failures_bin += n - recalled
        if recalled <= 0:
            return

        f_left = werner_to_fidelity(left.werner_w * self.memory.coherence_factor(wait_left_s))
        f_right = werner_to_fidelity(right.werner_w * self.memory.coherence_factor(wait_right_s))
        f_out = self.bsm.swapped_fidelity(f_left, f_right)
        if f_out < float(self.args.f_min):
            self.failures_bin += recalled
            return

        successes = _sample_binomial(self.rng, recalled, self.bsm.p_succ_bsm(1.0, 1.0))
        self.fail_bsm_bin += recalled - successes
        self.failures_bin += recalled - successes
        self.successes_bin += successes
        self.cumulative_successes += successes
        self.fidelity_sum_bin += successes * f_out
        self.success_wait_sum_bin += successes * 0.5 * (wait_left_s + wait_right_s)
        self.log("SWAP_BATCH", attempts=n, recalled=recalled, successes=successes, fidelity=f_out)

    def metrics_tick(self) -> None:
        ar1 = self.link_states.get("A-R1")
        r1b = self.link_states.get("R1-B")
        successes = self.successes_bin
        mean_fid = self.fidelity_sum_bin / successes if successes > 0 else 0.0
        mean_wait_s = self.success_wait_sum_bin / successes if successes > 0 else 0.0
        self.metrics.append(
            MetricRow(
                t_s=float(self.env.now) / PS_PER_S,
                ar1_visible=int(bool(ar1 and ar1.visible)),
                r1b_visible=int(bool(r1b and r1b.visible)),
                both_visible=int(bool(ar1 and r1b and ar1.visible and r1b.visible)),
                ar1_distance_km=float((ar1.distance_m if ar1 else 0.0) / 1e3),
                r1b_distance_km=float((r1b.distance_m if r1b else 0.0) / 1e3),
                ar1_p_link=float(ar1.p_link if ar1 else 0.0),
                r1b_p_link=float(r1b.p_link if r1b else 0.0),
                ar1_elevation_deg=float(ar1.elevation_deg if ar1 else 0.0),
                r1b_elevation_deg=float(r1b.elevation_deg if r1b else 0.0),
                ar1_loss_db=float(
                    self.fso.mean_loss_db(
                        distance_m=ar1.distance_m,
                        elevation_deg=ar1.elevation_deg,
                        intersatellite=False,
                        apply_min_elevation=False,
                    )
                    if ar1
                    else 150.0
                ),
                r1b_loss_db=float(
                    self.fso.mean_loss_db(
                        distance_m=r1b.distance_m,
                        elevation_deg=r1b.elevation_deg,
                        intersatellite=False,
                        apply_min_elevation=False,
                    )
                    if r1b
                    else 150.0
                ),
                ar1_fidelity=float(ar1.fidelity if ar1 else 0.25),
                r1b_fidelity=float(r1b.fidelity if r1b else 0.25),
                successes=int(successes),
                failures=int(self.failures_bin),
                mean_fidelity_success=float(mean_fid),
                mean_wait_success_s=float(mean_wait_s),
                throughput_pairs_per_s=successes / max(1e-12, float(self.args.metrics_dt_s)),
                cumulative_successes=int(self.cumulative_successes),
                memory_ar1=self.count_tokens("A-R1"),
                memory_r1b=self.count_tokens("R1-B"),
                fail_link=int(self.fail_link_bin),
                fail_recall=int(self.fail_recall_bin),
                fail_bsm=int(self.fail_bsm_bin),
                fail_storage_full=int(self.fail_storage_full_bin),
            )
        )
        self.successes_bin = 0
        self.failures_bin = 0
        self.fidelity_sum_bin = 0.0
        self.success_wait_sum_bin = 0.0
        self.fail_link_bin = 0
        self.fail_recall_bin = 0
        self.fail_bsm_bin = 0
        self.fail_storage_full_bin = 0
        self._schedule_next("metrics_tick", float(self.args.metrics_dt_s))

    def _schedule_next(self, handler: str, delay_s: float) -> None:
        next_t = int(self.env.now) + int(max(0.0, delay_s) * PS_PER_S)
        if next_t <= int(float(self.args.duration_s) * PS_PER_S):
            self.scheduler.schedule_at(next_t, EventHandler(self, handler))

    def count_tokens(self, label: str) -> int:
        return sum(b.count for b in self.tokens[label])

    def pop_tokens(self, label: str, count: int) -> List[TokenBatch]:
        out: List[TokenBatch] = []
        remaining = int(count)
        batches = self.tokens[label]
        while remaining > 0 and batches:
            b = batches[0]
            take = min(remaining, b.count)
            out.append(TokenBatch(b.label, take, b.ready_ps, b.werner_w, b.fidelity))
            b.count -= take
            remaining -= take
            if b.count <= 0:
                batches.pop(0)
        return out

    @staticmethod
    def pair_batches(left: List[TokenBatch], right: List[TokenBatch]) -> List[Tuple[int, TokenBatch, TokenBatch]]:
        pairs: List[Tuple[int, TokenBatch, TokenBatch]] = []
        i = j = 0
        while i < len(left) and j < len(right):
            n = min(left[i].count, right[j].count)
            if n > 0:
                pairs.append((n, left[i], right[j]))
            left[i].count -= n
            right[j].count -= n
            if left[i].count <= 0:
                i += 1
            if right[j].count <= 0:
                j += 1
        return pairs


def visibility_start_offset(
    *,
    selection: SatellitePassSelection,
    args: argparse.Namespace,
) -> float:
    if selection.joint_window is None:
        return 0.0
    return max(0.0, selection.joint_window.start_s - float(args.visibility_prebuffer_s))


def run_once(args: argparse.Namespace) -> DesPhysicsNoSdn:
    alice, bob = default_ground_stations(float(args.min_elevation_deg))
    sats = default_satellites(args.tle, args.omm_json)
    epoch_unix_s = (
        float(args.epoch_unix_s)
        if args.epoch_unix_s is not None
        else representative_epoch_unix_s(sats)
    )
    selection = select_satellite_for_run(
        sats=sats,
        alice=alice,
        bob=bob,
        epoch_unix_s=epoch_unix_s,
        args=args,
    )
    sat = selection.satellite
    selected_sat = SatelliteEphemeris(
        "R1",
        sat.tle_line1,
        sat.tle_line2,
        sat.source,
        {
            **(sat.metadata or {}),
            "original_name": sat.name,
            "pass_selection": selection.objective,
            "pass_candidates_considered": selection.candidates_considered,
            "joint_window_start_s": (
                selection.joint_window.start_s if selection.joint_window is not None else None
            ),
            "joint_window_end_s": (
                selection.joint_window.end_s if selection.joint_window is not None else None
            ),
            "joint_window_overlap_s": (
                float(selection.joint_window.overlap_duration_ps) / PS_PER_S
                if selection.joint_window is not None
                else None
            ),
            "joint_window_gap_s": (
                float(selection.joint_window.gap_ps) / PS_PER_S
                if selection.joint_window is not None
                else None
            ),
        },
    )
    if bool(args.auto_start_at_visibility) and not bool(args.force_visible):
        epoch_unix_s += visibility_start_offset(selection=selection, args=args)
    propagator = SkyfieldPropagator(
        satellites=[selected_sat],
        ground_nodes=[alice, bob],
        epoch_unix_s=epoch_unix_s,
    )
    env = DESEnv(name="des_physics_no_sdn", default=True)
    sim = DesPhysicsNoSdn(
        name="physics",
        env=env,
        args=args,
        geometry=GeometryEngine(propagator),
        fso=build_fso_model(args),
        memory=build_memory_model(args),
        bsm=build_bsm_model(args),
    )
    sim.selected_satellite_name = sat.name
    sim.selected_satellite_metadata = selected_sat.metadata
    env.init()
    env.run(end_time=int(float(args.duration_s) * PS_PER_S), logging=False, summary=True)
    return sim


def write_outputs(sim: DesPhysicsNoSdn, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(MetricRow.__dataclass_fields__.keys())
    if hasattr(sim, "selected_satellite_name"):
        payload = {
            "selected_satellite_name": getattr(sim, "selected_satellite_name"),
            "selected_satellite_metadata": getattr(sim, "selected_satellite_metadata", None),
        }
        (out_dir / "selected_pass.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    with (out_dir / "des_physics_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in sim.metrics:
            writer.writerow([getattr(row, k) for k in fields])
    if not bool(sim.args.no_event_log):
        with (out_dir / "des_physics_event_log.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_ps", "kind", "payload"])
            writer.writerows(sim.event_log)
    plot_outputs(sim, out_dir)


def summarize(sim: DesPhysicsNoSdn, x: float, y: float = 0.0) -> SweepPoint:
    success_count = int(sim.cumulative_successes)
    fail_link = sum(r.fail_link for r in sim.metrics)
    fail_recall = sum(r.fail_recall for r in sim.metrics)
    fail_bsm = sum(r.fail_bsm for r in sim.metrics)
    fail_storage_full = sum(r.fail_storage_full for r in sim.metrics)
    total = success_count + fail_link + fail_recall + fail_bsm + fail_storage_full
    fid_num = sum(r.mean_fidelity_success * r.successes for r in sim.metrics)
    mean_fid = fid_num / success_count if success_count > 0 else float("nan")
    return SweepPoint(
        x=float(x),
        y=float(y),
        success_rate=success_count / total if total > 0 else 0.0,
        mean_fidelity_success=float(mean_fid),
        throughput_pairs_per_s=success_count / max(1e-12, float(sim.args.duration_s)),
        success_count=success_count,
        total_attempts=total,
    )


def _parse_values(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _normalise_kind(kind: str) -> str:
    key = str(kind).strip().lower().removeprefix("--").replace("-", "_").replace("^", "")
    aliases = {
        "sigma_r2": "sigma_r2",
        "sigma_r_2": "sigma_r2",
        "r2": "sigma_r2",
        "rytov": "sigma_r2",
        "fso_sigma_r2": "sigma_r2",
        "d_r": "d_r",
        "dr": "d_r",
        "aperture": "d_r",
        "receiver_aperture": "d_r",
        "mem_t2": "mem_t2",
        "mem_t_2": "mem_t2",
        "t2": "mem_t2",
        "t_2": "mem_t2",
        "mem_t_mem": "mem_t_mem",
        "mem_tmem": "mem_t_mem",
    }
    if key not in aliases:
        raise ValueError("unsupported sweep kind; use sigma_r2, d_r, mem_t2, or mem_t_mem")
    return aliases[key]


def _set_kind(args: argparse.Namespace, kind: str, value: float) -> None:
    if kind == "sigma_r2":
        args.fso_sigma_r2 = float(value)
    elif kind == "d_r":
        args.fso_d_r = float(value)
    elif kind == "mem_t2":
        args.mem_t2 = float(value)
    elif kind == "mem_t_mem":
        args.mem_t_mem = float(value)
    else:
        raise ValueError(kind)


def write_sweep_outputs(
    *,
    out_dir: Path,
    x_kind: str,
    y_kind: str,
    grouped: Dict[float, List[SweepPoint]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"des_physics_sweep_{x_kind}_by_{y_kind or 'none'}.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                x_kind,
                y_kind or "group",
                "success_rate",
                "mean_fidelity_success",
                "throughput_pairs_per_s",
                "success_count",
                "total_attempts",
            ]
        )
        for y, points in grouped.items():
            for p in points:
                writer.writerow(
                    [
                        p.x,
                        y,
                        p.success_rate,
                        p.mean_fidelity_success,
                        p.throughput_pairs_per_s,
                        p.success_count,
                        p.total_attempts,
                    ]
                )

    specs = [
        ("mean_fidelity_success", "mean_fidelity_success"),
        ("success_rate", "success_rate"),
        ("throughput_pairs_per_s", "throughput_pairs_per_s"),
    ]
    x_log = x_kind in {"mem_t2", "mem_t_mem"}
    for attr, ylabel in specs:
        plt.figure()
        for y, points in sorted(grouped.items()):
            ordered = sorted(points, key=lambda p: p.x)
            label = f"{y_kind}={y:g}" if y_kind else "DES no-SDN"
            plt.plot([p.x for p in ordered], [getattr(p, attr) for p in ordered], marker="o", label=label)
        if x_log:
            plt.xscale("log")
        plt.xlabel(x_kind)
        plt.ylabel(ylabel)
        plt.title(f"DES no-SDN {ylabel} vs {x_kind}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"des_physics_sweep_{x_kind}_by_{y_kind or 'none'}_{attr}.png")
        plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (attr, ylabel) in zip(axes, specs):
        for y, points in sorted(grouped.items()):
            ordered = sorted(points, key=lambda p: p.x)
            label = f"{y_kind}={y:g}" if y_kind else "DES no-SDN"
            ax.plot([p.x for p in ordered], [getattr(p, attr) for p in ordered], marker="o", label=label)
        if x_log:
            ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel(x_kind)
    fig.suptitle(f"DES no-SDN sweep: {x_kind} by {y_kind or 'single'}")
    fig.tight_layout()
    fig.savefig(out_dir / f"des_physics_sweep_{x_kind}_by_{y_kind or 'none'}_combined.png")
    plt.close(fig)
    print(f"[des-physics] wrote sweep outputs to: {out_dir.resolve()}")


def run_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    x_kind = _normalise_kind(args.sweep)
    x_values = _parse_values(args.values)
    y_kind = _normalise_kind(args.sweep_y) if str(args.sweep_y).strip() else ""
    y_values = _parse_values(args.values_y) if y_kind else [0.0]
    repeats = max(1, int(args.sweep_repeats))
    grouped: Dict[float, List[SweepPoint]] = {}
    run_index = 0
    for y in y_values:
        points: List[SweepPoint] = []
        for x in x_values:
            repeated: List[SweepPoint] = []
            for repeat in range(repeats):
                run_args = argparse.Namespace(**vars(args))
                run_args.seed = int(args.seed) + run_index
                run_args.no_plots = True
                run_args.no_event_log = True
                _set_kind(run_args, x_kind, x)
                if y_kind:
                    _set_kind(run_args, y_kind, y)
                print(
                    f"[des-physics sweep] {x_kind}={x:g}"
                    + (f", {y_kind}={y:g}" if y_kind else "")
                    + f", repeat={repeat + 1}/{repeats}"
                )
                sim = run_once(run_args)
                repeated.append(summarize(sim, x=x, y=y))
                run_index += 1
            success_count = sum(p.success_count for p in repeated)
            total_attempts = sum(p.total_attempts for p in repeated)
            fid_num = sum(
                p.mean_fidelity_success * p.success_count
                for p in repeated
                if p.success_count > 0
            )
            points.append(
                SweepPoint(
                    x=x,
                    y=y,
                    success_rate=success_count / total_attempts if total_attempts > 0 else 0.0,
                    mean_fidelity_success=fid_num / success_count if success_count > 0 else float("nan"),
                    throughput_pairs_per_s=sum(p.throughput_pairs_per_s for p in repeated) / len(repeated),
                    success_count=success_count,
                    total_attempts=total_attempts,
                )
            )
        grouped[y] = points
    write_sweep_outputs(out_dir=out_dir, x_kind=x_kind, y_kind=y_kind, grouped=grouped)


def plot_outputs(sim: DesPhysicsNoSdn, out_dir: Path) -> None:
    if bool(sim.args.no_plots) or not sim.metrics:
        return
    ts = [r.t_s for r in sim.metrics]

    def save(name: str, title: str, ylabel: str, series: List[Tuple[str, List[float]]]) -> None:
        plt.figure()
        for label, values in series:
            plt.plot(ts, values, marker="o", markersize=2, label=label)
        plt.xlabel("time_s")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if len(series) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / name)
        plt.close()

    save("visibility_over_time.png", "DES visibility windows", "visible", [("A-R1", [r.ar1_visible for r in sim.metrics]), ("R1-B", [r.r1b_visible for r in sim.metrics])])
    save("throughput_over_time.png", "DES no-SDN entanglement throughput", "pairs_per_s", [("throughput", [r.throughput_pairs_per_s for r in sim.metrics])])
    save("fidelity_over_time.png", "DES no-SDN end-to-end fidelity", "mean_fidelity_success", [("fidelity", [r.mean_fidelity_success for r in sim.metrics])])
    save("memory_occupancy_over_time.png", "DES no-SDN memory occupancy", "tokens", [("A-R1", [r.memory_ar1 for r in sim.metrics]), ("R1-B", [r.memory_r1b for r in sim.metrics])])
    save("per_link_plink_over_time.png", "DES no-SDN per-link success probability", "p_link", [("A-R1", [r.ar1_p_link for r in sim.metrics]), ("R1-B", [r.r1b_p_link for r in sim.metrics])])
    save("per_link_loss_over_time.png", "DES no-SDN per-link downlink loss", "loss_dB", [("A-R1", [r.ar1_loss_db for r in sim.metrics]), ("R1-B", [r.r1b_loss_db for r in sim.metrics])])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Controller-free DES physics simulator for FSO + memory + BSM")
    p.add_argument("--out", default="./des_physics_no_sdn_out")
    p.add_argument("--sweep", default="", help="Optional sweep: sigma_r2, d_r, mem_t2, or mem_t_mem")
    p.add_argument("--values", default="0,0.5,1,1.5,2")
    p.add_argument("--sweep-y", default="", help="Optional grouped sweep parameter")
    p.add_argument("--values-y", default="10,100,1000")
    p.add_argument("--sweep-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--duration-s", type=float, default=5000.0)
    p.add_argument("--metrics-dt-s", type=float, default=1.0)
    p.add_argument("--link-update-dt-s", type=float, default=1.0)
    p.add_argument("--gen-delay-s", type=float, default=0.01)
    p.add_argument("--swap-dt-s", type=float, default=0.01)
    p.add_argument("--source-rate-hz", type=float, default=1_000_000.0)
    p.add_argument("--max-modes-per-label", type=int, default=2048)
    p.add_argument("--max-bsm-per-cycle", type=int, default=512)
    p.add_argument("--f-min", type=float, default=0.25)
    p.add_argument("--memory-effective-wait-s", type=float, default=0.0)
    p.add_argument("--min-elevation-deg", type=float, default=5.0)
    p.add_argument("--force-visible", action="store_true")
    p.add_argument("--tle", default=None)
    p.add_argument("--omm-json", default=None)
    p.add_argument("--sat-name", default=None)
    p.add_argument(
        "--pass-selection",
        choices=["memory_aware", "longest_dual", "highest_dual"],
        default="memory_aware",
    )
    p.add_argument("--pass-candidate-limit", type=int, default=64)
    p.add_argument("--pass-prefilter-step-s", type=float, default=300.0)
    p.add_argument("--epoch-unix-s", type=float, default=None)
    p.add_argument("--auto-start-at-visibility", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--visibility-scan-s", type=float, default=86400.0)
    p.add_argument("--visibility-scan-step-s", type=float, default=10.0)
    p.add_argument("--visibility-prebuffer-s", type=float, default=120.0)

    p.add_argument("--fso-eta-src", type=float, default=1.0)
    p.add_argument("--fso-eta-det", type=float, default=1.0)
    p.add_argument("--fso-eta-cpl", type=float, default=0.8)
    p.add_argument("--fso-eta-point", type=float, default=0.98)
    p.add_argument("--fso-d-t", type=float, default=0.20)
    p.add_argument("--fso-d-r", type=float, default=1.0)
    p.add_argument("--fso-theta-div", type=float, default=3e-6)
    p.add_argument("--fso-t-abs", type=float, default=0.95)
    p.add_argument("--fso-t-sca", type=float, default=0.95)
    p.add_argument("--fso-c-atm", type=float, default=0.05)
    p.add_argument("--fso-sigma-r2", type=float, default=0.05)
    p.add_argument("--fso-turbulence-loss-coeff", type=float, default=0.06)
    p.add_argument("--fso-turbulence-quality-coeff", type=float, default=0.025)
    p.add_argument("--fso-delta-t", type=float, default=1e-9)
    p.add_argument("--fso-n0", type=float, default=1e-5)
    p.add_argument("--fso-n-dc", type=float, default=100.0)
    p.add_argument("--fso-n-conv", type=float, default=0.0)
    p.add_argument("--fso-p-t", type=float, default=0.05)
    p.add_argument("--fso-pointing-jitter-rad", type=float, default=1e-8)
    p.add_argument("--disable-fso-pointing-jitter", action="store_true")
    p.add_argument("--disable-fso-stochastic-fading", action="store_true")

    p.add_argument("--mem-platform", choices=["REID", "COLD_ATOM", "TRAPPED_ION"], default="COLD_ATOM")
    p.add_argument("--mem-t2", type=float, default=100.0)
    p.add_argument("--mem-d", type=float, default=20.0)
    p.add_argument("--mem-t-mem", type=float, default=1000.0)
    p.add_argument("--mem-n", type=float, default=1.3)
    p.add_argument("--mem-m", type=float, default=1.0)
    p.add_argument("--mem-eta-mode0", type=float, default=0.9)
    p.add_argument("--mem-eta-ti", type=float, default=0.3)

    p.add_argument("--bsm-v-hom", type=float, default=0.95)
    p.add_argument("--bsm-eta-det", type=float, default=0.8)
    p.add_argument("--bsm-sigma-det-s", type=float, default=50e-12)
    p.add_argument("--bsm-delta-t-s", type=float, default=200e-12)
    p.add_argument("--bsm-p-dc", type=float, default=1e-6)
    p.add_argument("--no-event-log", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if str(args.sweep).strip():
        run_sweep(args, Path(args.out))
        return
    sim = run_once(args)
    out_dir = Path(args.out)
    write_outputs(sim, out_dir)
    print(f"[des-physics] wrote outputs to: {out_dir.resolve()}")
    print(f"[des-physics] cumulative_successes={sim.cumulative_successes}")


if __name__ == "__main__":
    main()
