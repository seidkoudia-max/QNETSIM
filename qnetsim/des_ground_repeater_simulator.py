from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channels.fiber import FiberAnalyticalChannelModel, FiberParams
from channels.fso_temporal import TemporalFSOChannel, TemporalFSOParams
from des_physics_no_sdn_simulator import (
    DESEnv,
    Entity,
    EventHandler,
    PS_PER_S,
    build_bsm_model,
    build_fso_model,
    build_memory_model,
    default_satellites,
    representative_epoch_unix_s,
    require_qnet_des,
)
from mobility.geometry import ContactGeometry, ContactWindow, GeometryEngine
from mobility.propagator import GroundStation, SkyfieldPropagator
from mobility.tle_loader import SatelliteEphemeris
from quantum.bell_diagonal import BellDiagonalState
from quantum.optical_instruments import CoincidenceBSM, PhysicalBSMParams, ThresholdDetector

WINDHOF_OGS_DEFAULT = {
    "name": "Windhof OGS",
    "lat_deg": 49.65,
    "lon_deg": 5.95,
    "alt_m": 320.0,
}

KIRCHBERG_END_DEFAULT = {
    "name": "Kirchberg End Node",
    "lat_deg": 49.6298,
    "lon_deg": 6.1720,
    "alt_m": 290.0,
}

GROUND_SDN_REQ_ID = 1
GROUND_SDN_EDGE_INFO: Dict[str, Tuple[str, str]] = {
    "S-R1": ("SAT_END", "R1"),
    "R1-G": ("R1", "GROUND_END"),
}


def _great_circle_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    r_km = 6371.0
    lat1 = math.radians(float(lat1_deg))
    lon1 = math.radians(float(lon1_deg))
    lat2 = math.radians(float(lat2_deg))
    lon2 = math.radians(float(lon2_deg))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def ground_repeater_sites(args: argparse.Namespace) -> Tuple[GroundStation, GroundStation]:
    ogs = GroundStation(
        "OGS",
        lat_deg=float(args.ogs_lat_deg),
        lon_deg=float(args.ogs_lon_deg),
        alt_m=float(args.ogs_alt_m),
        min_elevation_deg=float(args.min_elevation_deg),
    )
    ground_end = GroundStation(
        "GROUND_END",
        lat_deg=float(args.ground_end_lat_deg),
        lon_deg=float(args.ground_end_lon_deg),
        alt_m=float(args.ground_end_alt_m),
        min_elevation_deg=90.0,
    )
    return ogs, ground_end


def effective_fiber_length_km(args: argparse.Namespace) -> float:
    if hasattr(args, "effective_fiber_length_km"):
        return float(args.effective_fiber_length_km)
    if bool(getattr(args, "auto_fiber_length_from_sites", False)):
        return _great_circle_km(
            float(args.ogs_lat_deg),
            float(args.ogs_lon_deg),
            float(args.ground_end_lat_deg),
            float(args.ground_end_lon_deg),
        )
    return float(args.fiber_length_km)


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


@dataclass(frozen=True)
class GroundPassSelection:
    satellite: SatelliteEphemeris
    contact_window: Optional[ContactWindow]
    candidates_considered: int
    objective: str


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
    loss_db: float
    reference_loss_db: float = float("nan")
    fading_factor: float = 1.
    pointing_factor: float = 1.


@dataclass
class TokenBatch:
    label: str
    count: int
    ready_ps: int
    werner_w: float
    fidelity: float
    bell_state: Optional[BellDiagonalState] = None


@dataclass
class MetricRow:
    t_s: float
    sat_visible: int
    sat_distance_km: float
    sat_elevation_deg: float
    sat_p_link: float
    sat_loss_db: float
    sat_fidelity: float
    fiber_distance_km: float
    fiber_p_link: float
    fiber_loss_db: float
    fiber_fidelity: float
    successes: int
    failures: int
    mean_fidelity_success: float
    mean_wait_success_s: float
    throughput_pairs_per_s: float
    cumulative_successes: int
    memory_sat: int
    memory_fiber: int
    fail_link: int
    fail_recall: int
    fail_bsm: int
    fail_storage_full: int
    generation_attempts: int = 0
    link_successes: int = 0
    stored_tokens: int = 0
    swap_attempts: int = 0
    recalled_pairs: int = 0
    heralded_pairs: int = 0
    fail_write: int = 0
    fail_readout: int = 0
    fail_quality: int = 0
    target_fidelity_above_half: int = 0
    bell_phi_plus: float = float("nan")
    bell_psi_plus: float = float("nan")
    bell_phi_minus: float = float("nan")
    bell_psi_minus: float = float("nan")


def _single_station_score(window: ContactWindow, objective: str, memory_coherence_s: float) -> float:
    if objective == "highest_dual":
        return float(window.peak_elevation_deg)
    if objective == "longest_dual":
        return float(window.duration_ps)
    duration_s = float(window.duration_ps) / PS_PER_S
    memory_weight = math.exp(-max(0.0, duration_s) / max(1e-12, float(memory_coherence_s)))
    return float(window.peak_elevation_deg) * max(1.0, duration_s) * (0.5 + 0.5 * memory_weight)


def select_satellite_for_ground_run(
    *,
    sats: List[SatelliteEphemeris],
    ogs: GroundStation,
    epoch_unix_s: float,
    args: argparse.Namespace,
) -> GroundPassSelection:
    if not sats:
        raise ValueError("need at least one satellite")

    propagator = SkyfieldPropagator(
        satellites=sats,
        ground_nodes=[ogs],
        epoch_unix_s=epoch_unix_s,
    )
    geometry = GeometryEngine(propagator)
    coarse_step_ps = int(max(30.0, float(args.pass_prefilter_step_s)) * PS_PER_S)
    fine_step_ps = int(max(1.0, float(args.visibility_scan_step_s)) * PS_PER_S)
    scan_end_ps = int(float(args.visibility_scan_s) * PS_PER_S)

    scored: List[Tuple[float, float, SatelliteEphemeris]] = []
    for sat in sats:
        best_elev = -1e18
        t_ps = 0
        while t_ps <= scan_end_ps:
            best_elev = max(best_elev, propagator.elevation_deg(sat.name, ogs.name, t_ps))
            t_ps += coarse_step_ps
        scored.append((best_elev, -float(sat.name.__hash__() & 0xFFFF), sat))
    scored.sort(reverse=True)
    candidates = [sat for _score, _tiebreak, sat in scored[: max(1, int(args.pass_candidate_limit))]]

    best_sat = candidates[0]
    best_window: Optional[ContactWindow] = None
    best_score = -1e18
    for sat in candidates:
        windows = geometry.contact_windows(
            ogs.name,
            sat.name,
            start_ps=0,
            end_ps=scan_end_ps,
            step_ps=fine_step_ps,
        )
        for window in windows:
            score = _single_station_score(window, str(args.pass_selection), float(args.mem_t2))
            if score > best_score:
                best_score = score
                best_sat = sat
                best_window = window

    return GroundPassSelection(
        satellite=best_sat,
        contact_window=best_window,
        candidates_considered=len(candidates),
        objective=str(args.pass_selection),
    )


def visibility_start_offset(*, selection: GroundPassSelection, args: argparse.Namespace) -> float:
    if selection.contact_window is None:
        return 0.0
    return max(0.0, float(selection.contact_window.start_ps) / PS_PER_S - float(args.visibility_prebuffer_s))


class DesGroundRepeater(Entity):
    def __init__(
        self,
        *,
        name: str,
        env: DESEnv,
        args: argparse.Namespace,
        geometry: GeometryEngine,
        fso,
        fiber: FiberAnalyticalChannelModel,
        memory,
        bsm,
        controller_client=None,
    ):
        super().__init__(name=name, env=env)
        self.args = args
        self.geometry = geometry
        self.fso = fso
        self.fiber = fiber
        self.memory = memory
        self.bsm = bsm
        self.controller_backend = str(getattr(args, "ground_controller_backend", "des_only")).lower().strip()
        self.controller_client = controller_client
        self._controller_cache_t_ps: int | None = None
        self._controller_cache: List[Dict[str, object]] | None = None
        self.rng = random.Random(int(args.seed))
        self.link_states: Dict[str, LinkState] = {}
        self.tokens: Dict[str, List[TokenBatch]] = {"S-R1": [], "R1-G": []}
        self.metrics: List[MetricRow] = []
        self.event_log: List[List[object]] = []
        self.channel_mode = str(getattr(args, "ground_channel_mode", "mean"))
        if self.channel_mode not in {"mean", "correlated"}:
            raise ValueError("unknown Ground optical channel mode")
        self.temporal_fso = (TemporalFSOChannel(fso, seed=int(args.seed), params=TemporalFSOParams(
            fading_correlation_s=float(getattr(args, "fso_fading_correlation_s", 1.)),
            pointing_correlation_s=float(getattr(args, "fso_pointing_correlation_s", 1.))))
            if self.channel_mode == "correlated" else None)
        self.refresh_channel_per_generation = self.temporal_fso is not None or bool(getattr(args, "refresh_fso_per_generation", False))
        self.optical_trace: List[Dict[str, Any]] = []
        self.memory_trace: List[Dict[str, Any]] = []
        self.event_order = str(getattr(args, "ground_event_order", "legacy"))
        if self.event_order not in {"legacy", "phased"}:
            raise ValueError("unknown Ground event ordering")
        self.swap_policy = str(getattr(args, "ground_swap_policy", "legacy_memory_rate"))
        if self.swap_policy not in {"legacy_memory_rate", "per_cycle", "attempt_rate"}:
            raise ValueError("unknown Ground swap processing policy")
        self.bsm_attempt_rate_hz = getattr(args, "bsm_attempt_rate_hz", None)
        if self.swap_policy == "attempt_rate":
            if self.bsm_attempt_rate_hz is None or not math.isfinite(self.bsm_attempt_rate_hz) or self.bsm_attempt_rate_hz < 0:
                raise ValueError("attempt_rate requires an explicit finite nonnegative bsm_attempt_rate_hz")
        elif self.bsm_attempt_rate_hz is not None:
            raise ValueError("bsm_attempt_rate_hz requires ground_swap_policy=attempt_rate")
        if self.swap_policy != "legacy_memory_rate":
            if not math.isfinite(args.swap_dt_s) or args.swap_dt_s <= 0 or args.max_bsm_per_cycle < 0:
                raise ValueError("swap interval must be positive and cycle limit nonnegative")
        self._last_budget_ps = int(env.now)
        self._attempt_fraction = 0.
        self.memory_quality_threshold = float(getattr(args, "memory_quality_threshold", .5))
        if not 0 <= self.memory_quality_threshold <= 1:
            raise ValueError("memory quality threshold must be between zero and one")

        self.successes_bin = 0
        self.failures_bin = 0
        self.fidelity_sum_bin = 0.0
        self.success_wait_sum_bin = 0.0
        self.fail_link_bin = 0
        self.fail_recall_bin = 0
        self.fail_bsm_bin = 0
        self.fail_storage_full_bin = 0
        self.cumulative_successes = 0
        self.state_model = str(getattr(args, "ground_state_model", "legacy"))
        self.memory_noise = str(getattr(args, "ground_memory_noise", "phase"))
        if self.state_model not in {"legacy", "bell_diagonal"}:
            raise ValueError("unknown Ground state model")
        if self.memory_noise not in {"phase", "isotropic"}:
            raise ValueError("unknown Ground memory noise channel")
        self.bell_bsm = None
        if self.state_model == "bell_diagonal":
            window = float(args.bsm_delta_t_s)
            dark_probability = float(args.bsm_p_dc)
            if window <= 0 or not 0 <= dark_probability < 1:
                raise ValueError("Bell-vector mode requires a positive gate and 0 <= p_dc < 1")
            # Map the configured per-gate dark probability to an effective channel
            # rate. This is not a calibrated four-detector optical circuit.
            self.bell_bsm = CoincidenceBSM(
                PhysicalBSMParams(phase_visibility=bsm.effective_visibility(),
                    coincidence_window_s=window, noise_channel="phase"),
                ThresholdDetector(efficiency=float(args.bsm_eta_det),
                    jitter_sigma_s=float(args.bsm_sigma_det_s),
                    dark_rate_hz=-math.log1p(-dark_probability) / window,
                    dead_time_s=0.))
        self.bell_batches: List[Dict[str, Any]] = []
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.diagnostics = dict.fromkeys(("generation_attempts", "link_successes", "stored_tokens",
            "swap_attempts", "recalled_pairs", "heralded_pairs", "fail_write", "fail_readout",
            "fail_quality", "target_fidelity_above_half"), 0)
        self.bell_sum = [0.] * 4

    def per_tick_memory_rate_cap(self) -> int:
        """Historical numerical pacing only; memory slots are not a rate in Hz."""
        return max(1, int(math.floor(float(self.args.max_modes_per_label) * max(0.0, float(self.args.swap_dt_s)))))

    def swap_attempt_budget(self) -> int:
        """One shared operand-pair budget per tick, before recall and BSM trials.

        per_cycle removes the implicit modes-to-Hz conversion for diagnostics;
        its effective ceiling still depends on the tick duration. attempt_rate
        uses a separately supplied service assumption, not detector timing or
        memory capacity. Unused whole opportunities expire instead of banking.
        """
        cycle_limit = max(0, int(self.args.max_bsm_per_cycle))
        if self.swap_policy == "legacy_memory_rate":
            return min(cycle_limit, self.per_tick_memory_rate_cap())
        if self.swap_policy == "per_cycle":
            return cycle_limit
        now = int(self.env.now)
        if now < self._last_budget_ps:
            raise ValueError("swap budget time cannot move backwards")
        available = self._attempt_fraction + self.bsm_attempt_rate_hz * (now-self._last_budget_ps) / PS_PER_S
        # Avoid losing an integer slot to binary roundoff at exact boundaries.
        slots = int(math.floor(available + 1e-12))
        self._attempt_fraction = max(0., available-slots)
        self._last_budget_ps = now
        return min(cycle_limit, slots)

    def log(self, kind: str, **payload: object) -> None:
        if not bool(self.args.no_event_log):
            self.event_log.append([int(self.env.now), kind, payload])

    def init(self) -> None:
        if self.controller_client is not None and self.controller_backend != "des_only":
            self.controller_client.reset()
            self.controller_client.submit_request(
                req_id=GROUND_SDN_REQ_ID,
                src="SAT_END",
                dst="GROUND_END",
                f_min=float(self.args.f_min),
                timeout_ps=int(float(self.args.duration_s) * PS_PER_S),
                t_ps=0,
                target_pairs=0,
            )
        self.scheduler.schedule_now(EventHandler(self, "orbit_update"), priority=self.event_priority("orbit_update"))
        for handler, delay in (("generation_tick", self.args.gen_delay_s),
                               ("swap_tick", self.args.swap_dt_s), ("metrics_tick", self.args.metrics_dt_s)):
            self.scheduler.schedule_after(int(float(delay)*PS_PER_S), EventHandler(self, handler),
                                          priority=self.event_priority(handler))

    def event_priority(self, handler: str):
        # Keep archived equal-priority ordering by default. New studies can
        # make bin-end measurements independent of equal-time heap ordering.
        if self.event_order == "legacy":
            return None
        return {"orbit_update": 0, "generation_tick": 1, "swap_tick": 2, "metrics_tick": 3}[handler]

    def _controller_actions(self) -> List[Dict[str, object]]:
        if self.controller_client is None or self.controller_backend == "des_only":
            return []
        now_ps = int(self.env.now)
        if self._controller_cache_t_ps == now_ps and self._controller_cache is not None:
            return self._controller_cache
        payload = self.controller_client.poll(req_id=GROUND_SDN_REQ_ID, t_ps=now_ps)
        actions = payload if isinstance(payload, list) else []
        self._controller_cache_t_ps = now_ps
        self._controller_cache = actions
        return actions

    def _controller_generation_labels(self) -> set[str]:
        return {
            str(action.get("label", ""))
            for action in self._controller_actions()
            if str(action.get("msg_type", "")) == "StartSchedule"
        }

    def _controller_swap_actions(self) -> List[Dict[str, object]]:
        return [
            action
            for action in self._controller_actions()
            if str(action.get("msg_type", "")) == "InstructSwap"
        ]

    def _controller_link_payload(self, link: LinkState) -> Dict[str, object]:
        u, v = GROUND_SDN_EDGE_INFO[link.label]
        return {
            "label": str(link.label),
            "u": str(u),
            "v": str(v),
            "t_ps": int(self.env.now),
            "distance_m": float(link.distance_m),
            "visible": bool(link.visible),
            "elevation_deg": float(link.elevation_deg),
            "p_link": float(link.p_link),
            "fidelity": float(link.fidelity),
            "u_fid": float(link.fidelity),
            "node": "R1",
        }

    def compute_satellite_link(self) -> LinkState:
        geom: ContactGeometry = self.geometry.contact_geometry("OGS", "SAT_END", int(self.env.now))
        visible = geom.elevation_deg >= float(self.args.min_elevation_deg)
        if bool(self.args.force_visible):
            visible = True
        sample = None
        if self.temporal_fso is not None:
            self.temporal_fso.advance(float(self.env.now) / PS_PER_S)
        if visible and self.temporal_fso is not None:
            sample = self.temporal_fso.sample(time_s=float(self.env.now) / PS_PER_S,
                distance_m=geom.distance_m, elevation_deg=geom.elevation_deg)
            p_link, w, fid = sample.efficiency, sample.werner_w, sample.fidelity
            loss_db = -10*math.log10(p_link) if p_link > 0 else float("nan")
            reference_loss = (-10*math.log10(sample.reference_efficiency)
                              if sample.reference_efficiency > 0 else float("nan"))
        elif visible:
            p_link, w, fid = self.fso.mean_link_metrics(
                distance_m=geom.distance_m,
                elevation_deg=geom.elevation_deg,
                intersatellite=False,
            )
            loss_db = self.fso.mean_loss_db(
                distance_m=geom.distance_m,
                elevation_deg=geom.elevation_deg,
                intersatellite=False,
                apply_min_elevation=False,
            )
            reference_loss = loss_db
        else:
            p_link, w, fid, loss_db = 0.0, 0.0, 0.25, 150.0
            reference_loss = float("nan")
        link = LinkState(
            label="S-R1",
            t_ps=int(self.env.now),
            visible=visible,
            distance_m=geom.distance_m,
            elevation_deg=geom.elevation_deg,
            p_link=p_link,
            werner_w=w,
            fidelity=fid,
            loss_db=loss_db,
            reference_loss_db=reference_loss,
            fading_factor=sample.fading_factor if sample else 1.,
            pointing_factor=sample.pointing_factor if sample else 1.,
        )
        if bool(getattr(self.args, "record_optical_trace", False)):
            self.optical_trace.append(dict(t_s=float(self.env.now)/PS_PER_S, visible=int(visible),
                loss_db=loss_db if visible else float("nan"), reference_loss_db=reference_loss,
                p_link=p_link, fidelity=fid if visible else float("nan"),
                fading_factor=link.fading_factor, pointing_factor=link.pointing_factor,
                pointing_x_rad=sample.pointing_x_rad if sample else 0.,
                pointing_y_rad=sample.pointing_y_rad if sample else 0.,
                fading_clipped=int(sample.fading_clipped) if sample else 0,
                efficiency_clipped=int(sample.efficiency_clipped) if sample else 0))
        return link

    def compute_fiber_link(self) -> LinkState:
        length_km = effective_fiber_length_km(self.args)
        p_link, w, fid = self.fiber.mean_link_metrics(length_km=length_km)
        return LinkState(
            label="R1-G",
            t_ps=int(self.env.now),
            visible=True,
            distance_m=length_km * 1e3,
            elevation_deg=90.0,
            p_link=p_link,
            werner_w=w,
            fidelity=fid,
            loss_db=self.fiber.mean_loss_db(length_km=length_km),
        )

    def orbit_update(self) -> None:
        if not self.refresh_channel_per_generation or "S-R1" not in self.link_states:
            self.link_states["S-R1"] = self.compute_satellite_link()
        self.link_states["R1-G"] = self.compute_fiber_link()
        if self.controller_client is not None and self.controller_backend != "des_only":
            self._controller_cache_t_ps = None
            self._controller_cache = None
            for label in ("S-R1", "R1-G"):
                self.controller_client.update_link_state(self._controller_link_payload(self.link_states[label]))
        self.log(
            "LINK_UPDATE",
            sat_visible=self.link_states["S-R1"].visible,
            sat_elevation_deg=self.link_states["S-R1"].elevation_deg,
            sat_p_link=self.link_states["S-R1"].p_link,
            fiber_p_link=self.link_states["R1-G"].p_link,
        )
        self._schedule_next("orbit_update", float(self.args.link_update_dt_s))

    def generation_tick(self) -> None:
        if self.refresh_channel_per_generation:
            # Update even for a full buffer: environmental randomness must not
            # depend on whether this realization happens to request more pairs.
            self.link_states["S-R1"] = self.compute_satellite_link()
        self.record_memory_snapshot("pre_generation")
        dt_s = max(0.0, float(self.args.gen_delay_s))
        attempts = max(1, int(round(max(0.0, float(self.args.source_rate_hz)) * dt_s)))
        allowed_labels = (
            self._controller_generation_labels()
            if self.controller_client is not None and self.controller_backend != "des_only"
            else {"S-R1", "R1-G"}
        )
        for label in ("S-R1", "R1-G"):
            if label not in allowed_labels:
                continue
            link = self.link_states.get(label)
            if link is None:
                link = self.compute_satellite_link() if label == "S-R1" else self.compute_fiber_link()
                self.link_states[label] = link
            capacity = max(0, int(self.args.max_modes_per_label) - self.count_tokens(label))
            if capacity <= 0:
                continue
            if not link.visible:
                continue
            link_success = _sample_binomial(self.rng, attempts, link.p_link)
            stored = _sample_binomial(self.rng, link_success, self.memory.write_success_prob())
            accepted = min(stored, capacity)
            dropped = max(0, stored - accepted)
            self.diagnostics["generation_attempts"] += attempts
            self.diagnostics["link_successes"] += link_success
            self.diagnostics["stored_tokens"] += accepted
            self.diagnostics["fail_write"] += link_success - stored
            self.fail_link_bin += attempts - link_success
            self.fail_recall_bin += link_success - stored
            self.fail_storage_full_bin += dropped
            self.failures_bin += (attempts - link_success) + (link_success - stored) + dropped
            if accepted > 0:
                state = BellDiagonalState.werner(link.fidelity) if self.bell_bsm else None
                self.tokens[label].append(TokenBatch(label, accepted, int(self.env.now), link.werner_w, link.fidelity, state))
                if self.controller_client is not None and self.controller_backend != "des_only":
                    self.controller_client.report_storage(
                        req_id=GROUND_SDN_REQ_ID,
                        label=label,
                        accepted_count=int(accepted),
                    )
            self.log("GEN_BATCH", label=label, attempts=attempts, link_success=link_success, stored=accepted, dropped=dropped)
        self.record_memory_snapshot("post_generation")
        self._schedule_next("generation_tick", float(self.args.gen_delay_s))

    def swap_tick(self) -> None:
        before = self.cumulative_successes
        self.record_memory_snapshot("pre_swap")
        remaining = self.swap_attempt_budget()
        if self.controller_client is not None and self.controller_backend != "des_only":
            for action in self._controller_swap_actions():
                count = min(
                    self.count_tokens(str(action.get("left_label", "S-R1"))),
                    self.count_tokens(str(action.get("right_label", "R1-G"))),
                    int(action.get("count", 0)),
                    remaining,
                )
                if count <= 0:
                    continue
                remaining -= count
                sat_batches = self.pop_tokens(str(action.get("left_label", "S-R1")), count)
                fiber_batches = self.pop_tokens(str(action.get("right_label", "R1-G")), count)
                for n, left, right in self.pair_batches(sat_batches, fiber_batches):
                    self.execute_swap_batch(n, left, right)
        else:
            count = min(
                self.count_tokens("S-R1"),
                self.count_tokens("R1-G"),
                remaining,
            )
            if count > 0:
                sat_batches = self.pop_tokens("S-R1", count)
                fiber_batches = self.pop_tokens("R1-G", count)
                for n, left, right in self.pair_batches(sat_batches, fiber_batches):
                    self.execute_swap_batch(n, left, right)
        self.record_memory_snapshot("post_swap", delivered=self.cumulative_successes-before)
        self._schedule_next("swap_tick", float(self.args.swap_dt_s))

    def state_after_storage(self, batch: TokenBatch, wait_s: float) -> BellDiagonalState:
        state = batch.bell_state or BellDiagonalState.werner(batch.fidelity)
        coherence = self.memory.coherence_factor(wait_s)
        return state.dephase(coherence) if self.memory_noise == "phase" else state.depolarize(coherence)

    def record_memory_snapshot(self, phase: str, delivered: int = 0) -> None:
        if not bool(getattr(self.args, "record_memory_trace", False)):
            return
        row = dict(event_index=len(self.memory_trace), t_s=float(self.env.now)/PS_PER_S,
                   phase=phase, delivered=delivered, cumulative_delivered=self.cumulative_successes)
        for label, prefix in (("S-R1", "sat"), ("R1-G", "fiber")):
            occupied, retrievable, qualified, retrievable_qualified = 0, 0., 0, 0.
            for batch in self.tokens[label]:
                age = (int(self.env.now)-batch.ready_ps)/PS_PER_S + float(self.args.memory_effective_wait_s)
                probability = self.memory.conditional_recall_prob(age)
                fidelity = (self.state_after_storage(batch, age).fidelity if self.bell_bsm else
                            max(.25, batch.fidelity*self.memory.coherence_factor(age)))
                good = fidelity > self.memory_quality_threshold
                occupied += batch.count
                retrievable += batch.count*probability
                qualified += batch.count*int(good)
                retrievable_qualified += batch.count*probability*int(good)
            row.update({prefix+"_occupied": occupied, prefix+"_expected_retrievable": retrievable,
                        prefix+"_above_threshold": qualified,
                        prefix+"_expected_retrievable_above_threshold": retrievable_qualified})
        self.memory_trace.append(row)

    def execute_swap_batch(self, n: int, left: TokenBatch, right: TokenBatch) -> None:
        wait_left_s = (int(self.env.now) - left.ready_ps) / PS_PER_S + float(self.args.memory_effective_wait_s)
        wait_right_s = (int(self.env.now) - right.ready_ps) / PS_PER_S + float(self.args.memory_effective_wait_s)
        recall = self.memory.conditional_recall_prob(wait_left_s) * self.memory.conditional_recall_prob(wait_right_s)
        recalled = _sample_binomial(self.rng, n, recall)
        self.diagnostics["swap_attempts"] += n
        self.diagnostics["recalled_pairs"] += recalled
        self.diagnostics["fail_readout"] += n - recalled
        self.fail_recall_bin += n - recalled
        self.failures_bin += n - recalled
        if recalled <= 0:
            return

        state_out = None
        if self.bell_bsm is not None:
            states = []
            for batch, wait_s in ((left, wait_left_s), (right, wait_right_s)):
                states.append(self.state_after_storage(batch, wait_s))
            probability, state_out = self.bell_bsm.output(*states)
            f_out = state_out.fidelity
        else:
            f_left = max(0.25, min(1.0, left.fidelity * self.memory.coherence_factor(wait_left_s)))
            f_right = max(0.25, min(1.0, right.fidelity * self.memory.coherence_factor(wait_right_s)))
            f_out = self.bsm.swapped_fidelity(f_left, f_right)
            if f_out < float(self.args.f_min):
                self.diagnostics["fail_quality"] += recalled
                self.failures_bin += recalled
                return
            probability = self.bsm.p_succ_bsm(1.0, 1.0)

        heralded = _sample_binomial(self.rng, recalled, probability)
        self.diagnostics["heralded_pairs"] += heralded
        self.fail_bsm_bin += recalled - heralded
        self.failures_bin += recalled - heralded
        successes = heralded if f_out >= float(self.args.f_min) else 0
        self.diagnostics["fail_quality"] += heralded - successes
        if f_out > .5:
            self.diagnostics["target_fidelity_above_half"] += successes
        self.failures_bin += heralded - successes
        if state_out is not None:
            for i, value in enumerate(state_out.probabilities):
                self.bell_sum[i] += successes * value
            if bool(getattr(self.args, "record_bell_batches", False)):
                self.bell_batches.append(dict(t_s=float(self.env.now) / PS_PER_S,
                    attempts=n, recalled=recalled, heralded=heralded, delivered=successes,
                    wait_sat_s=wait_left_s, wait_fiber_s=wait_right_s,
                    **dict(zip(("phi_plus", "psi_plus", "phi_minus", "psi_minus"), state_out.probabilities))))
        self.successes_bin += successes
        self.cumulative_successes += successes
        self.fidelity_sum_bin += successes * f_out
        self.success_wait_sum_bin += successes * 0.5 * (wait_left_s + wait_right_s)
        if successes > 0 and self.controller_client is not None and self.controller_backend != "des_only":
            self.controller_client.report_swap_result(
                req_id=GROUND_SDN_REQ_ID,
                success_count=int(successes),
            )
        self.log("SWAP_BATCH", attempts=n, recalled=recalled, successes=successes, fidelity=f_out)

    def metrics_tick(self) -> None:
        sat = self.link_states.get("S-R1")
        fiber = self.link_states.get("R1-G")
        successes = self.successes_bin
        mean_fid = (self.fidelity_sum_bin / successes if successes > 0
                    else (float("nan") if self.bell_bsm else 0.0))
        mean_wait_s = self.success_wait_sum_bin / successes if successes > 0 else 0.0
        self.metrics.append(
            MetricRow(
                t_s=float(self.env.now) / PS_PER_S,
                sat_visible=int(bool(sat and sat.visible)),
                sat_distance_km=float((sat.distance_m if sat else 0.0) / 1e3),
                sat_elevation_deg=float(sat.elevation_deg if sat else 0.0),
                sat_p_link=float(sat.p_link if sat else 0.0),
                sat_loss_db=float(sat.loss_db if sat else 150.0),
                sat_fidelity=float(sat.fidelity if sat else 0.25),
                fiber_distance_km=float((fiber.distance_m if fiber else 0.0) / 1e3),
                fiber_p_link=float(fiber.p_link if fiber else 0.0),
                fiber_loss_db=float(fiber.loss_db if fiber else 150.0),
                fiber_fidelity=float(fiber.fidelity if fiber else 0.25),
                successes=int(successes),
                failures=int(self.failures_bin),
                mean_fidelity_success=float(mean_fid),
                mean_wait_success_s=float(mean_wait_s),
                throughput_pairs_per_s=successes / max(1e-12, float(self.args.metrics_dt_s)),
                cumulative_successes=int(self.cumulative_successes),
                memory_sat=self.count_tokens("S-R1"),
                memory_fiber=self.count_tokens("R1-G"),
                fail_link=int(self.fail_link_bin),
                fail_recall=int(self.fail_recall_bin),
                fail_bsm=int(self.fail_bsm_bin),
                fail_storage_full=int(self.fail_storage_full_bin),
                **self.diagnostics,
                **dict(zip(("bell_phi_plus", "bell_psi_plus", "bell_phi_minus", "bell_psi_minus"),
                    [p / successes for p in self.bell_sum] if successes and self.bell_bsm
                    else [float("nan")] * 4)),
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
        self._reset_diagnostics()
        self._schedule_next("metrics_tick", float(self.args.metrics_dt_s))

    def _schedule_next(self, handler: str, delay_s: float) -> None:
        next_t = int(self.env.now) + int(max(0.0, delay_s) * PS_PER_S)
        if next_t <= int(float(self.args.duration_s) * PS_PER_S):
            self.scheduler.schedule_at(next_t, EventHandler(self, handler), priority=self.event_priority(handler))

    def count_tokens(self, label: str) -> int:
        return sum(batch.count for batch in self.tokens[label])

    def pop_tokens(self, label: str, count: int) -> List[TokenBatch]:
        out: List[TokenBatch] = []
        remaining = int(count)
        batches = self.tokens[label]
        while remaining > 0 and batches:
            batch = batches[0]
            take = min(remaining, batch.count)
            out.append(TokenBatch(batch.label, take, batch.ready_ps, batch.werner_w, batch.fidelity, batch.bell_state))
            batch.count -= take
            remaining -= take
            if batch.count <= 0:
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


def run_once(args: argparse.Namespace, *, controller_client=None) -> DesGroundRepeater:
    ogs, ground_end = ground_repeater_sites(args)
    args.effective_fiber_length_km = effective_fiber_length_km(args)
    sats = default_satellites(args.tle, args.omm_json)
    epoch_unix_s = float(args.epoch_unix_s) if args.epoch_unix_s is not None else representative_epoch_unix_s(sats)
    selection = select_satellite_for_ground_run(
        sats=sats,
        ogs=ogs,
        epoch_unix_s=epoch_unix_s,
        args=args,
    )
    sat = selection.satellite
    selected_sat = SatelliteEphemeris(
        "SAT_END",
        sat.tle_line1,
        sat.tle_line2,
        sat.source,
        {
            **(sat.metadata or {}),
            "original_name": sat.name,
            "pass_selection": selection.objective,
            "pass_candidates_considered": selection.candidates_considered,
            "contact_window_start_s": (
                float(selection.contact_window.start_ps) / PS_PER_S if selection.contact_window is not None else None
            ),
            "contact_window_end_s": (
                float(selection.contact_window.end_ps) / PS_PER_S if selection.contact_window is not None else None
            ),
            "contact_window_duration_s": (
                float(selection.contact_window.duration_ps) / PS_PER_S if selection.contact_window is not None else None
            ),
            "ogs_site": str(args.ogs_name),
            "ogs_lat_deg": float(args.ogs_lat_deg),
            "ogs_lon_deg": float(args.ogs_lon_deg),
            "ground_end_site": str(args.ground_end_name),
            "ground_end_lat_deg": float(args.ground_end_lat_deg),
            "ground_end_lon_deg": float(args.ground_end_lon_deg),
            "fiber_length_km_effective": float(args.effective_fiber_length_km),
        },
    )
    if bool(args.auto_start_at_visibility) and not bool(args.force_visible):
        epoch_unix_s += visibility_start_offset(selection=selection, args=args)

    propagator = SkyfieldPropagator(
        satellites=[selected_sat],
        ground_nodes=[ogs],
        epoch_unix_s=epoch_unix_s,
    )
    env = DESEnv(name="des_ground_repeater", default=True)
    sim = DesGroundRepeater(
        name="ground",
        env=env,
        args=args,
        geometry=GeometryEngine(propagator),
        fso=build_fso_model(args),
        fiber=build_fiber_model(args),
        memory=build_memory_model(args),
        bsm=build_bsm_model(args),
        controller_client=controller_client,
    )
    sim.selected_satellite_name = sat.name
    sim.selected_satellite_metadata = selected_sat.metadata
    env.init()
    env.run(end_time=int(float(args.duration_s) * PS_PER_S), logging=False, summary=True)
    return sim


def build_fiber_model(args: argparse.Namespace) -> FiberAnalyticalChannelModel:
    return FiberAnalyticalChannelModel(
        FiberParams(
            eta_src=float(args.fiber_eta_src),
            eta_det=float(args.fiber_eta_det),
            eta_cpl=float(args.fiber_eta_cpl),
            eta_conn=float(args.fiber_eta_conn),
            loss_db_per_km=float(args.fiber_loss_db_per_km),
            insertion_loss_db=float(args.fiber_insertion_loss_db),
            polarization_error_rad=float(args.fiber_polarization_error_rad),
            phase_noise_rad=float(args.fiber_phase_noise_rad),
            temporal_mismatch_s=float(args.fiber_temporal_mismatch_s),
            coherence_time_s=float(args.fiber_coherence_time_s),
            p_t=float(args.fiber_p_t),
            n0=float(args.fiber_n0),
            n_dc=float(args.fiber_n_dc),
            n_conv=float(args.fiber_n_conv),
            delta_t=float(args.fiber_delta_t),
            enable_length_fluctuations=bool(args.fiber_enable_length_fluctuations),
            length_jitter_fraction=float(args.fiber_length_jitter_fraction),
            max_snr_for_mapping=float(args.fiber_max_snr_for_mapping),
        )
    )


def write_outputs(sim: DesGroundRepeater, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(sim, "selected_satellite_name"):
        payload = {
            "selected_satellite_name": getattr(sim, "selected_satellite_name"),
            "selected_satellite_metadata": getattr(sim, "selected_satellite_metadata", None),
        }
        (out_dir / "selected_ground_pass.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    fields = list(MetricRow.__dataclass_fields__.keys())
    for filename, rows in (("ground_optical_trace.csv", sim.optical_trace),
                           ("ground_memory_trace.csv", sim.memory_trace)):
        if rows:
            with (out_dir / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    with (out_dir / "ground_repeater_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in sim.metrics:
            writer.writerow([getattr(row, field) for field in fields])
    if sim.bell_batches:
        with (out_dir / "ground_bell_batches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sim.bell_batches[0]))
            writer.writeheader()
            writer.writerows(sim.bell_batches)
    if not bool(sim.args.no_event_log):
        with (out_dir / "ground_repeater_event_log.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_ps", "kind", "payload"])
            writer.writerows(sim.event_log)
    plot_outputs(sim, out_dir)


def plot_outputs(sim: DesGroundRepeater, out_dir: Path) -> None:
    if bool(sim.args.no_plots) or not sim.metrics:
        return
    ts = [row.t_s for row in sim.metrics]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ts, [row.sat_p_link for row in sim.metrics], label="Satellite-OGS FSO", color="#1d4ed8")
    axes[0].plot(ts, [row.fiber_p_link for row in sim.metrics], label="OGS-Ground fiber", color="#0f766e")
    axes[0].set_ylabel("link success proxy")
    axes[0].set_title("Ground scenario link budget")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(ts, [row.sat_loss_db if row.sat_visible else float("nan") for row in sim.metrics], label="Satellite-OGS loss", color="#dc2626")
    axes[1].plot(ts, [row.fiber_loss_db for row in sim.metrics], label="Fiber loss", color="#7c3aed")
    axes[1].fill_between(ts, 0, 1, where=[row.sat_visible for row in sim.metrics], color="#bfdbfe", alpha=0.2, transform=axes[1].get_xaxis_transform(), label="satellite visible")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("loss (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ground_link_budget.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ts, [row.throughput_pairs_per_s for row in sim.metrics], color="#1d4ed8")
    axes[0].set_ylabel("pairs / s")
    axes[0].set_title("Ground scenario end-to-end performance")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ts, [row.mean_fidelity_success if row.successes else float("nan") for row in sim.metrics], color="#dc2626")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("fidelity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ground_throughput_fidelity.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ts, [row.memory_sat for row in sim.metrics], label="Satellite-side buffer", color="#1d4ed8")
    axes[0].plot(ts, [row.memory_fiber for row in sim.metrics], label="Fiber-side buffer", color="#0f766e")
    axes[0].set_ylabel("stored tokens")
    axes[0].set_title("Ground scenario memory occupancy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(ts, [row.sat_elevation_deg for row in sim.metrics], label="Satellite elevation", color="#f59e0b")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("elevation (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ground_memory_buffers.png")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ground repeater DES simulator with FSO + fiber + memory + BSM")
    parser.add_argument("--out", default="./des_ground_repeater_out")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--duration-s", type=float, default=4000.0)
    parser.add_argument("--metrics-dt-s", type=float, default=10.0)
    parser.add_argument("--link-update-dt-s", type=float, default=5.0)
    parser.add_argument("--gen-delay-s", type=float, default=0.2)
    parser.add_argument("--swap-dt-s", type=float, default=0.2)
    parser.add_argument("--source-rate-hz", type=float, default=1_000_000.0)
    parser.add_argument("--max-modes-per-label", type=int, default=256)
    parser.add_argument("--max-bsm-per-cycle", type=int, default=256)
    parser.add_argument("--ground-swap-policy", choices=["legacy_memory_rate", "per_cycle", "attempt_rate"],
                        default="legacy_memory_rate",
                        help="Historical modes-to-Hz pacing (replay only), cycle-budget diagnostic, or independent attempts/s")
    parser.add_argument("--bsm-attempt-rate-hz", type=float, default=None,
                        help="Required for attempt_rate: explicit operand-pair service rate before readout/BSM, separate from memory slots")
    parser.add_argument("--f-min", type=float, default=0.25)
    parser.add_argument("--memory-effective-wait-s", type=float, default=0.0)
    parser.add_argument("--min-elevation-deg", type=float, default=10.0)
    parser.add_argument("--force-visible", action="store_true")
    parser.add_argument("--ogs-name", default=WINDHOF_OGS_DEFAULT["name"])
    parser.add_argument("--ogs-lat-deg", type=float, default=WINDHOF_OGS_DEFAULT["lat_deg"])
    parser.add_argument("--ogs-lon-deg", type=float, default=WINDHOF_OGS_DEFAULT["lon_deg"])
    parser.add_argument("--ogs-alt-m", type=float, default=WINDHOF_OGS_DEFAULT["alt_m"])
    parser.add_argument("--ground-end-name", default=KIRCHBERG_END_DEFAULT["name"])
    parser.add_argument("--ground-end-lat-deg", type=float, default=KIRCHBERG_END_DEFAULT["lat_deg"])
    parser.add_argument("--ground-end-lon-deg", type=float, default=KIRCHBERG_END_DEFAULT["lon_deg"])
    parser.add_argument("--ground-end-alt-m", type=float, default=KIRCHBERG_END_DEFAULT["alt_m"])
    parser.add_argument("--auto-fiber-length-from-sites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tle", default=None)
    parser.add_argument("--omm-json", default=None)
    parser.add_argument("--pass-selection", choices=["memory_aware", "longest_dual", "highest_dual"], default="memory_aware")
    parser.add_argument("--pass-candidate-limit", type=int, default=64)
    parser.add_argument("--pass-prefilter-step-s", type=float, default=300.0)
    parser.add_argument("--epoch-unix-s", type=float, default=None)
    parser.add_argument("--auto-start-at-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visibility-scan-s", type=float, default=86400.0)
    parser.add_argument("--visibility-scan-step-s", type=float, default=10.0)
    parser.add_argument("--visibility-prebuffer-s", type=float, default=120.0)

    parser.add_argument("--fso-eta-src", type=float, default=1.0)
    parser.add_argument("--fso-eta-det", type=float, default=1.0)
    parser.add_argument("--fso-eta-cpl", type=float, default=0.8)
    parser.add_argument("--fso-eta-point", type=float, default=0.98)
    parser.add_argument("--fso-d-t", type=float, default=0.20)
    parser.add_argument("--fso-d-r", type=float, default=1.0)
    parser.add_argument("--fso-theta-div", type=float, default=3e-6)
    parser.add_argument("--fso-t-abs", type=float, default=0.95)
    parser.add_argument("--fso-t-sca", type=float, default=0.95)
    parser.add_argument("--fso-c-atm", type=float, default=0.05)
    parser.add_argument("--fso-sigma-r2", type=float, default=0.05)
    parser.add_argument("--fso-turbulence-loss-coeff", type=float, default=0.06)
    parser.add_argument("--fso-turbulence-quality-coeff", type=float, default=0.025)
    parser.add_argument("--fso-delta-t", type=float, default=1e-9)
    parser.add_argument("--fso-n0", type=float, default=1e-5)
    parser.add_argument("--fso-n-dc", type=float, default=100.0)
    parser.add_argument("--fso-n-conv", type=float, default=0.0)
    parser.add_argument("--fso-p-t", type=float, default=0.05)
    parser.add_argument("--fso-pointing-jitter-rad", type=float, default=1e-8)
    parser.add_argument("--disable-fso-pointing-jitter", action="store_true")
    parser.add_argument("--disable-fso-stochastic-fading", action="store_true")

    parser.add_argument("--fiber-length-km", type=float, default=5.0)
    parser.add_argument("--fiber-loss-db-per-km", type=float, default=0.18)
    parser.add_argument("--fiber-insertion-loss-db", type=float, default=0.5)
    parser.add_argument("--fiber-eta-src", type=float, default=0.98)
    parser.add_argument("--fiber-eta-det", type=float, default=0.92)
    parser.add_argument("--fiber-eta-cpl", type=float, default=0.96)
    parser.add_argument("--fiber-eta-conn", type=float, default=0.99)
    parser.add_argument("--fiber-polarization-error-rad", type=float, default=0.005)
    parser.add_argument("--fiber-phase-noise-rad", type=float, default=0.02)
    parser.add_argument("--fiber-temporal-mismatch-s", type=float, default=5e-12)
    parser.add_argument("--fiber-coherence-time-s", type=float, default=100e-12)
    parser.add_argument("--fiber-p-t", type=float, default=0.08)
    parser.add_argument("--fiber-n0", type=float, default=1e-6)
    parser.add_argument("--fiber-n-dc", type=float, default=10.0)
    parser.add_argument("--fiber-n-conv", type=float, default=0.0)
    parser.add_argument("--fiber-delta-t", type=float, default=1e-9)
    parser.add_argument("--fiber-max-snr-for-mapping", type=float, default=1e12)
    parser.add_argument("--fiber-enable-length-fluctuations", action="store_true")
    parser.add_argument("--fiber-length-jitter-fraction", type=float, default=0.0)

    parser.add_argument("--mem-platform", choices=["REID", "COLD_ATOM", "TRAPPED_ION"], default="REID")
    parser.add_argument("--mem-t2", type=float, default=100.0)
    parser.add_argument("--mem-d", type=float, default=7.0)
    parser.add_argument("--mem-t-mem", type=float, default=200.0)
    parser.add_argument("--mem-n", type=float, default=1.5)
    parser.add_argument("--mem-m", type=float, default=1.0)
    parser.add_argument("--mem-eta-mode0", type=float, default=0.9)
    parser.add_argument("--mem-eta-ti", type=float, default=0.3)

    parser.add_argument("--bsm-v-hom", type=float, default=0.97)
    parser.add_argument("--bsm-eta-det", type=float, default=0.8)
    parser.add_argument("--bsm-sigma-det-s", type=float, default=50e-12)
    parser.add_argument("--bsm-delta-t-s", type=float, default=200e-12)
    parser.add_argument("--bsm-p-dc", type=float, default=1e-6)
    parser.add_argument("--bsm-swap-model", choices=["legacy", "paper"], default="paper")
    parser.add_argument("--ground-state-model", choices=["legacy", "bell_diagonal"], default="legacy",
                        help="Opt-in Bell vectors with a relative-window effective BSM; legacy preserves historical runs")
    parser.add_argument("--ground-memory-noise", choices=["phase", "isotropic"], default="phase")
    parser.add_argument("--record-bell-batches", action="store_true")
    parser.add_argument("--ground-channel-mode", choices=["mean", "correlated"], default="mean")
    parser.add_argument("--ground-event-order", choices=["legacy", "phased"], default="legacy",
                        help="Phased orders equal-time events as link, generation, swap, metrics.")
    parser.add_argument("--refresh-fso-per-generation", action="store_true",
                        help="Fine-grid deterministic reference; correlated mode always refreshes at generation ticks")
    parser.add_argument("--fso-fading-correlation-s", type=float, default=1.)
    parser.add_argument("--fso-pointing-correlation-s", type=float, default=1.)
    parser.add_argument("--record-optical-trace", action="store_true")
    parser.add_argument("--record-memory-trace", action="store_true")
    parser.add_argument("--memory-quality-threshold", type=float, default=.5)
    parser.add_argument("--ground-controller-backend", choices=["des_only", "sdn_ryu"], default="des_only")
    parser.add_argument("--no-event-log", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_qnet_des("the Ground repeater workflow")
    sim = run_once(args)
    out_dir = Path(args.out)
    write_outputs(sim, out_dir)
    print(f"[ground-repeater] wrote outputs to: {out_dir.resolve()}")
    print(f"[ground-repeater] cumulative_successes={sim.cumulative_successes}")


if __name__ == "__main__":
    main()
