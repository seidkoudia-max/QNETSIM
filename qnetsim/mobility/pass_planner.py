from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from mobility.geometry import ContactWindow, GeometryEngine
from mobility.propagator import GroundStation, PS_PER_S, SkyfieldPropagator
from mobility.tle_loader import SatelliteEphemeris


@dataclass(frozen=True)
class PassPlannerConfig:
    scan_start_ps: int = 0
    scan_end_ps: int = 86_400 * PS_PER_S
    fine_step_ps: int = 10 * PS_PER_S
    coarse_step_ps: int = 300 * PS_PER_S
    candidate_limit: int = 64
    objective: str = "memory_aware"
    memory_coherence_s: float = 1.0


@dataclass(frozen=True)
class JointAccessWindow:
    satellite_name: str
    ground_a: str
    ground_b: str
    start_ps: int
    end_ps: int
    duration_ps: int
    overlap_duration_ps: int
    gap_ps: int
    peak_min_elevation_deg: float
    score: float
    window_a: ContactWindow
    window_b: ContactWindow

    @property
    def start_s(self) -> float:
        return float(self.start_ps) / PS_PER_S

    @property
    def end_s(self) -> float:
        return float(self.end_ps) / PS_PER_S


@dataclass(frozen=True)
class SatellitePassSelection:
    satellite: SatelliteEphemeris
    joint_window: Optional[JointAccessWindow]
    candidates_considered: int
    objective: str


def _window_overlap_ps(a: ContactWindow, b: ContactWindow) -> int:
    return max(0, min(int(a.end_ps), int(b.end_ps)) - max(int(a.start_ps), int(b.start_ps)))


def _window_gap_ps(a: ContactWindow, b: ContactWindow) -> int:
    if int(a.end_ps) < int(b.start_ps):
        return int(b.start_ps) - int(a.end_ps)
    if int(b.end_ps) < int(a.start_ps):
        return int(a.start_ps) - int(b.end_ps)
    return 0


def _memory_weight(gap_ps: int, memory_coherence_s: float) -> float:
    t2_s = max(1e-12, float(memory_coherence_s))
    return math.exp(-(float(gap_ps) / PS_PER_S) / t2_s)


def _pair_score(
    a: ContactWindow,
    b: ContactWindow,
    *,
    objective: str,
    memory_coherence_s: float,
) -> float:
    overlap_ps = _window_overlap_ps(a, b)
    if objective == "longest_dual":
        return float(overlap_ps)
    if objective == "highest_dual":
        return float(min(a.peak_elevation_deg, b.peak_elevation_deg))

    gap_ps = _window_gap_ps(a, b)
    bridge_ps = min(int(a.duration_ps), int(b.duration_ps))
    return float(overlap_ps) + _memory_weight(gap_ps, memory_coherence_s) * float(bridge_ps)


def _window_key(window: JointAccessWindow) -> Tuple[float, int, float, int]:
    return (
        float(window.score),
        int(window.overlap_duration_ps),
        float(window.peak_min_elevation_deg),
        -int(window.gap_ps),
    )


def _joint_window_for_satellite(
    *,
    geometry: GeometryEngine,
    sat_name: str,
    ground_a: GroundStation,
    ground_b: GroundStation,
    cfg: PassPlannerConfig,
) -> Optional[JointAccessWindow]:
    windows_a = geometry.contact_windows(
        ground_a.name,
        sat_name,
        start_ps=cfg.scan_start_ps,
        end_ps=cfg.scan_end_ps,
        step_ps=cfg.fine_step_ps,
    )
    windows_b = geometry.contact_windows(
        ground_b.name,
        sat_name,
        start_ps=cfg.scan_start_ps,
        end_ps=cfg.scan_end_ps,
        step_ps=cfg.fine_step_ps,
    )
    if not windows_a or not windows_b:
        return None

    best: Optional[JointAccessWindow] = None
    for win_a in windows_a:
        for win_b in windows_b:
            overlap_ps = _window_overlap_ps(win_a, win_b)
            gap_ps = _window_gap_ps(win_a, win_b)
            start_ps = min(int(win_a.start_ps), int(win_b.start_ps))
            end_ps = max(int(win_a.end_ps), int(win_b.end_ps))
            candidate = JointAccessWindow(
                satellite_name=sat_name,
                ground_a=ground_a.name,
                ground_b=ground_b.name,
                start_ps=start_ps,
                end_ps=end_ps,
                duration_ps=end_ps - start_ps,
                overlap_duration_ps=overlap_ps,
                gap_ps=gap_ps,
                peak_min_elevation_deg=min(
                    float(win_a.peak_elevation_deg), float(win_b.peak_elevation_deg)
                ),
                score=_pair_score(
                    win_a,
                    win_b,
                    objective=cfg.objective,
                    memory_coherence_s=cfg.memory_coherence_s,
                ),
                window_a=win_a,
                window_b=win_b,
            )
            if best is None or _window_key(candidate) > _window_key(best):
                best = candidate
    return best


def _coarse_candidate_names(
    *,
    propagator: SkyfieldPropagator,
    satellites: Sequence[SatelliteEphemeris],
    ground_a: GroundStation,
    ground_b: GroundStation,
    cfg: PassPlannerConfig,
) -> List[str]:
    limit = int(cfg.candidate_limit)
    if limit <= 0 or len(satellites) <= limit:
        return [sat.name for sat in satellites]

    scores: List[Tuple[float, float, str]] = []
    for sat in satellites:
        best_dual = -1e18
        best_any = -1e18
        t_ps = int(cfg.scan_start_ps)
        while t_ps <= int(cfg.scan_end_ps):
            elev_a = propagator.elevation_deg(sat.name, ground_a.name, t_ps)
            elev_b = propagator.elevation_deg(sat.name, ground_b.name, t_ps)
            best_dual = max(best_dual, min(elev_a, elev_b))
            best_any = max(best_any, max(elev_a, elev_b))
            t_ps += int(cfg.coarse_step_ps)
        scores.append((best_dual, best_any, sat.name))
    scores.sort(reverse=True)
    return [name for _dual, _any, name in scores[:limit]]


def rank_satellite_candidates(
    *,
    satellites: Iterable[SatelliteEphemeris],
    ground_a: GroundStation,
    ground_b: GroundStation,
    epoch_unix_s: float,
    cfg: Optional[PassPlannerConfig] = None,
) -> List[SatelliteEphemeris]:
    cfg = cfg or PassPlannerConfig()
    sat_list = list(satellites)
    if not sat_list:
        return []

    propagator = SkyfieldPropagator(
        satellites=sat_list,
        ground_nodes=[ground_a, ground_b],
        epoch_unix_s=epoch_unix_s,
    )
    candidate_names = _coarse_candidate_names(
        propagator=propagator,
        satellites=sat_list,
        ground_a=ground_a,
        ground_b=ground_b,
        cfg=cfg,
    )
    by_name = {sat.name: sat for sat in sat_list}
    return [by_name[name] for name in candidate_names if name in by_name]


def select_best_satellite_pass(
    *,
    satellites: Iterable[SatelliteEphemeris],
    ground_a: GroundStation,
    ground_b: GroundStation,
    epoch_unix_s: float,
    cfg: Optional[PassPlannerConfig] = None,
    sat_name: Optional[str] = None,
) -> SatellitePassSelection:
    cfg = cfg or PassPlannerConfig()
    sat_list = list(satellites)
    if not sat_list:
        raise ValueError("no satellites available for pass selection")

    by_name = {sat.name: sat for sat in sat_list}
    if sat_name is not None:
        if sat_name not in by_name:
            raise KeyError(f"unknown satellite requested for pass selection: {sat_name}")
        candidate_names = [sat_name]
    else:
        propagator = SkyfieldPropagator(
            satellites=sat_list,
            ground_nodes=[ground_a, ground_b],
            epoch_unix_s=epoch_unix_s,
        )
        candidate_names = _coarse_candidate_names(
            propagator=propagator,
            satellites=sat_list,
            ground_a=ground_a,
            ground_b=ground_b,
            cfg=cfg,
        )

    candidate_sats = [by_name[name] for name in candidate_names]
    propagator = SkyfieldPropagator(
        satellites=candidate_sats,
        ground_nodes=[ground_a, ground_b],
        epoch_unix_s=epoch_unix_s,
    )
    geometry = GeometryEngine(propagator)

    best_sat = candidate_sats[0]
    best_window: Optional[JointAccessWindow] = None
    for sat in candidate_sats:
        candidate = _joint_window_for_satellite(
            geometry=geometry,
            sat_name=sat.name,
            ground_a=ground_a,
            ground_b=ground_b,
            cfg=cfg,
        )
        if candidate is None:
            continue
        if best_window is None or _window_key(candidate) > _window_key(best_window):
            best_sat = sat
            best_window = candidate

    return SatellitePassSelection(
        satellite=best_sat,
        joint_window=best_window,
        candidates_considered=len(candidate_sats),
        objective=str(cfg.objective),
    )


__all__ = [
    "JointAccessWindow",
    "PassPlannerConfig",
    "SatellitePassSelection",
    "rank_satellite_candidates",
    "select_best_satellite_pass",
]
