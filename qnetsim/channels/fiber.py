from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

from channels.fso import (
    clamp,
    efficiency_to_loss_db,
    werner_to_fidelity,
)


@dataclass(frozen=True)
class FiberParams:
    # Source / coupling / detection chain
    eta_src: float = 0.98
    eta_det: float = 0.92
    eta_cpl: float = 0.96
    eta_conn: float = 0.99

    # Fiber attenuation and fixed insertion loss
    loss_db_per_km: float = 0.18
    insertion_loss_db: float = 0.50

    # Interference / indistinguishability quality
    polarization_error_rad: float = 0.005
    phase_noise_rad: float = 0.02
    temporal_mismatch_s: float = 5e-12
    coherence_time_s: float = 100e-12

    # Brightness / noise / timing
    p_t: float = 0.08
    n0: float = 1e-6
    n_dc: float = 10.0
    n_conv: float = 0.0
    delta_t: float = 1e-9

    # Compact stochasticity
    enable_length_fluctuations: bool = False
    length_jitter_fraction: float = 0.0

    # Mapping guard
    max_snr_for_mapping: float = 1e12


class FiberAnalyticalChannelModel:
    """
    Compact analytical fiber channel complementary to the repository FSO model.

    The intent is to preserve the same downstream abstractions:
    - mean_link_metrics -> (effective_success_proxy, Werner weight, fidelity)
    - link_metrics -> stochastic per-event variant of the same tuple
    """

    def __init__(self, params: FiberParams):
        self.p = params

    def transmission_efficiency(self, length_km: float) -> float:
        loss_db = max(0.0, float(length_km)) * max(0.0, float(self.p.loss_db_per_km))
        return 10.0 ** (-loss_db / 10.0)

    def fixed_efficiency(self) -> float:
        return (
            clamp(self.p.eta_src, 0.0, 1.0)
            * clamp(self.p.eta_cpl, 0.0, 1.0)
            * clamp(self.p.eta_conn, 0.0, 1.0)
            * clamp(self.p.eta_det, 0.0, 1.0)
        )

    def insertion_efficiency(self) -> float:
        return 10.0 ** (-max(0.0, float(self.p.insertion_loss_db)) / 10.0)

    def mean_efficiency(self, *, length_km: float) -> float:
        eta = (
            self.fixed_efficiency()
            * self.insertion_efficiency()
            * self.transmission_efficiency(length_km)
        )
        return clamp(eta, 0.0, 1.0)

    def mean_loss_db(self, *, length_km: float, floor_db: float = 150.0) -> float:
        return efficiency_to_loss_db(self.mean_efficiency(length_km=length_km), floor_db=floor_db)

    def visibility_factor(self) -> float:
        v_pol = math.cos(max(0.0, float(self.p.polarization_error_rad))) ** 2
        v_phase = math.exp(-(max(0.0, float(self.p.phase_noise_rad)) ** 2))
        ratio = max(0.0, float(self.p.temporal_mismatch_s)) / max(1e-15, float(self.p.coherence_time_s))
        v_temp = math.exp(-(ratio**2))
        return clamp(v_pol * v_phase * v_temp, 0.0, 1.0)

    def snr_from_eta(self, eta_total: float) -> float:
        signal = max(0.0, float(self.p.p_t)) * max(0.0, float(eta_total))
        noise = max(1e-15, float(self.p.n0) + float(self.p.n_dc) * float(self.p.delta_t) + float(self.p.n_conv))
        return clamp(signal / noise, 0.0, float(self.p.max_snr_for_mapping))

    def werner_from_snr_and_visibility(self, snr: float, visibility: float) -> float:
        w_snr = snr / (1.0 + snr) if snr > 0.0 else 0.0
        return clamp(w_snr * clamp(visibility, 0.0, 1.0), 0.0, 1.0)

    def _sample_length_km(self, rng: random.Random, length_km: float) -> float:
        base = max(0.0, float(length_km))
        if not self.p.enable_length_fluctuations:
            return base
        sigma = max(0.0, float(self.p.length_jitter_fraction)) * base
        if sigma <= 0.0:
            return base
        return max(0.0, rng.gauss(base, sigma))

    def mean_link_metrics(self, *, length_km: float) -> Tuple[float, float, float]:
        eta = self.mean_efficiency(length_km=length_km)
        w = self.werner_from_snr_and_visibility(self.snr_from_eta(eta), self.visibility_factor())
        return eta, w, werner_to_fidelity(w)

    def link_metrics(self, rng: random.Random, *, length_km: float) -> Tuple[float, float, float]:
        sampled_length_km = self._sample_length_km(rng, length_km)
        eta = self.mean_efficiency(length_km=sampled_length_km)
        w = self.werner_from_snr_and_visibility(self.snr_from_eta(eta), self.visibility_factor())
        return eta, w, werner_to_fidelity(w)

    def sample_attempt(self, rng: random.Random, *, length_km: float) -> Tuple[bool, float, float]:
        p_link, w, f = self.link_metrics(rng, length_km=length_km)
        ok = rng.random() < p_link
        return (ok, w, f) if ok else (False, 0.0, 0.25)


__all__ = [
    "FiberParams",
    "FiberAnalyticalChannelModel",
]
