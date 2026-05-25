from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Free-Space Optical (FSO) channel physics
# ============================================================
# Purpose:
# - Provide analytical link budget + SNR → Werner → Fidelity mapping
# - Be usable both for static links and dynamic (mobility-driven) links
#
# Design:
# - Stateless model (pure functions of distance, elevation, params)
# - Separation between geometric loss and atmospheric loss
#
# Output:
# - p_link (success probability proxy)
# - Werner parameter W
# - Fidelity F


DEG2RAD = math.pi / 180.0


# ============================================================
# Core math helpers
# ============================================================
def werner_to_fidelity(w: float) -> float:
    w = max(0.0, min(1.0, float(w)))
    return (1.0 + 3.0 * w) / 4.0


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class FSOParams:
    # efficiencies
    eta_src: float = 0.9
    eta_det: float = 0.8
    eta_cpl: float = 0.7
    eta_point: float = 0.9

    # apertures / beam
    d_t: float = 0.08
    d_r: float = 0.5
    theta_div: float = 8e-6

    # atmosphere
    t_abs: float = 0.95
    t_sca: float = 0.95
    c_atm: float = 0.45
    sigma_r2: float = 0.2

    # noise / timing
    delta_t: float = 1e-9
    n0: float = 1e-3
    n_dc: float = 100.0
    n_conv: float = 0.0

    # brightness
    p_t: float = 0.05


# ============================================================
# Analytical FSO model
# ============================================================
class FSOAnalyticalModel:
    def __init__(self, params: FSOParams):
        self.p = params

    # ------------------------------------------------------------
    # Components
    # ------------------------------------------------------------
    def eta_geo(self, distance_m: float) -> float:
        w0 = self.p.d_t / 2.0
        w_l = math.sqrt(w0 * w0 + (self.p.theta_div * distance_m) ** 2)
        return max(
            0.0,
            min(1.0, 1.0 - math.exp(-(self.p.d_r**2) / (2.0 * w_l * w_l))),
        )

    def eta_atm(self, elevation_deg: float) -> float:
        sigma = max(0.0, float(self.p.sigma_r2))
        if elevation_deg >= 89.999:
            secz = 1.0
        else:
            secz = 1.0 / max(0.15, math.sin(max(1e-6, elevation_deg * DEG2RAD)))

        return max(
            0.0,
            min(
                1.0,
                self.p.t_abs
                * self.p.t_sca
                * math.exp(-self.p.c_atm * (sigma ** (12.0 / 5.0)) * secz),
            ),
        )

    def eta_total(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        include_atmosphere: bool = True,
    ) -> float:
        eta_atm = self.eta_atm(elevation_deg) if include_atmosphere else 1.0

        eta = (
            self.p.eta_src
            * self.eta_geo(distance_m)
            * eta_atm
            * self.p.eta_point
            * self.p.eta_cpl
            * self.p.eta_det
        )
        return max(0.0, min(1.0, eta))

    def snr(self, eta: float) -> float:
        signal = self.p.p_t * eta * self.p.eta_det
        noise = self.p.n0 + self.p.n_dc * self.p.delta_t + self.p.n_conv
        return max(0.0, signal / max(1e-12, noise))

    # ------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------
    def link_metrics(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
    ) -> Tuple[float, float, float]:
        """
        Returns:
            (p_link, werner_w, fidelity)
        """
        eta = self.eta_total(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            include_atmosphere=not intersatellite,
        )

        snr_val = self.snr(eta)
        w = snr_val / (1.0 + snr_val) if snr_val > 0 else 0.0
        w = max(0.0, min(1.0, w))

        p_link = eta
        f = werner_to_fidelity(w)

        return p_link, w, f


# ============================================================
# Utility wrapper
# ============================================================
class StaticFSOChannel:
    """
    Simple wrapper to match ChannelModel-like interface
    """

    def __init__(self, params: FSOParams, distance_m: float):
        self.model = FSOAnalyticalModel(params)
        self.distance_m = float(distance_m)

    def instantaneous_metrics(self) -> Tuple[float, float, float]:
        return self.model.link_metrics(
            distance_m=self.distance_m,
            elevation_deg=90.0,
            intersatellite=False,
        )

    def sample(self, rng) -> Tuple[bool, float, float]:
        p_link, w, f = self.instantaneous_metrics()
        ok = rng.random() < p_link
        return (ok, w, f) if ok else (False, 0.0, 0.25)


__all__ = [
    "FSOParams",
    "FSOAnalyticalModel",
    "StaticFSOChannel",
    "werner_to_fidelity",
]
