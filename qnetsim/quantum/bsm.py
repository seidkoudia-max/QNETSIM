from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Bell-State Measurement (BSM) model
# ============================================================
# Purpose:
# - Model linear-optics BSM with HOM interference, detector efficiency,
#   timing jitter, and dark counts.
# - Provide both success probability and swapped-fidelity mapping.
#
# Notes:
# - This is a compact, analytical proxy suitable for DES simulations.
# - The success probability follows a standard 50% linear-optics ceiling,
#   modulated by efficiencies and timing window.


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class BSMParams:
    # Hong–Ou–Mandel visibility
    v_hom: float = 0.95

    # Spatial / spectral mode-overlap proxy
    eta_mode: float = 1.0

    # Polarization indistinguishability proxy
    eta_pol: float = 1.0

    # Detector efficiency
    eta_det: float = 0.8

    # Timing jitter (std dev) [s]
    sigma_det_s: float = 50e-12

    # Coincidence window [s]
    delta_t_s: float = 200e-12

    # Dark count probability per window
    p_dc: float = 1e-6

    # Swap fidelity law:
    # - "legacy": historical compact proxy already used in the repo
    # - "paper": v_hom-weighted Werner-state swapping law
    swap_model: str = "legacy"


# ============================================================
# Model
# ============================================================
class BellStateMeasurementModel:
    def __init__(self, params: BSMParams):
        self.p = params

    def effective_visibility(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(self.p.v_hom) * float(self.p.eta_mode) * float(self.p.eta_pol),
            ),
        )

    # ------------------------------------------------------------
    # Timing acceptance
    # ------------------------------------------------------------
    def eta_time(self) -> float:
        """
        Probability that two photons fall within the coincidence window
        given Gaussian detector jitter.
        """
        sigma = max(1e-15, float(self.p.sigma_det_s))
        dt = max(1e-15, float(self.p.delta_t_s))
        x = dt / (2.0 * math.sqrt(2.0) * sigma)
        return max(0.0, min(1.0, math.erf(x)))

    # ------------------------------------------------------------
    # Success probability
    # ------------------------------------------------------------
    def p_succ_bsm(self, eta1: float, eta2: float) -> float:
        """
        Probability of a successful BSM given incoming photon efficiencies.

        eta1, eta2: effective transmission/detection efficiencies of the
                     two incoming modes (typically set to 1.0 if already
                     accounted for upstream).
        """
        eta_det_eff = max(0.0, min(1.0, float(self.p.eta_det) * self.eta_time()))
        p_dc = max(0.0, min(1.0, float(self.p.p_dc)))
        eta1 = max(0.0, min(1.0, float(eta1)))
        eta2 = max(0.0, min(1.0, float(eta2)))

        visibility = self.effective_visibility()
        return max(
            0.0,
            min(
                1.0,
                0.5 * eta1 * eta2 * visibility * (eta_det_eff**2) * (1.0 - p_dc),
            ),
        )

    # ------------------------------------------------------------
    # Fidelity transformation
    # ------------------------------------------------------------
    def swapped_fidelity(self, f01: float, f02: float) -> float:
        """
        Fidelity of the swapped pair after BSM.

        Uses HOM visibility to degrade ideal swapping.
        """
        v = self.effective_visibility()
        f01 = max(0.25, min(1.0, float(f01)))
        f02 = max(0.25, min(1.0, float(f02)))
        model = str(getattr(self.p, "swap_model", "legacy")).strip().lower()

        if model == "paper":
            ideal = f01 * f02 + ((1.0 - f01) * (1.0 - f02)) / 3.0
            return max(0.25, min(1.0, v * ideal + (1.0 - v) * 0.25))

        return max(0.25, min(1.0, 0.5 * (1.0 + v * (2.0 * f01 * f02 - 1.0))))

    # ------------------------------------------------------------
    # Convenience combined evaluation
    # ------------------------------------------------------------
    def attempt_swap(
        self,
        *,
        f_left: float,
        f_right: float,
        eta1: float = 1.0,
        eta2: float = 1.0,
        rng,
        f_min: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Perform a stochastic BSM attempt.

        Returns:
            (success, fidelity_out)
        """
        p = self.p_succ_bsm(eta1, eta2)
        if rng.random() >= p:
            return False, 0.25

        f_out = self.swapped_fidelity(f_left, f_right)
        if f_out < f_min:
            return False, 0.25

        return True, f_out


__all__ = [
    "BSMParams",
    "BellStateMeasurementModel",
]
