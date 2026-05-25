from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple


DEG2RAD = math.pi / 180.0


# ============================================================
# Helpers
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def werner_to_fidelity(w: float) -> float:
    w = clamp(w, 0.0, 1.0)
    return (1.0 + 3.0 * w) / 4.0


def fidelity_to_werner(f: float) -> float:
    f = clamp(f, 0.25, 1.0)
    return clamp((4.0 * f - 1.0) / 3.0, 0.0, 1.0)


def efficiency_to_loss_db(eta: float, *, floor_db: float = 150.0) -> float:
    eta = float(eta)
    if eta <= 0.0:
        return float(floor_db)
    return min(float(floor_db), -10.0 * math.log10(eta))


# ============================================================
# Simulator-facing interface
# ============================================================
class ChannelModelBase(Protocol):
    def sample_attempt(
        self,
        rng: random.Random,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
    ) -> Tuple[bool, float, float]:
        """
        Returns:
            (ok, werner_w, fidelity)
        """
        ...


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class FSOParams:
    # ---- source / optics / detection efficiencies
    eta_src: float = 0.90
    eta_det: float = 0.80
    eta_cpl: float = 0.80
    eta_point_static: float = 0.98

    # ---- transmitter / receiver geometry
    d_t: float = 0.20  # transmitter aperture diameter [m]
    d_r: float = 1.00  # receiver aperture diameter [m]
    theta_div: float = 3e-6  # beam divergence [rad]

    # ---- atmosphere / propagation
    wavelength_m: float = 810e-9
    t_abs: float = 0.95
    t_sca: float = 0.95
    c_atm: float = 0.05
    sigma_r2: float = 0.05  # Rytov variance proxy / turbulence strength
    ao_gain: float = 1.0  # multiplicative improvement on coupling (>=1 helps)
    turbulence_loss_coeff: float = 0.06
    turbulence_quality_coeff: float = 0.025

    # ---- stochastic fading / pointing
    enable_stochastic_fading: bool = True
    enable_pointing_jitter: bool = True
    pointing_jitter_rad: float = 1e-8
    turbulence_floor: float = 1e-4

    # ---- visibility / phase quality terms
    polarization_error_rad: float = 0.02
    phase_noise_rad: float = 0.10
    temporal_mismatch_s: float = 20e-12
    coherence_time_s: float = 100e-12

    # ---- brightness / noise / timing
    p_t: float = 0.05  # effective photon-pair brightness / signal scale
    n0: float = 1e-5  # background noise term
    n_dc: float = 100.0  # dark count rate [Hz]
    n_conv: float = 0.0  # converter/electronics noise proxy
    delta_t: float = 1e-9  # coincidence/integration window [s]

    # ---- simulation policy
    min_elevation_deg: float = 5.0
    max_snr_for_mapping: float = 1e12


# ============================================================
# Optional dynamic state passed per event
# ============================================================
@dataclass(frozen=True)
class FSOEventState:
    """
    Optional per-event dynamic state that can be provided by orbit/weather logic.
    Any field left as None falls back to the static FSOParams value.
    """

    sigma_r2: Optional[float] = None
    pointing_jitter_rad: Optional[float] = None
    ao_gain: Optional[float] = None
    polarization_error_rad: Optional[float] = None
    phase_noise_rad: Optional[float] = None
    temporal_mismatch_s: Optional[float] = None


# ============================================================
# Analytical + stochastic FSO model
# ============================================================
class FSOAnalyticalChannelModel:
    """
    Deliverable-aligned compact FSO channel model.

    Design intent:
    - Deterministic mean link budget:
        eta_total = eta_src * eta_geo * eta_atm * eta_point * eta_cpl * eta_det
    - Stochastic per-attempt variations:
        pointing jitter, turbulence fading, and quality/visibility fluctuations
    - Per-attempt output:
        (ok, W0, F0)

    Notes:
    - For intersatellite links, atmosphere is bypassed.
    - This is still a compact system-level model, not a full wave-optics propagator.
    """

    def __init__(self, params: FSOParams):
        self.p = params

    # ------------------------------------------------------------
    # Mean deterministic components
    # ------------------------------------------------------------
    def eta_geo(self, distance_m: float) -> float:
        distance_m = max(0.0, float(distance_m))
        w0 = self.p.d_t / 2.0
        w_l = math.sqrt(w0 * w0 + (self.p.theta_div * distance_m) ** 2)
        if w_l <= 0.0:
            return 0.0
        eta = 1.0 - math.exp(-(self.p.d_r**2) / (2.0 * w_l * w_l))
        return clamp(eta, 0.0, 1.0)

    def slant_factor(self, elevation_deg: float) -> float:
        if elevation_deg >= 89.999:
            return 1.0
        sin_el = math.sin(max(1e-6, elevation_deg * DEG2RAD))
        return 1.0 / max(0.15, sin_el)

    def eta_atm_mean(
        self,
        elevation_deg: float,
        sigma_r2: float,
    ) -> float:
        sigma_r2 = max(0.0, float(sigma_r2))
        secz = self.slant_factor(elevation_deg)
        eta = (
            self.p.t_abs
            * self.p.t_sca
            * math.exp(-self.p.c_atm * (sigma_r2 ** (12.0 / 5.0)) * secz)
        )
        return clamp(eta, 0.0, 1.0)

    def eta_turbulence_collection_mean(
        self,
        elevation_deg: float,
        sigma_r2: float,
    ) -> float:
        sigma_r2 = max(0.0, float(sigma_r2))
        secz = math.sqrt(self.slant_factor(elevation_deg))
        eta = math.exp(-self.p.turbulence_loss_coeff * sigma_r2 * secz)
        return clamp(eta, 0.0, 1.0)

    def turbulence_visibility_factor(
        self,
        elevation_deg: float,
        sigma_r2: float,
        *,
        intersatellite: bool,
    ) -> float:
        if intersatellite:
            return 1.0
        sigma_r2 = max(0.0, float(sigma_r2))
        secz = math.sqrt(self.slant_factor(elevation_deg))
        visibility = math.exp(-self.p.turbulence_quality_coeff * sigma_r2 * secz)
        return clamp(visibility, 0.0, 1.0)

    def eta_pointing_mean(self) -> float:
        return clamp(self.p.eta_point_static, 0.0, 1.0)

    def eta_coupling_mean(self, ao_gain: float) -> float:
        eta = self.p.eta_cpl * max(1.0, float(ao_gain))
        return clamp(eta, 0.0, 1.0)

    def eta_total_mean(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
    ) -> float:
        sigma_r2 = (
            self.p.sigma_r2
            if state is None or state.sigma_r2 is None
            else state.sigma_r2
        )
        ao_gain = (
            self.p.ao_gain if state is None or state.ao_gain is None else state.ao_gain
        )

        eta_atm = 1.0 if intersatellite else self.eta_atm_mean(elevation_deg, sigma_r2)
        eta_turb = (
            1.0
            if intersatellite
            else self.eta_turbulence_collection_mean(elevation_deg, sigma_r2)
        )

        eta = (
            self.p.eta_src
            * self.eta_geo(distance_m)
            * eta_atm
            * eta_turb
            * self.eta_pointing_mean()
            * self.eta_coupling_mean(ao_gain)
            * self.p.eta_det
        )
        return clamp(eta, 0.0, 1.0)

    def mean_efficiency(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
        apply_min_elevation: bool = True,
    ) -> float:
        if (
            apply_min_elevation
            and (not intersatellite)
            and elevation_deg < self.p.min_elevation_deg
        ):
            return 0.0
        return self.eta_total_mean(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            intersatellite=intersatellite,
            state=state,
        )

    def mean_loss_db(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
        apply_min_elevation: bool = True,
        floor_db: float = 150.0,
    ) -> float:
        eta = self.mean_efficiency(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            intersatellite=intersatellite,
            state=state,
            apply_min_elevation=apply_min_elevation,
        )
        return efficiency_to_loss_db(eta, floor_db=floor_db)

    # ------------------------------------------------------------
    # Stochastic factors sampled per attempt
    # ------------------------------------------------------------
    def sample_pointing_factor(
        self,
        rng: random.Random,
        *,
        distance_m: float,
        jitter_rad: float,
    ) -> float:
        """
        Compact Gaussian beam overlap penalty due to angular jitter.
        """
        if not self.p.enable_pointing_jitter:
            return 1.0

        jitter_rad = max(0.0, float(jitter_rad))
        if jitter_rad == 0.0 or distance_m <= 0.0:
            return 1.0

        w0 = self.p.d_t / 2.0
        w_l = math.sqrt(w0 * w0 + (self.p.theta_div * distance_m) ** 2)
        if w_l <= 0.0:
            return 0.0

        # Sample transverse beam-center displacement at receiver plane
        sigma_xy = jitter_rad * distance_m
        dx = rng.gauss(0.0, sigma_xy)
        dy = rng.gauss(0.0, sigma_xy)
        r2 = dx * dx + dy * dy

        # Gaussian overlap penalty
        penalty = math.exp(-r2 / (2.0 * w_l * w_l))
        return clamp(penalty, 0.0, 1.0)

    def sample_turbulence_factor(
        self,
        rng: random.Random,
        *,
        sigma_r2: float,
        intersatellite: bool,
    ) -> float:
        """
        Compact stochastic irradiance factor.
        Uses a log-normal style multiplicative fading proxy in weak/moderate turbulence.
        """
        if intersatellite or not self.p.enable_stochastic_fading:
            return 1.0

        sigma_r2 = max(0.0, float(sigma_r2))
        if sigma_r2 <= 0.0:
            return 1.0

        # Mean-1 lognormal approximation:
        # If X ~ N(-s^2/2, s^2), then exp(X) has mean 1.
        # Use a compact mapping from turbulence proxy to log-irradiance std.
        s = math.sqrt(max(self.p.turbulence_floor, min(2.0, 0.5 * sigma_r2)))
        x = rng.gauss(-0.5 * s * s, s)
        fade = math.exp(x)

        # Prevent pathologically huge boosts in compact system model
        return clamp(fade, self.p.turbulence_floor, 3.0)

    def sample_visibility_factor(
        self,
        *,
        state: Optional[FSOEventState] = None,
    ) -> float:
        pol_err = self.p.polarization_error_rad
        ph_err = self.p.phase_noise_rad
        dtau = self.p.temporal_mismatch_s

        if state is not None:
            if state.polarization_error_rad is not None:
                pol_err = state.polarization_error_rad
            if state.phase_noise_rad is not None:
                ph_err = state.phase_noise_rad
            if state.temporal_mismatch_s is not None:
                dtau = state.temporal_mismatch_s

        # Compact HOM / indistinguishability penalties
        v_pol = math.cos(pol_err) ** 2
        v_phase = math.exp(-(ph_err**2))
        ratio = dtau / max(1e-15, self.p.coherence_time_s)
        v_temp = math.exp(-(ratio**2))

        return clamp(v_pol * v_phase * v_temp, 0.0, 1.0)

    # ------------------------------------------------------------
    # SNR and state-quality mapping
    # ------------------------------------------------------------
    def snr_from_eta(self, eta_total: float) -> float:
        """
        eta_total already includes detector efficiency, so do not multiply eta_det again.
        """
        signal = self.p.p_t * eta_total
        noise = self.p.n0 + self.p.n_dc * self.p.delta_t + self.p.n_conv
        snr = signal / max(1e-15, noise)
        return clamp(snr, 0.0, self.p.max_snr_for_mapping)

    def werner_from_snr_and_visibility(self, snr: float, visibility: float) -> float:
        """
        Compact system-level mapping.
        - SNR controls mixture with noise
        - visibility degrades indistinguishability / interference quality
        """
        w_snr = snr / (1.0 + snr) if snr > 0.0 else 0.0
        w = w_snr * clamp(visibility, 0.0, 1.0)
        return clamp(w, 0.0, 1.0)

    # ------------------------------------------------------------
    # Main per-event evaluation
    # ------------------------------------------------------------
    def link_metrics(
        self,
        *,
        rng: random.Random,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
    ) -> Tuple[float, float, float]:
        """
        Returns:
            (p_link, werner_w, fidelity)

        p_link is the instantaneous success-probability proxy for this event.
        """
        if (not intersatellite) and elevation_deg < self.p.min_elevation_deg:
            return 0.0, 0.0, 0.25

        sigma_r2 = (
            self.p.sigma_r2
            if state is None or state.sigma_r2 is None
            else state.sigma_r2
        )
        jitter_rad = (
            self.p.pointing_jitter_rad
            if state is None or state.pointing_jitter_rad is None
            else state.pointing_jitter_rad
        )

        eta_mean = self.mean_efficiency(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            intersatellite=intersatellite,
            state=state,
        )

        point_factor = self.sample_pointing_factor(
            rng,
            distance_m=distance_m,
            jitter_rad=jitter_rad,
        )
        turb_factor = self.sample_turbulence_factor(
            rng,
            sigma_r2=sigma_r2,
            intersatellite=intersatellite,
        )

        eta_inst = clamp(eta_mean * point_factor * turb_factor, 0.0, 1.0)

        snr = self.snr_from_eta(eta_inst)
        visibility = self.sample_visibility_factor(
            state=state
        ) * self.turbulence_visibility_factor(
            elevation_deg,
            sigma_r2,
            intersatellite=intersatellite,
        )
        w = self.werner_from_snr_and_visibility(snr, visibility)
        f = werner_to_fidelity(w)

        return eta_inst, w, f

    def mean_link_metrics(
        self,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
    ) -> Tuple[float, float, float]:
        """
        Mean link estimate for routing/control and batched source scheduling.

        This intentionally excludes per-attempt stochastic fading and pointing
        draws. Use sample_attempt/link_metrics for individual event samples.
        """
        if (not intersatellite) and elevation_deg < self.p.min_elevation_deg:
            return 0.0, 0.0, 0.25

        sigma_r2 = (
            self.p.sigma_r2
            if state is None or state.sigma_r2 is None
            else state.sigma_r2
        )

        eta_mean = self.eta_total_mean(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            intersatellite=intersatellite,
            state=state,
        )
        snr = self.snr_from_eta(eta_mean)
        visibility = self.sample_visibility_factor(
            state=state
        ) * self.turbulence_visibility_factor(
            elevation_deg,
            sigma_r2,
            intersatellite=intersatellite,
        )
        w = self.werner_from_snr_and_visibility(snr, visibility)
        f = werner_to_fidelity(w)
        return eta_mean, w, f

    def sample_attempt(
        self,
        rng: random.Random,
        *,
        distance_m: float,
        elevation_deg: float,
        intersatellite: bool,
        state: Optional[FSOEventState] = None,
    ) -> Tuple[bool, float, float]:
        """
        Simulator-facing stochastic attempt interface:
            (ok, W0, F0)
        """
        p_link, w, f = self.link_metrics(
            rng=rng,
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            intersatellite=intersatellite,
            state=state,
        )

        ok = rng.random() < p_link
        return (ok, w, f) if ok else (False, 0.0, 0.25)


# ============================================================
# Static wrapper for fixed links
# ============================================================
class StaticFSOChannel:
    """
    Convenience wrapper for fixed geometry links.
    """

    def __init__(
        self,
        params: FSOParams,
        distance_m: float,
        elevation_deg: float = 90.0,
        intersatellite: bool = False,
    ):
        self.model = FSOAnalyticalChannelModel(params)
        self.distance_m = float(distance_m)
        self.elevation_deg = float(elevation_deg)
        self.intersatellite = bool(intersatellite)

    def instantaneous_metrics(
        self,
        rng: random.Random,
        state: Optional[FSOEventState] = None,
    ) -> Tuple[float, float, float]:
        return self.model.link_metrics(
            rng=rng,
            distance_m=self.distance_m,
            elevation_deg=self.elevation_deg,
            intersatellite=self.intersatellite,
            state=state,
        )

    def sample(
        self,
        rng: random.Random,
        state: Optional[FSOEventState] = None,
    ) -> Tuple[bool, float, float]:
        return self.model.sample_attempt(
            rng,
            distance_m=self.distance_m,
            elevation_deg=self.elevation_deg,
            intersatellite=self.intersatellite,
            state=state,
        )


FSOAnalyticalModel = FSOAnalyticalChannelModel


__all__ = [
    "FSOParams",
    "FSOEventState",
    "FSOAnalyticalModel",
    "FSOAnalyticalChannelModel",
    "StaticFSOChannel",
    "ChannelModelBase",
    "werner_to_fidelity",
    "fidelity_to_werner",
]
