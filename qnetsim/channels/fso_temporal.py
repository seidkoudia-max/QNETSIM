"""Correlated block-fading extension of the repository's compact FSO model.

Gaussian latent processes have exponential correlation exp(-|dt|/tau). Their
lognormal and pointing transforms retain the existing FSO parameterization.
The correlation times are scenario assumptions, not derived from Rytov variance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from channels.fso import FSOAnalyticalChannelModel, clamp, werner_to_fidelity


@dataclass(frozen=True)
class TemporalFSOParams:
    fading_correlation_s: float = 1.
    pointing_correlation_s: float = 1.

    def __post_init__(self):
        for value in (self.fading_correlation_s, self.pointing_correlation_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("FSO correlation times must be finite and positive")


@dataclass(frozen=True)
class TemporalFSOSample:
    efficiency: float
    reference_efficiency: float
    werner_w: float
    fidelity: float
    fading_factor: float
    pointing_factor: float
    pointing_x_rad: float
    pointing_y_rad: float
    fading_clipped: bool
    efficiency_clipped: bool


class TemporalFSOChannel:
    def __init__(self, model: FSOAnalyticalChannelModel, *, seed: int, params=TemporalFSOParams()):
        self.model, self.params = model, params
        self.rng = random.Random(f"qnetsim:fso-temporal:{seed}")
        self.last_time_s = None
        self.latent = None

    def advance(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("channel time must be finite and nonnegative")
        if self.last_time_s is not None and time_s < self.last_time_s:
            raise ValueError("channel time cannot go backwards")
        if self.last_time_s == time_s:
            return self.latent
        if self.last_time_s is None:
            self.latent = tuple(self.rng.gauss(0., 1.) for _ in range(3))
        else:
            dt = time_s-self.last_time_s
            taus = (self.params.fading_correlation_s, self.params.pointing_correlation_s,
                    self.params.pointing_correlation_s)
            self.latent = tuple(math.exp(-dt/tau)*old + math.sqrt(-math.expm1(-2*dt/tau))*self.rng.gauss(0.,1.)
                                for old, tau in zip(self.latent, taus))
        self.last_time_s = time_s
        return self.latent

    def sample(self, *, time_s, distance_m, elevation_deg):
        z, x, y = self.advance(time_s)
        model, p = self.model, self.model.p
        reference = model.eta_total_mean(distance_m=distance_m, elevation_deg=elevation_deg,
                                         intersatellite=False)
        fade, fade_clipped = 1., False
        if p.enable_stochastic_fading and p.sigma_r2 > 0:
            variance = max(p.turbulence_floor, min(2., .5*p.sigma_r2))
            raw_fade = math.exp(-.5*variance + math.sqrt(variance)*z)
            fade = clamp(raw_fade, p.turbulence_floor, 3.)
            fade_clipped = fade != raw_fade
        jitter = max(0., p.pointing_jitter_rad) if p.enable_pointing_jitter else 0.
        dx, dy = jitter*x, jitter*y
        beam_radius_sq = (p.d_t/2)**2 + (p.theta_div*distance_m)**2
        pointing = math.exp(-distance_m**2*(dx*dx+dy*dy)/(2*beam_radius_sq)) if beam_radius_sq > 0 else 1.
        raw_eta = reference*fade*pointing
        eta = clamp(raw_eta, 0., 1.)
        visibility = model.sample_visibility_factor() * model.turbulence_visibility_factor(
            elevation_deg, p.sigma_r2, intersatellite=False)
        w = model.werner_from_snr_and_visibility(model.snr_from_eta(eta), visibility)
        return TemporalFSOSample(eta, reference, w, werner_to_fidelity(w), fade, pointing,
                                 dx, dy, fade_clipped, eta != raw_eta)
