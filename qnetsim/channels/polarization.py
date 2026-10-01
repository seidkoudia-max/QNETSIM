"""Lossless two-arm polarization evolution and sampled compensation.

This is a parametric sensitivity model, not an orbit-to-mirror calibration.
Angles are in degrees at the public interface; Jones matrices act on (H, V).
Scalar propagation/detection losses belong to the separate FSO link budget.
"""
from dataclasses import dataclass, field
import math

import numpy as np


BELL_STATES = {
    "phi_plus": np.array([1, 0, 0, 1], complex)/np.sqrt(2),
    "psi_minus": np.array([0, 1, -1, 0], complex)/np.sqrt(2),
}


@dataclass(frozen=True)
class ArmPolarization:
    rotation_deg: float = 0.
    rotation_rate_deg_s: float = 0.
    retardance_deg: float = 0.
    retardance_rate_deg_s: float = 0.
    wander_rms_deg: float = 0.


@dataclass(frozen=True)
class PolarizationParams:
    enabled: bool = False
    compensate: bool = False
    target: str = "phi_plus"
    reference_epoch_s: float = 1377.
    matera: ArmPolarization = field(default_factory=lambda: ArmPolarization(5., .04, 10., .01, .5))
    helmos: ArmPolarization = field(default_factory=lambda: ArmPolarization(-5., -.03, -5., -.01, .5))
    wander_correlation_s: float = 30.
    wander_grid_s: float = 1.
    update_interval_s: float = 10.
    latency_s: float = 1.
    estimation_rms_deg: float = .2

    def validate(self, end_s):
        if self.target not in BELL_STATES:
            raise ValueError("Unsupported polarization Bell target.")
        for name, lo, hi in [("reference_epoch_s", 0, 172800), ("wander_correlation_s", .01, 86400),
                             ("wander_grid_s", .001, 60), ("update_interval_s", .001, 3600),
                             ("latency_s", 0, 3600), ("estimation_rms_deg", 0, 180)]:
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Polarization {name} must be between {lo} and {hi}.")
        for arm in (self.matera, self.helmos):
            for name, value in vars(arm).items():
                bound = 10. if "rate" in name else 360.
                if not math.isfinite(value) or abs(value) > bound or (name == "wander_rms_deg" and value < 0):
                    raise ValueError(f"Invalid polarization {name}.")
        if self.enabled and end_s/self.wander_grid_s > 500000:
            raise ValueError("Polarization trace exceeds 500,000 samples; increase the wander grid interval.")
        if self.enabled and self.compensate and end_s/self.update_interval_s > 100000:
            raise ValueError("Too many polarization updates; increase the correction interval.")


def jones(rotation_deg, retardance_deg):
    """J = R(theta) D(delta), with a linear rotator and H/V retarder."""
    theta, delta = np.broadcast_arrays(np.deg2rad(rotation_deg), np.deg2rad(retardance_deg))
    c, s, p = np.cos(theta), np.sin(theta), np.exp(-.5j*delta)
    out = np.empty(theta.shape+(2, 2), complex)
    out[..., 0, 0], out[..., 0, 1] = c*p, -s*p.conj()
    out[..., 1, 0], out[..., 1, 1] = s*p, c*p.conj()
    return out


def source_density(fidelity, target="phi_plus"):
    """Isotropic Bell + white-noise family with the specified Bell fidelity."""
    if not .25 <= fidelity <= 1:
        raise ValueError("Source fidelity must be in [0.25, 1].")
    ket = BELL_STATES[target]
    weight = (4*fidelity-1)/3
    return weight*np.outer(ket, ket.conj())+(1-weight)*np.eye(4)/4


def propagate_density(rho, a, b):
    joint = np.kron(a, b)
    return joint @ rho @ joint.conj().T


def genuine_fidelity(fidelity, a, b, target="phi_plus"):
    """Vectorized Bell overlap, algebraically identical to density propagation."""
    ket = BELL_STATES[target].reshape(2, 2)
    evolved = a @ ket @ np.swapaxes(b, -1, -2)
    overlap = np.abs(np.sum(ket.conj()*evolved, axis=(-2, -1)))**2
    weight = (4*fidelity-1)/3
    return np.clip(weight*overlap+(1-weight)/4, 0., 1.)


def mixture_fidelity(genuine, true_count, accidental_count, accidental_fidelity=.25):
    """Fidelity to a pure target is linear in an incoherent mixture."""
    total = true_count+accidental_count
    return (genuine*true_count+accidental_fidelity*accidental_count)/total if total else float("nan")


class PolarizationTrace:
    """Causal sample-and-hold inverse Jones control, evaluated at reception.

    OU rotation wander is sampled on an epoch-anchored grid and linearly
    interpolated. Separate RNG streams keep counts and control errors isolated.
    Beacon estimates are available only inside supplied visibility intervals;
    acquisition is ideal at each interval start. No beacon photons are modeled.
    """
    def __init__(self, params, end_s, seed, wavelength, windows):
        params.validate(end_s)
        self.params = params
        self.grid = np.arange(math.ceil(end_s/params.wander_grid_s)+2)*params.wander_grid_s
        self.noise, self.updates, self.events = [], [], []
        for arm, spec in enumerate((params.matera, params.helmos)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, wavelength, 701, arm]))
            z = rng.normal(size=len(self.grid))
            alpha = math.exp(-params.wander_grid_s/params.wander_correlation_s)
            for i in range(1, len(z)):
                z[i] = alpha*z[i-1]+math.sqrt(1-alpha*alpha)*z[i]
            self.noise.append(z*spec.wander_rms_deg)
            applied, estimates = [], []
            errors = np.random.default_rng(np.random.SeedSequence([seed, wavelength, 702, arm]))
            if params.enabled and params.compensate:
                for begin, end in windows[arm]:
                    for measured in np.arange(begin, end, params.update_interval_s):
                        apply = float(measured+params.latency_s)
                        if apply >= end:
                            continue
                        theta, delta = self.angles(arm, measured)
                        estimate = jones(theta+errors.normal(0, params.estimation_rms_deg),
                                         delta+errors.normal(0, params.estimation_rms_deg))
                        applied.append(apply)
                        estimates.append(estimate.conj().T)
                        self.events.append(dict(arm=arm, measured_epoch_s=float(measured), applied_epoch_s=apply))
            self.updates.append((np.array(applied), np.array(estimates)))

    def angles(self, arm, epoch_s):
        spec = (self.params.matera, self.params.helmos)[arm]
        t = np.asarray(epoch_s)
        dt = t-self.params.reference_epoch_s
        return (spec.rotation_deg+spec.rotation_rate_deg_s*dt+np.interp(t, self.grid, self.noise[arm]),
                spec.retardance_deg+spec.retardance_rate_deg_s*dt)

    def matrix(self, arm, epoch_s):
        t = np.asarray(epoch_s)
        identity = np.broadcast_to(np.eye(2, dtype=complex), t.shape+(2, 2))
        if not self.params.enabled:
            return identity
        channel = jones(*self.angles(arm, t))
        applied, inverses = self.updates[arm]
        if not len(applied):
            return channel
        index = np.searchsorted(applied, t, side="right")-1
        correction = np.where((index >= 0)[..., None, None], inverses[np.maximum(0, index)], identity)
        return correction @ channel

    def fidelity(self, source_fidelity, reception_a_s, reception_b_s):
        if not self.params.enabled:
            return np.full(np.broadcast_shapes(np.shape(reception_a_s), np.shape(reception_b_s)), source_fidelity)
        return genuine_fidelity(source_fidelity, self.matrix(0, reception_a_s),
                                self.matrix(1, reception_b_s), self.params.target)
