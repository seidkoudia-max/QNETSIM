"""Explicit source, detector, and conditional Bell-measurement contracts.

The BSM is a calibrated effective instrument with two effective coincidence
channels. It is not a simulation of a specific four-detector optical circuit.
Optical indistinguishability and electronic time tagging are separate inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from quantum.bell_diagonal import BellDiagonalState


@dataclass(frozen=True)
class SinglePairSource:
    pulse_rate_hz: float = 40000.
    emission_probability: float = .05
    fidelity: float = .99

    def __post_init__(self):
        if self.pulse_rate_hz <= 0 or not math.isfinite(self.pulse_rate_hz):
            raise ValueError("pulse rate must be finite and positive")
        if not 0 <= self.emission_probability <= 1 or not 0 <= self.fidelity <= 1:
            raise ValueError("invalid source probability or fidelity")

    @property
    def pair_rate_hz(self):
        return self.pulse_rate_hz*self.emission_probability

    @property
    def state(self):
        return BellDiagonalState.werner(self.fidelity)


def spdc_pair_probability(n: int, mean_pairs: float, modes: int = 1):
    """Negative-binomial pair statistics for independent thermal Schmidt modes.

This is a source diagnostic; the single-pair network executor does not silently
substitute it for a multiphoton state and threshold-detector simulation.
"""
    if n < 0 or int(n) != n or modes < 1 or int(modes) != modes or mean_pairs < 0:
        raise ValueError("invalid photon-pair statistics parameters")
    if mean_pairs == 0:
        return float(n==0)
    q=mean_pairs/(modes+mean_pairs)
    return math.exp(math.lgamma(n+modes)-math.lgamma(modes)-math.lgamma(n+1)
                    +modes*math.log1p(-q)+n*math.log(q))


@dataclass(frozen=True)
class ThresholdDetector:
    efficiency: float = .8
    dark_rate_hz: float = 100.
    jitter_sigma_s: float = 50e-12
    dead_time_s: float = 50e-9

    def __post_init__(self):
        if not 0 <= self.efficiency <= 1 or min(self.dark_rate_hz,self.jitter_sigma_s,self.dead_time_s)<0:
            raise ValueError("invalid detector parameters")

    def dark_probability(self, gate_s: float):
        if gate_s < 0:
            raise ValueError("gate duration must be nonnegative")
        return -math.expm1(-self.dark_rate_hz*gate_s)

    def click_probability(self, photon_number: int, gate_s: float):
        if photon_number < 0 or photon_number != int(photon_number):
            raise ValueError("photon number must be a nonnegative integer")
        return 1-(1-self.dark_probability(gate_s))*(1-self.efficiency)**photon_number

    def nonparalyzable_count_rate(self, incident_rate_hz: float):
        if incident_rate_hz < 0:
            raise ValueError("rate must be nonnegative")
        return incident_rate_hz/(1+incident_rate_hz*self.dead_time_s)


@dataclass(frozen=True)
class PhysicalBSMParams:
    phase_visibility: float = .95
    intrinsic_success: float = .5
    coincidence_window_s: float = 200e-12
    synchronization_sigma_s: float = 0.
    service_time_s: float = 1e-4
    noise_channel: str = "phase"

    def __post_init__(self):
        if not 0 <= self.phase_visibility <= 1 or not 0 <= self.intrinsic_success <= .5:
            raise ValueError("invalid BSM visibility or passive success ceiling")
        if self.coincidence_window_s <= 0 or self.service_time_s < 0 or self.synchronization_sigma_s < 0:
            raise ValueError("invalid BSM timing")
        if self.noise_channel not in {"phase","isotropic"}:
            raise ValueError("unknown BSM conditional noise channel")


class CoincidenceBSM:
    def __init__(self, params=PhysicalBSMParams(), detector=ThresholdDetector()):
        self.p=params
        self.detector=detector
        if params.service_time_s < detector.dead_time_s:
            raise ValueError("BSM service interval must cover detector recovery")

    def time_acceptance(self):
        # Difference of two independent detector tags, plus synchronization error.
        sigma=math.sqrt(2*self.detector.jitter_sigma_s**2+self.p.synchronization_sigma_s**2)
        return 1. if sigma==0 else math.erf(self.p.coincidence_window_s/(2*math.sqrt(2)*sigma))

    def probabilities(self, eta_left=1., eta_right=1.):
        if not 0 <= eta_left <= 1 or not 0 <= eta_right <= 1:
            raise ValueError("incoming efficiencies must lie in [0, 1]")
        s1=eta_left*self.detector.efficiency
        s2=eta_right*self.detector.efficiency
        d=self.detector.dark_probability(self.p.coincidence_window_s)
        true=self.p.intrinsic_success*s1*s2*self.time_acceptance()
        # Effective two-channel accidental model. Signal-dark and dark-dark
        # events are uniform within an already selected coincidence gate.
        false=self.p.intrinsic_success*(s1*(1-s2)*d+s2*(1-s1)*d+(1-s1)*(1-s2)*d*d)
        return true, false

    def output(self, left, right, eta_left=1., eta_right=1.):
        true,false=self.probabilities(eta_left,eta_right)
        total=true+false
        state=left.swap(right)
        state=(state.dephase(self.p.phase_visibility) if self.p.noise_channel=="phase"
               else state.depolarize(self.p.phase_visibility))
        # An accepted accidental carries no information about the Bell label.
        return total, state.depolarize(true/total) if total else BellDiagonalState.werner(.25)


def gaussian_mode_overlap(delay_s: float, intensity_sigma_s: float, frequency_offset_hz=0.):
    """Squared overlap of normalized Gaussian single-photon amplitudes.

intensity_sigma_s is the RMS width of |psi(t)|^2. Electronic detector jitter
does not enter this optical overlap.
"""
    if intensity_sigma_s <= 0:
        raise ValueError("wavepacket width must be positive")
    return math.exp(-delay_s**2/(4*intensity_sigma_s**2)
                    -(2*math.pi*frequency_offset_hz*intensity_sigma_s)**2)
