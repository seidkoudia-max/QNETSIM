"""Flagged erasure plus conditional phase noise, with independent calibration.

Efficiencies apply to one physical endpoint. Write is sampled at admission;
read is sampled exactly once at destructive readout. Population/mode survival
and logical phase coherence are separately specified curves.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicalMemoryParams:
    platform: str
    eta_write: float
    eta_read: float
    t2_s: float
    dephasing_exponent: float = 1.
    lifetime_s: float = math.inf
    loss_exponent: float = 1.
    # Optional fitted collective-mode loss. Zero means independent erasure.
    # Never infer this coefficient from T2 alone.
    collective_loss_coefficient: float = 0.
    noise: str = "dephasing"

    def __post_init__(self):
        for name in ("eta_write", "eta_read"):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("t2_s", "lifetime_s", "dephasing_exponent", "loss_exponent"):
            value=getattr(self,name)
            if math.isnan(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.collective_loss_coefficient < 0 or not math.isfinite(self.collective_loss_coefficient):
            raise ValueError("collective loss coefficient must be finite and nonnegative")
        if self.noise not in {"dephasing", "depolarizing"}:
            raise ValueError("unknown conditional noise channel")


class ErasureDephasingMemory:
    def __init__(self, params: PhysicalMemoryParams):
        self.p = params

    def chi(self, age_s: float):
        return (max(0.,age_s)/self.p.t2_s)**self.p.dephasing_exponent

    def coherence_factor(self, age_s: float):
        return math.exp(-self.chi(age_s))

    def incremental_coherence(self, old_age_s: float, new_age_s: float):
        if new_age_s < old_age_s:
            raise ValueError("memory age cannot decrease")
        return math.exp(-(self.chi(new_age_s)-self.chi(old_age_s)))

    def survival(self, age_s: float):
        age_s=max(0.,age_s)
        exponent=(age_s/self.p.lifetime_s)**self.p.loss_exponent
        exponent+=self.p.collective_loss_coefficient*self.chi(age_s)
        return math.exp(-exponent)

    def write_success_prob(self):
        return self.p.eta_write

    def conditional_recall_prob(self, age_s: float):
        return self.p.eta_read*self.survival(age_s)

    def p_succ(self, age_s: float):
        return self.p.eta_write*self.conditional_recall_prob(age_s)

    def conditional_entanglement_fidelity(self, age_s: float):
        c=self.coherence_factor(age_s)
        return (1+c)/2 if self.p.noise=="dephasing" else (1+3*c)/4

    def evolve_pair(self, state, old_ages, new_ages):
        c=math.prod(self.incremental_coherence(a,b) for a,b in zip(old_ages,new_ages))
        return state.dephase(c) if self.p.noise=="dephasing" else state.depolarize(c)

    def kraus(self, age_s: float):
        """2-dimensional input to qubit-plus-erasure output; dephasing model."""
        import numpy as np
        if self.p.noise != "dephasing":
            raise ValueError("explicit Kraus helper implements the dephasing variant")
        s=self.p_succ(age_s);q=(1-self.coherence_factor(age_s))/2
        embed=np.array([[1,0],[0,1],[0,0]],complex)
        return [math.sqrt(s*(1-q))*embed,
                math.sqrt(s*q)*embed@np.diag([1,-1]),
                math.sqrt(1-s)*np.array([[0,0],[0,0],[1,0]],complex),
                math.sqrt(1-s)*np.array([[0,0],[0,0],[0,1]],complex)]


def crib_backward_efficiency(optical_depth: float):
    """Ideal backward CRIB write-read efficiency, not universal EIT efficiency."""
    if optical_depth < 0:
        raise ValueError("optical depth must be nonnegative")
    return (-math.expm1(-optical_depth))**2


def afc_efficiency(optical_depth: float, finesse: float, *, backward=True,
                   control_transfer=1., background_depth=0.):
    """Gaussian-comb approximation (Afzelius et al., PRA 79, 052329).

Two control pulses each have probability control_transfer. Finesse is FWHM
spacing/linewidth. The Gaussian-comb effective depth includes sqrt(pi/4ln2).
"""
    if optical_depth < 0 or finesse <= 0 or background_depth < 0 or not 0 <= control_transfer <= 1:
        raise ValueError("invalid AFC parameters")
    d=optical_depth/finesse*math.sqrt(math.pi/(4*math.log(2)))
    recall=(-math.expm1(-d))**2 if backward else d*d*math.exp(-d)
    return recall*math.exp(-math.pi**2/(2*math.log(2)*finesse*finesse)-background_depth)*control_transfer**2


def cold_atom_motion_chi(age_s: float, *, delta_k_m_inv: float, velocity_rms_m_s=0.,
                         diffusion_m2_s=0., detuning_rms_rad_s=0., homogeneous_rate_s_inv=0.):
    """Ballistic or diffusive motion plus quasi-static detuning; rates in SI.

Use either ballistic or diffusive motion for the same physical mechanism,
unless independent contributions are explicitly justified.
"""
    if min(age_s,delta_k_m_inv,velocity_rms_m_s,diffusion_m2_s,detuning_rms_rad_s,homogeneous_rate_s_inv)<0:
        raise ValueError("parameters must be nonnegative")
    if velocity_rms_m_s and diffusion_m2_s:
        raise ValueError("choose ballistic or diffusive motion, not both")
    return (.5*(delta_k_m_inv*velocity_rms_m_s*age_s)**2
            +diffusion_m2_s*delta_k_m_inv**2*age_s
            +.5*(detuning_rms_rad_s*age_s)**2+homogeneous_rate_s_inv*age_s)
