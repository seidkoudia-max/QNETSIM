from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================
# Quantum memory models
# ============================================================
# Purpose:
# - Provide platform-specific write / recall / decoherence behavior
# - Keep memory physics separate from token storage and swapping logic
# - Offer a common interface for all memory technologies


class MemoryModelBase:
    def write_success_prob(self) -> float:
        raise NotImplementedError

    def p_succ(self, t_s: float) -> float:
        raise NotImplementedError

    def coherence_factor(self, t_s: float) -> float:
        raise NotImplementedError

    def conditional_entanglement_fidelity(self, t_s: float) -> float:
        lam = max(0.0, min(1.0, self.coherence_factor(t_s)))
        return 0.5 * (1.0 + lam)

    def conditional_recall_prob(self, t_s: float) -> float:
        p0 = max(0.0, self.p_succ(0.0))
        if p0 <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.p_succ(t_s) / p0))


# ============================================================
# REID memory
# ============================================================
@dataclass(frozen=True)
class REIDParams:
    d: float = 7.0
    t2_spin_dd_s: float = 3.0
    n: float = 1.5
    t_mem_s: float = 10.0
    m: float = 1.0


class REIDMemoryModel(MemoryModelBase):
    def __init__(self, params: REIDParams):
        self.p = params

    def _eta_crib(self) -> float:
        d = max(0.0, float(self.p.d))
        return (1.0 - math.exp(-d)) ** 2

    def _eta_dd(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        t_mem = max(1e-12, float(self.p.t_mem_s))
        m = max(0.1, float(self.p.m))
        return math.exp(-((t / t_mem) ** m))

    def _chi(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        t2 = max(1e-12, float(self.p.t2_spin_dd_s))
        n = max(0.5, float(self.p.n))
        return (t / t2) ** n

    def write_success_prob(self) -> float:
        return max(0.0, min(1.0, self._eta_crib() * self._eta_dd(0.0)))

    def p_succ(self, t_s: float) -> float:
        eta = self._eta_crib() * self._eta_dd(t_s)
        return max(0.0, min(1.0, eta * math.exp(-2.0 * self._chi(t_s))))

    def coherence_factor(self, t_s: float) -> float:
        return max(0.0, min(1.0, math.exp(-self._chi(t_s))))


# ============================================================
# Cold atom memory
# ============================================================
@dataclass(frozen=True)
class ColdAtomParams:
    d: float = 20.0
    t2_sw_s: float = 0.05
    n: float = 1.3
    eta_mode0: float = 0.9
    t_mode_s: float = 1.0
    m_mode: float = 1.0


class ColdAtomMemoryModel(MemoryModelBase):
    def __init__(self, params: ColdAtomParams):
        self.p = params

    def _eta_eit(self) -> float:
        d = max(0.0, float(self.p.d))
        return (1.0 - math.exp(-d)) ** 2

    def _eta_mode(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        eta0 = max(0.0, min(1.0, float(self.p.eta_mode0)))
        tmode = max(1e-12, float(self.p.t_mode_s))
        m = max(0.1, float(self.p.m_mode))
        return eta0 * math.exp(-((t / tmode) ** m))

    def _chi(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        t2 = max(1e-12, float(self.p.t2_sw_s))
        n = max(0.5, float(self.p.n))
        return (t / t2) ** n

    def write_success_prob(self) -> float:
        return max(0.0, min(1.0, self._eta_eit() * self._eta_mode(0.0)))

    def p_succ(self, t_s: float) -> float:
        eta = self._eta_eit() * self._eta_mode(t_s)
        return max(0.0, min(1.0, eta * math.exp(-2.0 * self._chi(t_s))))

    def coherence_factor(self, t_s: float) -> float:
        return max(0.0, min(1.0, math.exp(-self._chi(t_s))))


# ============================================================
# Trapped-ion memory
# ============================================================
@dataclass(frozen=True)
class TrappedIonParams:
    t2_ti_s: float = 10.0
    eta_ti: float = 0.3


class TrappedIonMemoryModel(MemoryModelBase):
    def __init__(self, params: TrappedIonParams):
        self.p = params

    def write_success_prob(self) -> float:
        return max(0.0, min(1.0, float(self.p.eta_ti)))

    def p_succ(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        t2 = max(1e-12, float(self.p.t2_ti_s))
        eta = max(0.0, min(1.0, float(self.p.eta_ti)))
        return max(0.0, min(1.0, eta * math.exp(-2.0 * t / t2)))

    def coherence_factor(self, t_s: float) -> float:
        t = max(0.0, float(t_s))
        t2 = max(1e-12, float(self.p.t2_ti_s))
        return max(0.0, min(1.0, math.exp(-t / t2)))


# ============================================================
# Factory
# ============================================================
def build_memory_model(platform: str) -> MemoryModelBase:
    plat = str(platform).upper().strip()
    if plat == "COLD_ATOM":
        return ColdAtomMemoryModel(ColdAtomParams())
    if plat == "TRAPPED_ION":
        return TrappedIonMemoryModel(TrappedIonParams())
    if plat == "REID":
        return REIDMemoryModel(REIDParams())
    raise ValueError(f"unknown memory platform: {platform}")


__all__ = [
    "MemoryModelBase",
    "REIDParams",
    "REIDMemoryModel",
    "ColdAtomParams",
    "ColdAtomMemoryModel",
    "TrappedIonParams",
    "TrappedIonMemoryModel",
    "build_memory_model",
]
