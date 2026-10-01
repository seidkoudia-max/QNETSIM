"""Exact Bell-diagonal operations for Pauli noise and corrected Bell projections.

Order: Phi+, Psi+, Phi-, Psi- (Pauli labels I, X, Z, XZ). These operations
preserve anisotropic noise; converting to one Werner parameter is unnecessary.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BellDiagonalState:
    probabilities: tuple[float, float, float, float] = (1., 0., 0., 0.)

    def __post_init__(self):
        p = self.probabilities
        if len(p) != 4 or any(not math.isfinite(x) or x < -1e-12 for x in p):
            raise ValueError("Bell probabilities must be four finite nonnegative values")
        if not math.isclose(sum(p), 1., abs_tol=1e-10):
            raise ValueError("Bell probabilities must sum to one")

    @classmethod
    def werner(cls, fidelity: float):
        if not 0 <= fidelity <= 1:
            raise ValueError("fidelity must lie in [0, 1]")
        other = (1-fidelity)/3
        return cls((fidelity, other, other, other))

    @property
    def fidelity(self):
        return self.probabilities[0]

    def dephase(self, coherence: float):
        if not 0 <= coherence <= 1:
            raise ValueError("coherence must lie in [0, 1]")
        q = (1-coherence)/2
        p = self.probabilities
        return BellDiagonalState(tuple((1-q)*p[i]+q*p[i ^ 2] for i in range(4)))

    def depolarize(self, weight: float):
        if not 0 <= weight <= 1:
            raise ValueError("weight must lie in [0, 1]")
        return BellDiagonalState(tuple(weight*p+(1-weight)/4 for p in self.probabilities))

    def pauli(self, errors: tuple[float, float, float, float]):
        return self.swap(BellDiagonalState(errors))

    def swap(self, other: "BellDiagonalState"):
        # Bell measurement with its known Pauli-frame correction: XOR convolution.
        a, b = self.probabilities, other.probabilities
        return BellDiagonalState(tuple(sum(a[i]*b[i ^ k] for i in range(4)) for k in range(4)))

    def twirl(self):
        """Optional bilateral randomization; a physical operation, not compression."""
        return self.werner(self.fidelity)

    @property
    def teleportation_fidelity(self):
        """Standard ideal teleportation protocol targeting Phi+, Haar-averaged input."""
        return (2*self.fidelity+1)/3

    @property
    def chsh_max(self):
        """Ideal optimized two-qubit CHSH value; not a loophole-free test prediction."""
        a,b,c,d = self.probabilities
        correlations = (a+b-c-d, -a+b+c-d, a-b+c-d)
        squares = sorted(x*x for x in correlations)
        return 2*math.sqrt(sum(squares[-2:]))

    def density_matrix(self):
        import numpy as np
        vectors = np.array([[1,0,0,1],[0,1,1,0],[1,0,0,-1],[0,1,-1,0]],float)/math.sqrt(2)
        return sum(p*np.outer(v,v) for p,v in zip(self.probabilities,vectors))
