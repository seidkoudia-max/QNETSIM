# QUASAR polarization and coincidence fidelity

## Status and assumptions

This is an optional, parametric sensitivity model, not a calibrated prediction
of the Matera/Helmos mission. The workbook supplies a source fidelity but not a
Bell-state identity, density matrix, attitude history, telescope Jones matrices,
polarization drift spectrum or compensation specifications. Polarization is off
by default; the existing three applications are unchanged.

The on/off control applies only to polarization. HV/Rytov values remain
diagnostics: adding polarization does not add a calibrated turbulence fading
model. A drift angle is not inferred from Cn2 or Rytov variance.

## Source and propagation

The default fixed target is Phi+ = (HH + VV)/sqrt(2); Psi- is also selectable.
With only F0 known, assume an isotropic Bell-plus-white-noise source:

    rho0 = w |B><B| + (1-w) I4/4,    w = (4 F0 - 1)/3.

This choice has <B|rho0|B> = F0. A single fidelity value does not uniquely specify
a density matrix, so this state-family assumption is explicit.

Each arm has a lossless Jones matrix in the H/V basis:

    J_i(t) = R(theta_i(t)) D(delta_i(t)),
    R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]],
    D(delta) = diag(exp(-i delta/2), exp(+i delta/2)).

Rotation = offset + drift_rate * (t - reference_time) + OU wander. Retardance =
offset + drift_rate * (t - reference_time). The OU rotation process has the
specified stationary RMS and exponential correlation at grid samples. Linear
interpolation between grid samples is a numerical approximation, not an exact
continuous OU realization. The grid is anchored to workbook epoch zero.
Separate seeded streams isolate wander, estimator errors and DES photon counts.
No chromatic drift law is presumed; parameters can differ at 810 and 1550 nm.

Compensation estimates J_i at measurement time t_k, with independent Gaussian
errors on rotation and retardance. At t_k + latency it applies inverse(J_est),
held until the next update. Estimator updates are logged in the DES event queue.
Beacon acquisition is assumed ideal at each station's bin-resolved visibility
opening; updates stop at closure. No correction is available before the first
estimate arrives. The previous setting can remain held during a visibility gap.
There is no beacon photon simulation, Jones-estimation algorithm, mirror train,
actuator dynamics, insertion loss or polarization-dependent loss in this model.

    U_i(t) = inverse(J_est_i) J_i(t), or J_i(t) without compensation.
    rho_g(t) = (U_M(t_M) tensor U_H(t_H)) rho0 (U_M tensor U_H)^dagger.
    F_g(t) = <B|rho_g(t)|B>.

Reception times t_M and t_H include each photon's flight delay. Photon DES
evaluates the overlap for each genuine source pair at its individual reception
times. Batched DES freezes polarization at each bin midpoint, consistent with
the frozen-bin link calculation. Use smaller batches and wander grids to test
convergence, especially near correction updates. A pure local unitary does not
destroy entanglement: reduced overlap to a fixed Bell state can be a correctable
basis mismatch, not irreversible depolarization.

## How polarization enters F_mix

Let C be accepted genuine coincidences and A be accidental coincidences over
the SAME interval (or use rates consistently). With independent unpolarized
accidentals and polarization-independent detection:

    rho_acc = I4/4,
    rho_mix = (C rho_g + A rho_acc)/(C+A),
    F_mix = <B|rho_mix|B> = (C F_g + A/4)/(C+A).

This is the squared-fidelity convention for a pure target, not its square root.
The weighting is a consequence of density-matrix mixing and the linearity of
<B|rho|B>, not an empirical QNETSIM fit or a universal atmospheric-fidelity law.
F_g replaces F0 inside the numerator. Do not multiply the old F_mix by F_g,
apply a second source-fidelity penalty, or attenuate fidelity by eta_M eta_H.
Scalar losses already enter C and A through the DES detection probabilities.

For the isotropic source and local unitary channels, each photon marginal is
I2/2. Independent photons from different source pairs therefore give I4/4.
Unpolarized sky/dark contributions use the same assumption. Polarized sky,
polarization-dependent loss, biased analyzer selection or correlated noise
require a different rho_acc and, potentially, a state-dependent count model.
All-pairs time-tag matching can reuse detections: this is an effective ensemble
of coincidence records, not a claim that every accidental is a physical pair.
The detector allocator remains an aggregate count approximation with uniform
channels, not a polarization-analyzer/tomography Monte Carlo.

For a pass use (sum_j C_j F_g,j + sum_j A_j/4) / sum_j(C_j+A_j).
Empty coincidence intervals have undefined F_mix. F_g may still be computed as
a hypothetical conditional channel overlap, but plots mask no-visibility bins.

## Illustrative sensitivity settings

Reference time: 1377 s after workbook epoch. Matera: rotation +5 deg, slope
+0.04 deg/s, retardance +10 deg, slope +0.01 deg/s. Helmos: rotation -5 deg,
slope -0.03 deg/s, retardance -5 deg, slope -0.01 deg/s. Each arm: 0.5 deg RMS
rotation wander, 30 s correlation, 1 s trace grid. Compensation: 10 s update,
1 s latency, 0.2 deg RMS estimation error on each angle. These values are NOT
from the workbook or from the cited experiments. Identical parameter settings
at both wavelengths do not establish identical real optical behavior.

The default 1e9 pairs/s and 1 ns coincidence gate remain unchanged. Under the
CW Poisson approximation, other-pair accidentals are comparable to genuine
coincidences. Compensation can recover F_g toward 0.99 but cannot remove that
accidental floor; F_mix can therefore remain near 0.62 even with good alignment.

## Literature basis

1. Saggio and Walther, *Few-Copy Entanglement Detection in the Presence of
   Noise*, Annalen der Physik 534, 2100597 (2022), Sec. 3, Eq. (6), describes
   white-noise density-matrix admixtures. Our count-weighted F_mix follows from
   this modeling choice and pure-target overlap; that paper does not calibrate
   QUASAR's accidentals or use our C/A notation.
   https://doi.org/10.1002/andp.202100597
2. Han et al., *Polarization design for ground-to-satellite quantum entanglement
   distribution*, describes polarization maintenance and motion compensation.
   Its experimental performance is not assigned to this simulation.
   https://arxiv.org/abs/1912.06318
3. *Real-time polarization control for satellite QKD with liquid-crystal beacon
   stabilization*, Scientific Reports (2026), uses independent Jones matrices
   for the two arms and models finite polarimetric accuracy. Our two-arm
   sampled inverse controller is a simplified sensitivity model, not a
   reproduction of that experimental instrument.
   https://doi.org/10.1038/s41598-026-55812-2
4. *Quantum and Classical Noise in Practical Quantum Cryptography Systems
   Based on Polarization-Entangled Photons*, Physical Review A (2003), treats
   accidentals in practical SPDC-based systems. Actual pulsed-source statistics
   should replace our CW approximation if the workbook MHz value is a clock.
   https://www.nist.gov/publications/quantum-and-classical-noise-practical-quantum-cryptography-systems-based-polarization
