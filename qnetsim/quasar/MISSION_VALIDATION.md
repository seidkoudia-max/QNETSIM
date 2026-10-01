# QUASAR mission-fidelity validation

## Present status

The simulations are assumption-based studies, not validated mission predictions.
The user confirmed: **1000 MHz is the emitted-pair rate**, i.e. 1e9 pairs/s.
This resolves the rate interpretation only. It does not establish CW versus
pulsed operation, photon-number statistics, source calibration uncertainty, or
the fraction of emitted pairs inside the receiver passband. Do not change this
rate to force a higher fidelity.

`mission_evidence.json` records the clarification against the original workbook
hash and C21 value. Each run exports `mission_validation.json` and
`MISSION_VALIDATION_STATUS.md`; overrides or a changed workbook do not inherit
confirmation automatically. There is no checkbox that certifies mission validity.

## What the mission team should provide

Provide the following separately for Matera/Helmos and 810/1550 nm where relevant,
including units, uncertainty, conditions, instrument/configuration IDs and data
provenance. Laboratory or representative-system measurements can support
pre-flight validation; a future mission overpass need not already have occurred.

| Area | Evidence needed |
| --- | --- |
| Source | CW/pulsed operation; temporal and pair-number statistics at 1e9 pairs/s (including pulse rate/width if pulsed); source spectrum and in-band brightness; Bell target/density matrix or analyzer counts, with uncertainty; separate wavelength configurations or non-degenerate pairs. |
| Optical receiver | Wavelength-resolved TX/RX/filter throughput, PDE, aperture/coupling mode, field-of-view convention, equivalent passband and source fraction passed. |
| Detector/timing | Dark and background rates per detector, timing histograms including tails, jitter, dead time, afterpulsing and saturation; synchronization residuals; gate definition and actual matching/reuse rules. |
| Channel | Loss and background measurements versus elevation; extinction/weather/sky conditions; beam divergence and pointing convention; turbulence/aperture-averaged fading and coupling/AO calibration. |
| Polarization | Source/telescope Jones or Mueller calibration and reference frames; attitude/tracking dependence; compensation residuals, latency, bandwidth, insertion loss and acquisition behavior. |
| Orbit | Authoritative ephemeris or element frame/epoch/mean-osculating interpretation and intended passes, including uncertainty and visibility/acquisition criteria. |
| End-to-end reference | Representative paired observations of singles, coincidence time-difference histograms, accidental estimates and raw Bell fidelity or basis-resolved tomography across the intended loss/noise/polarization range. Reserve independent data for validation. |

For a reference time series, include absolute UTC or a declared relative epoch,
interval duration (s), station, wavelength (nm), operating-point/calibration ID,
per-detector singles counts, coincidence counts, accidental-estimation method,
full gate (s), both arm losses (dB), fidelity estimator/target/basis conventions,
reported fidelity uncertainty and its confidence level. Raw time tags and
analyzer settings are preferable where available. Do not synthesize missing
observations or fill unavailable uncertainty with zero.

## Validation procedure

1. Define the intended prediction and operating envelope: raw coincidence-mixture
   fidelity versus genuine-pair or accidental-subtracted fidelity; Bell target;
   wavelength, loss, elevation, background and acquisition conditions. These
   observables are not interchangeable.
2. Agree accuracy and uncertainty-coverage criteria with the mission team before
   fitting or comparing results. There is no justified universal tolerance for
   this scenario; none is invented here.
3. Calibrate source, detector, channel and polarization models on an identified
   calibration subset. Replace unsupported model assumptions, including CW
   statistics, all-pairs selection and I/4 accidentals if measurements contradict
   them. Turbulence is currently diagnostic-only and requires implementation and
   verification of the applicable fading/coupling model, or an evidence-based
   negligible-effect bound. The drift toggle is not that implementation.
4. Verify equations and code independently where possible: optical/noise units,
   loss limits, density-matrix physicality, identity/perfect-compensation limits,
   event matching and dead time. Compare batch approximations against timestamp
   DES over representative operating points; test time-step and seed convergence.
   Shared-code comparisons check consistency, not independent physical accuracy.
5. Compare held-out reference observations and predictions at matched settings
   using the same estimator and postselection rules. Check singles and timing
   histograms as well as fidelity, so compensating errors cannot hide in one
   fitted number. Inspect residuals versus loss, rate, elevation and noise.
6. Propagate measured input uncertainties and correlations; distinguish Monte
   Carlo sampling, parameter uncertainty and model discrepancy. Report prediction
   intervals and their assessed coverage. A seed spread alone is not a mission
   uncertainty band. Unconstrained weather scenarios are sensitivity cases, not
   calibrated confidence intervals.
7. Publish the dataset/configuration/model versions, comparisons, acceptance
   results, uncertainty budget, applicable envelope and exclusions. Independent
   technical review should approve the claim; a simulation completing or a unit
   test passing is not certification. Extrapolations beyond the validated envelope
   must remain explicitly qualified.

## Interpretation of the current fidelity

For CW Poisson emission and approximate all-pairs coincidence processing,
`A_other/C ~ R*gate/q`, with timing acceptance q. At 1e9 pairs/s and a 1 ns gate,
q near one gives F_mix near 0.62 for source fidelity 0.99 and I/4 accidentals.
This may be a physically relevant operating-point limitation, not a plotting
error. Its applicability depends on source statistics and the receiver's actual
selection rules. Background-limited operation can instead yield fidelity that
improves with transmission. Do not tune the model to an expected curve shape.

## References and scope

The verification/empirical-validation distinction and credibility reporting
follow the general approach described by [NASA's modeling and simulation
definitions](https://public.ksc.nasa.gov/mns/ms-definitions/) and
[NASA-STD-7009B](https://standards.nasa.gov/standard/nasa/nasa-std-7009).
These references are guidance, not a claim of NASA compliance or certification.

Source-dependent accidental modeling is discussed by
[Castelletto et al., Physical Review A 67 (2003)](https://www.nist.gov/publications/quantum-and-classical-noise-practical-quantum-cryptography-systems-based-polarization).
The implemented density-mixture assumptions and polarization equations are
documented separately in `POLARIZATION_MODEL.md`. Literature support for an
equation is not validation of this mission's parameter choices.
