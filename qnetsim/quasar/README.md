# QNETSIM QUASAR

Separate DES-only GUI for a satellite source with direct optical downlinks to
Matera and Helmos. The existing Single Repeater, STBD and Ground applications
are unchanged. No Ryu, SDN, memory or repeater BSM is used in this scenario.

## Launch

From the QNETSIM directory on this machine:

```sh
cd QNETSIM/qnetsim
../.venv39/bin/python -m streamlit run quasar_gui.py --server.port 8504
```

Open http://localhost:8504. The original workbook values are preloaded. An
updated copy of the same Excel template can be loaded in the input audit panel.
Parameter formulas are rejected, not evaluated. Review receiver assumptions,
tick the acknowledgment, and select **Run QUASAR - DES only**.

For a different installation, create a virtual environment and install
`quasar/requirements.txt`, then run `python -m streamlit run quasar_gui.py` from
the repository root. This extension currently uses the native framework under
`framework/core/des.py` and the isolated orbit/link
adapter under `studies/quasar_matera_helmos/`; keep these directories present.
This is not yet a new standalone-bundle export or a GitHub update.

## Mission Validation Status

The original workbook's 1000 MHz has been confirmed by the user as **1e9 emitted
pairs/s**, not a pump clock. The clarification is hash/value-bound in
`mission_evidence.json`; it does not confirm CW/pulsed statistics, in-band
brightness or a replacement workbook. The physical calculation is unchanged.

All new plots and the GUI label the results **assumption-based, not
mission-validated**. Each export includes machine-readable
`mission_validation.json`, a run-specific `MISSION_VALIDATION_STATUS.md`, and
`MISSION_VALIDATION.md` describing required calibration/reference data and a
validation procedure. There is no automatic promotion to validated status and
no fabricated measurement or uncertainty band. Numerical verification is
tracked separately from empirical accuracy. Tests can be run with:

```sh
python -m unittest quasar.test_polarization quasar.test_des quasar.test_fidelity quasar.test_gui quasar.test_mission_validation quasar.test_detectors quasar.test_timing_study
```

## Two DES Resolutions

**Batched DES - full pass** uses scheduled source intervals, delayed receiver
batch arrivals and coincidence processing through QNETSIM's native event queue.
Correlated Poisson categories retain the shared source-pair contribution.
Nonparalyzable detector live fractions and accidental counts are approximated
at batch level. There are no individual photon timestamps in this mode.
The default scans the first orbit and runs a 1,200 s interval around shared
visibility, with 1 s batches at the workbook's listed source brightness.

**Timestamp-resolved DES - short interval** samples detectable source categories
and schedules individual photon arrivals. Undetectable photons are removed by
Poisson thinning, without reducing source brightness. Per-detector busy state
persists across reporting bins. Noisy time tags with ideal flight-delay
compensation are matched after reception using an inclusive full-width,
all-pairs coincidence gate. The default interval is 100 microseconds near pass
center, with 10 microsecond reporting bins. At the listed 1e9 pairs/s, a full
pass of explicit detections is not practical; an event cap rejects oversized
runs instead of silently substituting the batch approximation.

Rates are **counts / elapsed bin duration**, including a partial last bin.
Pairs per event are separately labeled. Coincidence fidelity is a conditional
mixture model: `(F_genuine(t) * true + 0.25 * accidental) / total`. Optional
Jones-state propagation determines F_genuine; off uses F_source. This is not
simulated analyzer tomography or a secure key rate. No detections means an
undefined fidelity, not zero.

## Optional Timing / Brightness Study

In the **Timing and brightness study** section, tick **Coincidence-window /
source-brightness study**, edit the windows in ps and emitted-pair rates in MHz,
then select **Run timing / brightness study**. This is a separate button and
does not alter or rerun the main simulation. It runs every grid point through
the selected DES backend, holding detector, geometry, noise and polarization
settings fixed. Results include side-by-side fidelity/rate plots per wavelength,
a trade-off plot, all-row CSV and a complete audit ZIP. Each rate is averaged
over the common shared-visibility interval; fidelity is coincidence-count
weighted. Default axes have five values each, and the current settings are
also simulated as a baseline. See `TIMING_STUDY.md` for details and limitations.

## Conditional Physical Model

Circular two-body orbit from the supplied semi-major axis, epoch and elements;
WGS84 stations; optional MOD-frame or osculating J2 interpretation. The nominal
560 km statement does not override the supplied semi-major axis. Both ends
must meet the 20-degree elevation mask for true coincidences.

The link uses equivalent Gaussian annular-aperture collection, pointing jitter
averaging, elevation-dependent atmospheric extinction, and separately entered
TX/RX/detector efficiencies. Sky counts follow spectral radiance times clear
collecting area, solid angle, equivalent filter bandwidth and detection
efficiency, divided by photon energy. Dark counts and detector dead time are
included. Geometry and efficiencies are frozen per bin; photon flight delay is
interpolated between bin boundaries.

The workbook lacks receiver efficiency, sky radiance, passband, field of view,
detector count, dead time and timing specifications. The GUI now offers
**Thorlabs SPDMH2 at 810 nm** and **Single Quantum SNSPD specification bounds at
1550 nm**, with wavelength-specific timing, noise and efficiency. These are
sensitivity presets, not unit calibration; Single Quantum does not guarantee
all headline specifications simultaneously. See `DETECTOR_PRESETS.md` for
sources, assumptions, FWHM-to-RMS conversion and limits. Legacy settings remain
selectable; `RunConfig()` and CLI retain legacy defaults. Each wavelength is an independent
equal-wavelength dual-downlink configuration, not an 810/1550 non-degenerate
pair. Source photons are assumed to lie inside the filter passband.

**Cn2 and wind currently produce HV/Rytov diagnostics only.** Turbulence fades,
AO and fiber coupling are not calibrated from these inputs. Extra per-arm loss
can be varied as an explicit sensitivity, not as a measured turbulence model.
The exports list all assumptions and questions to return to the reviewer.

## Reproduce and Test

```sh
python quasar_des_simulator.py --workbook "/path/to/QUASAR simulation parameters.xlsx"
python quasar_des_simulator.py --mode photons --start 1377 --duration 0.0001 --bin 0.00001
python quasar_des_simulator.py --detectors vendor
python -m unittest quasar.test_des -v
python -m unittest quasar.test_gui -v
```

Each run creates a new directory under `Output/quasar_des_runs/`, containing
geometry, component losses, optical power, signal/noise metrics, event logs,
eight PNG figures, configuration, workbook audit and hash, model source hashes,
reviewer questions, detector provenance/overrides, and a ZIP download. The old expected-value results in
`Output/quasar_matera_helmos/` are not overwritten or relabeled as DES outputs.

## Understanding Fidelity

The two wavelengths have separate full-scale panels and a shared-scale,
shared-visibility zoom. Noise-only coincidences outside shared visibility are
shown in gray in the full view, not treated as successful link fidelity. The
export includes `FIDELITY_CHECK.md` and `fidelity_diagnostics.json`, separating
true pairs, matches between different source pairs, and sky/dark accidentals.

For CW Poisson emission, the approximate background-free all-pairs ratio is
`A_other / C = R * gate / q`, where `q = erf(gate / (4 * jitter))`.
The leading loss and detector live-fraction factors cancel. At 1e9 pairs/s,
a 1 ns window, and q near 1, this gives comparable true and accidental counts,
and mixture fidelity near `(0.99 + 0.25) / 2 = 0.62` at both wavelengths.
This explains a nearly flat overlap despite changing throughput. DES counts
still fluctuate; full-pass bins contain enough events that the relative
fluctuations are small. A flat result is not proof of model validity, either:
calibrated polarization dynamics and source/timing details require additional input.
The plots do not introduce artificial noise or change brightness to raise F.

## Optional Polarization

Use **Enable polarization drift** to turn on the new physical-layer module in
`channels/polarization.py`. **Enable polarization compensation** applies causal
sample-and-hold inverse Jones corrections at each receiver, including latency
and noisy angle estimates. Per-wavelength, per-station rotation/retardance
offsets and slopes, rotation wander, correlation time and correction settings
are editable. Defaults are an explicitly uncalibrated sensitivity example.
The default source family is an isotropic Phi+ Bell state with the workbook's
source fidelity; Psi- is also available. This source-state assumption is needed
because scalar F0 alone does not specify a density matrix.

**Also generate off / uncompensated / compensated comparison** produces all
three same-seed DES runs, an additional comparison figure, table and ZIP. Pure
unitary polarization changes do not alter the total detected pair rate in this
polarization-independent loss model; the comparison checks count equality.
Compensation corrects alignment, not accidental coincidences. Its inclusion
does not make turbulence diagnostics into a calibrated fading model.

```sh
python quasar_des_simulator.py --polarization uncompensated
python quasar_des_simulator.py --polarization compensated
python quasar_des_simulator.py --polarization-comparison
python -m unittest quasar.test_polarization quasar.test_des quasar.test_fidelity quasar.test_gui
```

`POLARIZATION_MODEL.md`, included in each result archive, defines the source
state, both Jones matrices, compensation assumptions and the derivation of
`F_mix = (C F_genuine + A/4)/(C+A)`, with literature references. This is a
pure-target density-mixture identity under a white-noise accidental assumption,
not a universal empirical fidelity law or a calibration from those references.
