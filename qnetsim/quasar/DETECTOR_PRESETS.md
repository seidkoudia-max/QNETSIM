# QUASAR wavelength-specific receiver presets

These are manufacturer-informed sensitivity cases, **not calibrated mission
receivers**. The user specified Si-APD / Thorlabs at 810 nm and SNSPD / Single
Quantum at 1550 nm. QNETSIM selected the representative presets below, assuming
identical receivers at Matera and Helmos. Sources checked 2026-09-11.

| Input | 810 nm: Thorlabs SPDMH2, free space | 1550 nm: Single Quantum Eos portfolio |
| --- | --- | --- |
| Efficiency | 0.60, typical PDE at 810 nm | 0.90, stated SDE lower bound |
| Timing FWHM | 1000 ps, typical resolution; wavelength/operating-point dependence not calibrated | 15 ps, stated upper bound |
| Gaussian RMS used | 424.6609 ps | 6.3699 ps |
| Dark counts per detector | 100/s, catalog maximum used | 1/s, stated upper bound used |
| Effective dead time | 45 ns typical, approximated as a step | 50 ns **assumed**, retained from the original baseline |
| Full coincidence window | 1 ns **assumed** | 1 ns **assumed** |
| Analyzer channels per OGS | 4 **assumed** | 4 **assumed**, not nanowire count |

## Source and operating-point cautions

[Thorlabs catalog, SPDMHx column](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5255&partnumber=SPDMH3)
provides the SPDMH2 values. The efficiency refers to free-space modules, not
the FC/PC variant. The manufacturer's
[SPDMHx manual MTN028160-D02, mirrored PDF, sections 3.4.2 and 6](https://manuals.plus/m/61ed70d5d283173a4e6f0e6ad81e7e0bdbf3dc1619016d6611b09929dcc7b1c5.pdf)
describes timing resolution as FWHM and warns of wavelength, focusing and
count-rate dependence. Above 1 MHz, a fixed low-rate width is not reliable;
the simulation warns when its per-channel mean exceeds that level. It also
flags mean rates above the catalog 20 MHz maximum. A unit production report
and an 810 nm timing histogram at the intended count rate remain necessary.
The stated 0.2% typical afterpulsing is documented but not simulated.

[Single Quantum brochure, final specification page](https://www.singlequantum.com/wp-content/uploads/2026/03/SQ_combined_brochure_with_specs.pdf)
gives the 1550 nm bounds and explicitly cautions that not all specifications
are simultaneously achievable. This preset combines them only for sensitivity
analysis; it is not a guaranteed Eos configuration. The listed maximum count
rate is at least 50 MHz; this does not specify a step recovery time and is not
inverted to obtain one. The assumed 50 ns dead time is not fitted to that
specification. A quoted joint operating point is needed.
[Single Quantum's timing definition](https://www.singlequantum.com/technology/interleaved-detectors/)
uses FWHM. Interleaved-detector performance is not substituted into this preset.

## How the presets enter DES and fidelity

PDE/SDE multiplies each arm's signal and sky detection probability exactly once.
RX efficiency must describe only losses upstream of the detector-system
reference plane: do not double-count SNSPD coupling already included in SDE.
Telescope-to-system coupling is not calibrated by selecting a detector brand.

The GUI takes single-detector jitter in FWHM. DES uses an **assumed Gaussian**
distribution with sigma = FWHM / sqrt(8 ln 2). For two independent identical
detectors and full coincidence width tau, true-pair acceptance is
q = erf(tau / (4 sigma)). No jitter contribution from a time tagger,
synchronization residual or source correlation width is added implicitly.

Both detector technologies use the existing nonparalyzable detector abstraction:
exact persistent dead time in timestamp DES, approximate live fractions in
batched DES. No APD afterpulsing, SNSPD latching or measured efficiency-recovery
curve is modeled. Four detector channels are assigned uniformly/independently;
polarization-analyzer count correlations are not simulated.

Accepted genuine coincidences C and accidental coincidences A still give
F_mix = (C F_genuine + A/4) / (C + A). Changing the detector does not justify
changing this mixture into a link-loss penalty. A 1 ns gate and the confirmed
1e9 emitted pairs/s can still yield strong other-pair contamination; no gate
or source-rate adjustment is made merely to raise the reported fidelity.

## Reproducibility

GUI defaults select the vendor sensitivity presets, with all numbers editable.
Select `Legacy assumptions` separately at each wavelength to reproduce the
previous shared defaults. Python `RunConfig()` retains those defaults for
backward compatibility; opt in with `with_vendor_detectors(cfg)` from
`quasar.detector_presets`. Changing a preset resets its inputs to that preset's
defaults on first use; returning to one retains its edits within the session.

Every export records effective per-wavelength settings, source URLs, preset
revision, typical/bound/assumed provenance, custom overrides, numerical source
hashes and operating warnings. Source-rate clarification is not calibration;
all mission-validation blockers remain outstanding until supported by evidence.
