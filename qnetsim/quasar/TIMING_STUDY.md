# QUASAR coincidence-window and source-brightness study

Optional additional DES study; it does not change the main simulation settings,
overwrite previous results, or add an adaptive-optics model. All outputs remain
assumption-based, not mission-validated.

## Running in the GUI

Enable **Coincidence-window / source-brightness study**. Enter comma-separated
full coincidence windows in ps and emitted-pair brightnesses in MHz. After
acknowledging the model limitations, select **Run timing / brightness study**.
This button is independent of **Run QUASAR - DES only**.

Defaults: 50, 100, 200, 500, 1000 ps and 10, 100, 250, 500, 1000 MHz. For both
wavelengths this is 50 grid simulations plus two baseline simulations. There is
one seeded realization per setting, not an ensemble uncertainty estimate.
Each axis accepts at most eight unique values. Total reporting bins are capped;
photon-mode event limits remain active. The selected DES mode and resolution
are never silently changed. Timestamp mode studies the chosen short interval,
not a complete pass; batched mode can study the entire shared pass.

## Interpretation

The baseline star shows the current GUI source rate and each wavelength's
current coincidence width. Each grid setting changes only the full software
coincidence window and source emitted-pair rate. Geometry, detector settings,
background, source fidelity and polarization controls are held fixed.
Automatic pass selection is resolved once so all cells use identical times.
Compensation stays on if selected; it is never enabled implicitly.

Plots show raw mixture fidelity and retained genuine-pair rate side by side for
each wavelength, plus a fidelity/rate trade-off view. The rate is the sum of
accepted genuine pairs divided by the shared-visibility time, including
zero-delivery bins and the actual duration of partial bins. Fidelity is pooled
over coincidence counts: (sum C_i F_i + sum A_i/4) / sum(C_i + A_i), not an
unweighted average of bin fidelities. Empty coincidence sets have undefined
fidelity. All-pairs matching counts are not entanglement-distilled pairs or a
secret-key rate.

The timing-acceptance column is the Gaussian-jitter analytical acceptance
erf(window / (4 single-detector RMS jitter)); the plotted rate and fidelity are
from actual DES counts, not this approximation substituted for simulation.
Narrow windows can reject genuine pairs, especially for the Si-APD preset.
Lower brightness reduces multi-pair accidentals but also changes throughput
and saturation. The selected source-state and detector parameters are held
fixed assumptions: they are not known to remain constant at every brightness.

Same seeds are used across cells to make results reproducible. Different
brightnesses/windows may consume random draws differently; the markers are
not exactly paired samples or statistically independent confidence replicates.
Finite-count fluctuations need not be monotonic. The study does not optimize
away fluctuations or promise a hardware-achievable operating point.

Single Quantum headline specifications may not hold simultaneously; Si-APD
jitter varies with wavelength, illumination and count rate. These and all
other mission-validation limitations remain in the per-case exports.
Changing the source rate is a labeled sensitivity override, not a correction
to the confirmed 1000 MHz workbook value. Timing electronics and synchronization
uncertainty must be supplied before very narrow windows are operationally credible.

## Downloads and audit

`study.csv` and `study_results.json` contain every baseline/grid result, timing
acceptance, counts, visibility duration and detector warnings. `study.json`
contains the complete base settings, axes, provenance, source hash and completion
status. Each case folder contains configuration, source hashes, metrics, event
logs, fidelity diagnostics, assumptions and validation status, without duplicating
the eight main plots. The study ZIP includes everything.

Study results are marked stale if the active settings or grid change. A failed
study leaves a partial audit labeled `failed`, not a completed result.
