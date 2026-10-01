# Windhof-Kirchberg S3 Replay Inputs

This compact fixture supports the physical-timing DES study and its GUI.
It was exported from the local sampled S3 simulation of 2026-09-26.

- `t2_600/arguments.json`: archived model configuration.
- `t2_600/ground_optical_trace.csv`: simulated reference and sampled link trace.
- `figure3_pass_geometry.csv`: simulated pass geometry used for interpolation.

The physical-timing study uses these as inputs and computes new delivery,
storage and fidelity records through the bundled DES framework. These are
synthetic scenario inputs, not measurements or validated mission predictions.
Randomness and model settings remain recorded in the run metadata.
