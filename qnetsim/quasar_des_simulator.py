"""Run the separate QUASAR DES-only scenario without opening the GUI."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from quasar.config import RunConfig, default_inputs, read_workbook, values_from_inputs
from quasar.outputs import export
from quasar.simulator import run
from quasar.detector_presets import with_vendor_detectors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbook", type=Path)
    p.add_argument("--mode", choices=["batched", "photons"], default="batched")
    p.add_argument("--start", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--bin", type=float)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--detectors", choices=["legacy", "vendor"], default="legacy",
                   help="Use legacy assumptions or Thorlabs/Single Quantum sensitivity presets (GUI default: vendor)")
    p.add_argument("--polarization", choices=["off", "uncompensated", "compensated"], default="off")
    p.add_argument("--polarization-comparison", action="store_true", help="Run all three illustrative polarization cases with the same seed")
    args = p.parse_args()
    inputs = read_workbook(args.workbook.read_bytes()) if args.workbook else default_inputs()
    photons = args.mode == "photons"
    cfg = RunConfig(mode=args.mode, seed=args.seed, auto_pass=not photons and args.start is None,
                    start_s=args.start if args.start is not None else 1377.,
                    duration_s=args.duration if args.duration is not None else (.0001 if photons else 1200.),
                    bin_s=args.bin if args.bin is not None else (.00001 if photons else 1.))
    values = values_from_inputs(inputs)
    if args.detectors == "vendor":
        cfg = with_vendor_detectors(cfg)
    cfg = replace(cfg, polarization={nm:replace(pol, enabled=args.polarization != "off", compensate=args.polarization == "compensated")
                                    for nm,pol in cfg.polarization.items()})
    if args.polarization_comparison:
        from quasar.polarization_study import run_comparison
        print(json.dumps(run_comparison(cfg, values, inputs), indent=2))
        return
    result = export(run(cfg, values), inputs, values)
    print(json.dumps({"out_dir":result["out_dir"], "summaries":result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
