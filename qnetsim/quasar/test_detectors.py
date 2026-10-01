"""Detector routing, reference conventions and provenance (not mission validation)."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quasar.config import DetectorSettings, RunConfig, default_inputs, values_from_inputs
from quasar.detector_presets import DEFAULT_PRESET, FWHM_TO_RMS, preset_settings, with_vendor_detectors, operating_warnings
from quasar.fidelity import diagnostics
from quasar.mission_validation import assessment, report_markdown
from quasar.outputs import export
from quasar.physics import prepare
from quasar.simulator import run


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = default_inputs()
        cls.values = values_from_inputs(cls.inputs)
        cls.legacy = RunConfig(auto_pass=False, start_s=1377., duration_s=2.)
        cls.vendor = with_vendor_detectors(cls.legacy)

    def test_specification_conversion_and_assumptions(self):
        a, b = [self.vendor.detector_for(nm) for nm in (810, 1550)]
        self.assertAlmostEqual(a.jitter_s*1e12, 424.660900144, places=7)
        self.assertAlmostEqual(b.jitter_s*1e12, 6.369913502, places=7)
        self.assertEqual((a.deadtime_s, b.deadtime_s), (45e-9, 50e-9))
        self.assertEqual((a.dark_per_detector_s, b.dark_per_detector_s), (100., 1.))
        self.assertEqual((a.gate_s, b.gate_s), (1e-9, 1e-9))
        self.assertEqual(self.vendor.optics[1550].eta_detector, .9)
        self.assertEqual(self.vendor.optics[810], self.legacy.optics[810])
        audit = self.vendor.detector_audit(1550)
        self.assertIn("ASSUMED", audit["parameter_basis"]["deadtime_s"])
        self.assertTrue(any("simultaneously" in s for s in audit["caveats"]))
        self.assertEqual(audit["overrides"], {})

    def test_legacy_shared_fields_are_still_used(self):
        cfg = replace(self.legacy, gate_s=2e-9, jitter_s=30e-12, dark_per_detector_s=3., detectors=2, deadtime_s=8e-9)
        for nm in (810, 1550):
            det = cfg.detector_for(nm)
            self.assertEqual((det.gate_s, det.jitter_s, det.deadtime_s), (2e-9, 30e-12, 8e-9))
            self.assertEqual((det.detectors, det.dark_per_detector_s), (2, 3.))
            self.assertEqual(cfg.detector_audit(nm)["numerical_specifications_status"], "assumed_not_calibrated")

    def test_explicit_legacy_equivalent_reproduces_des(self):
        cfg = replace(self.legacy, detector_settings={nm:self.legacy.detector_for(nm) for nm in (810,1550)})
        a,b = run(self.legacy,self.values), run(cfg,self.values)
        for left,right in zip(a,b):
            self.assertEqual(left.metrics, right.metrics)
            self.assertEqual(left.events, right.events)

    def test_invalid_preset_and_technology(self):
        a = self.vendor.detector_for(810)
        for det in [replace(a, technology="SNSPD"), replace(a, jitter_s=-1),
                    replace(a, dark_per_detector_s=float("nan")), replace(a, gate_s=0),
                    replace(a, detectors=1.5), replace(a, preset_id=DEFAULT_PRESET[1550]),
                    replace(a, preset_id="unknown")]:
            with self.subTest(det=det), self.assertRaises(ValueError):
                replace(self.vendor, detector_settings={810:det}).validate(self.values)

    def test_physics_acceptance_and_efficiency(self):
        for nm in (810,1550):
            p = prepare(self.vendor,self.values,nm)
            det = self.vendor.detector_for(nm)
            self.assertAlmostEqual(p["acceptance"], math.erf(det.gate_s/(4*det.jitter_s)))
        base, changed = prepare(self.legacy,self.values,1550), prepare(self.vendor,self.values,1550)
        self.assertAlmostEqual(changed["arms"][0]["eta"][0]/base["arms"][0]["eta"][0], 1.5)
        self.assertAlmostEqual(changed["sky"][0]/base["sky"][0], 1.5)

    def test_wavelength_isolation_both_modes(self):
        for mode in ("batched", "photons"):
            cfg = replace(self.vendor, mode=mode, duration_s=2. if mode=="batched" else .0001,
                          bin_s=1. if mode=="batched" else .00001)
            settings = dict(cfg.detector_settings)
            settings[810] = replace(settings[810], gate_s=50e-12, jitter_s=2e-9, dark_per_detector_s=1e5, deadtime_s=150e-9, detectors=2)
            changed = replace(cfg, detector_settings=settings)
            a, b = run(cfg,self.values), run(changed,self.values)
            self.assertEqual(a[1].events, b[1].events)
            self.assertEqual(a[1].summary, b[1].summary)
            self.assertNotEqual(a[0].events, b[0].events)
            self.assertGreater(a[0].summary["true_pairs"], b[0].summary["true_pairs"])
            for r in b:
                self.assertEqual(r.config.detector_settings, settings)

    def test_photon_deadtime_uses_each_wavelength(self):
        cfg = replace(self.vendor, mode="photons", duration_s=.0001, bin_s=.00001)
        for result in run(cfg,self.values):
            last = {}
            for event in result.events:
                if event["event"] != "photon_arrival" or not event["accepted"]:
                    continue
                key = event["station"],event["detector"]
                delay = round(cfg.detector_for(result.wavelength_nm).deadtime_s*1e12)
                self.assertGreaterEqual(event["des_time_ps"]-last.get(key,-10**12), delay)
                last[key] = event["des_time_ps"]

    def test_distinct_gate_diagnostics_and_validation(self):
        settings = dict(self.vendor.detector_settings)
        settings[1550] = replace(settings[1550], gate_s=100e-12)
        cfg = replace(self.vendor, detector_settings=settings)
        report = assessment(cfg,self.values,self.inputs)
        self.assertIsNone(report["gate_s"])
        self.assertIsNone(report["mean_emitted_pairs_per_gate"])
        self.assertEqual([r["mean_emitted_pairs_per_gate"] for r in report["detectors"]], [1.,.1])
        self.assertIn("1e-10 s", report_markdown(report))
        for result in run(cfg,self.values):
            d = diagnostics(result,1e9,.99)
            self.assertEqual(d["full_gate_s"], cfg.detector_for(result.wavelength_nm).gate_s)
        self.assertEqual(report["independent_validation"]["status"], "not_performed")

    def test_custom_overrides_and_roundtrip_provenance(self):
        a = self.vendor.detector_for(810)
        roundtrip = replace(a, jitter_s=(a.jitter_s/FWHM_TO_RMS*1e12)*1e-12*FWHM_TO_RMS)
        cfg = replace(self.vendor, detector_settings={810:roundtrip})
        self.assertEqual(cfg.detector_audit(810)["overrides"], {})
        cfg = replace(cfg, detector_settings={810:replace(a, dark_per_detector_s=11.)})
        audit = cfg.detector_audit(810)
        self.assertEqual(audit["overrides"]["dark_per_detector_s"], {"reference":100., "effective":11.})
        self.assertIn("User override", audit["parameter_basis"]["dark_per_detector_s"])

    def test_operating_envelope_warnings(self):
        a = preset_settings(DEFAULT_PRESET[810])
        self.assertFalse(operating_warnings(a, 1e5))
        self.assertEqual(len(operating_warnings(a, 2e6)),1)
        self.assertEqual(len(operating_warnings(a, 21e6)),2)

    def test_export_effective_settings_and_sources(self):
        results = run(self.vendor,self.values)
        with tempfile.TemporaryDirectory() as tmp, patch("quasar.outputs.plot", return_value={}):
            result = export(results,self.inputs,self.values,Path(tmp)/"run")
            config = result["configuration"]
            self.assertEqual([x["eta_detector"] for x in config["effective_detectors"]], [.6,.9])
            self.assertEqual(len(config["effective_detectors"][1]["sources"]),2)
            self.assertIn("quasar/detector_presets.py", config["source_sha256"])
            self.assertTrue((Path(result["out_dir"])/"DETECTOR_PRESETS.md").exists())
            json.dumps(config, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
