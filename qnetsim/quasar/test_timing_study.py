"""Actual DES grid checks and bookkeeping; no claim of mission calibration."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from quasar.config import RunConfig, default_inputs, values_from_inputs
from quasar.detector_presets import with_vendor_detectors
from quasar.simulator import run
from quasar.timing_study import TimingStudy, configurations, parse_axis, plot_study, run_study, study_signature, summarize


class TimingStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = default_inputs()
        cls.values = values_from_inputs(cls.provenance)
        cls.cfg = with_vendor_detectors(RunConfig(auto_pass=False, start_s=1377., duration_s=2.5))
        cls.spec = TimingStudy((100., 1000.), (100., 1000.))

    def test_parser_rejects_ambiguous_or_oversized_axes(self):
        for text in ["", "100,", "50,50", "nan", "inf", "0", "-5", "1e8", "1;2", "1,2,3,4,5,6,7,8,9"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_axis(text, "windows", 1, 1e7)
        self.assertEqual(parse_axis("1000, 5e1,100", "windows", 1, 1e7), (50.,100.,1000.))
        self.assertEqual(parse_axis("0, 1e3", "brightness", 0,10000), (0.,1000.))

    def test_configurations_only_change_window_and_brightness(self):
        original = self.cfg.as_dict(), deepcopy(self.values)
        cases = configurations(self.cfg,self.values,self.spec)
        self.assertEqual(len(cases),5)
        for key,cfg,values in cases:
            self.assertEqual(cfg.optics,self.cfg.optics)
            self.assertEqual(cfg.polarization,self.cfg.polarization)
            self.assertEqual((cfg.start_s,cfg.duration_s,cfg.bin_s,cfg.seed,cfg.mode),
                             (self.cfg.start_s,self.cfg.duration_s,self.cfg.bin_s,self.cfg.seed,self.cfg.mode))
            self.assertEqual({k:v for k,v in values.items() if k != "C21"},
                             {k:v for k,v in self.values.items() if k != "C21"})
            for nm in cfg.wavelengths_nm:
                self.assertEqual(replace(cfg.detector_for(nm),gate_s=self.cfg.detector_for(nm).gate_s), self.cfg.detector_for(nm))
        self.assertEqual((self.cfg.as_dict(),self.values),original)

    def test_auto_pass_resolved_once_for_all_cells(self):
        from quasar.physics import resolve_window
        cfg = replace(self.cfg,auto_pass=True)
        with patch("quasar.timing_study.resolve_window", wraps=resolve_window) as resolve:
            cases = configurations(cfg,self.values,self.spec)
        self.assertEqual(resolve.call_count,1)
        self.assertEqual(len({(c.start_s,c.duration_s,c.auto_pass) for _,c,_ in cases}),1)
        self.assertTrue(all(not c.auto_pass for _,c,_ in cases))

    def test_resource_limits_and_mode_not_silently_changed(self):
        cfg = replace(self.cfg,duration_s=10000.)
        with self.assertRaisesRegex(ValueError,"reporting bins"):
            configurations(cfg,self.values,TimingStudy(tuple(range(1,9)),tuple(range(1,9))))
        cfg = replace(self.cfg,mode="photons",duration_s=.00005,bin_s=.00001)
        self.assertTrue(all(c.mode=="photons" for _,c,_ in configurations(cfg,self.values,self.spec)))

    def test_exact_count_weighted_aggregation(self):
        result = run(self.cfg,self.values)[0]
        row = summarize(result,self.values,"test")
        counts = sum(m["true_pairs"] for m in result.metrics)
        acc = sum(m["other_pair_accidentals"]+m["noise_accidentals"] for m in result.metrics)
        weighted = sum(m["genuine_fidelity_sum"] for m in result.metrics)
        self.assertEqual(row["shared_visibility_s"],2.5)
        self.assertEqual(row["true_pairs_per_s"],counts/2.5)
        self.assertAlmostEqual(row["raw_mixture_fidelity"],(weighted+.25*acc)/(counts+acc))

    def test_complete_grid_exports_and_reference_behavior(self):
        before = deepcopy(self.values)
        progress = []
        with tempfile.TemporaryDirectory() as tmp, patch("quasar.timing_study.plot_study",return_value={}):
            result = run_study(self.cfg,self.values,self.provenance,self.spec,Path(tmp)/"study",
                               progress=lambda i,total:progress.append((i,total)))
            self.assertEqual(len(result["rows"]),10)
            self.assertEqual(progress,[(i,5) for i in range(1,6)])
            self.assertEqual(result["manifest"]["status"],"complete")
            self.assertEqual(result["manifest"]["completed_wavelength_runs"],10)
            rows = [r for r in result["rows"] if r["wavelength_nm"]==1550 and not r["is_baseline"]]
            short = next(r for r in rows if r["source_rate_mhz"]==1000 and math.isclose(r["full_window_ps"],100))
            wide = next(r for r in rows if r["source_rate_mhz"]==1000 and math.isclose(r["full_window_ps"],1000))
            dim = next(r for r in rows if r["source_rate_mhz"]==100 and math.isclose(r["full_window_ps"],100))
            self.assertGreater(short["raw_mixture_fidelity"],wide["raw_mixture_fidelity"])
            self.assertGreater(dim["raw_mixture_fidelity"],short["raw_mixture_fidelity"])
            self.assertLess(dim["true_pairs_per_s"],short["true_pairs_per_s"])
            # A grid cell matching the original settings reproduces that DES run.
            baseline = next(r for r in result["rows"] if r["wavelength_nm"]==1550 and r["is_baseline"])
            self.assertEqual(baseline["true_pairs"],wide["true_pairs"])
            config = json.loads((Path(result["out_dir"])/"case_001/configuration.json").read_text())
            self.assertEqual(config["effective_workbook_values"]["C21"],100.)
            self.assertFalse(config["mission_validation"]["source_rate_interpretation"]["applies_to_run"])
            self.assertEqual(config["effective_detectors"][0]["gate_s"],100e-12)
            with zipfile.ZipFile(result["archive"]) as archive:
                for name in ["study.csv","study_results.json","study.json","baseline/events_810nm.csv",
                             "case_004/metrics_1550nm.csv","case_001/MISSION_VALIDATION_STATUS.md"]:
                    self.assertIn(name,archive.namelist())
                self.assertFalse(any(name.endswith(".zip") for name in archive.namelist()))
            with self.assertRaises(FileExistsError):
                run_study(self.cfg,self.values,self.provenance,self.spec,Path(result["out_dir"]))
        self.assertEqual(self.values,before)

    def test_photon_study_is_actual_event_mode(self):
        cfg = replace(self.cfg,mode="photons",duration_s=.00005,bin_s=.00001,wavelengths_nm=(1550,))
        with tempfile.TemporaryDirectory() as tmp, patch("quasar.timing_study.plot_study",return_value={}):
            result = run_study(cfg,self.values,self.provenance,TimingStudy((100.,),(1000.,)),Path(tmp)/"study")
            self.assertTrue(all(r["mode"]=="photons" for r in result["rows"]))
            summary = json.loads((Path(result["out_dir"])/"case_001/summary.json").read_text())[0]
            self.assertGreater(summary["individual_photon_events"],0)

    def test_failure_keeps_partial_audit_not_success(self):
        baseline = run(self.cfg,self.values)
        with tempfile.TemporaryDirectory() as tmp, patch("quasar.timing_study.run",side_effect=[baseline,ValueError("event limit")]):
            path = Path(tmp)/"study"
            with self.assertRaisesRegex(RuntimeError,"Partial audit"):
                run_study(self.cfg,self.values,self.provenance,self.spec,path)
            manifest = json.loads((path/"study.json").read_text())
            self.assertEqual(manifest["status"],"failed")
            self.assertEqual(manifest["completed_cases"],["baseline"])
            self.assertFalse((path/"quasar_timing_brightness_study.zip").exists())

    def test_empty_visibility_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"study"
            with self.assertRaisesRegex(ValueError,"No shared visibility"):
                run_study(replace(self.cfg,start_s=0),self.values,self.provenance,self.spec,path)
            self.assertFalse(path.exists())

    def test_zero_coincidences_are_undefined_and_plot_safe(self):
        cfg = replace(self.cfg, wavelengths_nm=(810,),
                      optics={810:replace(self.cfg.optics[810],sky_matera=0,sky_helmos=0)},
                      detector_settings={810:replace(self.cfg.detector_for(810),dark_per_detector_s=0)})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_study(cfg,{**self.values,"C21":0},self.provenance,TimingStudy((100.,),(0.,)),Path(tmp)/"study")
            self.assertTrue(all(r["raw_mixture_fidelity"] is None for r in result["rows"]))
            self.assertTrue(all(r["true_pairs_per_s"]==0 for r in result["rows"]))
            self.assertTrue(all(Path(p).is_file() for p in result["figures"].values()))
            json.dumps(result,allow_nan=False)

    def test_signature_changes_with_axis_detector_and_polarization(self):
        original = study_signature(self.cfg,self.values,self.provenance,self.spec)
        for cfg,values,spec in [(self.cfg,self.values,TimingStudy((50.,),(100.,))),
                                (replace(self.cfg,seed=17),self.values,self.spec),
                                (self.cfg,{**self.values,"C21":50},self.spec),
                                (replace(self.cfg,polarization={nm:replace(p,enabled=True) for nm,p in self.cfg.polarization.items()}),self.values,self.spec)]:
            self.assertNotEqual(study_signature(cfg,values,self.provenance,spec),original)


if __name__ == "__main__":
    unittest.main()
