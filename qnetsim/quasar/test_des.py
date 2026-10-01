"""Isolated QUASAR regression tests; no other GUI or controller is launched."""
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from quasar.config import Optics, RunConfig, default_inputs, read_workbook, values_from_inputs
from quasar.outputs import export
from quasar.physics import prepare
from quasar.simulator import EventLimitError, Experiment, match_tags, run


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.v = values_from_inputs(default_inputs())

    def test_default_inputs(self):
        RunConfig().validate(self.v)
        self.assertEqual(self.v["C21"], 1000)
        self.assertEqual(self.v["C26"], 6939.12837495153)

    def test_invalid_values(self):
        for cfg in [RunConfig(bin_s=0), RunConfig(duration_s=float("nan")),
                    RunConfig(seed=1.5), RunConfig(wavelengths_nm=()),
                    RunConfig(mode="ryu"), RunConfig(mode="photons"),
                    RunConfig(duration_s=20000, bin_s=1), RunConfig(optics={}),
                    RunConfig(detectors=1.5), RunConfig(max_photon_events=2.5)]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate(self.v)

    def test_valid_zero_physical_inputs(self):
        self.v["C21"] = 0
        self.v["C36"] = 0
        RunConfig(jitter_s=0, deadtime_s=0, dark_per_detector_s=0).validate(self.v)

    def test_reject_non_circular_and_bad_epoch(self):
        for key, value in [("C29", .1), ("C33", "2032-01-01T22:59:28+02:00"),
                           ("C6", -1), ("C22", float("nan"))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                RunConfig().validate({**self.v, key:value})

    def test_workbook_parser_against_supplied_file(self):
        path = Path(__file__).parent/"optional_inputs/QUASAR simulation parameters.xlsx"
        if not path.exists():
            self.skipTest("Private reviewer workbook not present on this machine")
        parsed = read_workbook(path.read_bytes())
        self.assertEqual(parsed["sha256"], default_inputs()["sha256"])
        self.assertEqual(values_from_inputs(parsed), self.v)

    def test_workbook_invalid_zip(self):
        with self.assertRaisesRegex(ValueError, "Invalid workbook"):
            read_workbook(b"not a workbook")

    def test_workbook_formula_rejected(self):
        contents = io.BytesIO()
        with zipfile.ZipFile(contents, "w") as z:
            z.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" r:id="s1"/></sheets></workbook>')
            z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="s1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="C21"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>')
        with self.assertRaisesRegex(ValueError, "Formula in C21"):
            read_workbook(contents.getvalue())


class MatchingTests(unittest.TestCase):
    def test_full_width_boundaries_and_true_identity(self):
        a = [(100, 1, True, 0)]
        b = [(90, 2, True, 0), (110, 1, True, 0), (111, 3, True, 0)]
        self.assertEqual(match_tags(a,b,20), [(0,1,1,0,100)])

    def test_noise_never_identifies_a_pair(self):
        self.assertEqual(match_tags([(0,-1,False,0)], [(0,-1,False,0)],10), [(0,0,0,1,0)])

    def test_all_pairs_not_one_to_one(self):
        a = [(0,1,True,0), (0,2,True,1)]
        b = [(0,1,True,0), (0,2,True,1), (0,-1,False,0)]
        rows = match_tags(a,b,1)
        self.assertEqual(sum(x[1] for x in rows), 2)
        self.assertEqual(sum(x[2] for x in rows), 2)
        self.assertEqual(sum(x[3] for x in rows), 2)

    def test_empty_tags(self):
        self.assertEqual(match_tags([],[],1), [])
        self.assertEqual(match_tags([(0,0,True,0)],[],1), [(0,0,0,0,0)])


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = values_from_inputs(default_inputs())
        cls.batch = RunConfig(auto_pass=False, start_s=1377, duration_s=2.5)
        cls.photon = replace(cls.batch, mode="photons", duration_s=.0001, bin_s=.00001,
                            wavelengths_nm=(810,))
        cls.bp = prepare(cls.batch, cls.v, 810)
        cls.pp = prepare(cls.photon, cls.v, 810)

    def experiment(self, cfg=None, values=None, physical=None):
        return Experiment(cfg or self.batch, values or self.v, 810, physical or self.bp).run()

    def test_seed_reproducibility_and_stochasticity(self):
        a, b = self.experiment(), self.experiment()
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.metrics, b.metrics)
        c = self.experiment(replace(self.batch, seed=100))
        self.assertNotEqual(a.events, c.events)

    def test_batch_event_count_order_and_delay(self):
        result = self.experiment()
        self.assertEqual(result.summary["scheduler_events"], 4*3)
        self.assertEqual(result.summary["backend"], "des_only")
        timestamps = [e["des_time_ps"] for e in result.events]
        self.assertEqual(timestamps, sorted(timestamps))
        for i, (edge, width) in enumerate(zip(self.bp["edges"],self.bp["widths"])):
            receiver = [e for e in result.events if e["bin_index"]==i and e["event"]=="receiver_batch"]
            match = next(e for e in result.events if e["bin_index"]==i and e["event"]=="coincidence_batch")
            self.assertTrue(all(e["des_time_ps"] > (edge+width)*1e12 for e in receiver))
            self.assertEqual(match["des_time_ps"], max(e["des_time_ps"] for e in receiver))

    def test_pair_bounds_and_partial_bin(self):
        result = self.experiment()
        self.assertEqual([r["bin_width_s"] for r in result.metrics], [1,1,.5])
        for row in result.metrics:
            self.assertLessEqual(row["true_pairs"], row["Matera_accepted_counts"])
            self.assertLessEqual(row["true_pairs"], row["Helmos_accepted_counts"])
            self.assertGreaterEqual(row["coincidence_mixture_fidelity"], .25)
            self.assertLessEqual(row["coincidence_mixture_fidelity"], self.v["C22"])
        integral = sum(r["true_pairs_per_s"]*r["bin_width_s"] for r in result.metrics)
        self.assertAlmostEqual(integral/2.5, result.summary["mean_true_pairs_per_s"])

    def test_zero_source_zero_noise_has_no_fidelity(self):
        cfg = replace(self.batch, dark_per_detector_s=0,
                      optics={810:Optics(sky_matera=0, sky_helmos=0)}, wavelengths_nm=(810,))
        v = {**self.v, "C21":0}
        result = self.experiment(cfg, v, prepare(cfg,v,810))
        self.assertEqual(result.summary["true_pairs"], 0)
        self.assertEqual(result.summary["accidentals"], 0)
        self.assertIsNone(result.summary["coincidence_mixture_fidelity"])
        self.assertTrue(all(math.isnan(r["coincidence_mixture_fidelity"]) for r in result.metrics))

    def test_no_visibility_no_true_pairs(self):
        cfg = replace(self.batch, start_s=0, wavelengths_nm=(810,))
        result = run(cfg,self.v)[0]
        self.assertEqual(result.summary["true_pairs"], 0)
        self.assertEqual(result.summary["shared_visibility_s"], 0)

    def test_batched_counts_close_to_reference(self):
        result = self.experiment()
        expected = sum(r["expected_true_pairs_per_s"]*r["bin_width_s"] for r in result.metrics)
        self.assertLess(abs(result.summary["true_pairs"]-expected), 8*np.sqrt(expected))

    def test_wavelength_order_independence(self):
        a = run(self.batch, self.v)
        b = run(replace(self.batch, wavelengths_nm=(1550,810)), self.v)
        self.assertEqual(a[0].events,b[1].events)
        self.assertEqual(a[1].events,b[0].events)

    def test_photon_timestamps_and_exact_deadtime(self):
        result = self.experiment(self.photon,physical=self.pp)
        arrivals = [e for e in result.events if e["event"]=="photon_arrival"]
        self.assertEqual(len(arrivals), result.summary["individual_photon_events"])
        self.assertGreater(len(set(e["des_time_ps"] for e in arrivals)),100)
        self.assertGreater(sum(not e["accepted"] for e in arrivals),0)
        self.assertEqual([e["des_time_ps"] for e in arrivals], sorted(e["des_time_ps"] for e in arrivals))
        last = {}
        for e in arrivals:
            key = e["station"],e["detector"]
            if e["accepted"]:
                self.assertGreaterEqual(e["des_time_ps"]-last.get(key,-10**12), round(self.photon.deadtime_s*1e12))
                last[key] = e["des_time_ps"]
        logs = [e for e in result.events if e["event"]=="true_coincidence"]
        self.assertEqual(len(logs), result.summary["true_pairs"])
        self.assertGreater(len(set(e["corrected_tag_ps"] for e in logs)),1)

    def test_no_deadtime_has_no_rejections(self):
        cfg = replace(self.photon, deadtime_s=0, jitter_s=0)
        p = prepare(cfg,self.v,810)
        result = self.experiment(cfg,physical=p)
        self.assertEqual(sum(result.summary["deadtime_rejections"].values()),0)
        ids = [{e["pair_id"] for e in result.events if e["event"]=="photon_arrival" and
                e["station"]==station and e["is_signal"]} for station in ["Matera","Helmos"]]
        self.assertEqual(result.summary["true_pairs"],len(ids[0]&ids[1]))

    def test_event_limit_no_silent_fallback(self):
        with self.assertRaises(EventLimitError):
            self.experiment(replace(self.photon, max_photon_events=100),physical=self.pp)

    def test_export_audits_and_does_not_overwrite(self):
        result = self.experiment()
        with tempfile.TemporaryDirectory() as directory, patch("quasar.outputs.plot", return_value={}):
            out = Path(directory)/"run"
            exported = export([result],default_inputs(),{**self.v,"C21":500},out)
            config = json.loads((out/"configuration.json").read_text())
            self.assertEqual(config["backend"],"des_only")
            self.assertEqual(config["overrides"]["C21"]["effective"],500)
            self.assertIn("quasar/simulator.py",config["source_sha256"])
            with zipfile.ZipFile(exported["archive"]) as z:
                self.assertIn("ASSUMPTIONS.md",z.namelist())
                self.assertIn("events_810nm.csv",z.namelist())
            with self.assertRaises(FileExistsError):
                export([result],default_inputs(),self.v,out)


if __name__ == "__main__":
    unittest.main()
