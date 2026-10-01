"""Evidence provenance must not be confused with empirical validation."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from quasar.config import RunConfig, default_inputs, values_from_inputs
from quasar.mission_validation import STATUS, assessment, report_markdown, source_interpretation
from quasar.outputs import export
from quasar.simulator import run


class MissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.provenance = default_inputs()
        self.values = values_from_inputs(self.provenance)

    def test_original_source_interpretation_confirmed_only(self):
        report = assessment(RunConfig(), self.values, self.provenance)
        source = report["source_rate_interpretation"]
        self.assertEqual(source["status"], "user_confirmed")
        self.assertTrue(source["applies_to_run"])
        self.assertEqual(source["effective_rate_pairs_s"], 1e9)
        self.assertEqual(report["mean_emitted_pairs_per_gate"], 1)
        self.assertEqual(report["mission_status"], STATUS)
        self.assertEqual(report["independent_validation"]["status"], "not_performed")
        self.assertEqual(report["independent_validation"]["dataset_ids"], [])
        self.assertEqual(report["numerical_verification"]["status"], "not_executed_by_this_assessment")

    def test_rate_override_does_not_inherit_confirmation(self):
        source = source_interpretation({**self.values, "C21":500.}, self.provenance)
        self.assertFalse(source["applies_to_run"])
        self.assertEqual(source["effective_rate_pairs_s"], 5e8)

    def test_other_workbook_does_not_inherit_confirmation(self):
        for provenance in [{**self.provenance, "sha256":"different"},
                           {k:v for k,v in self.provenance.items() if k != "sha256"}]:
            self.assertFalse(source_interpretation(self.values, provenance)["applies_to_run"])

    def test_modified_source_row_does_not_inherit_confirmation(self):
        provenance = deepcopy(self.provenance)
        next(r for r in provenance["rows"] if r["cell"] == "C21")["value"] = 500.
        self.assertFalse(source_interpretation(self.values, provenance)["applies_to_run"])

    def test_unrelated_override_retains_rate_clarification_not_validation(self):
        values = {**self.values, "C22":.98}
        cfg = RunConfig()
        cfg = replace(cfg, polarization={nm:replace(p, enabled=True, compensate=True)
                                         for nm,p in cfg.polarization.items()})
        report = assessment(cfg, values, self.provenance)
        self.assertTrue(report["source_rate_interpretation"]["applies_to_run"])
        self.assertEqual(report["mission_status"], STATUS)
        self.assertEqual(report["source_fidelity_assumed"], .98)
        self.assertTrue(all(item["status"] == "outstanding" for item in report["blocking_items"]))

    def test_report_does_not_modify_inputs_and_is_strict_json(self):
        original = deepcopy(self.values)
        provenance = deepcopy(self.provenance)
        report = assessment(RunConfig(), self.values, self.provenance)
        json.dumps(report, allow_nan=False)
        self.assertEqual(self.values, original)
        self.assertEqual(self.provenance, provenance)
        text = report_markdown(report)
        self.assertIn("not mission-validated", text)
        self.assertIn("CW/pulsed", text)
        self.assertIn("not performed", text)

    def test_report_exports_with_run_and_archive(self):
        cfg = RunConfig(auto_pass=False, start_s=1377, duration_s=2, wavelengths_nm=(810,))
        results = run(cfg, self.values)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            output = export(results, self.provenance, self.values, out)
            audit = json.loads((out/"mission_validation.json").read_text())
            self.assertEqual(audit, output["mission_validation"])
            self.assertEqual(audit, output["configuration"]["mission_validation"])
            self.assertEqual(audit["mission_status"], STATUS)
            self.assertTrue(audit["source_rate_interpretation"]["applies_to_run"])
            self.assertIn("quasar/mission_evidence.json", output["configuration"]["source_sha256"])
            with zipfile.ZipFile(output["archive"]) as archive:
                for name in ["mission_validation.json", "MISSION_VALIDATION_STATUS.md", "MISSION_VALIDATION.md"]:
                    self.assertIn(name, archive.namelist())


if __name__ == "__main__":
    unittest.main()
