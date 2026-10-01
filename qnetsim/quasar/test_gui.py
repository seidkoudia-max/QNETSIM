"""Exercise the real GUI's two run paths, validation and stale-result message."""
import unittest

from streamlit.testing.v1 import AppTest

from quasar.config import ROOT


def labeled(elements, label):
    return next(element for element in elements if element.label == label)


class GuiTests(unittest.TestCase):
    def test_timing_study_opt_in_runs_and_preserves_main_settings(self):
        app = AppTest.from_file(str(ROOT/"quasar_gui.py")).run(timeout=60)
        self.assertNotIn("quasar_timing_result", app.session_state)
        app.checkbox(key="quasar_timing_enabled").check().run()
        self.assertTrue(app.button(key="quasar_timing_run").disabled)
        labeled(app.checkbox,"Find first shared pass automatically").uncheck().run()
        app.number_input(key="quasar_batch_start").set_value(1377.)
        app.number_input(key="quasar_batch_duration").set_value(2.5)
        app.text_input(key="quasar_timing_windows").set_value("100, 1000")
        app.text_input(key="quasar_timing_rates").set_value("100, 1000").run()
        labeled(app.checkbox,"I understand the receiver/noise assumptions and the limitations of the selected DES resolution.").check().run()
        app.button(key="quasar_timing_run").click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = app.session_state["quasar_timing_result"]
        self.assertEqual(len(result["rows"]),10)
        self.assertEqual(result["manifest"]["status"],"complete")
        self.assertNotIn("quasar_result", app.session_state)
        self.assertEqual(app.number_input(key="quasar_input_C21").value,1000.)
        self.assertAlmostEqual(app.number_input(key="quasar_detector_1550_single_quantum_1550_v1_gate").value,1000.)
        app.text_input(key="quasar_timing_windows").set_value("50,100").run()
        self.assertTrue(any("displayed study belongs to previous settings" in m.value for m in app.info))
        app.text_input(key="quasar_timing_rates").set_value("nan").run()
        self.assertTrue(app.button(key="quasar_timing_run").disabled)
        self.assertTrue(any("finite" in m.value for m in app.error))
        self.assertEqual(app.session_state["quasar_timing_result"]["out_dir"],result["out_dir"])

    def test_full_pass_and_photon_runs(self):
        app = AppTest.from_file(str(ROOT/"quasar_gui.py")).run(timeout=60)
        self.assertEqual(len(app.exception),0)
        run_label = "Run QUASAR - DES only"
        self.assertTrue(labeled(app.button,run_label).disabled)
        acknowledgment = "I understand the receiver/noise assumptions and the limitations of the selected DES resolution."
        labeled(app.checkbox,acknowledgment).check().run()
        labeled(app.button,run_label).click().run(timeout=120)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        result = app.session_state["quasar_result"]
        self.assertEqual(len(result["summaries"]),2)
        self.assertEqual([s["detector_technology"] for s in result["summaries"]], ["Si-APD", "SNSPD"])
        self.assertEqual([s["detector_settings"]["eta_detector"] for s in result["summaries"]], [.6, .9])
        self.assertTrue(all(s["detector_settings"]["numerical_specifications_status"] == "vendor_sensitivity_not_calibrated"
                            for s in result["summaries"]))
        self.assertTrue(all(s["mode"]=="batched" for s in result["summaries"]))
        self.assertTrue(all(s["true_pairs"]>0 for s in result["summaries"]))
        self.assertEqual(set(result["figures"]), {"geometry","link_budget","rates","fidelity_noise","fidelity_zoom","background_noise","events","polarization"})
        self.assertEqual(len(result["fidelity_diagnostics"]),2)
        self.assertEqual(result["mission_validation"]["mission_status"], "assumption_based_not_mission_validated")
        self.assertTrue(result["mission_validation"]["source_rate_interpretation"]["applies_to_run"])
        self.assertTrue(any("not mission-validated" in notice.value for notice in app.warning))
        self.assertTrue(any("Source rate confirmed by you" in notice.value for notice in app.info))
        self.assertTrue(all(d["shared_bin_fidelity_min"] is not None for d in result["fidelity_diagnostics"]))
        labeled(app.selectbox,"DES resolution").select("Timestamp-resolved DES - short interval").run()
        self.assertTrue(any("previous settings" in notice.value for notice in app.info))
        labeled(app.button,run_label).click().run(timeout=120)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        result = app.session_state["quasar_result"]
        self.assertTrue(all(s["mode"]=="photons" for s in result["summaries"]))
        self.assertTrue(all(s["individual_photon_events"]>0 for s in result["summaries"]))

    def test_no_wavelength_shows_validation_error(self):
        app = AppTest.from_file(str(ROOT/"quasar_gui.py")).run(timeout=60)
        labeled(app.multiselect,"Wavelengths (nm)").set_value([]).run()
        labeled(app.checkbox,"I understand the receiver/noise assumptions and the limitations of the selected DES resolution.").check().run()
        labeled(app.button,"Run QUASAR - DES only").click().run()
        self.assertTrue(any("Select 810 nm" in notice.value for notice in app.error))
        self.assertNotIn("quasar_result", app.session_state)

    def test_detector_presets_and_custom_overrides(self):
        app = AppTest.from_file(str(ROOT/"quasar_gui.py")).run(timeout=60)
        self.assertEqual(app.selectbox(key="quasar_detector_preset_810").value, "thorlabs_spdmh2_v1")
        self.assertEqual(app.selectbox(key="quasar_detector_preset_1550").value, "single_quantum_1550_v1")
        labeled(app.checkbox,"Find first shared pass automatically").uncheck().run()
        app.number_input(key="quasar_batch_start").set_value(1377.)
        app.number_input(key="quasar_batch_duration").set_value(1.)
        app.number_input(key="quasar_detector_1550_single_quantum_1550_v1_gate").set_value(100.).run()
        labeled(app.checkbox,"I understand the receiver/noise assumptions and the limitations of the selected DES resolution.").check().run()
        labeled(app.button,"Run QUASAR - DES only").click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = app.session_state["quasar_result"]
        self.assertEqual([d["full_gate_s"] for d in result["fidelity_diagnostics"]], [1e-9,1e-10])
        audit = result["configuration"]["effective_detectors"][1]
        self.assertEqual(audit["overrides"]["gate_s"]["effective"], 1e-10)
        app.selectbox(key="quasar_detector_preset_810").select("legacy").run()
        self.assertTrue(any("previous settings" in notice.value for notice in app.info))
        self.assertAlmostEqual(app.number_input(key="quasar_detector_810_legacy_jitter_fwhm").value, 117.741, places=3)
        self.assertEqual(app.number_input(key="quasar_detector_1550_single_quantum_1550_v1_gate").value, 100.)

    def test_polarization_controls_and_comparison(self):
        app = AppTest.from_file(str(ROOT/"quasar_gui.py")).run(timeout=60)
        labeled(app.checkbox,"Enable polarization drift").check().run()
        labeled(app.checkbox,"Enable polarization compensation").uncheck().run()
        labeled(app.checkbox,"Also generate off / uncompensated / compensated comparison").check().run()
        labeled(app.checkbox,"Find first shared pass automatically").uncheck().run()
        app.number_input(key="quasar_batch_start").set_value(1377.)
        app.number_input(key="quasar_batch_duration").set_value(3.).run()
        labeled(app.checkbox,"I understand the receiver/noise assumptions and the limitations of the selected DES resolution.").check().run()
        labeled(app.button,"Run QUASAR - DES only").click().run(timeout=180)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        result = app.session_state["quasar_result"]
        self.assertTrue(all(s["polarization_enabled"] and not s["polarization_compensated"] for s in result["summaries"]))
        self.assertEqual(len(result["polarization_comparison"]["rows"]),6)
        labeled(app.checkbox,"Enable polarization drift").uncheck().run()
        self.assertTrue(any("previous settings" in notice.value for notice in app.info))


if __name__ == "__main__":
    unittest.main()
