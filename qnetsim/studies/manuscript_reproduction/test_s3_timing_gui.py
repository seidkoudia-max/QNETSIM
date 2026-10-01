"""GUI input wiring without launching an expensive full-pass campaign."""
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(AppTest is None, "Streamlit test runtime is optional")
class GroundTimingGUITests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT/"ground_physical_studio.py"), default_timeout=30).run()

    def test_historical_and_new_profiles_render(self):
        app = AppTest.from_file(str(ROOT/"paper_ground_studio.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.sidebar.selectbox(key="ground_execution_profile").select("Physical timing S3 (DES only)").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Physical Timing", app.title[0].value)

    def test_invalid_resources_do_not_launch(self):
        app = self.app()
        next(x for x in app.number_input if x.label == "Independent loading ports per link").set_value(513)
        with patch("ground_physical_studio.run") as launch:
            app.button[0].click().run()
        self.assertFalse(launch.called)
        self.assertTrue(any("enough memory slots" in x.value for x in app.error))
        self.assertEqual(len(app.exception), 0)

    def test_form_dispatches_its_own_configuration(self):
        app = self.app()
        next(x for x in app.number_input if x.label == "Independent loading ports per link").set_value(8)
        app.text_input[0].set_value("1, 20")
        app.checkbox[0].uncheck()
        # The standalone script imports its runner into __main__; patch the source.
        with patch("studies.manuscript_reproduction.s3_physical_timing.run") as launch:
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        launch.assert_called_once()
        _, repeats, config, controls = launch.call_args.args
        self.assertEqual(config.loading_ports, 8)
        self.assertEqual(config.t2_values, (1.,20.))
        self.assertEqual(repeats, 4)
        self.assertFalse(controls)


if __name__ == "__main__":
    unittest.main()
