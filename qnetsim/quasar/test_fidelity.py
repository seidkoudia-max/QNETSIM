"""Protect the distinction between a plot overlap and a physical model change."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quasar.config import RunConfig, default_inputs, values_from_inputs
from quasar.fidelity import background_free_reference, diagnostics, zoom_series, zoom_limits
from quasar.outputs import plot
from quasar.simulator import run


class FidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = values_from_inputs(default_inputs())
        cls.results = run(RunConfig(), cls.v)
        cls.diag = [diagnostics(r,1e9,.99) for r in cls.results]

    def test_equal_true_and_accidental_limit(self):
        q,f = background_free_reference(1e9,.99,1e-9,50e-12)
        self.assertAlmostEqual(q,1)
        self.assertAlmostEqual(f,.62)

    def test_narrow_gate_includes_true_pair_rejection(self):
        q,f = background_free_reference(1e9,.99,100e-12,50e-12)
        self.assertAlmostEqual(q,math.erf(.5))
        self.assertAlmostEqual(f,(.99*q+.025)/(q+.1))
        self.assertLess(f,(.99+.025)/1.1)

    def test_lower_brightness_limit(self):
        self.assertAlmostEqual(background_free_reference(1e8,.99,1e-9,0)[1],(.99+.025)/1.1)
        self.assertAlmostEqual(background_free_reference(1e7,.99,1e-9,0)[1],(.99+.0025)/1.01)

    def test_zero_source_undefined(self):
        self.assertIsNone(background_free_reference(0,.99,1e-9,0)[1])

    def test_both_wavelengths_exist_and_fluctuate(self):
        self.assertEqual([d["wavelength_nm"] for d in self.diag],[810,1550])
        for d in self.diag:
            self.assertGreater(d["shared_bin_fidelity_std"],0)
            self.assertLess(d["shared_bin_fidelity_std"],.002)
            self.assertAlmostEqual(d["shared_weighted_mixture_fidelity"],.62,places=3)
            self.assertLess(d["shared_noise_fraction_of_accidentals"],.001)

    def test_counts_unchanged_from_original_simulation(self):
        self.assertEqual([r.summary["true_pairs"] for r in self.results],[248348468,91370470])
        self.assertEqual([r.summary["accidentals"] for r in self.results],[248351193,91392372])

    def test_zoom_preserves_data_and_gaps(self):
        for result in self.results:
            zoom = zoom_series(result)
            for sample,row in zip(zoom,result.metrics):
                if row["joint_visible"]:
                    self.assertEqual(sample,row["coincidence_mixture_fidelity"])
                else:
                    self.assertTrue(np.isnan(sample))
        lo,hi = zoom_limits(self.results)
        self.assertLess(lo,min(d["shared_bin_fidelity_min"] for d in self.diag))
        self.assertGreater(hi,max(d["shared_bin_fidelity_max"] for d in self.diag))

    def test_no_visibility_zoom(self):
        r = run(RunConfig(auto_pass=False,start_s=0,duration_s=2,wavelengths_nm=(810,)),self.v)[0]
        d = diagnostics(r,1e9,.99)
        self.assertIsNone(d["shared_weighted_mixture_fidelity"])
        self.assertTrue(np.all(np.isnan(zoom_series(r))))
        self.assertEqual(zoom_limits([r]),(.25,1))

    def test_separate_panels_with_same_zoom_scale(self):
        import matplotlib.pyplot as plt
        saved = {}
        labels = {}
        def capture(fig,path,**kwargs):
            saved[Path(path).stem] = [(ax.get_title(),ax.get_ylim(),[line.get_ydata().copy() for line in ax.lines]) for ax in fig.axes]
            labels[Path(path).stem] = [text.get_text() for text in fig.texts]
        with tempfile.TemporaryDirectory() as directory, patch.object(plt.Figure,"savefig",capture):
            paths = plot(self.results,Path(directory),self.diag)
        self.assertEqual(len(saved["fidelity_noise"]),2)
        self.assertIn("810",saved["fidelity_noise"][0][0])
        self.assertIn("1550",saved["fidelity_noise"][1][0])
        self.assertEqual(saved["fidelity_zoom"][0][1],saved["fidelity_zoom"][1][1])
        for panel,result in zip(saved["fidelity_zoom"],self.results):
            np.testing.assert_equal(panel[2][0],zoom_series(result))
        self.assertIn("background_noise",paths)
        self.assertTrue(all(any("not mission-validated" in text for text in texts) for texts in labels.values()))

    def test_single_wavelength_plot(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory, patch.object(plt.Figure,"savefig"):
            paths=plot(self.results[:1],Path(directory),self.diag[:1])
        self.assertIn("fidelity_zoom",paths)


if __name__ == "__main__":
    unittest.main()
