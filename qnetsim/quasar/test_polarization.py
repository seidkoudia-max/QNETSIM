"""Physical identities, causal control and DES integration regressions."""
from dataclasses import replace
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from channels.polarization import (ArmPolarization, PolarizationParams, PolarizationTrace,
                                   BELL_STATES, jones, source_density, propagate_density,
                                   genuine_fidelity, mixture_fidelity)
from quasar.config import RunConfig, default_inputs, values_from_inputs
from quasar.fidelity import diagnostics
from quasar.simulator import run


class PolarizationPhysicsTests(unittest.TestCase):
    def test_jones_unitary(self):
        for rotation, phase in [(0,0), (90,180), (15,25), (-80,600)]:
            u = jones(rotation,phase)
            np.testing.assert_allclose(u.conj().T@u, np.eye(2), atol=1e-14)

    def test_source_fidelity_trace_and_positivity(self):
        for target,ket in BELL_STATES.items():
            for f in [.25,.6,.99,1.]:
                rho = source_density(f,target)
                self.assertAlmostEqual(np.trace(rho).real,1)
                self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-14)
                self.assertAlmostEqual(np.vdot(ket,rho@ket).real,f)

    def test_propagation_matches_vectorized_overlap(self):
        a,b = jones(35,20),jones(-12,100)
        for target,ket in BELL_STATES.items():
            rho = propagate_density(source_density(.99,target),a,b)
            np.testing.assert_allclose(rho,rho.conj().T,atol=1e-14)
            self.assertAlmostEqual(np.trace(rho).real,1)
            self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-14)
            self.assertAlmostEqual(np.vdot(ket,rho@ket).real,float(genuine_fidelity(.99,a,b,target)))

    def test_single_rotation_and_retardance_analytic_limits(self):
        identity = np.eye(2)
        for angle in [0,15,45,90]:
            self.assertAlmostEqual(float(genuine_fidelity(1,jones(angle,0),identity)),math.cos(math.radians(angle))**2)
            self.assertAlmostEqual(float(genuine_fidelity(1,jones(0,angle),identity)),math.cos(math.radians(angle)/2)**2)

    def test_common_rotation_not_two_independent_fidelity_penalties(self):
        a = jones(42,0)
        self.assertAlmostEqual(float(genuine_fidelity(.99,a,a)),.99)
        a = jones(33,27)
        self.assertAlmostEqual(float(genuine_fidelity(.99,a,a,"psi_minus")),.99)

    def test_mixture_matches_density_matrix(self):
        ket = BELL_STATES["phi_plus"]
        rho = propagate_density(source_density(.99),jones(20,30),jones(-5,10))
        fg = float(np.vdot(ket,rho@ket).real)
        mixed = (70*rho+30*np.eye(4)/4)/100
        self.assertAlmostEqual(mixture_fidelity(fg,70,30),np.vdot(ket,mixed@ket).real)
        self.assertAlmostEqual(mixture_fidelity(.99,75,75),.62)
        self.assertEqual(mixture_fidelity(.8,10,0),.8)
        self.assertEqual(mixture_fidelity(.8,0,10),.25)
        self.assertTrue(math.isnan(mixture_fidelity(.8,0,0)))

    def trace(self, **kwargs):
        p = PolarizationParams(enabled=True, matera=ArmPolarization(25,.1,40,.02), helmos=ArmPolarization(),
                               reference_epoch_s=0, estimation_rms_deg=0, **kwargs)
        return PolarizationTrace(p,100,1,810,[[(10,90)],[(10,90)]])

    def test_compensation_is_causal_and_uses_inverse_order(self):
        trace = self.trace(compensate=True,latency_s=2,update_interval_s=10)
        np.testing.assert_allclose(trace.matrix(0,11),jones(*trace.angles(0,11)))
        expected = jones(*trace.angles(0,10)).conj().T@jones(*trace.angles(0,12))
        np.testing.assert_allclose(trace.matrix(0,12),expected)
        expected = jones(*trace.angles(0,10)).conj().T@jones(*trace.angles(0,21))
        np.testing.assert_allclose(trace.matrix(0,21),expected)
        self.assertTrue(all(e["applied_epoch_s"]==e["measured_epoch_s"]+2 for e in trace.events))

    def test_exact_compensation_at_measurement_recovers_source(self):
        trace = self.trace(compensate=True,latency_s=0,update_interval_s=10)
        for t in [10,20,30]:
            self.assertAlmostEqual(float(trace.fidelity(.99,t,t)),.99)

    def test_off_is_identity_even_with_nonzero_drift(self):
        p = PolarizationParams()
        trace = PolarizationTrace(p,100,1,810,[[(0,100)],[(0,100)]])
        self.assertEqual(len(trace.events),0)
        np.testing.assert_equal(trace.matrix(0,25),np.eye(2))
        self.assertEqual(float(trace.fidelity(.99,25,25)),.99)

    def test_seed_and_time_resolution_independent_paths(self):
        p = PolarizationParams(enabled=True)
        a = PolarizationTrace(p,100,1,810,[[],[]])
        b = PolarizationTrace(p,200,1,810,[[],[]])
        c = PolarizationTrace(p,100,2,810,[[],[]])
        t = np.arange(0,90,.5)
        np.testing.assert_equal(a.angles(0,t)[0],b.angles(0,t)[0])
        self.assertFalse(np.array_equal(a.angles(0,t)[0],c.angles(0,t)[0]))

    def test_visibility_and_latency_prevent_premature_updates(self):
        trace = self.trace(compensate=True,latency_s=100)
        self.assertEqual(trace.events,[])

    def test_invalid_and_oversized_configs(self):
        for p in [PolarizationParams(target="unknown"),PolarizationParams(update_interval_s=0),
                  PolarizationParams(latency_s=-1),PolarizationParams(estimation_rms_deg=float("nan")),
                  PolarizationParams(matera=ArmPolarization(wander_rms_deg=-1)),
                  PolarizationParams(enabled=True,wander_grid_s=.001)]:
            with self.subTest(p=p),self.assertRaises(ValueError):
                p.validate(2000)


class PolarizationDESTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = values_from_inputs(default_inputs())
        cls.cfg = RunConfig(auto_pass=False,start_s=1377,duration_s=30)
        cls.cases = []
        for enabled,compensate in [(False,False),(True,False),(True,True)]:
            cfg = replace(cls.cfg,polarization={nm:PolarizationParams(enabled=enabled,compensate=compensate) for nm in (810,1550)})
            cls.cases.append(run(cfg,cls.v))

    def test_all_counts_identical_between_modes(self):
        for wavelength in range(2):
            for rows in zip(*(r[wavelength].metrics for r in self.cases)):
                for key in ["true_pairs","other_pair_accidentals","noise_accidentals","Matera_deadtime_rejections","Helmos_deadtime_rejections"]:
                    self.assertEqual(rows[0][key],rows[1][key])
                    self.assertEqual(rows[0][key],rows[2][key])

    def test_compensation_improves_bell_overlap_in_example(self):
        off,raw,corrected = (r[0] for r in self.cases)
        self.assertLess(raw.summary["genuine_pair_fidelity"],off.summary["genuine_pair_fidelity"])
        self.assertGreater(corrected.summary["genuine_pair_fidelity"],raw.summary["genuine_pair_fidelity"])
        self.assertGreater(np.ptp([r["genuine_pair_fidelity"] for r in raw.metrics]),.001)

    def test_weighted_summary_and_diagnostics(self):
        result = self.cases[1][0]
        numerator = sum(r["genuine_pair_fidelity"]*r["true_pairs"]+.25*(r["other_pair_accidentals"]+r["noise_accidentals"]) for r in result.metrics)
        denominator = result.summary["true_pairs"]+result.summary["accidentals"]
        self.assertAlmostEqual(result.summary["coincidence_mixture_fidelity"],numerator/denominator)
        d = diagnostics(result,1e9,.99)
        self.assertAlmostEqual(d["shared_weighted_mixture_fidelity"],numerator/denominator)

    def test_updates_are_logged_as_des_events(self):
        result = self.cases[2][0]
        updates = [e for e in result.events if e["event"]=="polarization_update"]
        self.assertEqual(len(updates),6)
        self.assertEqual(result.summary["event_types"]["polarization_update"],6)
        self.assertEqual(result.summary["scheduler_events"],126)
        for e in updates:
            self.assertAlmostEqual(e["des_time_ps"]/1e12+self.cfg.start_s,e["applied_epoch_s"])
            self.assertGreater(e["applied_epoch_s"],e["measured_epoch_s"])

    def test_photon_fidelity_is_individual_and_count_stream_unchanged(self):
        cfg = replace(self.cfg,mode="photons",duration_s=.0001,bin_s=.00001,wavelengths_nm=(810,))
        off = run(cfg,self.v)[0]
        cfg = replace(cfg,polarization={810:PolarizationParams(enabled=True)})
        result = run(cfg,self.v)[0]
        self.assertEqual(off.summary["true_pairs"],result.summary["true_pairs"])
        self.assertEqual(off.summary["accidentals"],result.summary["accidentals"])
        events = [e for e in result.events if e["event"]=="true_coincidence"]
        self.assertGreater(np.ptp([e["genuine_pair_fidelity"] for e in events]),0)
        self.assertAlmostEqual(sum(e["genuine_pair_fidelity"] for e in events)/len(events),result.summary["genuine_pair_fidelity"])

    def test_batch_resolution_validation(self):
        cfg = replace(self.cfg,bin_s=10,polarization={810:PolarizationParams(enabled=True)})
        with self.assertRaisesRegex(ValueError,"Reduce the DES batch"):
            cfg.validate(self.v)

    def test_comparison_export(self):
        from quasar.polarization_study import run_comparison
        cfg = replace(self.cfg,duration_s=2,wavelengths_nm=(810,))
        with tempfile.TemporaryDirectory() as directory, patch("quasar.outputs.plot", return_value={}):
            output = run_comparison(cfg,self.v,default_inputs(),Path(directory)/"comparison")
            self.assertEqual(len(output["rows"]),3)
            self.assertTrue(Path(output["figure"]).exists())
            self.assertTrue(Path(output["archive"]).exists())


if __name__ == "__main__":
    unittest.main()
