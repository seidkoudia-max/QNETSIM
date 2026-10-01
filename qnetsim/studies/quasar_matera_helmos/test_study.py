import json
import math
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

import run_study as s


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance, cls.v = s.inputs()
        cls.gs = s.stations(cls.v)

    def test_workbook_units(self):
        self.assertEqual(float(self.v["C21"])*1e6,1e9)
        self.assertAlmostEqual((.74/2.3)**2,float(self.v["C13"]),places=4)
        self.assertEqual(len(self.provenance["rows"]),31)

    def test_orbit_is_circular_and_periodic(self):
        a=float(self.v["C26"])*1000
        period=2*np.pi/np.sqrt(s.MU/a**3)
        g=s.orbit(self.v,np.array([0,period/4,period]))
        np.testing.assert_allclose(g["radius_m"],a,rtol=1e-14)
        self.assertTrue(np.all(g["Matera_range_m"]>0))

    def test_geometry_refined_convergence(self):
        durations=[]
        for step in [1.,.25]:
            t=np.arange(1100,1600,step)
            g=s.orbit(self.v,t)
            durations.append(s.intervals(t,g["both_visible"])[0]["duration_midpoint_s"])
        self.assertLessEqual(abs(durations[0]-durations[1]),1.)

    def test_collection_matches_independent_gaussian_integral(self):
        gs=self.gs[0]
        r=700000.
        g={"Matera_range_m":np.array([r]),"Matera_elevation_deg":np.array([60.]),
           "Matera_visible":np.array([True])}
        l=s.link(gs,g,810e-9,self.v,zenith_transmission=1,eta_tx=1,eta_rx=1,eta_detector=1)
        w=l["beam_radius_m"][0]
        outer=gs["diameter_m"]/2
        inner=outer*np.sqrt(gs["obscuration_area"])
        direct=quad(lambda x:4*x/w**2*np.exp(-2*x*x/w**2),inner,outer)[0]
        self.assertAlmostEqual(l["eta"][0],direct,places=12)

    def test_pointing_convolution_monte_carlo(self):
        gs=self.gs[0]
        distance=700000.
        g={"Matera_range_m":np.array([distance]),"Matera_elevation_deg":np.array([60.]),
           "Matera_visible":np.array([True])}
        l=s.link(gs,g,810e-9,self.v,zenith_transmission=1,eta_tx=1,eta_rx=1,eta_detector=1)
        w0=.25/2
        w=np.sqrt(w0*w0+(810e-9/(np.pi*w0)*distance)**2)
        rng=np.random.default_rng(20260910)
        offsets=rng.normal(0,1e-6*distance,size=(200000,2))
        noncentral=np.sum(offsets**2,axis=1)/(w/2)**2
        outer=(gs["diameter_m"]/2/(w/2))**2
        inner=outer*gs["obscuration_area"]
        measured=ncx2.cdf(outer,2,noncentral)-ncx2.cdf(inner,2,noncentral)
        self.assertLess(abs(np.mean(measured)-l["eta"][0]),5*np.std(measured)/np.sqrt(len(measured)))

    def test_background_units_and_scaling(self):
        gs=self.gs[0]
        base=s.background(gs,810e-9,1e-6)
        self.assertAlmostEqual(s.background(gs,810e-9,2e-6),2*base)
        self.assertAlmostEqual(s.background(gs,810e-9,1e-6,bandwidth_nm=2),2*base)
        self.assertAlmostEqual(s.background(gs,810e-9,1e-6,fov_half_angle_urad=20)/base,4.,places=7)
        self.assertEqual(s.background(gs,810e-9,0),0)
        self.assertAlmostEqual(s.background(gs,1550e-9,1e-6)/base,1550/810)
        area=np.pi*(1.5/2)**2*.9
        independent=1e-6*area*np.pi*(10e-6)**2*1*.5*.6/(s.H_PLANCK*s.C_LIGHT/810e-9)
        self.assertAlmostEqual(base,independent,places=6)

    def test_coincidence_signal_and_accidental_limit(self):
        c=s.coincidence(np.array([.01]),np.array([.02]),1e9,.99,0,0,dark_each_s=0)
        self.assertAlmostEqual(c["true"][0],200000.,places=4)
        self.assertAlmostEqual(c["other_pair_accidental"][0],200000.)
        self.assertAlmostEqual(c["fidelity"][0],.62,places=9)
        self.assertEqual(c["background_accidental"][0],0)

    def test_no_link_means_no_true_pairs(self):
        c=s.coincidence(np.array([0]),np.array([.1]),1e9,.99,1000,2000)
        self.assertEqual(c["true"][0],0)
        self.assertGreater(c["accidental"][0],0)
        self.assertAlmostEqual(c["fidelity"][0],.25)

    def test_no_loss_no_source_and_deadtime(self):
        zero=s.coincidence(np.array([.01]),np.array([.02]),0,.99,0,0,dark_each_s=0)
        self.assertEqual(zero["true"][0],0)
        self.assertTrue(np.isnan(zero["fidelity"][0]))
        linear=s.coincidence(np.array([.01]),np.array([.02]),1e9,.99,0,0)
        dead=s.coincidence(np.array([.01]),np.array([.02]),1e9,.99,0,0,deadtime_s=50e-9)
        self.assertLess(dead["true"][0],linear["true"][0])
        self.assertTrue(0<dead["live_a"][0]<=1)

    def test_turbulence_strength_and_wavelength(self):
        a,r0=s.turbulence_integrals(self.gs[0],810e-9,self.v)
        b,r1=s.turbulence_integrals(self.gs[0],1550e-9,self.v)
        self.assertGreater(a,b)
        self.assertLess(r0,r1)
        local,_=s.turbulence_integrals(self.gs[0],810e-9,self.v,local_surface=True)
        self.assertGreater(local,a)

    def test_output_budget_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(s, "OUT", Path(directory)), patch.object(s, "plots"):
                s.main()
                summary=json.loads((s.OUT/"summary.json").read_text())
        for case in summary["scenarios"].values():
            product=1
            for arm in case["at_joint_peak"].values():
                loss=(arm["diffraction_loss_db"]+arm["pointing_increment_db"]+
                      arm["atmospheric_loss_db"]-10*np.log10(.8*.5*.6))
                self.assertAlmostEqual(loss,arm["detection_loss_db"])
                self.assertAlmostEqual(10**(-loss/10),arm["eta"])
                product*=arm["eta"]
            self.assertAlmostEqual(case["peak_true_pairs_s"],1e9*product,places=3)


if __name__=="__main__":
    unittest.main(verbosity=2)
