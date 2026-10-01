from dataclasses import replace
import math
import unittest

import numpy as np

from des_ground_physical import DESEnv, GroundTimingDES, OpticalSegment, PS, TimingConfig


def execute(config=None, probability=.3):
    config = config or TimingConfig(duration_s=1., capacity_per_bank=8, loading_ports=2,
        source_rate_hz=20000., cutoff_s=.2, metrics_dt_s=.1, trace_start_s=0., trace_end_s=1.)
    def channel(link, t):
        return OpticalSegment(config.duration_s+1., probability if link == 0 else .8,
                              .99, 500e3 if link == 0 else 16000.)
    env = DESEnv(default=True)
    sim = GroundTimingDES(env, config, channel)
    env.init()
    env.run(end_time=round(config.duration_s*PS), summary=False)
    sim.validate()
    return sim


class TimingTests(unittest.TestCase):
    def test_invalid_inputs(self):
        for kw in (dict(source_rate_hz=0), dict(loading_ports=0), dict(loading_ports=513),
                   dict(bsm_service_s=0), dict(eta_read=1.2), dict(herald_electronics_s=-1),
                   dict(t2_values=(0.,)), dict(t2_values=(1.,1.)), dict(duration_s=math.inf)):
            with self.assertRaises(ValueError):
                TimingConfig(**kw)

    def test_four_bank_resource_and_count_balance(self):
        sim = execute()
        self.assertGreater(sim.counts["raw_delivered"], 0)
        self.assertTrue(np.all(sim.peak_occupied <= sim.p.capacity_per_bank))
        self.assertTrue(sim.validate()["ownership"])
        self.assertEqual(len(sim.agenda), 0)

    def test_positive_herald_and_readout_ages(self):
        sim = execute()
        loads = [r for r in sim.trace if r["event"] == "load_herald"]
        self.assertTrue(loads)
        for r in loads:
            self.assertGreater(r["t_s"], max(r["remote_write_s"], r["repeater_write_s"]))
        swaps = [r for r in sim.trace if r["event"] == "bsm_complete"]
        for r in swaps:
            self.assertGreaterEqual(r["t_s"]-r["started_s"]+1e-12, sim.p.bsm_service_s)
            self.assertGreater(r["sat_age_s"], 0.)
            self.assertGreater(r["fiber_age_s"], 0.)

    def test_t2_changes_quality_not_raw_timing(self):
        a = execute()
        b = execute(replace(a.p, t2_values=(.0001, 600.)))
        self.assertEqual(a.counts, b.counts)
        self.assertEqual([r["raw_count"] for r in a.metrics], [r["raw_count"] for r in b.metrics])
        self.assertLess(b.quality[.0001]["fidelity_sum"], b.quality[600.]["fidelity_sum"])
        self.assertLess(b.quality[.0001]["qualified"], b.quality[600.]["qualified"])

    def test_metrics_cadence_is_observational(self):
        a = execute()
        b = execute(replace(a.p, metrics_dt_s=.07))
        self.assertEqual(a.counts["raw_delivered"], b.counts["raw_delivered"])
        self.assertEqual(a.trace, b.trace)
        self.assertEqual(a.quality, b.quality)
        self.assertAlmostEqual(a.integral_busy, b.integral_busy, places=12)

    def test_horizon_pending_events_and_partial_bin(self):
        sim = execute(TimingConfig(duration_s=.001, capacity_per_bank=4, loading_ports=2,
            metrics_dt_s=.1, source_rate_hz=5000.))
        self.assertEqual(sim.metrics[-1]["t_s"], .001)
        self.assertEqual(sim.metrics[-1]["bin_s"], .001)
        self.assertEqual(sim.counts["raw_delivered"], 0)
        sim.validate()

    def test_serialized_processing_limit_independent_of_capacity(self):
        p = TimingConfig(duration_s=2., capacity_per_bank=32, loading_ports=8,
                         source_rate_hz=1e5, bsm_service_s=.02, metrics_dt_s=.2)
        sim = execute(p, probability=1.)
        self.assertLessEqual(sim.counts["bsm_completed"], 100)
        self.assertGreater(sim.integral_busy/p.duration_s, .8)

    def test_no_contact_and_no_write(self):
        sim = execute(probability=0.)
        self.assertEqual(sim.counts["raw_delivered"], 0)
        sim = execute(replace(sim.p, eta_write=0.))
        self.assertEqual(sim.counts["loaded_0"], 0)
        self.assertEqual(sim.counts["loaded_1"], 0)

    def test_expiry_remote_release_is_delayed(self):
        sim = execute(TimingConfig(duration_s=.5, capacity_per_bank=8, loading_ports=2,
            cutoff_s=.005, bsm_service_s=.01, metrics_dt_s=.001), probability=.01)
        self.assertGreater(sim.counts["expired_1"], 0)
        self.assertTrue(any(r["fiber_remote_occupied"] > r["fiber_ogs_occupied"] for r in sim.metrics))

    def test_expired_before_herald_cannot_enter_bsm(self):
        sim = execute(TimingConfig(duration_s=.1, capacity_per_bank=8, loading_ports=2,
            cutoff_s=1e-5, metrics_dt_s=.01), probability=1.)
        self.assertGreater(sim.counts["loaded_0"], 0)
        self.assertEqual(sim.counts["loaded_0"], sim.counts["expired_0"])
        self.assertEqual(sim.counts["bsm_started"], 0)

    def test_accelerated_geometric_trials_match_explicit_distribution(self):
        base = TimingConfig(duration_s=.4, capacity_per_bank=8, loading_ports=2,
                            source_rate_hz=20000., metrics_dt_s=.1)
        results = []
        for accelerated in (False, True):
            counts = [execute(replace(base, seed=i+730, accelerate_failures=accelerated)).counts["loaded_0"]
                      for i in range(80)]
            results.append(np.array(counts))
        se = math.sqrt(sum(np.var(v, ddof=1)/len(v) for v in results))
        self.assertLess(abs(results[0].mean()-results[1].mean()), 4*se)

    def test_raw_rate_responds_to_loss_when_bsm_is_fast(self):
        base = TimingConfig(duration_s=3., capacity_per_bank=64, loading_ports=4, cutoff_s=1.)
        low, high = execute(base, probability=.05), execute(base, probability=.8)
        self.assertGreater(high.counts["raw_delivered"], 4*low.counts["raw_delivered"])


if __name__ == "__main__":
    unittest.main()
