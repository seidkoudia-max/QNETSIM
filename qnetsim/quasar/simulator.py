"""DES-only QUASAR receiver simulation; no controller imports or subprocesses.

Full-pass batched DES is a stochastic count approximation. Short-interval DES
uses Poisson thinning of undetectable photons, explicit arrival events,
per-detector nonparalyzable dead time and all-pairs coincidence matching.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np

from framework.core.des import DESEnv, Entity, EventHandler
from quasar.physics import prepare, resolve_window
from quasar.detector_presets import operating_warnings
from channels.polarization import mixture_fidelity

PS = 10**12
NAMES = ("Matera", "Helmos")


class EventLimitError(ValueError):
    pass


@dataclass
class Result:
    wavelength_nm: int
    metrics: list
    events: list
    summary: dict
    config: object


def match_tags(a, b, gate_ps):
    """Count all cross-station matches and separately identify true source pairs.

    Tags are (corrected_ps, pair_id, is_signal, bin_index). Negative IDs are
    background and can never identify a genuine pair. No N-by-M matrix.
    """
    a = sorted(a)
    b = sorted(b)
    bt = np.array([x[0] for x in b], dtype=np.int64)
    b_signal = np.array([x[2] for x in b], dtype=np.int64)
    prefix = np.r_[0, np.cumsum(b_signal)]
    b_pairs = {x[1]: x for x in b if x[1] >= 0}
    matches = []
    for tag, pair_id, is_signal, bucket in a:
        # Integer comparison implements an inclusive full-width gate.
        low = np.searchsorted(bt, tag-gate_ps/2, side="left")
        high = np.searchsorted(bt, tag+gate_ps/2, side="right")
        total = int(high-low)
        other_signal = int(prefix[high]-prefix[low]) if is_signal else 0
        partner = b_pairs.get(pair_id) if pair_id >= 0 else None
        genuine = int(partner is not None and abs(partner[0]-tag)*2 <= gate_ps)
        matches.append((bucket, genuine, other_signal-genuine, total-other_signal, tag))
    return matches


class Experiment(Entity):
    def __init__(self, cfg, v, wavelength, physical):
        self.cfg, self.v, self.wavelength, self.p = cfg, v, wavelength, physical
        self.detector = cfg.detector_for(wavelength)
        self.env = DESEnv("quasar_des")
        super().__init__("QUASAR", self.env)
        self.rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, wavelength]))
        self.rate = float(v["C21"])*1e6
        self.fidelity = float(v["C22"])
        self.events = []
        self.metrics = []
        self.types = Counter()
        self.pending = {}
        self.tags = [[], []]
        self.busy = np.full((2, int(self.detector.detectors)), -1, dtype=np.int64)
        self.next_id = 0
        self.candidate_count = 0
        self.accepted = np.zeros((len(physical["edges"]), 2), dtype=np.int64)
        self.dropped = np.zeros_like(self.accepted)
        self.noise = np.zeros_like(self.accepted)
        self.pair_fidelity = {}

    def schedule(self, seconds, method, *args):
        self.scheduler.schedule_at(round(seconds*PS), EventHandler(self, method, args))

    def init(self):
        if self.p.get("polarization") is not None:
            for event in self.p["polarization"].events:
                self.schedule(event["applied_epoch_s"]-self.cfg.start_s, "polarization_update", event)
        for i, start in enumerate(self.p["edges"]):
            self.schedule(float(start), "emit", i)

    def polarization_update(self, event):
        self.types["polarization_update"] += 1
        self.events.append(dict(event="polarization_update", des_time_ps=int(self.env.now),
                                station=NAMES[event["arm"]], measured_epoch_s=event["measured_epoch_s"],
                                applied_epoch_s=event["applied_epoch_s"], count=0))

    def genuine_at(self, i, emission_s=None):
        if self.p.get("polarization") is None:
            return self.fidelity
        start = self.p["edges"][i]
        duration = self.p["widths"][i]
        t = start+duration/2 if emission_s is None else emission_s
        delays = [np.interp(t, [start, start+duration], arm[i:i+2]) for arm in self.p["delay_boundaries"]]
        return float(self.p["polarization"].fidelity(self.fidelity, self.cfg.start_s+t+delays[0],
                                                    self.cfg.start_s+t+delays[1]))

    def base_row(self, i):
        g = self.p["geometry"]
        row = dict(wavelength_nm=self.wavelength, detector_technology=self.detector.technology,
                   bin_start_epoch_s=self.cfg.start_s+float(self.p["edges"][i]),
                   bin_width_s=float(self.p["widths"][i]),
                   utc_midpoint=str(g["utc"][i]),
                   minimum_elevation_deg=float(self.v["C14"]),
                   satellite_altitude_wgs84_m=float(g["altitude_wgs84_m"][i]),
                   joint_visible=bool(g["both_visible"][i]))
        for arm, key in enumerate(NAMES):
            model = self.p["arms"][arm]
            row.update({key+"_elevation_deg": float(g[key+"_elevation_deg"][i]),
                        key+"_range_m": float(g[key+"_range_m"][i]),
                        key+"_eta": float(model["eta"][i]),
                        key+"_optical_loss_db": float(model["optical_loss_db"][i]),
                        key+"_detection_loss_db": float(model["detection_loss_db"][i]),
                        key+"_rytov_diagnostic": float(model["rytov"][i]),
                        key+"_sky_rate_s": float(self.p["sky"][arm]),
                        key+"_visible": bool(g[key+"_visible"][i]),
                        key+"_diffraction_loss_db": float(model["diffraction_loss_db"][i]),
                        key+"_pointing_increment_db": float(model["pointing_increment_db"][i]),
                        key+"_atmospheric_loss_db": float(model["atmospheric_loss_db"][i]),
                        key+"_fried_r0_m_diagnostic": float(model["fried_r0_m"][i]),
                        key+"_beam_radius_m": float(model["beam_radius_m"][i]),
                        key+"_signal_candidate_rate_s": self.rate*float(model["eta"][i]),
                        key+"_received_signal_power_before_detection_W": self.rate*float(model["optical_eta"][i])*6.62607015e-34*299792458/(self.wavelength*1e-9)})
        return row

    def emit(self, i):
        self.types["source_interval"] += 1
        if self.cfg.mode == "batched":
            self.emit_batch(i)
        else:
            self.emit_photons(i)

    def probabilities(self, i):
        ea, eb = [float(x["eta"][i]) for x in self.p["arms"]]
        return ea*eb, ea*(1-eb), eb*(1-ea)

    def emit_batch(self, i):
        duration = self.p["widths"][i]
        ea, eb = [float(x["eta"][i]) for x in self.p["arms"]]
        both, only_a, only_b = self.rng.poisson(self.rate*duration*np.array(self.probabilities(i)))
        noise_raw = self.rng.poisson((np.array(self.p["sky"])+self.detector.detectors*self.detector.dark_per_detector_s)*duration)
        live = 1/(1+(self.rate*np.array([ea, eb])+np.array(self.p["sky"])+
                    self.detector.detectors*self.detector.dark_per_detector_s)*self.detector.deadtime_s/self.detector.detectors)
        la, lb = live
        shared, a_lone, b_lone, _lost = self.rng.multinomial(both, [la*lb, la*(1-lb), (1-la)*lb, (1-la)*(1-lb)])
        signal = np.array([shared+a_lone+self.rng.binomial(only_a, la),
                           shared+b_lone+self.rng.binomial(only_b, lb)])
        noise = self.rng.binomial(noise_raw, live)
        genuine = int(self.rng.binomial(shared, self.p["acceptance"]))
        self.pending[i] = dict(arrived=set(), signal=signal, noise=noise, shared=shared, true=genuine,
                               raw=np.array([both+only_a, both+only_b])+noise_raw,
                               expected=self.rate*ea*eb*self.p["acceptance"]*la*lb)
        end = self.p["edges"][i]+duration
        for arm in [0, 1]:
            self.schedule(float(end+self.p["delays"][arm][i]), "receive_batch", i, arm)

    def receive_batch(self, i, arm):
        self.types["receiver_batch"] += 1
        packet = self.pending[i]
        received = int(packet["signal"][arm]+packet["noise"][arm])
        self.accepted[i, arm] = received
        self.noise[i, arm] = packet["noise"][arm]
        self.dropped[i, arm] = packet["raw"][arm]-received
        self.events.append(dict(event="receiver_batch", des_time_ps=int(self.env.now),
                                station=NAMES[arm], bin_index=i, count=received))
        packet["arrived"].add(arm)
        if len(packet["arrived"]) == 2:
            self.schedule(self.env.now/PS, "match_batch", i)

    def match_batch(self, i):
        self.types["coincidence_batch"] += 1
        packet = self.pending.pop(i)
        sa, sb = map(int, packet["signal"])
        na, nb = map(int, packet["noise"])
        factor = self.detector.gate_s/self.p["widths"][i]
        # Exclude the diagonal same-source pairs from independent accidentals.
        other = int(self.rng.poisson(max(0, sa*sb-int(packet["shared"]))*factor))
        bg = int(self.rng.poisson((sa*nb+sb*na+na*nb)*factor))
        self.finish_row(i, packet["true"], other, bg, packet["expected"])
        self.events.append(dict(event="coincidence_batch", des_time_ps=int(self.env.now),
                                station="Matera-Helmos", bin_index=i, count=int(packet["true"])))

    def emit_photons(self, i):
        duration = float(self.p["widths"][i])
        start = float(self.p["edges"][i])
        category_counts = self.rng.poisson(self.rate*duration*np.array(self.probabilities(i)))
        noise_counts = self.rng.poisson((np.array(self.p["sky"])+self.detector.detectors*self.detector.dark_per_detector_s)*duration)
        candidates = int(2*category_counts[0]+category_counts[1]+category_counts[2]+sum(noise_counts))
        if self.candidate_count+candidates > self.cfg.max_photon_events:
            raise EventLimitError("Realized photon events exceed the safety limit. Shorten the interval; no approximation was substituted.")
        self.candidate_count += candidates
        def delay_at(arm, local_time):
            d0, d1 = self.p["delay_boundaries"][arm][i:i+2]
            return float(d0+(d1-d0)*(local_time-start)/duration)
        for count, arms in zip(category_counts, [(0, 1), (0,), (1,)]):
            times = start+self.rng.uniform(0, duration, int(count))
            for emission in times:
                pair_id = self.next_id
                self.next_id += 1
                if len(arms) == 2 and self.p.get("polarization") is not None:
                    self.pair_fidelity[pair_id] = self.genuine_at(i, float(emission))
                for arm in arms:
                    arrival = float(emission+delay_at(arm, emission))
                    tag = round((float(emission)+self.rng.normal(0, self.detector.jitter_s))*PS)
                    detector = int(self.rng.integers(int(self.detector.detectors)))
                    self.schedule(arrival, "detect", i, arm, pair_id, detector, tag, True)
        for arm, count in enumerate(noise_counts):
            for local_time in start+self.rng.uniform(0, duration, int(count)):
                # Observe a delay-shifted interval, aligned with corrected signal tags.
                arrival = float(local_time+delay_at(arm, local_time))
                tag = round((float(local_time)+self.rng.normal(0, self.detector.jitter_s))*PS)
                detector = int(self.rng.integers(int(self.detector.detectors)))
                self.schedule(arrival, "detect", i, arm, -1, detector, tag, False)

    def detect(self, i, arm, pair_id, detector, tag, is_signal):
        self.types["photon_arrival"] += 1
        now = int(self.env.now)
        accepted = now >= self.busy[arm, detector]
        if accepted:
            self.busy[arm, detector] = now+round(self.detector.deadtime_s*PS)
            self.accepted[i, arm] += 1
            self.noise[i, arm] += int(not is_signal)
            self.tags[arm].append((tag, pair_id, is_signal, i))
        else:
            self.dropped[i, arm] += 1
        self.events.append(dict(event="photon_arrival", des_time_ps=now, station=NAMES[arm],
                                bin_index=i, pair_id=pair_id, detector=detector,
                                corrected_tag_ps=tag, is_signal=is_signal, accepted=bool(accepted), count=1))

    def finish_row(self, i, genuine, other, bg, expected, genuine_fidelity_sum=None):
        duration = float(self.p["widths"][i])
        total = int(genuine+other+bg)
        row = self.base_row(i)
        f_genuine = self.genuine_at(i)
        if genuine_fidelity_sum is not None and genuine:
            f_genuine = genuine_fidelity_sum/genuine
        pol = self.p.get("polarization")
        row.update(genuine_pair_fidelity=f_genuine,
                   genuine_fidelity_sum=f_genuine*genuine,
                   polarization_enabled=pol is not None,
                   polarization_compensated=bool(pol and pol.params.compensate))
        if pol is not None:
            midpoint = self.cfg.start_s+self.p["edges"][i]+duration/2
            for arm, key in enumerate(NAMES):
                theta, delta = pol.angles(arm, midpoint+self.p["delays"][arm][i])
                row[key+"_polarization_rotation_deg"] = float(theta)
                row[key+"_polarization_retardance_deg"] = float(delta)
        row.update(true_pairs=int(genuine), other_pair_accidentals=int(other), noise_accidentals=int(bg),
                   true_pairs_per_s=genuine/duration, accidentals_per_s=(other+bg)/duration,
                   expected_true_pairs_per_s=float(expected),
                   coincidence_mixture_fidelity=mixture_fidelity(f_genuine, genuine, other+bg))
        for arm, key in enumerate(NAMES):
            row[key+"_accepted_counts"] = int(self.accepted[i, arm])
            row[key+"_noise_counts"] = int(self.noise[i, arm])
            row[key+"_deadtime_rejections"] = int(self.dropped[i, arm])
        self.metrics.append(row)

    def finish_photons(self):
        self.types["coincidence_match"] += 1
        counts = np.zeros((len(self.p["edges"]), 3), dtype=np.int64)
        fidelity_sums = np.zeros(len(counts))
        for source_tag, (bucket, true, other, noise, tag) in zip(sorted(self.tags[0]), match_tags(*self.tags, round(self.detector.gate_s*PS))):
            counts[bucket] += [true, other, noise]
            if true:
                pair_f = self.pair_fidelity.get(source_tag[1], self.fidelity)
                fidelity_sums[bucket] += pair_f
                self.events.append(dict(event="true_coincidence", des_time_ps=int(self.env.now),
                                        station="Matera-Helmos", bin_index=bucket,
                                        corrected_tag_ps=int(tag), genuine_pair_fidelity=pair_f, count=1))
        for i, (genuine, other, noise) in enumerate(counts):
            ea, eb = [float(x["eta"][i]) for x in self.p["arms"]]
            expected = self.rate*ea*eb*self.p["acceptance"]
            self.finish_row(i, genuine, other, noise, expected, fidelity_sums[i])

    def run(self):
        if self.cfg.mode == "photons":
            expected = sum((self.rate*np.array([a["eta"] for a in self.p["arms"]]).sum(axis=0)+
                            sum(self.p["sky"])+2*self.detector.detectors*self.detector.dark_per_detector_s)*self.p["widths"])
            if expected+8*np.sqrt(expected) > self.cfg.max_photon_events:
                suggested = self.cfg.duration_s*self.cfg.max_photon_events/max(expected, 1)*.8
                raise EventLimitError(f"About {expected:,.0f} arrival events expected (limit {self.cfg.max_photon_events:,}). "
                                      f"Try a duration below {suggested:.6g} s, or select batched DES for the full pass.")
        self.env.init()
        if self.cfg.mode == "photons":
            stop = self.cfg.duration_s+max(float(max(x)) for x in self.p["delay_boundaries"])+1e-6
            self.schedule(stop, "finish_photons")
        self.env.run(logging=False, summary=False)
        self.metrics.sort(key=lambda r: r["bin_start_epoch_s"])
        true = sum(r["true_pairs"] for r in self.metrics)
        acc = sum(r["other_pair_accidentals"]+r["noise_accidentals"] for r in self.metrics)
        fidelity_sum = sum(r["genuine_fidelity_sum"] for r in self.metrics)
        peak_detector_rate = max(r[name+"_accepted_counts"]/r["bin_width_s"]/self.detector.detectors
                                 for r in self.metrics for name in NAMES)
        summary = dict(backend="des_only", mode=self.cfg.mode, wavelength_nm=self.wavelength,
                       detector_technology=self.detector.technology,
                       detector_settings=self.cfg.detector_audit(self.wavelength),
                       peak_mean_counts_per_detector_s=peak_detector_rate,
                       detector_operating_warnings=operating_warnings(self.detector, peak_detector_rate),
                       scheduler="QNETSIM native DESEnv", scheduler_events=sum(self.types.values()),
                       event_types=dict(self.types), individual_photon_events=self.candidate_count,
                       candidate_detection_count=int(np.sum(self.accepted+self.dropped)),
                       true_pairs=true, accidentals=acc, mean_true_pairs_per_s=true/self.cfg.duration_s,
                       peak_binned_true_pairs_per_s=max(r["true_pairs_per_s"] for r in self.metrics),
                       genuine_pair_fidelity=fidelity_sum/true if true else None,
                       polarization_enabled=self.cfg.polarization[self.wavelength].enabled,
                       polarization_compensated=self.cfg.polarization[self.wavelength].enabled and self.cfg.polarization[self.wavelength].compensate,
                       coincidence_mixture_fidelity=(fidelity_sum+.25*acc)/(true+acc) if true+acc else None,
                       shared_visibility_s=float(sum(self.p["widths"]*self.p["geometry"]["both_visible"])),
                       accepted_counts={key:int(sum(self.accepted[:,arm])) for arm,key in enumerate(NAMES)},
                       deadtime_rejections={key:int(sum(self.dropped[:,arm])) for arm,key in enumerate(NAMES)},
                       event_time_origin="Seconds after simulation start; epoch offset stored in configuration",
                       expectation_reference="Dead-time live-fraction approximation" if self.cfg.mode=="batched" else "Ideal zero-dead-time reference")
        return Result(self.wavelength, self.metrics, self.events, summary, self.cfg)


def run(cfg, values):
    cfg.validate(values)
    cfg = resolve_window(cfg, values)
    cfg.validate(values)
    results = []
    for wavelength in cfg.wavelengths_nm:
        physical = prepare(cfg, values, wavelength)
        results.append(Experiment(cfg, values, wavelength, physical).run())
    return results
