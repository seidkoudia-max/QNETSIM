"""Opt-in S3 timing executor: endpoint reservations, heralds and serialized BSMs.

Source pulses are round-robin partitioned among independent loading ports.
An idle port reserves a slot at both endpoints and retries until success. Failed
trials within one constant optical segment can be skipped geometrically; their
count and last emission/completion times are retained. This is exact for the
declared piecewise-constant channel and periodic stop-and-wait port protocol,
not a multiphoton optical circuit or a continuous calibrated atmosphere.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import random

import numpy as np

from framework.core.des import DESEnv, Entity, EventHandler
from quantum.bell_diagonal import BellDiagonalState
from quantum.optical_instruments import CoincidenceBSM, PhysicalBSMParams, ThresholdDetector
from quantum.physical_memory import ErasureDephasingMemory, PhysicalMemoryParams

PS = 10**12
C = 299792458.


@dataclass(frozen=True)
class TimingConfig:
    duration_s: float = 1800.
    source_rate_hz: float = 5e6  # Per elementary link, before round-robin admission.
    capacity_per_bank: int = 512  # Four banks: two OGS, satellite and Kirchberg.
    loading_ports: int = 16  # Per link; independent of memory capacity.
    bsm_servers: int = 1
    bsm_service_s: float = 1e-4
    herald_electronics_s: float = 1e-4
    fiber_group_index: float = 1.468
    eta_write: float = .9  # Per endpoint, independently sampled at loading.
    eta_read: float = .9  # Per endpoint, independently sampled at destructive readout.
    lifetime_s: float = 400.
    t2_values: tuple = (.6, 6., 60., 600.)
    dephasing_exponent: float = 1.5
    fidelity_threshold: float = .8
    cutoff_s: float = 10.  # Queued elementary-pair age cutoff, independent of T2.
    selection: str = "fifo"
    metrics_dt_s: float = 1.
    accelerate_failures: bool = True
    seed: int = 12345
    trace_start_s: float = 300.
    trace_end_s: float = 302.

    def __post_init__(self):
        for name in ("duration_s", "source_rate_hz", "bsm_service_s", "metrics_dt_s", "lifetime_s",
                     "fiber_group_index", "dephasing_exponent", "cutoff_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("capacity_per_bank", "loading_ports", "bsm_servers"):
            value = getattr(self, name)
            if value < 1 or int(value) != value:
                raise ValueError(f"{name} must be a positive integer")
        if self.loading_ports > self.capacity_per_bank:
            raise ValueError("loading ports require enough memory slots")
        if not math.isfinite(self.herald_electronics_s) or self.herald_electronics_s < 0:
            raise ValueError("herald electronics delay must be finite and nonnegative")
        if not self.t2_values or any(not math.isfinite(t) or t <= 0 for t in self.t2_values):
            raise ValueError("T2 values must be finite and positive")
        if len(set(self.t2_values)) != len(self.t2_values):
            raise ValueError("T2 values must be unique")
        for name in ("eta_write", "eta_read", "fidelity_threshold"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.selection not in {"fifo", "freshest"}:
            raise ValueError("unknown selection policy")
        if self.bsm_service_s < 50e-9:
            raise ValueError("service must cover the effective detector's 50 ns recovery")


@dataclass(frozen=True)
class OpticalSegment:
    end_s: float
    probability: float
    fidelity: float
    distance_m: float
    visible: bool = True


@dataclass
class Pair:
    identity: int
    link: int
    remote_born: float
    repeater_born: float
    state: BellDiagonalState
    flight_s: float
    status: str = "ready"


class GroundTimingDES(Entity):
    """Two-link, four-bank model; final product is a recalled endpoint pair.

    Loading trials include optical transport and two endpoint writes. Both slots
    stay reserved through failed-trial heralds. A ready pair keeps its remote
    slot until the BSM result arrives. T2 changes conditional quality only: the
    same stochastic trajectory is evaluated at all T2 values without changing
    admission, read survival, expiry or scheduling in response to quality.
    """
    def __init__(self, env, config: TimingConfig, channel):
        super().__init__("ground_physical_timing", env)
        self.p, self.channel = config, channel
        # Native queue drives events; disable optional growing history lists.
        self.agenda = self.signed_events = deque(maxlen=0)
        env.future_events = deque(maxlen=0)
        self.rng = random.Random(config.seed)
        self.memory = ErasureDephasingMemory(PhysicalMemoryParams(
            "illustrative_S3", config.eta_write, config.eta_read, 1.,
            config.dephasing_exponent, config.lifetime_s))
        self.bsm = CoincidenceBSM(PhysicalBSMParams(phase_visibility=.97,
            coincidence_window_s=200e-12, service_time_s=config.bsm_service_s),
            ThresholdDetector(efficiency=.8, jitter_sigma_s=50e-12,
                              dark_rate_hz=-math.log1p(-1e-6)/200e-12))
        self.counts = Counter()
        self.occupied = np.zeros((2, 2), dtype=int)  # [link, remote/repeater]
        self.peak_occupied = np.zeros((2, 2), dtype=int)
        self.integral_occupied = np.zeros((2, 2), dtype=float)
        self.last_integral = 0.
        self.busy = 0
        self.integral_busy = 0.
        self.pending = {}  # One reserved trial group per port.
        self.waking = set()
        self.pools = [dict(), dict()]  # Insertion-ordered ready pairs.
        self.in_service = {}
        self.in_notice = {}
        self.sequence = 0
        self.quality = {t: Counter() for t in config.t2_values}
        self.last_counts = Counter()
        self.last_quality = {t: Counter() for t in config.t2_values}
        self.metrics = []
        self.trace = []
        self.delivery_sample = []
        self.age_edges = np.r_[0., np.geomspace(1e-6, config.duration_s*2, 100)]
        self.age_hist = {k: np.zeros(len(self.age_edges)-1, dtype=int)
                         for k in ("sat_repeater", "fiber_repeater", "sat_endpoint", "ground_endpoint")}
        self.age_sums = Counter()
        self.last_age_sums = Counter()

    @property
    def now(self):
        return self.env.now/PS

    def schedule(self, at, method, params=(), priority=0):
        when = round(at*PS)
        if when < self.env.now:
            raise AssertionError("noncausal event")
        if when <= round(self.p.duration_s*PS):
            self.scheduler.schedule_at(when, EventHandler(self, method, params), priority)
            self.counts["scheduled_events"] += 1

    def integrate(self):
        dt = self.now-self.last_integral
        self.integral_occupied += dt*self.occupied
        self.integral_busy += dt*self.busy
        self.last_integral = self.now

    def change_occupancy(self, link, side, delta):
        self.integrate()
        self.occupied[link, side] += delta
        if not 0 <= self.occupied[link, side] <= self.p.capacity_per_bank:
            raise AssertionError("memory capacity or ownership violation")
        self.peak_occupied = np.maximum(self.peak_occupied, self.occupied)

    def log(self, kind, **data):
        if self.p.trace_start_s <= self.now <= self.p.trace_end_s:
            self.trace.append(dict(t_s=self.now, event=kind, **data))

    def init(self):
        for link in (0, 1):
            for port in range(self.p.loading_ports):
                self.wake(link, port, 0.)
        self.schedule(min(self.p.metrics_dt_s, self.p.duration_s), "measure", priority=3)

    def wake(self, link, port, at=None):
        key = link, port
        if key not in self.pending and key not in self.waking:
            self.waking.add(key)
            self.schedule(self.now if at is None else at, "load", [link, port], priority=1)

    def wake_idle(self, link):
        if max(self.occupied[link]) < self.p.capacity_per_bank:
            for port in range(self.p.loading_ports):
                self.wake(link, port)

    def load(self, link, port):
        key = link, port
        self.waking.discard(key)
        period = self.p.loading_ports/self.p.source_rate_hz
        phase = (port+.5)/self.p.source_rate_hz
        emission = phase + max(0, math.ceil((self.now-phase)/period-1e-10))*period
        emission = max(self.now, emission)
        segment = self.channel(link, emission)
        if not segment.visible:
            if key in self.pending:
                del self.pending[key]
                self.change_occupancy(link, 0, -1)
                self.change_occupancy(link, 1, -1)
            self.wake(link, port, segment.end_s)
            return
        if key not in self.pending:
            if max(self.occupied[link]) >= self.p.capacity_per_bank:
                return
            self.pending[key] = 0
            self.change_occupancy(link, 0, 1)
            self.change_occupancy(link, 1, 1)
        if segment.end_s <= emission or not 0 <= segment.probability <= 1:
            raise ValueError("invalid optical segment")
        flight = segment.distance_m/C*(1. if link == 0 else self.p.fiber_group_index)
        delay = 2*flight+self.p.herald_electronics_s
        cycle = max(1, math.ceil(delay/period))*period
        # Only group emissions within the same optical segment and completed by
        # the horizon. Every skipped trial occupies its slots through its herald.
        last_emission = min(np.nextafter(segment.end_s, -math.inf), self.p.duration_s-delay)
        available = max(0, math.floor((last_emission-emission)/cycle)+1)
        if not self.p.accelerate_failures:
            available = min(available, 1)
        if available == 0:
            return  # Reserved unfinished trial at the horizon, explicitly counted.
        probability = segment.probability*self.p.eta_write**2
        k = (1 if probability >= 1 else
             math.floor(math.log1p(-self.rng.random())/math.log1p(-probability))+1 if probability > 0 else available+1)
        n, ok = min(k, available), k <= available
        last = emission+(n-1)*cycle
        self.pending[key] = n
        self.schedule(last+delay, "loading_notice", [link, port, n, ok, last, flight, segment.fidelity])

    def loading_notice(self, link, port, n, ok, emission, flight, fidelity):
        self.pending[link, port] = 0
        self.counts[f"trials_{link}"] += n
        self.counts[f"loading_failures_{link}"] += n-int(ok)
        if not ok:
            self.load(link, port)
            return
        del self.pending[link, port]
        self.counts[f"loaded_{link}"] += 1
        self.sequence += 1
        remote_birth = emission if link == 0 else emission+flight
        repeater_birth = emission+flight if link == 0 else emission
        pair = Pair(self.sequence, link, remote_birth, repeater_birth, BellDiagonalState.werner(fidelity), flight)
        self.pools[link][pair.identity] = pair
        self.log("load_herald", link=link, pair=pair.identity, emission_s=emission,
                 remote_write_s=remote_birth, repeater_write_s=repeater_birth)
        expiry = min(remote_birth, repeater_birth)+self.p.cutoff_s
        if round(expiry*PS) <= self.env.now:
            # An expired herald must not briefly become an eligible BSM operand.
            self.expire(pair)
            self.wake(link, port)
            return
        self.schedule(expiry, "expire", [pair], priority=-1)
        self.dispatch()
        self.wake(link, port)

    def expire(self, pair):
        if pair.status != "ready":
            return
        self.pools[pair.link].pop(pair.identity)
        pair.status = "expired"
        self.counts[f"expired_{pair.link}"] += 1
        self.change_occupancy(pair.link, 1, -1)
        # Remote release waits for an expiry notification; never instantaneous.
        self.in_notice[-pair.identity] = (pair,)
        self.schedule(self.now+pair.flight_s+self.p.herald_electronics_s, "expiry_notice", [pair])
        self.wake_idle(pair.link)

    def expiry_notice(self, pair):
        del self.in_notice[-pair.identity]
        self.change_occupancy(pair.link, 0, -1)
        self.wake_idle(pair.link)

    def dispatch(self):
        while self.busy < self.p.bsm_servers and self.pools[0] and self.pools[1]:
            selected = []
            for pool in self.pools:
                key = next(iter(pool)) if self.p.selection == "fifo" else next(reversed(pool))
                pair = pool.pop(key)
                pair.status = "service"
                selected.append(pair)
            self.integrate()
            self.busy += 1
            left, right = selected
            self.in_service[left.identity] = selected
            self.counts["bsm_started"] += 1
            self.schedule(self.now+self.p.bsm_service_s, "bsm_complete", [left, right, self.now])

    def record_age(self, key, age):
        if age < 0:
            raise AssertionError("negative storage age")
        index = min(len(self.age_edges)-2, max(0, np.searchsorted(self.age_edges, age, side="right")-1))
        self.age_hist[key][index] += 1
        self.age_sums[key] += age
        self.age_sums[key+"_n"] += 1

    def bsm_complete(self, left, right, started):
        del self.in_service[left.identity]
        self.integrate()
        self.busy -= 1
        ages = [self.now-p.repeater_born for p in (left, right)]
        for key, age in zip(("sat_repeater", "fiber_repeater"), ages):
            self.record_age(key, age)
        recalls = [int(self.rng.random() < self.memory.conditional_recall_prob(age)) for age in ages]
        self.counts["bsm_completed"] += 1
        self.counts["both_recalled"] += int(all(recalls))
        true, false = self.bsm.probabilities(*recalls)
        heralded = self.rng.random() < true+false
        self.counts["bsm_heralded"] += int(heralded)
        self.counts["bsm_failed"] += int(not heralded)
        for pair in (left, right):
            pair.status = "notice"
            self.change_occupancy(pair.link, 1, -1)
        self.in_notice[left.identity] = left, right
        delay = max(left.flight_s, right.flight_s)+self.p.herald_electronics_s
        self.schedule(self.now+delay, "delivery_notice", [left, right, self.now, recalls, heralded])
        self.log("bsm_complete", left=left.identity, right=right.identity, started_s=started,
                 sat_age_s=ages[0], fiber_age_s=ages[1], heralded=int(heralded))
        for link in (0, 1):
            self.wake_idle(link)
        self.dispatch()

    def fidelity_at_delivery(self, left, right, bsm_time, recalls, t2):
        memory = ErasureDephasingMemory(PhysicalMemoryParams("S3", self.p.eta_write,
            self.p.eta_read, t2, self.p.dephasing_exponent, self.p.lifetime_s))
        states = []
        for p in (left, right):
            state = p.state.dephase(memory.coherence_factor(bsm_time-p.remote_born)*
                                    memory.coherence_factor(bsm_time-p.repeater_born))
            states.append(state)
        _, state = self.bsm.output(*states, *recalls)
        extra = math.prod(memory.incremental_coherence(bsm_time-p.remote_born, self.now-p.remote_born)
                          for p in (left, right))
        return state.dephase(extra)

    def delivery_notice(self, left, right, bsm_time, recalls, heralded):
        del self.in_notice[left.identity]
        if heralded:
            self.counts["herald_notices"] += 1
            read = [self.rng.random() < self.memory.conditional_recall_prob(self.now-p.remote_born)
                    for p in (left, right)]
            if all(read):
                self.counts["raw_delivered"] += 1
                for key, pair in zip(("sat_endpoint", "ground_endpoint"), (left, right)):
                    self.record_age(key, self.now-pair.remote_born)
                for t2 in self.p.t2_values:
                    state = self.fidelity_at_delivery(left, right, bsm_time, recalls, t2)
                    self.quality[t2]["fidelity_sum"] += state.fidelity
                    self.quality[t2]["qualified"] += int(state.fidelity >= self.p.fidelity_threshold)
                    self.quality[t2]["above_half"] += int(state.fidelity > .5)
                self.log("delivery", left=left.identity, right=right.identity,
                         sat_age_s=self.now-left.remote_born, ground_age_s=self.now-right.remote_born)
                if self.p.trace_start_s <= self.now <= self.p.trace_end_s:
                    self.delivery_sample.append(dict(t_s=self.now, sat_age_s=self.now-left.remote_born,
                        ground_age_s=self.now-right.remote_born,
                        **{f"fidelity_t2_{t:g}":self.fidelity_at_delivery(left,right,bsm_time,recalls,t).fidelity
                           for t in self.p.t2_values}))
            else:
                self.counts["endpoint_read_failures"] += 1
        for p in (left, right):
            self.change_occupancy(p.link, 0, -1)
            p.status = "finished"
        for link in (0, 1):
            self.wake_idle(link)

    def measure(self):
        self.integrate()
        previous = self.metrics[-1]["t_s"] if self.metrics else 0.
        dt = self.now-previous
        raw = self.counts["raw_delivered"]-self.last_counts["raw_delivered"]
        row = dict(t_s=self.now, bin_s=dt, raw_count=raw, raw_rate=raw/dt,
                   cumulative_raw=self.counts["raw_delivered"],
                   bsm_attempts=self.counts["bsm_completed"]-self.last_counts["bsm_completed"],
                   bsm_busy_fraction=(self.integral_busy-self.last_counts["busy_integral"])/dt/self.p.bsm_servers)
        for link, name in ((0,"sat"),(1,"fiber")):
            for side, label in ((0,"remote"),(1,"ogs")):
                row[f"{name}_{label}_occupied"] = int(self.occupied[link,side])
            row[f"{name}_ready"] = len(self.pools[link])
            row[f"{name}_loading_trials"] = self.counts[f"trials_{link}"]-self.last_counts[f"trials_{link}"]
            row[f"{name}_loaded"] = self.counts[f"loaded_{link}"]-self.last_counts[f"loaded_{link}"]
        for key in self.age_hist:
            n = self.age_sums[key+"_n"]-self.last_age_sums[key+"_n"]
            row[key+"_mean_age_s"] = (self.age_sums[key]-self.last_age_sums[key])/n if n else float("nan")
        for t2 in self.p.t2_values:
            q, prev = self.quality[t2], self.last_quality[t2]
            row[f"qualified_rate_{t2:g}"] = (q["qualified"]-prev["qualified"])/dt
            row[f"fidelity_{t2:g}"] = (q["fidelity_sum"]-prev["fidelity_sum"])/raw if raw else float("nan")
            row[f"cumulative_qualified_{t2:g}"] = q["qualified"]
            self.last_quality[t2] = q.copy()
        self.last_counts = self.counts.copy()
        self.last_counts["busy_integral"] = self.integral_busy
        self.last_age_sums = self.age_sums.copy()
        self.metrics.append(row)
        if self.now < self.p.duration_s:
            self.schedule(min(self.p.duration_s, self.now+self.p.metrics_dt_s), "measure", priority=3)

    def validate(self):
        for link in (0,1):
            assert self.counts[f"trials_{link}"] == self.counts[f"loading_failures_{link}"]+self.counts[f"loaded_{link}"]
            pending = sum(key[0] == link for key in self.pending)
            service = sum(p.link == link for pair in self.in_service.values() for p in pair)
            notice = sum(p.link == link for pair in self.in_notice.values() for p in pair)
            assert self.occupied[link,1] == pending+len(self.pools[link])+service
            assert self.occupied[link,0] == self.occupied[link,1]+notice
            assert self.counts[f"loaded_{link}"] == (len(self.pools[link])+self.counts["bsm_started"]+
                                                     self.counts[f"expired_{link}"])
        assert self.counts["bsm_started"] == self.counts["bsm_completed"]+len(self.in_service)
        assert self.counts["bsm_completed"] == self.counts["bsm_heralded"]+self.counts["bsm_failed"]
        assert self.counts["herald_notices"] == self.counts["raw_delivered"]+self.counts["endpoint_read_failures"]
        assert sum(m["raw_count"] for m in self.metrics) == self.counts["raw_delivered"]
        assert all(q["qualified"] <= self.counts["raw_delivered"] for q in self.quality.values())
        assert np.all(self.peak_occupied <= self.p.capacity_per_bank)
        assert 0 <= self.integral_busy <= self.p.duration_s*self.p.bsm_servers+1e-8
        return dict(ownership=True, count_balance=True, capacities=True, metrics=True,
                    pending_trial_groups=len(self.pending), unfinished_bsms=len(self.in_service),
                    unfinished_notices=len(self.in_notice), unfinished_ready=sum(map(len,self.pools)),
                    unfinished_reservations=self.occupied.tolist())

    def summary(self):
        self.validate()
        n = self.counts["raw_delivered"]
        return dict(counts=dict(self.counts), quality={str(t):dict(q,
            mean_fidelity=q["fidelity_sum"]/n if n else None) for t,q in self.quality.items()},
            occupancy_time_average=(self.integral_occupied/self.p.duration_s).tolist(),
            bsm_busy_fraction=self.integral_busy/self.p.duration_s/self.p.bsm_servers,
            storage_age_means={k:self.age_sums[k]/self.age_sums[k+"_n"] if self.age_sums[k+"_n"] else None
                               for k in self.age_hist}, checks=self.validate())
