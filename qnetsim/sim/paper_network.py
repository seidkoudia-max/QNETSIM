"""Opt-in, two-ended native-DES executor for reproducible paper campaigns.

This is a synchronized-cohort protocol, not a pulse-resolved source. A cohort
reserves both memories until its herald, including unsuccessful attempts.
Existing GUI executors are intentionally unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
import numpy as np

from framework.core.des import DESEnv, Entity, EventHandler
from quantum.bell_diagonal import BellDiagonalState
from quantum.physical_memory import ErasureDephasingMemory, PhysicalMemoryParams

PS = 10**12
C = 299792458.


def keyed_rng(seed, *identity):
    key = repr((int(seed), identity)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big"))


@dataclass(frozen=True)
class NetworkConfig:
    duration_s: float = 800.
    step_s: float = .5
    capacity_per_bank: int = 256
    total_attempt_rate_s: float = 2000.
    emission_probability: float = .85
    source_fidelity: float = .99
    t2_s: float = 30.
    lifetime_s: float = 10.
    exponent: float = 1.5
    eta_write: float = .8
    eta_read: float = .8
    bsm_detector_efficiency: float = .75
    bsm_visibility: float = .97
    bsm_service_s: float = .001
    bsm_window_s: float = 1e-9
    detector_jitter_rms_s: float = 50e-12
    f_min: float = .8
    cutoff_s: float = math.inf
    policy: str = "fifo"
    state_model: str = "bell"
    late_start_s: float = 0.
    control_delay_s: float = 0.
    # A recalled photon pair at the outer interfaces is the common S1 product.
    final_recall: bool = True

    def __post_init__(self):
        if min(self.duration_s, self.step_s, self.t2_s, self.lifetime_s,
               self.total_attempt_rate_s, self.cutoff_s) <= 0:
            raise ValueError("durations, lifetimes and rates must be positive")
        if self.capacity_per_bank < 1 or self.bsm_service_s < 0 or self.control_delay_s < 0:
            raise ValueError("invalid hardware budget or delay")
        if self.state_model not in {"bell", "scalar"}:
            raise ValueError("unknown state model")
        if self.policy not in {"fifo", "cutoff", "quality", "balanced", "late"}:
            raise ValueError("unknown scheduling policy")
        for x in (self.emission_probability, self.source_fidelity, self.eta_write,
                  self.eta_read, self.bsm_detector_efficiency, self.bsm_visibility, self.f_min):
            if not 0 <= x <= 1:
                raise ValueError("probabilities must be in [0,1]")


@dataclass
class Pair:
    identity: str
    span: tuple[int, int]
    slots: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    born: tuple[float, float]
    evolved: tuple[float, float]
    state: BellDiagonalState
    scalar: float
    origin_s: float
    path_m: float
    physical_ready_s: float


def nesting_operations(segment_ends):
    """Balanced local nesting followed by chronological segment stitching."""
    ops = []
    def tree(a, b):
        if b-a <= 1:
            return
        m = (a+b)//2
        tree(a, m); tree(m, b)
        ops.append((a, m, b))
    start = 0
    for end in segment_ends:
        tree(start, end)
        if start:
            ops.append((0, start, end))
        start = end
    return ops


class PaperNetworkDES(Entity):
    """One cold-start demand stream on one reserved simple route.

    One terminal per neighboring route link, one serialized BSM per repeater.
    An endpoint bank is shared by all elementary/derived labels using that port.
    No request multiplexing or measured device calibration is implied.
    """
    def __init__(self, env, config, nodes, samples, *, seed, segment_ends=None,
                 active_windows=None, trace=False, controller=None):
        super().__init__("paper_network", env)
        self.p = config
        self.nodes = list(nodes)
        if len(set(nodes)) != len(nodes) or len(nodes) < 2:
            raise ValueError("executor requires a simple route")
        self.samples = samples
        self.seed = seed
        self.record = trace
        self.controller = controller
        self.memory = ErasureDephasingMemory(PhysicalMemoryParams(
            "effective_balanced_rail", config.eta_write, config.eta_read,
            config.t2_s, config.exponent, config.lifetime_s))
        self.ops = nesting_operations(segment_ends or [len(nodes)-1])
        self.windows = active_windows or {i: [(0., config.duration_s)] for i in range(len(nodes)-1)}
        self.free = {(i, j): deque(range(config.capacity_per_bank))
                     for i in range(len(nodes)) for j in (i-1, i+1) if 0 <= j < len(nodes)}
        self.versions = Counter()
        self.allocated = set()
        self.pools = defaultdict(deque)
        self.busy = set()
        self.counts = Counter()
        self.events = []
        self.deliveries = []
        self.bins = []
        self.control_trace = []
        self.mode_seconds = 0.
        self.integral_time = 0.
        self.max_occupancy = 0
        self.sequence = 0

    @property
    def now(self):
        return self.env.now / PS

    def schedule(self, at, method, args=(), priority=1):
        t = round(at*PS)
        if t < self.env.now:
            raise ValueError("noncausal event")
        if t <= round(self.p.duration_s*PS):
            self.counts["scheduled_events"] += 1
            self.scheduler.schedule_at(t, EventHandler(self, method, args), priority)

    def log(self, kind, **data):
        if self.record:
            self.events.append(dict(t_ps=self.env.now, order=self.sequence, kind=kind, **data))
        self.sequence += 1

    def integrate(self):
        self.mode_seconds += (self.now-self.integral_time)*len(self.allocated)
        self.integral_time = self.now

    def allocate(self, i, j):
        self.integrate()
        mode = self.free[i, j].popleft()
        key = (i, j, mode)
        self.versions[key] += 1
        slot = (i, j, mode, self.versions[key])
        self.allocated.add(slot)
        self.max_occupancy = max(self.max_occupancy, len(self.allocated))
        return slot

    def release(self, slot):
        self.integrate()
        if slot not in self.allocated:
            raise AssertionError("stale allocation or duplicate destructive release")
        self.allocated.remove(slot)
        self.free[slot[0], slot[1]].append(slot[2])

    def discard(self, pair, reason):
        for slot in pair.slots:
            self.release(slot)
        self.counts[reason] += 1
        self.log(reason, pair=pair.identity)

    def init(self):
        self.schedule(0., "tick", [0], 2)
        self.schedule(0., "metrics", [], 3)

    def evolve(self, pair):
        ages = tuple(max(0., self.now-b) for b in pair.born)
        state = self.memory.evolve_pair(pair.state, pair.evolved, ages)
        factor = math.prod(self.memory.incremental_coherence(a, b) for a, b in zip(pair.evolved, ages))
        scalar = .25 + (pair.scalar-.25)*factor
        return state, scalar, ages

    def expire(self):
        if not math.isfinite(self.p.cutoff_s) and self.p.policy != "quality":
            return
        for span, pool in self.pools.items():
            keep = deque()
            while pool:
                pair = pool.popleft()
                ages = tuple(max(0., self.now-b) for b in pair.born)
                fidelity = 1.
                if self.p.policy == "quality":
                    state, scalar, ages = self.evolve(pair)
                    fidelity = state.fidelity if self.p.state_model == "bell" else scalar
                reason = ("expiry" if max(ages) >= self.p.cutoff_s else
                          "quality_drop" if self.p.policy == "quality" and fidelity < self.p.f_min else None)
                if reason:
                    self.discard(pair, reason)
                else:
                    keep.append(pair)
            self.pools[span] = keep

    def tick(self, k):
        self.expire()
        nlinks = len(self.nodes)-1
        active = [any(a <= self.now < b for a, b in self.windows[i]) for i in range(nlinks)]
        if self.controller:
            snapshot = self.controller.poll(req_id=1, t_s=self.now)
            self.control_trace.append(dict(t_s=self.now, **snapshot))
            active = [a and str(i) in snapshot["active_labels"] for i, a in enumerate(active)]
        weights = np.full(nlinks, 1/nlinks)
        if self.p.policy == "balanced":
            # Same total source budget; route forecast, not hidden random draws.
            eta = np.array([max(1e-5, self.samples[i][k][0]) for i in range(nlinks)])
            weights = 1/eta
            weights /= weights.sum()
        for i in range(nlinks):
            if not active[i]:
                continue
            eta, distance, visible = self.samples[i][k]
            if not visible:
                continue
            rate = self.p.total_attempt_rate_s*weights[i]
            offered = math.floor((k+1)*self.p.step_s*rate)-math.floor(k*self.p.step_s*rate)
            n = min(offered, len(self.free[i, i+1]), len(self.free[i+1, i]))
            self.counts["offered"] += offered
            self.counts["blocked"] += offered-n
            self.counts["admitted"] += n
            if not n:
                continue
            edge_key = tuple(sorted(self.nodes[i:i+2]))
            success = keyed_rng(self.seed, "source_write", edge_key, k).random(n) < (
                self.p.emission_probability*eta*self.p.eta_write**2)
            allocated = [(self.allocate(i, i+1), self.allocate(i+1, i)) for _ in range(n)]
            # Emission marks are reproducible per route edge/cohort/opportunity.
            flight = distance/C
            satellite_right = self.nodes[i+1].startswith(("SAT", "OPT"))
            satellite_left = self.nodes[i].startswith(("SAT", "OPT"))
            births = ((self.now+flight, self.now) if satellite_right and not satellite_left
                      else (self.now, self.now+flight))
            self.schedule(self.now+2*flight+self.p.control_delay_s, "herald",
                          [i, k, allocated, success, births, distance, self.now], 0)
        self.dispatch()
        next_t = (k+1)*self.p.step_s
        if next_t < self.p.duration_s:
            self.schedule(next_t, "tick", [k+1], 2)

    def herald(self, i, k, allocated, successes, births, distance, origin):
        for mark, (slots, ok) in enumerate(zip(allocated, successes)):
            if not ok:
                for slot in slots:
                    self.release(slot)
                self.counts["loading_failure"] += 1
                continue
            self.counts["loaded"] += 1
            pair = Pair(f"{self.nodes[i]}:{self.nodes[i+1]}:k{k}:m{mark}", (i, i+1), slots, births, (0., 0.),
                        BellDiagonalState.werner(self.p.source_fidelity), self.p.source_fidelity,
                        origin, distance, self.now)
            self.log("load", pair=pair.identity, slots=slots, write_times=births)
            self.pools[i, i+1].append(pair)
        self.dispatch()

    def dispatch(self):
        self.expire()
        if self.p.policy == "late" and self.now < self.p.late_start_s:
            return
        for a, m, b in self.ops:
            if m in self.busy or not self.pools[a, m] or not self.pools[m, b]:
                continue
            left = self.pools[a, m].popleft()
            right = self.pools[m, b].popleft()
            self.busy.add(m)
            self.schedule(self.now+self.p.bsm_service_s, "swap", [a, m, b, left, right], 0)

    def swap(self, a, m, b, left, right):
        ls, lf, la = self.evolve(left)
        rs, rf, ra = self.evolve(right)
        er = self.memory.conditional_recall_prob(la[1])*self.memory.conditional_recall_prob(ra[0])
        gate = 1. if self.p.detector_jitter_rms_s == 0 else math.erf(
            self.p.bsm_window_s/(4*self.p.detector_jitter_rms_s))
        prob = .5*er*self.p.bsm_detector_efficiency**2*gate
        identity = hashlib.blake2b((left.identity+"|"+right.identity).encode(), digest_size=12).hexdigest()
        ok = keyed_rng(self.seed, "bsm", identity).random() < prob
        self.release(left.slots[1]); self.release(right.slots[0])
        self.busy.remove(m)
        self.counts["bsm_attempts"] += 1
        herald = max(left.path_m, right.path_m)/C+self.p.control_delay_s
        self.log("bsm", pair=identity, left=left.identity, right=right.identity, node=m,
                 success=bool(ok), probability=prob, consumed=[left.slots[1], right.slots[0]],
                 survivor_births=[left.born[0], right.born[1]], ages=[la, ra])
        if not ok:
            self.counts["bsm_failure"] += 1
            # Remote allocations stay committed until failure information arrives.
            self.schedule(self.now+herald, "failure_notice", [left.slots[0], right.slots[1]], 0)
        else:
            state = ls.swap(rs).dephase(self.p.bsm_visibility)
            scalar = .5*(1+self.p.bsm_visibility*(2*lf*rf-1))
            pair = Pair(identity, (a, b), (left.slots[0], right.slots[1]),
                        (left.born[0], right.born[1]), (la[0], ra[1]), state, scalar,
                        min(left.origin_s, right.origin_s), left.path_m+right.path_m, self.now)
            self.schedule(self.now+herald, "swap_notice", [pair], 0)
        self.dispatch()

    def failure_notice(self, left, right):
        self.release(left); self.release(right)

    def swap_notice(self, pair):
        if pair.span != (0, len(self.nodes)-1):
            self.pools[pair.span].append(pair)
        else:
            state, scalar, ages = self.evolve(pair)
            retrieval = math.prod(self.memory.conditional_recall_prob(a) for a in ages) if self.p.final_recall else 1.
            ok = keyed_rng(self.seed, "endpoint_recall", pair.identity).random() < retrieval
            for slot in pair.slots:
                self.release(slot)
            f = state.fidelity if self.p.state_model == "bell" else scalar
            self.counts["final_recall_failure" if not ok else "delivered"] += 1
            if ok:
                usable = f >= self.p.f_min and max(ages) < self.p.cutoff_s
                self.counts["usable"] += int(usable)
                self.deliveries.append(dict(time_s=self.now, pair=pair.identity, fidelity=f,
                    scalar_fidelity=scalar, bell_fidelity=state.fidelity, usable=int(usable),
                    physical_ready_s=pair.physical_ready_s, frame_known_s=self.now,
                    latency_s=self.now-pair.origin_s, write_a_s=pair.born[0], write_b_s=pair.born[1],
                    **{f"bell_{i}": p for i, p in enumerate(state.probabilities)}))
                self.log("delivery", pair=pair.identity, fidelity=f, usable=bool(usable))
        self.dispatch()

    def metrics(self):
        self.bins.append(dict(time_s=self.now, delivered=self.counts["delivered"],
            usable=self.counts["usable"], occupied=len(self.allocated),
            **{f"bank_{i}_{j}": self.p.capacity_per_bank-len(v) for (i, j), v in self.free.items()}))
        self.schedule(self.now+1., "metrics", [], 3)

    def finish(self):
        self.mode_seconds += (self.p.duration_s-self.integral_time)*len(self.allocated)
        for (i, j), free in self.free.items():
            assert len(free)+sum(s[:2] == (i, j) for s in self.allocated) == self.p.capacity_per_bank
        assert self.counts["offered"] == self.counts["admitted"]+self.counts["blocked"]
        assert self.counts["delivered"] == len(self.deliveries)
        assert self.counts["usable"] == sum(d["usable"] for d in self.deliveries)
        assert all(d["frame_known_s"] >= d["physical_ready_s"] for d in self.deliveries)
        n = len(self.deliveries)
        return dict(dict(self.counts), delivered=n, usable=self.counts["usable"],
            rate_pairs_s=self.counts["usable"]/self.p.duration_s,
            fidelity=sum(d["fidelity"] for d in self.deliveries)/n if n else None,
            latency_s=sum(d["latency_s"] for d in self.deliveries)/n if n else None,
            mode_seconds=self.mode_seconds, mode_seconds_per_usable=self.mode_seconds/self.counts["usable"] if self.counts["usable"] else None,
            unfinished_slots=len(self.allocated), max_occupancy=self.max_occupancy)
