"""QNET STBD executor with endpoint memories and Bell-diagonal state tracking.

This explicit execution profile complements the historical scalar batch profile.
It reuses the STBD planner and segmented swap dependencies. Each live pair owns
one physical slot at each endpoint. Photon-flight, herald, BSM completion and
delivery are distinct events; generation opportunities remain grouped in batches.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
import numpy as np
from des_physics_no_sdn_simulator import DESEnv, Entity, EventHandler, HAVE_QNET_DES
from des_stbd_physics_simulator import build_route_edges, build_snapshot_segmented_execution_plan, _label_windows
from quantum.bell_diagonal import BellDiagonalState
from quantum.optical_instruments import SinglePairSource, CoincidenceBSM
from quantum.physical_memory import ErasureDephasingMemory

PS=10**12
C=299792458.


@dataclass
class PhysicalPair:
    label: str
    nodes: tuple[str,str]
    ports: tuple[str,str]
    born_s: tuple[float,float]
    ages_s: tuple[float,float]
    state: BellDiagonalState
    distance_m: float
    origin_s: float


@dataclass(frozen=True)
class PhysicalExecutionParams:
    duration_s: float
    seed: int
    batch_s: float = .01
    capacity_per_node: int = 128
    policy: str = "segmented"
    final_fidelity: float = .5
    cutoff_s: float = math.inf
    control_delay_s: float = 0.
    planned_windows_only: bool = False

    def __post_init__(self):
        if self.duration_s<=0 or self.batch_s<=0 or self.capacity_per_node<2:
            raise ValueError("invalid execution duration, batch size or capacity")
        if self.policy not in {"segmented","delayed","freshest"}:
            raise ValueError("unknown memory scheduling policy")
        if not 0 <= self.final_fidelity <= 1 or self.cutoff_s<=0 or self.control_delay_s<0:
            raise ValueError("invalid acceptance, cutoff or control delay")


class PhysicalSTBDDES(Entity):
    def __init__(self, *, env, geometry, route, channel, memory: ErasureDephasingMemory,
                 source=SinglePairSource(), bsm=None, params: PhysicalExecutionParams):
        if not HAVE_QNET_DES:
            raise RuntimeError("Physical STBD study requires the real QNET event engine")
        super().__init__(name="physical_stbd",env=env)
        self.p=params;self.geometry=geometry;self.route=route;self.channel=channel
        self.memory=memory;self.source=source;self.bsm=bsm or CoincidenceBSM()
        self.rng=np.random.default_rng(params.seed)
        # Channel accepts random.Random.gauss; wrap Generator without a second seed.
        class OpticalRNG:
            def __init__(self, generator):self.generator=generator
            def gauss(self,mu,sigma):return float(self.generator.normal(mu,sigma))
        self.optical_rng=OpticalRNG(self.rng)
        self.edges=build_route_edges(route.physical_nodes)
        self.plan,self.final_label=build_snapshot_segmented_execution_plan(route,self.edges)
        self.windows=_label_windows(route,self.edges)
        neighbors=defaultdict(set)
        for edge in self.edges:
            neighbors[edge.u].add(edge.v);neighbors[edge.v].add(edge.u)
        self.port_limit={node:params.capacity_per_node//len(adj) for node,adj in neighbors.items()}
        self.occupancy=Counter();self.port_occupancy=Counter()
        self.tokens=defaultdict(deque);self.busy=set()
        self.counts=Counter();self.deliveries=[];self.trace=[];self.operations=[]
        self.max_node_occupancy=0;self.max_total_occupancy=0
        self.final_start=min(s.start_s for s in route.segments if route.physical_nodes[-1] in s.nodes)
        self.generation_index=0

    @property
    def now(self):return int(self.env.now)/PS

    def schedule(self, at_s, method, args=None, priority=1):
        time_ps=int(round(at_s*PS))
        if time_ps < int(self.env.now):raise ValueError("cannot schedule in the past")
        if time_ps <= round(self.p.duration_s*PS):
            self.scheduler.schedule_at(time_ps,EventHandler(self,method,args or []),priority=priority)
            self.counts["scheduled_events"]+=1

    def init(self):
        self.schedule(self.p.batch_s,"generate",priority=2)
        self.schedule(0.,"metrics",priority=3)

    def reserve(self,node,port,n):
        self.occupancy[node]+=n;self.port_occupancy[(node,port)]+=n
        if self.occupancy[node]>self.p.capacity_per_node or self.port_occupancy[(node,port)]>self.port_limit[node]:
            raise AssertionError("physical memory capacity exceeded")
        self.max_node_occupancy=max(self.max_node_occupancy,self.occupancy[node])
        self.max_total_occupancy=max(self.max_total_occupancy,sum(self.occupancy.values()))

    def release(self,node,port,n=1):
        self.occupancy[node]-=n;self.port_occupancy[(node,port)]-=n
        if min(self.occupancy[node],self.port_occupancy[(node,port)])<0:
            raise AssertionError("memory slot released more than once")

    def release_pair(self,pair):
        for node,port in zip(pair.nodes,pair.ports):self.release(node,port)

    def free(self,node,port):
        return min(self.p.capacity_per_node-self.occupancy[node],
                   self.port_limit[node]-self.port_occupancy[(node,port)])

    def generation_enabled(self, edge, t_s):
        """Control hook; the autonomous executor admits every visible link."""
        return True

    def swap_enabled(self, operation):
        """Control hook called only when both inputs and the BSM are free."""
        return True

    def generate(self):
        self.counts["processed_events"]+=1
        self.generation_index+=1
        end=self.generation_index*self.p.batch_s
        start=(self.generation_index-1)*self.p.batch_s
        offered=(math.floor(end*self.source.pulse_rate_hz+1e-7)
                 -math.floor(start*self.source.pulse_rate_hz+1e-7))
        t_geom=max(0.,end-self.p.batch_s/2)
        self.expire()
        for edge in self.edges:
            u,v=edge.u,edge.v
            geom=self.geometry.contact_geometry(u,v,int(round(t_geom*PS)))
            active=geom.visible
            if self.p.planned_windows_only:
                active=active and any(a<=t_geom<b for a,b in self.windows.get(edge.label,[]))
            if not active or not self.generation_enabled(edge,t_geom):continue
            self.counts["offered_pulses"]+=offered
            n=min(offered,self.free(u,v),self.free(v,u))
            self.counts["blocked_pulses"]+=offered-n
            if n==0:continue
            self.reserve(u,v,n);self.reserve(v,u,n)
            self.counts["admitted_pulses"]+=n
            isl=self.geometry.is_satellite(u) and self.geometry.is_satellite(v)
            eta=self.channel.sample_transmission(self.optical_rng,distance_m=geom.distance_m,
                elevation_deg=geom.elevation_deg,intersatellite=isl)
            p=self.source.emission_probability*eta*self.memory.p.eta_write**2
            successes=int(self.rng.binomial(n,p))
            flight=geom.distance_m/C
            # For downlinks the emitter is the satellite endpoint. Both are
            # memory endpoints; no terminal photodetector absorbs a stored photon.
            emitter=v if self.geometry.is_ground(u) and self.geometry.is_satellite(v) else u
            births=(self.now+(0. if u==emitter else flight),self.now+(0. if v==emitter else flight))
            self.schedule(self.now+2*flight+self.p.control_delay_s,"generation_ready",
                          [edge.label,u,v,n,successes,births,geom.distance_m,self.now],priority=0)
        self.dispatch()
        self.schedule((self.generation_index+1)*self.p.batch_s,"generate",priority=2)

    def generation_ready(self,label,u,v,n,successes,births,distance,origin):
        self.counts["processed_events"]+=1
        self.release(u,v,n-successes);self.release(v,u,n-successes)
        self.counts["generation_failures"]+=n-successes
        self.counts["elementary_pairs"]+=successes
        for _ in range(successes):
            pair=PhysicalPair(label,(u,v),(v,u),births,(0.,0.),self.source.state,distance,origin)
            if label==self.final_label:self.deliver(pair)
            else:self.tokens[label].append(pair)
        self.dispatch()

    def evolve(self,pair):
        ages=tuple(max(0.,self.now-t) for t in pair.born_s)
        state=self.memory.evolve_pair(pair.state,pair.ages_s,ages)
        return state,ages

    def expire(self):
        if not math.isfinite(self.p.cutoff_s):return
        for label,pool in self.tokens.items():
            kept=deque()
            while pool:
                pair=pool.popleft()
                if max(self.now-t for t in pair.born_s)>self.p.cutoff_s:
                    self.release_pair(pair);self.counts["expired_pairs"]+=1
                else:kept.append(pair)
            self.tokens[label]=kept

    def dispatch(self):
        self.expire()
        if self.p.policy=="delayed" and self.now<self.final_start:return
        for op in self.plan:
            if op.middle in self.busy:continue
            if not self.tokens[op.left_label] or not self.tokens[op.right_label]:continue
            if not self.swap_enabled(op):continue
            take="pop" if self.p.policy=="freshest" else "popleft"
            left=getattr(self.tokens[op.left_label],take)()
            right=getattr(self.tokens[op.right_label],take)()
            assert set(left.nodes)&set(right.nodes)=={op.middle}
            self.busy.add(op.middle)
            ls,la=self.evolve(left);rs,ra=self.evolve(right)
            il=left.nodes.index(op.middle);ir=right.nodes.index(op.middle)
            el=self.memory.conditional_recall_prob(la[il]);er=self.memory.conditional_recall_prob(ra[ir])
            probability,state=self.bsm.output(ls,rs,el,er)
            ok=bool(self.rng.random()<probability)
            self.counts["bsm_attempts"]+=1
            self.schedule(self.now+self.bsm.p.service_time_s,"swap_complete",
                          [op,left,right,la,ra,state,ok,self.now],priority=0)

    def swap_complete(self,op,left,right,la,ra,state,ok,started):
        self.counts["processed_events"]+=1
        self.busy.remove(op.middle)
        il=left.nodes.index(op.middle);ir=right.nodes.index(op.middle)
        self.release(op.middle,left.ports[il]);self.release(op.middle,right.ports[ir])
        self.operations.append({"node":op.middle,"start_s":started,"complete_s":self.now,
                                "success":int(ok),"output":op.out_label})
        if not ok:
            self.release(left.nodes[1-il],left.ports[1-il])
            self.release(right.nodes[1-ir],right.ports[1-ir])
            self.counts["bsm_failures"]+=1
        else:
            self.counts["bsm_heralds"]+=1
            pair=PhysicalPair(op.out_label,(left.nodes[1-il],right.nodes[1-ir]),
                (left.ports[1-il],right.ports[1-ir]),(left.born_s[1-il],right.born_s[1-ir]),
                (la[1-il],ra[1-ir]),state,left.distance_m+right.distance_m,min(left.origin_s,right.origin_s))
            herald=max(left.distance_m,right.distance_m)/C+self.p.control_delay_s
            self.schedule(self.now+herald,"swap_ready",[pair],priority=0)
        self.dispatch()

    def swap_ready(self,pair):
        self.counts["processed_events"]+=1
        if pair.label==self.final_label:self.deliver(pair)
        else:self.tokens[pair.label].append(pair)
        self.dispatch()

    def deliver(self,pair):
        state,ages=self.evolve(pair)
        # Destructive retrieval of the two outer memories is counted once.
        # Middle memories were read during swaps; these outer memories were not.
        retrieval=math.prod(self.memory.conditional_recall_prob(age) for age in ages)
        self.release_pair(pair)
        self.counts["final_heralds"]+=1
        if self.rng.random()>=retrieval:
            self.counts["final_retrieval_failures"]+=1;return
        if state.fidelity < self.p.final_fidelity:
            self.counts["final_threshold_rejections"]+=1;return
        self.counts["accepted_pairs"]+=1
        self.deliveries.append({"time_s":self.now,"fidelity":state.fidelity,
            "teleportation_fidelity":state.teleportation_fidelity,"chsh_max":state.chsh_max,
            "latency_s":self.now-pair.origin_s,"oldest_endpoint_age_s":max(ages),
            **{f"bell_{i}":value for i,value in enumerate(state.probabilities)}})

    def metrics(self):
        self.counts["processed_events"]+=1
        self.trace.append({"t_s":self.now,**dict(self.counts),"memory_slots":sum(self.occupancy.values()),
                           **{f"memory_{node}":value for node,value in self.occupancy.items()}})
        self.schedule(self.now+1.,"metrics",priority=3)

    def validate_accounting(self):
        assert self.counts["accepted_pairs"]==len(self.deliveries)
        assert self.counts["final_heralds"]==sum(self.counts[k] for k in
            ("accepted_pairs","final_retrieval_failures","final_threshold_rejections"))
        assert self.counts["offered_pulses"]==self.counts["admitted_pulses"]+self.counts["blocked_pulses"]
        assert 0<=self.max_node_occupancy<=self.p.capacity_per_node
        # Unfinished reservations may remain at the finite observation boundary.
        assert all(v>=0 for v in self.occupancy.values())
        last={}
        for op in self.operations:
            assert op["start_s"]>=last.get(op["node"],-1)-1e-12
            assert op["complete_s"]>=op["start_s"]+self.bsm.p.service_time_s-1e-12
            last[op["node"]]=op["complete_s"]
