"""Causal Ground/S3 pass study using the native event engine, without SDN."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

import des_ground_repeater_simulator as historical
from des_ground_physical import DESEnv, GroundTimingDES, OpticalSegment, PS, TimingConfig
from studies.manuscript_reproduction.s3_study_utils import mean_ci
from studies.manuscript_reproduction.s3_study_utils import read_rows

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/"data/s3_windhof_kirchberg"
OUT = ROOT/"Output/reviewer_s3_physical_timing_20260927"
COLORS = ["#306998", "#d58128", "#287e65", "#b94943"]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


class ArchivedPass:
    """Piecewise optical input; trial timestamps are independent of this grid."""
    def __init__(self, archive=ARCHIVE, step_s=.2, duration_s=1800.):
        if not math.isfinite(step_s) or step_s <= 0:
            raise ValueError("channel step must be finite and positive")
        self.step, self.duration = step_s, duration_s
        self.files = [archive/"t2_600/arguments.json", archive/"t2_600/ground_optical_trace.csv",
                      archive/"figure3_pass_geometry.csv"]
        self.args = argparse.Namespace(**json.loads(self.files[0].read_text()))
        self.optics = read_rows(self.files[1])
        self.geometry = read_rows(self.files[2])
        self.ts = np.array([r["t_s"] for r in self.optics])
        self.p = np.array([r["p_link"] for r in self.optics])
        self.f = np.array([r["fidelity"] if r["visible"] else .25 for r in self.optics])
        self.geo_t = np.array([r["t_s"] for r in self.geometry])
        self.geo_d = np.array([r["slant_range_km"]*1000 for r in self.geometry])
        fiber = historical.build_fiber_model(self.args)
        self.fiber_p, _, self.fiber_f = fiber.mean_link_metrics(length_km=self.args.effective_fiber_length_km)

    @lru_cache(maxsize=None)
    def segment(self, index):
        t = index*self.step
        # Retain the archived contact mask; interpolate transport/quality only.
        i = min(len(self.optics)-1, max(0, int(np.searchsorted(self.ts, t+1e-10, side="right")-1)))
        visible = bool(self.optics[i]["visible"])
        return OpticalSegment((index+1)*self.step,
            float(np.interp(t,self.ts,self.p)) if visible else 0.,
            float(np.interp(t,self.ts,self.f)) if visible else .25,
            float(np.interp(t,self.geo_t,self.geo_d)), visible)

    def __call__(self, link, t):
        if link == 1:
            return OpticalSegment(self.duration+1., self.fiber_p, self.fiber_f,
                                  self.args.effective_fiber_length_km*1000)
        return self.segment(math.floor(t/self.step))


def execute(config, channel):
    env = DESEnv("s3_causal_timing", default=True)
    sim = GroundTimingDES(env, config, channel)
    env.init()
    env.run(end_time=round(config.duration_s*PS), summary=False)
    sim.validate()
    return sim


def aggregate(rows, config):
    groups = {}
    for row in rows:
        groups.setdefault(row["t_s"], []).append(row)
    result = []
    for t, group in groups.items():
        row = dict(t_s=t)
        for key in ("raw_rate", "cumulative_raw", "bsm_busy_fraction", "sat_remote_occupied", "sat_ogs_occupied",
                    "fiber_remote_occupied", "fiber_ogs_occupied", "sat_repeater_mean_age_s", "fiber_repeater_mean_age_s"):
            row[key], row[key+"_ci95"] = mean_ci([r[key] for r in group])
        for t2 in config.t2_values:
            for field in (f"qualified_rate_{t2:g}", f"cumulative_qualified_{t2:g}"):
                row[field], row[field+"_ci95"] = mean_ci([r[field] for r in group])
            count = sum(r["raw_count"] for r in group)
            row[f"fidelity_{t2:g}"] = sum(r["raw_count"]*r[f"fidelity_{t2:g}"] for r in group if r["raw_count"])/count if count else float("nan")
        result.append(row)
    return result


def figures(folder, config, channel, ensembles, raw, hist, *, include_controls=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    main = ensembles["baseline"]
    x = [r["t_s"] for r in main]
    fig, axes = plt.subplots(2,2,figsize=(12,7.2))
    optics = [r for r in channel.optics if r["t_s"] <= min(config.duration_s,800.)]
    axes[0,0].plot([r["t_s"] for r in optics], [r["loss_db"] for r in optics], lw=.6, label="Sampled downlink")
    axes[0,0].plot([r["t_s"] for r in optics], [r["reference_loss_db"] for r in optics], color="black", ls="--", lw=.9, label="Reference")
    axes[0,0].axvline(377,color="gray",ls=":",lw=.8)
    axes[0,0].set(title="(a) Windhof downlink", ylabel="Loss (dB)")
    single = [r for r in raw if r["repeat"] == 0]
    axes[0,1].plot([r["t_s"] for r in single], [r["raw_rate"] for r in single], color="#b5c9d0", lw=.6, label="One counting realization")
    axes[0,1].plot(x,[r["raw_rate"] for r in main],color="#224b5c",lw=1.,label="Across-realization mean")
    axes[0,1].axvline(377,color="gray",ls=":",lw=.8)
    axes[0,1].set(title="(b) Raw endpoint-pair delivery", ylabel="Pairs/s (1 s bins)",ylim=(0,None))
    for color,t2 in zip(COLORS,config.t2_values):
        label=rf"$T_2$ = {t2:g} s"
        y=np.array([r[f"qualified_rate_{t2:g}"] for r in main])
        ci=np.array([r[f"qualified_rate_{t2:g}_ci95"] for r in main])
        axes[1,0].plot(x,y,color=color,label=label,lw=1)
        axes[1,0].fill_between(x,np.maximum(0,y-ci),y+ci,color=color,alpha=.08)
        axes[1,1].plot(x,[r[f"fidelity_{t2:g}"] for r in main],color=color,label=label,lw=.9)
    axes[1,0].set(title=f"(c) Qualified delivery, F >= {config.fidelity_threshold:g}", ylabel="Qualified pairs/s",ylim=(0,None))
    axes[1,1].set(title="(d) Conditional endpoint Bell fidelity",ylabel="Count-weighted fidelity",ylim=(.2,1))
    axes[1,1].axhline(config.fidelity_threshold,color="gray",ls=":",lw=.8)
    for ax in axes.flat:
        ax.set(xlabel="Time (s)",xlim=(0,min(config.duration_s,850.)))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.suptitle("S3 | Windhof-Kirchberg | causal loading and DES delivery",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ("png","pdf","svg"):
        fig.savefig(folder/f"s3_physical_timing.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    if not include_controls:
        return

    fig,axes=plt.subplots(2,2,figsize=(12,7.6))
    for color,t2 in zip(COLORS,config.t2_values):
        axes[0,0].plot(x,[r[f"cumulative_qualified_{t2:g}"] for r in main],color=color,label=rf"$T_2$ = {t2:g} s")
    axes[0,0].set(title="(a) Cumulative qualified deliveries",xlabel="Time (s)",ylabel="Pairs")
    for key,label in (("sat_repeater","Satellite-link OGS"),("fiber_repeater","Fiber-link OGS"),
                      ("sat_endpoint","Satellite endpoint"),("ground_endpoint","Kirchberg endpoint")):
        rows=[r for r in hist if r["bank"]==key]
        count=np.array([r["count"] for r in rows])
        if sum(count):
            axes[0,1].step([r["upper_s"] for r in rows],np.cumsum(count)/sum(count),where="post",label=label)
    axes[0,1].set(title="(b) Storage-age distributions",xlabel="Age (s)",ylabel="Cumulative fraction",xscale="log",xlim=(1e-4,config.duration_s))
    for name,label in (("baseline",f"{config.selection.upper()}, {config.bsm_service_s*1e6:g} us service"),("freshest","Freshest-first control"),
                        ("slow_bsm","50 ms BSM service"),("fine_channel","0.1 s optical interpolation")):
        if name not in ensembles:
            continue
        rows=ensembles[name]
        axes[1,0].plot([r["t_s"] for r in rows],[r["raw_rate"] for r in rows],lw=.8,label=label)
        axes[1,1].plot([r["t_s"] for r in rows],[r["bsm_busy_fraction"] for r in rows],lw=.8,label=label)
    axes[1,0].set(title="(c) Declared resource/policy controls",xlabel="Time (s)",ylabel="Raw pairs/s",xlim=(0,min(config.duration_s,850.)))
    axes[1,1].set(title="(d) BSM occupancy",xlabel="Time (s)",ylabel="Busy-time fraction",xlim=(0,min(config.duration_s,850.)),ylim=(0,1.05))
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(folder/f"s3_timing_controls.{ext}",dpi=200,bbox_inches="tight")
    plt.close(fig)


def report(folder, config, repeats, summaries, totals):
    baseline = read_rows(folder/"baseline_ensemble.csv")
    peak = max(baseline, key=lambda r:r["raw_rate"])
    max_busy = max(r["bsm_busy_fraction"] for r in baseline)
    evidence = (f"The across-realization mean in {config.metrics_dt_s:g} s bins peaks at "
                f"{peak['t_s']:g} s (bin end), at {peak['raw_rate']:.2f} raw pairs/s. "
                f"Its maximum mean BSM busy fraction is {100*max_busy:.2f}%. "
                "The peak is measured from the output, not prescribed in the scheduler.")
    trace_link = (' | <a href="baseline_events.csv">Short event trace</a>'
                  if (folder/"baseline_events.csv").exists() else '')
    rows="".join(f"<tr><td>{r['variant']}</td><td>{r['t2_s']:g}</td><td>{r['raw_mean']:,.1f}</td>"
                 f"<td>{r['qualified_mean']:,.1f} &plusmn; {r['qualified_ci95']:,.1f}</td><td>{r['fidelity']:.5f}</td></tr>" for r in totals)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S3 causal timing study</title><style>body{{max-width:1180px;margin:32px auto;padding:0 22px;font:17px/1.55 Georgia,serif;color:#203e48;background:#f8f7f2}}img{{width:100%}}td,th{{padding:8px 12px;border-bottom:1px solid #bbb}}a{{color:#07677b}}.note{{padding:18px;background:#fff1dc}}code{{font-size:14px}}</style>
<h1>S3: causal loading, finite memories and independent BSM service</h1>
<p class="note"><strong>Exploratory model, not hardware-calibrated mission performance.</strong> This separate DES-only
profile removes the implicit modes-to-Hz rule and synchronized zero-age swap batches. Hardware values below are
explicit assumptions. The historical figures and their withdrawn plateau interpretation remain unchanged.</p>
<h2>Measured outcome</h2><p>{evidence} Compare both qualified counts and fidelity below: equal
counts above one threshold do not imply equal state quality. The freshest-first control tests whether an apparent
T2 limitation is caused by waiting behind old pairs. Its effect must be reported with the selection policy.</p>
<img src="s3_physical_timing.png" alt="Downlink, raw delivery, fidelity-qualified delivery and conditional fidelity">
<p><a href="s3_physical_timing.pdf">Vector PDF</a> | <a href="s3_physical_timing.svg">SVG</a> |
<a href="summary.csv">Numerical summary</a> | <a href="provenance.json">Inputs and source hashes</a> |
<a href="checks.json">Conservation checks</a></p>
<h2>What is now causal</h2><p>Each of two elementary links has its own {config.source_rate_hz:g} Hz source opportunity budget,
round-robin divided among {config.loading_ports} independent loading ports. A port reserves one slot at each end,
emits at its next assigned pulse time, and waits for a round-trip herald before retrying. Failed attempts occupy
their slots too. Success creates an elementary pair with separate remote and OGS write timestamps.
The satellite emits for the FSO link; Windhof emits for the fiber link. This is an ideal heralded-loading primitive,
not a claimed nondestructive detector design.</p>
<p>A free BSM server takes one ready pair from each OGS bank. Destructive readout happens after its service duration;
BSM results reach the endpoints after classical propagation. Both endpoint memories are conservatively retained
until the later result notification, then recalled to define the delivered product. State dephasing uses all four
actual storage ages; remote birth times survive the swap. Failed BSMs also retain remote reservations until notification.</p>
<h2>Declared parameters</h2><ul>
<li>{config.capacity_per_bank} slots in each of four banks: two at Windhof, one on the satellite, one at Kirchberg.</li>
<li>{config.loading_ports} loading ports per link, independent of slot capacity; {config.bsm_servers} BSM server(s),
{config.bsm_service_s:g} s service each. These are illustrative resource assumptions, not inferred from the manuscript.</li>
<li>Endpoint write/read efficiencies {config.eta_write:g}/{config.eta_read:g}; independent survival lifetime
{config.lifetime_s:g} s. These per-endpoint efficiencies are newly declared assumptions, not a reinterpretation of
the old lumped memory write coefficient.</li>
<li>Pure phase memory noise with exponent {config.dephasing_exponent:g}; T2 values {list(config.t2_values)} s.
T2 does not silently change survival or read efficiency.</li>
<li>Classical/optical flight uses c in free space and c/{config.fiber_group_index:g} in fiber, plus
{config.herald_electronics_s:g} s electronics delay per notice. Elementary heralds use a conservative round trip.</li>
<li>Queued elementary pairs expire at age {config.cutoff_s:g} s, measured from the earlier write. This cutoff is fixed
across T2. Remote slots are freed only after an expiry notice. Already selected BSM operands are not cancelled mid-service.</li>
<li>Baseline {config.selection} matching; final quality threshold F &ge; {config.fidelity_threshold:g}.
The threshold classifies delivered pairs and does not feed back into scheduling.</li>
<li>Effective two-channel Bell BSM: intrinsic 1/2, detector efficiency 0.8, 50 ps RMS jitter, 200 ps full relative
window, phase visibility 0.97, per-gate dark probability 1e-6; 50 ns recovery is shorter than configured service.</li></ul>
<h2>Optical continuity and changed semantics</h2><p>The saved Windhof-Kirchberg pass and its 0.2 s sampled optical trace
are reused, with reference closest approach at 377 s and a 16.143 km fiber proxy. Range is interpolated from the saved
1 s geometry. The saved transport factors include the historical eta_src/eta_det/coupling convention; here they remain
<em>effective link-success factors</em>, not an assertion that a terminal detector absorbs a photon subsequently stored.
Separate endpoint writes are newly modeled. Input Bell states use the saved compact channel fidelities.</p>
<p>The fixed optical realization is held piecewise constant within each optical segment. Actual loading attempts,
heralds, expiry, BSM completions and delivery notifications occur at their own timestamps, not at that segment boundary.
The 0.1 s control linearly interpolates the archived transport trace; it checks sensitivity to its representation,
not convergence of a new atmospheric model. The input trajectory and weather are not independently sampled across repeats.</p>
<h2>How to interpret T2</h2><p>The same stochastic trajectory is evaluated for every T2 because coherence affects
quality rather than retrieval or the scheduling policy. Raw deliveries are therefore shared, by construction of the
declared physical separation, rather than forced to be different. Qualified rates and fidelities can differ whenever
actual storage ages are comparable with T2. Long-T2 overlap is meaningful only relative to those measured age distributions
and these resource assumptions. The pass duration is not a pair's storage time.</p>
<p>Statistics use {repeats} counting realizations per control with the same fixed optical input. Confidence intervals
are 95% Student-t intervals over counting realizations, not uncertainty in hardware, weather or mission parameters.
Empty-bin fidelity is undefined. Histograms combine all BSM operand ages, while endpoint ages refer to raw successful
deliveries; this distinction avoids silently treating survival-selected ages as all attempted ages.</p>
<table><tr><th>Variant</th><th>T2 (s)</th><th>Raw pairs</th><th>Qualified pairs &plusmn; CI</th><th>Pooled fidelity</th></tr>{rows}</table>
<img src="s3_timing_controls.png" alt="Cumulative yields, memory ages, resource controls and BSM busy fractions">
<p><a href="s3_timing_controls.pdf">Control figure PDF</a> | <a href="run_summaries.json">Per-realization counters and ages</a> |
<a href="age_histograms.csv">Baseline age histograms</a> | <a href="baseline_metrics.csv">Baseline per-realization time bins</a> |
<a href="baseline_ensemble.csv">Baseline pooled curves</a>{trace_link}.</p>
<h2>Numerical verification and remaining scope</h2><p>Constant-channel failed attempts are sampled using a geometric
first-success draw, bounded by the optical segment and the horizon. The equivalent explicit one-attempt-at-a-time
mode is retained and compared statistically in tests. This skips failed events, not physical waiting times.
Counters include completed trials and identify unfinished reservations at the horizon; source opportunities assigned
to blocked or busy ports are not admitted trials. Tests cover two-ended ownership, delayed release, nonzero ages,
independent BSM limits, metric-cadence invariance, empty delivery and T2-independent raw counts.</p>
<p>These are new protocol/resource assumptions, so numerical differences from the historical figure cannot be attributed
to one correction. The source primitive is not a multipair optical simulation, memories are not calibrated device models,
and the effective Bell BSM is not a complete optical circuit. The S3-versus-S1 claim remains withdrawn until both cases
are evaluated under matched physical resources and timing. This study supplies the causal S3 evidence needed for that work.</p>
<h2>Reproduce</h2><p><code>python -m studies.manuscript_reproduction.s3_physical_timing --out NEW_DIRECTORY --provenance {folder/'provenance.json'}</code>.</p>
<p>Ground GUI: select <strong>Physical timing S3 (DES only)</strong>. The new profile is separate from historical and SDN
controls. A GUI run uses the values shown in its own configuration panel; other Ground-form values do not silently apply.</p>
<p>Historical audit results are not included in the standalone source export.</p></html>'''
    (folder/"REPORT.html").write_text(text)


def run(folder=OUT, repeats=4, config=None, controls=True, progress=print):
    config=config or TimingConfig()
    if repeats < 2:
        raise ValueError("At least two realizations are required")
    if folder.exists():
        raise FileExistsError("Choose a fresh output directory")
    folder.mkdir(parents=True)
    channel=ArchivedPass(duration_s=config.duration_s)
    variants=[("baseline",config,.2)]
    if controls:
        variants += [("freshest",replace(config,selection="freshest"),.2),
                     ("slow_bsm",replace(config,bsm_service_s=.05),.2),
                     ("fine_channel",config,.1)]
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in channel.files}
    sources=[Path(__file__),ROOT/"des_ground_physical.py",ROOT/"quantum/physical_memory.py",
             ROOT/"quantum/bell_diagonal.py",ROOT/"quantum/optical_instruments.py",
             ROOT/"framework/core/des.py"]
    save(folder/"provenance.json",dict(config=asdict(config),repeats=repeats,
        variants={name:dict(config=asdict(c),optical_step_s=dt) for name,c,dt in variants},
        input_sha256=inputs,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        assumptions="Illustrative hardware resources; fixed archived optical realization; per-port round-robin stop-and-wait loading; ideal nondestructive herald primitive",
        scope="Exploratory physical timing; not mission-calibrated; T2 quality evaluated on the same stochastic trajectories"))
    summaries,ensembles,validation=[],{},{}
    baseline_raw,hist=[],[]
    for name,p,dt in variants:
        raw=[]
        ch=ArchivedPass(step_s=dt,duration_s=p.duration_s)
        for repeat in range(repeats):
            start=time.monotonic()
            sim=execute(replace(p,seed=p.seed+repeat),ch)
            summary=sim.summary()
            summaries.append(dict(variant=name,repeat=repeat,seed=p.seed+repeat,**summary))
            validation[f"{name}/{repeat}"]=sim.validate()
            raw.extend(dict(repeat=repeat,**r) for r in sim.metrics)
            if name=="baseline":
                for key,counts in sim.age_hist.items():
                    for i,count in enumerate(counts):
                        hist.append(dict(repeat=repeat,bank=key,lower_s=sim.age_edges[i],upper_s=sim.age_edges[i+1],count=int(count)))
            if repeat==0:
                write_csv(folder/f"{name}_events.csv",sim.trace)
                write_csv(folder/f"{name}_delivery_sample.csv",sim.delivery_sample)
            progress(f"{name} {repeat+1}/{repeats}: {summary['counts'].get('raw_delivered',0):,} raw pairs; {time.monotonic()-start:.1f} s")
            save(folder/"run_summaries.json",summaries)
            save(folder/"checks.json",validation)
        if name=="baseline":
            baseline_raw=raw
        ensembles[name]=aggregate(raw,p)
        write_csv(folder/f"{name}_metrics.csv",raw)
        write_csv(folder/f"{name}_ensemble.csv",ensembles[name])
    summary_rows=[]
    for name,_,_ in variants:
        group=[s for s in summaries if s["variant"]==name]
        raw_count=sum(s["counts"].get("raw_delivered",0) for s in group)
        for t2 in config.t2_values:
            mean,ci=mean_ci([s["quality"][str(t2)].get("qualified",0) for s in group])
            summary_rows.append(dict(variant=name,t2_s=t2,
                raw_mean=statistics.mean(s["counts"].get("raw_delivered",0) for s in group),qualified_mean=mean,qualified_ci95=ci,
                fidelity=sum(s["quality"][str(t2)].get("fidelity_sum",0) for s in group)/raw_count if raw_count else float("nan")))
    grouped={}
    for r in hist:
        key=(r["bank"],r["lower_s"],r["upper_s"])
        grouped[key]=grouped.get(key,0)+r["count"]
    hist_rows=[dict(bank=k[0],lower_s=k[1],upper_s=k[2],count=n) for k,n in grouped.items()]
    write_csv(folder/"summary.csv",summary_rows)
    write_csv(folder/"age_histograms.csv",hist_rows)
    assert inputs=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in channel.files}
    figures(folder,config,channel,ensembles,baseline_raw,hist_rows)
    report(folder,config,repeats,summaries,summary_rows)
    progress(str(folder/"REPORT.html"))
    return folder


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--duration-s",type=float,default=1800.)
    parser.add_argument("--baseline-only",action="store_true")
    parser.add_argument("--provenance",type=Path,help="Replay saved configuration and controls; inputs must still match")
    cli=parser.parse_args()
    if cli.provenance:
        saved=json.loads(cli.provenance.read_text())
        for path,digest in saved["input_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Archived input has changed: {path}")
        run(cli.out,saved["repeats"],TimingConfig(**saved["config"]),len(saved["variants"])>1)
    else:
        run(cli.out,cli.repeats,TimingConfig(duration_s=cli.duration_s),not cli.baseline_only)
