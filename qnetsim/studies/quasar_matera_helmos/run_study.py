"""Conditional dual-downlink study. No changes to existing QNETSIM scenarios.

The orbital elements are not a TLE. Propagate a circular two-body orbit directly,
and explicitly compare two possible element reference frames. All link/noise
outputs are expected values, not measured data or individual photon DES events.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "Output" / "quasar_matera_helmos"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import erf
from skyfield.api import load, wgs84
from skyfield.constants import AU_M
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric

from channels.fso import FSOAnalyticalChannelModel, FSOParams

MU = 3.986004418e14
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
J2 = 1.08262668e-3
R_EARTH = 6378137.0


def inputs():
    data = json.loads((HERE / "inputs.json").read_text())
    return data, {row["cell"]: row["value"] for row in data["rows"]}


def stations(v):
    return [dict(name=name, latitude=float(v[f"C{row}"]), longitude=float(v[f"C{row+1}"]),
                 altitude_m=float(v[f"C{row+2}"]), diameter_m=float(v[f"C{row+3}"]),
                 obscuration_area=float(v[f"C{row+4}"]))
            for name, row in [("Matera", 3), ("Helmos", 9)]]


def rotation_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def orbit(v, seconds, frame="J2000", include_j2=False):
    epoch = dt.datetime.fromisoformat(v["C33"]).replace(tzinfo=dt.timezone.utc)
    ts = load.timescale(builtin=True)
    epoch_time = ts.from_datetime(epoch)
    times = ts.from_datetimes([epoch + dt.timedelta(seconds=float(s)) for s in seconds])
    a = float(v["C26"]) * 1000
    n = np.sqrt(MU / a**3)
    inc, raan, omega, nu = np.radians([v["C30"], v["C27"], v["C31"], v["C32"]])
    tilt = np.array([[1, 0, 0], [0, np.cos(inc), -np.sin(inc)], [0, np.sin(inc), np.cos(inc)]])
    u = omega + nu + n * np.asarray(seconds)
    xyz = rotation_z(raan) @ tilt @ (a * np.array([np.cos(u), np.sin(u), np.zeros_like(u)]))
    if frame == "MOD":
        # P maps J2000 to mean equator/equinox of date. Freeze at element epoch.
        xyz = epoch_time.P.T @ xyz
    elif frame != "J2000":
        raise ValueError(frame)
    if include_j2:
        u0 = omega+nu
        r0 = rotation_z(raan) @ tilt @ (a*np.array([np.cos(u0),np.sin(u0),0]))
        vel0 = rotation_z(raan) @ tilt @ (a*n*np.array([-np.sin(u0),np.cos(u0),0]))
        if frame == "MOD":
            r0,vel0 = epoch_time.P.T@r0,epoch_time.P.T@vel0
        pole_rotation = epoch_time.P
        def derivative(_t, state):
            r = state[:3]
            radius = np.linalg.norm(r)
            x,y,z = pole_rotation@r
            z2 = (z/radius)**2
            factor = 1.5*J2*MU*R_EARTH**2/radius**5
            aj2 = pole_rotation.T @ (factor*np.array([x*(5*z2-1),y*(5*z2-1),z*(5*z2-3)]))
            return np.r_[state[3:],-MU*r/radius**3+aj2]
        if min(seconds)<0:
            raise ValueError("J2 scan is forward from epoch only.")
        solution = solve_ivp(derivative,(0,float(max(seconds))),np.r_[r0,vel0],
                             t_eval=seconds,rtol=1e-10,atol=1e-6,method="DOP853")
        if not solution.success:
            raise RuntimeError(solution.message)
        xyz = solution.y[:3]
    sat = Geocentric(xyz / AU_M, t=times, center=399)
    out = {"t_s": np.asarray(seconds, float), "utc": np.array([x.isoformat() for x in times.utc_datetime()]),
           "radius_m": np.linalg.norm(xyz, axis=0),
           "altitude_wgs84_m": wgs84.height_of(sat).m}
    ecef = sat.frame_xyz(itrs).m
    for gs in stations(v):
        observer = wgs84.latlon(gs["latitude"], gs["longitude"], elevation_m=gs["altitude_m"])
        altitude, azimuth, distance = (sat - observer.at(times)).altaz()
        key = gs["name"]
        out[key + "_elevation_deg"] = altitude.degrees
        out[key + "_azimuth_deg"] = azimuth.degrees
        out[key + "_range_m"] = distance.m
        out[key + "_visible"] = altitude.degrees >= float(v["C14"])
        # Independent geometric elevation check in the rotating frame.
        lat, lon = np.radians([gs["latitude"], gs["longitude"]])
        up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        diff = ecef - observer.at(times).frame_xyz(itrs).m
        direct = np.degrees(np.arcsin(up @ diff / np.linalg.norm(diff, axis=0)))
        assert np.max(np.abs(direct - altitude.degrees)) < 1e-6
    out["both_visible"] = out["Matera_visible"] & out["Helmos_visible"]
    return out


def hv_cn2(h, a, wind, base_height=0.0):
    """Hufnagel-Valley profile; local surface term can be shifted as sensitivity."""
    return (0.00594 * (wind/27)**2 * (1e-5*h)**10 * np.exp(-h/1000)
            + 2.7e-16*np.exp(-h/1500) + a*np.exp(-(h-base_height)/100))


def turbulence_integrals(gs, wavelength, v, local_surface=False):
    h0 = gs["altitude_m"]
    a = float(str(v["C36"]).replace(" ", ""))
    wind = float(v["C37"])
    profile = lambda h: hv_cn2(h, a, wind, h0 if local_surface else 0.0)
    # Scale the small integrands to avoid default quadrature absolute tolerances.
    def integrate(power):
        return quad(lambda z: profile(z+h0)*1e14*z**power, 0, 30000-h0,
                    epsabs=1e-8, epsrel=1e-8, points=[100, 500, 1000, 5000])[0]*1e-14
    k = 2*np.pi/wavelength
    return 2.25*k**(7/6)*integrate(5/6), (0.423*k*k*integrate(0))**(-3/5)


def link(gs, g, wavelength, v, *, m2=1.0, zenith_transmission=0.90,
         eta_tx=0.80, eta_rx=0.50, eta_detector=0.60, pointing_scale=1.0):
    """Gaussian bucket collection with annular receiver and averaged TX jitter.

    Reuses QNETSIM's Gaussian collection primitive, NOT its SNR/noise proxies.
    Diameter equals 2*w0. This is an ideal untruncated equivalent Gaussian.
    Jitter is assumed independent zero-mean Gaussian with 1 urad per axis.
    """
    key = gs["name"]
    r = np.asarray(g[key+"_range_m"])
    e = np.asarray(g[key+"_elevation_deg"])
    vis = np.asarray(g[key+"_visible"])
    tx_row = 17 if key == "Matera" else 19
    dtx = float(v[f"C{tx_row}"])
    sigma = float(v[f"C{tx_row+1}"]) * 1e-6 * pointing_scale
    w0 = dtx/2
    divergence = m2*wavelength/(np.pi*w0)
    divergence_effective = np.sqrt(divergence**2+4*sigma**2)
    dr = gs["diameter_m"]
    di = dr*np.sqrt(gs["obscuration_area"])
    def aperture(d, theta):
        model = FSOAnalyticalChannelModel(FSOParams(d_t=dtx, d_r=d, theta_div=theta))
        return np.array([model.eta_geo(x) for x in r])
    diffraction = aperture(dr, divergence)-aperture(di, divergence)
    collection = aperture(dr, divergence_effective)-aperture(di, divergence_effective)
    air_mass = 1/np.sin(np.radians(np.maximum(e, float(v["C14"]))))
    transmission = zenith_transmission**air_mass
    optical = eta_tx*collection*transmission*eta_rx
    detected = optical*eta_detector
    zero = lambda x: np.where(vis, x, 0.0)
    loss = lambda x: np.where(vis, -10*np.log10(np.maximum(x, 1e-300)), np.nan)
    rytov_z, r0_z = turbulence_integrals(gs, wavelength, v)
    rytov_local_z, _ = turbulence_integrals(gs, wavelength, v, local_surface=True)
    return dict(eta=zero(detected), optical_eta=zero(optical),
                diffraction_loss_db=loss(diffraction),
                pointing_increment_db=loss(collection/diffraction),
                atmospheric_loss_db=loss(transmission), optical_loss_db=loss(optical),
                detection_loss_db=loss(detected),
                rytov=np.where(vis, rytov_z*air_mass**(11/6), np.nan),
                rytov_local=np.where(vis, rytov_local_z*air_mass**(11/6), np.nan),
                fried_r0_m=np.where(vis, r0_z*air_mass**(-3/5), np.nan),
                beam_radius_m=np.sqrt(w0*w0+(divergence_effective*r)**2),
                divergence_rad=divergence, sigma_pointing_rad=sigma)


def background(gs, wavelength, radiance, *, fov_half_angle_urad=10.0,
               bandwidth_nm=1.0, eta_rx=0.50, eta_detector=0.60):
    """Total station sky counts, all analyzer outputs; radiance in W/m2/sr/nm."""
    area = np.pi*(gs["diameter_m"]/2)**2*(1-gs["obscuration_area"])
    alpha = fov_half_angle_urad*1e-6
    solid_angle = 4*np.pi*np.sin(alpha/2)**2
    return radiance*area*solid_angle*bandwidth_nm*eta_rx*eta_detector*wavelength/(H_PLANCK*C_LIGHT)


def coincidence(eta_a, eta_b, source_rate, source_fidelity, sky_a, sky_b,
                *, gate_s=1e-9, jitter_each_s=50e-12, dark_each_s=100.0,
                detectors=4, deadtime_s=0.0):
    """CW Poisson expectation; all-pairs time-tag matching, no basis sifting.

    Dead-time option is only a first-order independent nonparalyzable live-time
    approximation, not a replacement for a detector-specific high-flux DES.
    """
    sa, sb = source_rate*eta_a, source_rate*eta_b
    na, nb = sky_a+detectors*dark_each_s, sky_b+detectors*dark_each_s
    la = 1/(1+(sa+na)*deadtime_s/detectors)
    lb = 1/(1+(sb+nb)*deadtime_s/detectors)
    acceptance = erf(gate_s/(2*np.sqrt(2)*np.sqrt(2)*jitter_each_s))
    true = source_rate*eta_a*eta_b*acceptance*la*lb
    acc_other_pairs = sa*sb*gate_s*la*lb
    acc_background = (sa*nb+sb*na+na*nb)*gate_s*la*lb
    acc = acc_other_pairs+acc_background
    total = true+acc
    fidelity = np.divide(source_fidelity*true+.25*acc, total,
                         out=np.full_like(np.asarray(total, float), np.nan), where=total>0)
    return dict(true=true, accidental=acc, other_pair_accidental=acc_other_pairs,
                background_accidental=acc_background, fidelity=fidelity,
                singles_a=(sa+na)*la, singles_b=(sb+nb)*lb, live_a=la, live_b=lb,
                gate_acceptance=float(acceptance))


def intervals(seconds, mask):
    mask = np.asarray(mask, bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    # Threshold crossings are bracketed by the one-second samples.
    step = float(seconds[1]-seconds[0])
    return [dict(first_sample_s=float(seconds[a]), last_sample_s=float(seconds[b]),
                 duration_midpoint_s=float(seconds[b]-seconds[a]+step),
                 start_estimate_s=float(seconds[a]-step/2),
                 end_estimate_s=float(seconds[b]+step/2)) for a, b in zip(starts, ends)]


def write_csv(path, columns):
    keys = list(columns)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(np.atleast_1d(columns[k]) for k in keys)))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance, v = inputs()
    if abs(float(v["C29"])) > 1e-8:
        raise ValueError("This dedicated circular-orbit implementation requires negligible eccentricity.")
    a = float(v["C26"])*1000
    period = 2*np.pi/np.sqrt(MU/a**3)
    scan_seconds = np.arange(0,period+10,10.)
    scan = orbit(v,scan_seconds)
    joint = scan["both_visible"]
    if not np.any(joint):
        raise RuntimeError("No simultaneous pass in first orbit. Confirm epoch and reference frame.")
    score = np.where(joint,scan["Matera_range_m"]+scan["Helmos_range_m"],np.inf)
    center = float(scan_seconds[np.argmin(score)])
    seconds = np.arange(max(0,center-600),center+601,float(v["C40"]))
    g = orbit(v, seconds)
    g_mod = orbit(v, seconds, frame="MOD")
    g_j2 = orbit(v, seconds, include_j2=True)
    summary = {"input_sha256": provenance["sha256"], "orbit_frame_assumed": "J2000 approximately ICRS",
               "epoch_assumed_utc": v["C33"]+"Z", "step_s": float(v["C40"]),
               "simulated_range_s": [float(seconds[0]),float(seconds[-1])], "scenarios": {}, "geometry": {}}
    for frame, geom in [("J2000", g), ("MOD", g_mod), ("J2000_J2_osculating",g_j2)]:
        summary["geometry"][frame] = {
            "windows": {key: intervals(seconds, geom[key+"_visible"])
                        for key in ["Matera", "Helmos", "both"]},
            "max_elevation_deg": {key: float(max(geom[key+"_elevation_deg"])) for key in ["Matera", "Helmos"]},
            "height_range_m": [float(min(geom["altitude_wgs84_m"])), float(max(geom["altitude_wgs84_m"]))]}
    a = float(v["C26"])*1000
    n = np.sqrt(MU/a**3)
    summary["orbit_period_s"] = float(2*np.pi/n)
    summary["j2_nodal_precession_deg_per_day"] = float(np.degrees(-1.5*J2*n*(R_EARTH/a)**2*np.cos(np.radians(v["C30"])))*86400)
    summary["earth_radius_implied_by_altitude_km"] = float(v["C26"])-float(v["C28"])
    summary["assumptions"] = dict(eta_tx=.8, eta_rx=.5, eta_detector=.6,
        zenith_transmission=.9, gaussian_m2=1.0, waist_to_tx_radius=1.0,
        pointing_sigma_per_axis_urad=1.0, fov_half_angle_urad=10,
        bandwidth_nm=1.0, dark_rate_per_detector_s=100, detectors_per_station=4,
        gate_s=1e-9, timing_jitter_sigma_per_detector_s=50e-12,
        source_statistics="continuous-wave Poisson", source_in_band_fraction=1.0,
        deadtime_s=0, tracking_and_coincidence_compensation="perfect",
        mean_extra_turbulence_loss_db=0, background_radiance_w_m2_sr_nm=1e-6)
    write_csv(OUT/"geometry.csv", g)
    write_csv(OUT/"geometry_MOD_sensitivity.csv", g_mod)
    write_csv(OUT/"geometry_J2_sensitivity.csv",g_j2)
    gs_a, gs_b = stations(v)
    results = {}
    noise_rows, sensitivity_rows, timing_rows, deadtime_rows = [], [], [], []
    rate = float(v["C21"])*1e6
    for nm in [810, 1550]:
        wl = nm*1e-9
        la, lb = [link(gs, g, wl, v) for gs in [gs_a, gs_b]]
        noise_a, noise_b = [background(gs, wl, 1e-6) for gs in [gs_a, gs_b]]
        c = coincidence(la["eta"], lb["eta"], rate, float(v["C22"]), noise_a, noise_b)
        c["fidelity"] = np.where(g["both_visible"], c["fidelity"], np.nan)
        peak = int(np.argmax(c["true"]))
        both = g["both_visible"]
        count = int(np.count_nonzero(both))
        summary["scenarios"][str(nm)] = dict(
            peak_time_s=float(seconds[peak]), peak_utc=str(g["utc"][peak]),
            peak_true_pairs_s=float(c["true"][peak]), integrated_true_pairs=float(sum(c["true"])*float(v["C40"])),
            mean_true_pairs_s_during_joint_visibility=float(np.mean(c["true"][both])) if count else None,
            peak_accidentals_s=float(c["accidental"][peak]), illustrative_fidelity_at_peak=float(c["fidelity"][peak]),
            source_rate_pairs_s=rate, background_s={"Matera":noise_a,"Helmos":noise_b},
            source_power_per_arm_w=rate*H_PLANCK*C_LIGHT/wl,
            power_at_detector_input_w={"Matera":float(rate*H_PLANCK*C_LIGHT/wl*la["optical_eta"][peak]),
                                       "Helmos":float(rate*H_PLANCK*C_LIGHT/wl*lb["optical_eta"][peak])},
            peak_singles_s={"Matera":float(c["singles_a"][peak]),"Helmos":float(c["singles_b"][peak])},
            at_joint_peak={key: {metric:float(l[metric][peak]) for metric in
                          ["eta", "diffraction_loss_db", "pointing_increment_db", "atmospheric_loss_db",
                           "optical_loss_db", "detection_loss_db", "rytov", "fried_r0_m"]}
                           for key,l in [("Matera",la),("Helmos",lb)]},
            rytov_range={key:[float(np.nanmin(l["rytov"])), float(np.nanmax(l["rytov"]))]
                         for key,l in [("Matera",la),("Helmos",lb)]})
        cols = dict(t_s=seconds, utc=g["utc"], joint_visible=both)
        for key, l in [("Matera",la),("Helmos",lb)]:
            cols.update({key+"_"+k:val for k,val in l.items() if isinstance(val,np.ndarray)})
            cols[key+"_power_at_detector_input_w"] = rate*H_PLANCK*C_LIGHT/wl*l["optical_eta"]
            cols[key+"_signal_counts_s"] = rate*l["eta"]
        cols.update({k:val for k,val in c.items() if isinstance(val,np.ndarray)})
        write_csv(OUT/f"link_budget_{nm}nm.csv", cols)
        results[nm] = (la, lb, c)
        for radiance in np.logspace(-9,-2,71):
            ba, bb = [background(gs, wl, radiance) for gs in [gs_a,gs_b]]
            cp = coincidence(la["eta"][peak],lb["eta"][peak],rate,float(v["C22"]),ba,bb)
            noise_rows.append(dict(wavelength_nm=nm,radiance_w_m2_sr_nm=float(radiance),
                Matera_sky_counts_s=ba,Helmos_sky_counts_s=bb,
                Matera_noise_in_1ns=(ba+400)*1e-9,Helmos_noise_in_1ns=(bb+400)*1e-9,
                true_pairs_s=float(cp["true"]),accidentals_s=float(cp["accidental"]),
                illustrative_fidelity=float(cp["fidelity"])))
        for gate in [50e-12,100e-12,200e-12,500e-12,1e-9,2e-9,5e-9]:
            for brightness in [1e6,1e7,1e8,rate]:
                cp = coincidence(la["eta"][peak],lb["eta"][peak],brightness,float(v["C22"]),noise_a,noise_b,gate_s=gate)
                timing_rows.append(dict(wavelength_nm=nm,gate_ps=gate*1e12,source_rate_pairs_s=brightness,
                    gate_acceptance=cp["gate_acceptance"],true_pairs_s=float(cp["true"]),
                    accidentals_s=float(cp["accidental"]),illustrative_fidelity=float(cp["fidelity"])))
        for deadtime in [0,10e-9,50e-9,100e-9]:
            cp = coincidence(la["eta"][peak],lb["eta"][peak],rate,float(v["C22"]),noise_a,noise_b,deadtime_s=deadtime)
            deadtime_rows.append(dict(wavelength_nm=nm,deadtime_ns=deadtime*1e9,
                live_Matera=float(cp["live_a"]),live_Helmos=float(cp["live_b"]),
                estimated_true_pairs_s=float(cp["true"])))
        variants = [("baseline",{}),("M2_1.5",dict(m2=1.5)),("M2_2",dict(m2=2)),
                    ("zenith_transmission_0.7",dict(zenith_transmission=.7)),
                    ("zenith_transmission_0.98",dict(zenith_transmission=.98)),
                    ("pointing_radial_RMS",dict(pointing_scale=1/np.sqrt(2))),
                    ("detector_efficiency_0.3",dict(eta_detector=.3)),
                    ("detector_efficiency_0.8",dict(eta_detector=.8))]
        for name,kw in variants:
            ea,eb = [link(gs,g,wl,v,**kw)["eta"] for gs in [gs_a,gs_b]]
            cr = rate*ea*eb*c["gate_acceptance"]
            sensitivity_rows.append(dict(wavelength_nm=nm,case=name,peak_true_pairs_s=float(max(cr)),
                                         integrated_true_pairs=float(sum(cr)*float(v["C40"]))))
        for margin in [1.,3.,6.]:
            factor = 10**(-2*margin/10)
            sensitivity_rows.append(dict(wavelength_nm=nm,case=f"extra_loss_{margin:g}dB_per_arm",
                peak_true_pairs_s=float(max(c["true"])*factor),
                integrated_true_pairs=float(sum(c["true"])*float(v["C40"])*factor)))
        for name,geom in [("MOD_frame",g_mod),("J2_osculating",g_j2)]:
            ea,eb = [link(gs,geom,wl,v)["eta"] for gs in [gs_a,gs_b]]
            cr = rate*ea*eb*c["gate_acceptance"]
            sensitivity_rows.append(dict(wavelength_nm=nm,case=name,peak_true_pairs_s=float(max(cr)),
                                         integrated_true_pairs=float(sum(cr)*float(v["C40"]))))
    write_rows(OUT/"background_sensitivity.csv",noise_rows)
    write_rows(OUT/"timing_brightness_sensitivity.csv",timing_rows)
    write_rows(OUT/"detector_deadtime_sensitivity.csv",deadtime_rows)
    write_rows(OUT/"link_assumption_sensitivity.csv",sensitivity_rows)
    summary["source_hashes"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__),ROOT/"channels"/"fso.py"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    plots(g,results,noise_rows,timing_rows,v)
    print(json.dumps(summary,indent=2))


def plots(g, results, noise, timing, v):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.2,"figure.dpi":150})
    colors = {"Matera":"#1865a7", "Helmos":"#c15a23"}
    t = g["t_s"]
    visible = g["Matera_visible"] | g["Helmos_visible"]
    if not np.any(visible):
        raise RuntimeError("No pass in selected time span; inspect geometry before link results.")
    left,right = t[visible][0]-30,t[visible][-1]+30
    def shade(ax):
        for window in intervals(t,g["both_visible"]):
            ax.axvspan(window["start_estimate_s"],window["end_estimate_s"],color="#51a887",alpha=.09)
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(OUT/(name+".png"),dpi=190,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for key in colors:
        axes[0].plot(t,g[key+"_elevation_deg"],label=key,color=colors[key])
        axes[1].plot(t,g[key+"_range_m"]/1000,label=key,color=colors[key])
    axes[0].axhline(float(v["C14"]),color="gray",linestyle="--",label="20 deg mask")
    for ax in axes:
        ax.set_xlim(left,right); ax.set_xlabel("Seconds from supplied epoch (assumed UTC)"); shade(ax); ax.legend()
    axes[0].set(ylabel="Elevation (deg)",ylim=(0,90),title="Nominal overpass: J2000 element assumption")
    axes[1].set(ylabel="Slant range (km)",title="Green shading: simultaneous visibility")
    display=(t>=left)&(t<=right)
    max_range=max(np.max(g[key+"_range_m"][display])/1000 for key in colors)
    axes[1].set_ylim(500,max_range*1.05)
    finish(fig,"01_geometry")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(nm,(a,b,c)) in zip(axes,results.items()):
        for key,l in [("Matera",a),("Helmos",b)]:
            ax.plot(t,l["optical_loss_db"],color=colors[key],label=key+" optical")
            ax.plot(t,l["detection_loss_db"],color=colors[key],linestyle="--",label=key+" incl. detector")
        ax.set(xlim=(left,right),xlabel="Seconds from epoch",ylabel="Per-arm loss (dB)",
               title=f"{nm} nm | conditional Gaussian budget")
        shade(ax); ax.legend(fontsize=8)
    finish(fig,"02_link_budget")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for nm,(a,b,c) in results.items():
        axes[0].plot(t,c["true"]/1e6,label=f"{nm} nm")
        axes[1].plot(t,c["singles_a"]/1e6,label=f"Matera {nm} nm")
        axes[1].plot(t,c["singles_b"]/1e6,linestyle="--",label=f"Helmos {nm} nm")
    for ax in axes:
        ax.set_xlim(left,right); ax.set_xlabel("Seconds from epoch"); shade(ax); ax.legend(fontsize=8)
    axes[0].set(ylabel="Million true pairs / s",title="Ideal linear detectors; source = 1e9 pairs/s")
    axes[1].set(ylabel="Million counts / s, total per station",title="Detector loading before dead-time effects")
    finish(fig,"03_expected_signal")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for nm in results:
        rows=[r for r in noise if r["wavelength_nm"]==nm]
        x=[r["radiance_w_m2_sr_nm"] for r in rows]
        for key,style in [("Matera","-"),("Helmos","--")]:
            axes[0].loglog(x,[r[key+"_sky_counts_s"] for r in rows],linestyle=style,label=f"{key} {nm} nm")
        axes[1].semilogx(x,[r["illustrative_fidelity"] for r in rows],label=f"{nm} nm, 1 ns")
    axes[0].axhline(400,color="gray",linestyle=":",label="Assumed total dark counts")
    axes[0].set(ylabel="Sky counts / s, total per station",title="10 urad half-FOV; 1 nm filter; 30% Rx x PDE")
    axes[1].set(ylabel="Illustrative coincidence-mixture fidelity",ylim=(.24,1),
               title="CW Poisson / uniform accidentals; not a measured fidelity")
    for ax in axes:
        ax.set_xlabel("Assumed sky radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)"); ax.legend(fontsize=8)
    finish(fig,"04_background_noise")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,nm in zip(axes,[810,1550]):
        for rate in [1e6,1e7,1e8,1e9]:
            rows=[r for r in timing if r["wavelength_nm"]==nm and r["source_rate_pairs_s"]==rate]
            ax.semilogx([r["gate_ps"] for r in rows],[r["illustrative_fidelity"] for r in rows],
                        "o-",label=f"{rate/1e6:g} MHz pairs")
        ax.axhline(.99,linestyle=":",color="gray")
        ax.set(xlabel="Full coincidence window (ps)",ylabel="Illustrative mixture fidelity",ylim=(.24,1.01),
               title=f"{nm} nm | 50 ps RMS detector jitter each")
        ax.legend(fontsize=8)
    finish(fig,"05_timing_brightness")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for nm,(a,b,c) in results.items():
        for key,l in [("Matera",a),("Helmos",b)]:
            axes[0].plot(t,l["rytov"],label=f"{key} {nm} nm")
            axes[1].plot(t,l["fried_r0_m"]*100,label=f"{key} {nm} nm")
    for ax in axes:
        ax.set_xlim(left,right); ax.set_xlabel("Seconds from epoch"); ax.legend(fontsize=8)
    axes[0].set(ylabel="Point-receiver Rytov variance",title="HV profile diagnostics, no aperture averaging")
    axes[1].set(ylabel="Plane-wave Fried parameter (cm)",title="Not a downlink beam-divergence correction")
    finish(fig,"06_turbulence_diagnostics")


if __name__ == "__main__":
    main()
