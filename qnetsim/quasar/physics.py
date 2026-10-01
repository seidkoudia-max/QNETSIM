"""Input mapping and frozen-bin physical states shared by both DES resolutions."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.special import erf
from channels.polarization import PolarizationTrace

from studies.quasar_matera_helmos.run_study import MU, orbit, stations, link, background, C_LIGHT


def resolve_window(cfg, v):
    if not cfg.auto_pass:
        return cfg
    period = 2*np.pi/np.sqrt(MU/(float(v["C26"])*1000)**3)
    t = np.arange(0, period+10, 10.)
    g = orbit(v, t, cfg.frame, include_j2=cfg.include_j2)
    score = np.where(g["both_visible"], g["Matera_range_m"]+g["Helmos_range_m"], np.inf)
    if not np.any(np.isfinite(score)):
        raise ValueError("No shared visibility in the first orbit. Check the orbital frame, epoch and elevation mask.")
    center = float(t[np.argmin(score)])
    return replace(cfg, auto_pass=False, start_s=max(0, center-600), duration_s=1200.)


def prepare(cfg, v, wavelength):
    edges = np.arange(0, cfg.duration_s, cfg.bin_s)
    widths = np.minimum(cfg.bin_s, cfg.duration_s-edges)
    midpoints = cfg.start_s+edges+widths/2
    g = orbit(v, midpoints, cfg.frame, include_j2=cfg.include_j2)
    boundaries = cfg.start_s+np.r_[edges, cfg.duration_s]
    boundary_geometry = orbit(v, boundaries, cfg.frame, include_j2=cfg.include_j2)
    o = cfg.optics[wavelength]
    arms = [link(gs, g, wavelength*1e-9, v, m2=o.m2,
                 zenith_transmission=o.zenith_transmission, eta_tx=o.eta_tx,
                 eta_rx=o.eta_rx, eta_detector=o.eta_detector) for gs in stations(v)]
    extra = 10**(-o.extra_loss_db_per_arm/10)
    for arm in arms:
        arm["eta"] *= extra
        arm["optical_eta"] *= extra
        arm["detection_loss_db"] += o.extra_loss_db_per_arm
        arm["optical_loss_db"] += o.extra_loss_db_per_arm
    sky = [background(gs, wavelength*1e-9, radiance,
                      fov_half_angle_urad=o.fov_half_angle_urad, bandwidth_nm=o.bandwidth_nm,
                      eta_rx=o.eta_rx, eta_detector=o.eta_detector)
           for gs, radiance in zip(stations(v), [o.sky_matera, o.sky_helmos])]
    detector = cfg.detector_for(wavelength)
    acceptance = 1. if detector.jitter_s == 0 else float(erf(detector.gate_s/(4*detector.jitter_s)))
    pol = None
    if cfg.polarization[wavelength].enabled:
        windows = []
        for key in ["Matera", "Helmos"]:
            visible = np.r_[False, g[key+"_visible"], False].astype(int)
            begins, ends = np.where(np.diff(visible) == 1)[0], np.where(np.diff(visible) == -1)[0]
            arrivals = boundaries+boundary_geometry[key+"_range_m"]/C_LIGHT
            windows.append([(float(arrivals[a]), float(arrivals[b])) for a,b in zip(begins, ends)])
        last = cfg.start_s+cfg.duration_s+max(float(max(boundary_geometry[key+"_range_m"]))/C_LIGHT for key in ["Matera", "Helmos"])
        pol = PolarizationTrace(cfg.polarization[wavelength], last, cfg.seed, wavelength, windows)
    return dict(edges=edges, widths=widths, geometry=g, arms=arms, sky=sky, acceptance=acceptance,
                polarization=pol,
                delays=[g[key+"_range_m"]/C_LIGHT for key in ["Matera", "Helmos"]],
                delay_boundaries=[boundary_geometry[key+"_range_m"]/C_LIGHT for key in ["Matera", "Helmos"]])
