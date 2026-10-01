"""Dimensioned Gaussian-aperture FSO transport with bounded irradiance fading.

Transport excludes source emission, memory write/read and detector efficiency.
Polarization-independent loss does not by itself decohere the surviving qubit.
The scintillation index is an aperture-level input, not an inferred Rytov value.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from functools import lru_cache
import numpy as np
from scipy.special import chndtr, ndtr


@dataclass(frozen=True)
class PhysicalFSOParams:
    wavelength_m: float = 810e-9
    transmitter_diameter_m: float = .20
    waist_radius_m: float = .07
    beam_quality_m2: float = 1.2
    receiver_diameter_m: float = 1.
    optical_efficiency: float = .8
    zenith_transmission: float = .90
    atmosphere_height_m: float = 10000.
    pointing_sigma_rad: float = .5e-6
    boresight_rad: float = 0.
    scintillation_index: float = .05
    min_elevation_deg: float = 10.

    def __post_init__(self):
        for name in ("wavelength_m","transmitter_diameter_m","waist_radius_m","receiver_diameter_m","atmosphere_height_m"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        if self.waist_radius_m > self.transmitter_diameter_m/2:
            raise ValueError("Gaussian waist must fit the transmitter radius")
        if self.beam_quality_m2 < 1:
            raise ValueError("M squared must be at least one")
        if not 0 <= self.optical_efficiency <= 1 or not 0 < self.zenith_transmission <= 1:
            raise ValueError("invalid optical transmission")
        if min(self.pointing_sigma_rad,self.boresight_rad,self.scintillation_index)<0:
            raise ValueError("jitter, offset and scintillation must be nonnegative")


def aperture_fraction(radius_m: float, beam_radius_m: float, offset_m: float = 0.):
    """Integral of 2/(pi w^2) exp(-2 |r-r0|^2/w^2) over a circular aperture."""
    if radius_m < 0 or beam_radius_m <= 0 or offset_m < 0:
        raise ValueError("invalid Gaussian-aperture geometry")
    x=4*radius_m**2/beam_radius_m**2
    if offset_m==0:
        return -math.expm1(-x/2)
    return min(1.,max(0.,float(chndtr(x,2,4*offset_m**2/beam_radius_m**2))))


def mean_clipped_lognormal(scale: float, scintillation_index: float):
    """E[min(1, scale H)] for unit-mean lognormal H of variance SI."""
    if scale < 0 or scintillation_index < 0:
        raise ValueError("scale and scintillation index must be nonnegative")
    if scale==0:
        return 0.
    if scintillation_index==0:
        return min(1.,scale)
    v=math.log1p(scintillation_index);s=math.sqrt(v);threshold=-math.log(scale)
    return float(scale*ndtr((threshold-v/2)/s)+ndtr(-(threshold+v/2)/s))


class GaussianApertureFSO:
    def __init__(self, params=PhysicalFSOParams()):
        self.p=params

    @property
    def divergence_rad(self):
        return self.p.beam_quality_m2*self.p.wavelength_m/(math.pi*self.p.waist_radius_m)

    def beam_radius(self, distance_m: float):
        if distance_m < 0:
            raise ValueError("distance must be nonnegative")
        return math.hypot(self.p.waist_radius_m,self.divergence_rad*distance_m)

    def slant_factor(self, elevation_deg: float):
        # Length through a spherical shell avoids a divergent low-elevation secant.
        re=6371000.;h=self.p.atmosphere_height_m;s=math.sin(math.radians(elevation_deg))
        return (math.sqrt((re*s)**2+2*re*h+h*h)-re*s)/h

    def extinction(self, elevation_deg: float, intersatellite: bool):
        return 1. if intersatellite else self.p.zenith_transmission**self.slant_factor(elevation_deg)

    def sample_transmission(self, rng, *, distance_m, elevation_deg, intersatellite=False):
        if not intersatellite and elevation_deg < self.p.min_elevation_deg:
            return 0.
        w=self.beam_radius(distance_m)
        sigma=self.p.pointing_sigma_rad*distance_m
        dx=rng.gauss(self.p.boresight_rad*distance_m,sigma) if sigma else self.p.boresight_rad*distance_m
        dy=rng.gauss(0.,sigma) if sigma else 0.
        collection=aperture_fraction(self.p.receiver_diameter_m/2,w,math.hypot(dx,dy))
        si=0. if intersatellite else self.p.scintillation_index
        if si:
            v=math.log1p(si)
            collection=min(1.,collection*math.exp(rng.gauss(-v/2,math.sqrt(v))))
        return self.p.optical_efficiency*self.extinction(elevation_deg,intersatellite)*collection

    def mean_transmission(self, *, distance_m, elevation_deg, intersatellite=False):
        if not intersatellite and elevation_deg < self.p.min_elevation_deg:
            return 0.
        w=self.beam_radius(distance_m)
        si=0. if intersatellite else self.p.scintillation_index
        sigma=self.p.pointing_sigma_rad*distance_m
        if not sigma:
            a=aperture_fraction(self.p.receiver_diameter_m/2,w,self.p.boresight_rad*distance_m)
            collection=mean_clipped_lognormal(a,si)
        elif si==0:
            collection=aperture_fraction(self.p.receiver_diameter_m/2,
                math.sqrt(w*w+4*sigma*sigma),self.p.boresight_rad*distance_m)
        elif self.p.boresight_rad==0:
            nodes,weights=_laguerre()
            collection=sum(weight*mean_clipped_lognormal(
                aperture_fraction(self.p.receiver_diameter_m/2,w,math.sqrt(2*u)*sigma),si)
                for u,weight in zip(nodes,weights))
        else:
            nodes,weights=_hermite()
            collection=sum(wx*wy*mean_clipped_lognormal(aperture_fraction(
                self.p.receiver_diameter_m/2,w,math.hypot(self.p.boresight_rad*distance_m+math.sqrt(2)*sigma*x,
                math.sqrt(2)*sigma*y)),si) for x,wx in zip(nodes,weights) for y,wy in zip(nodes,weights))/math.pi
        return self.p.optical_efficiency*self.extinction(elevation_deg,intersatellite)*collection

    def mean_link_metrics(self, **kwargs):
        p=self.mean_transmission(**kwargs)
        # Planner gets transport probability and ideal conditional transport state.
        return p,1.,1.


@lru_cache(None)
def _laguerre():
    return np.polynomial.laguerre.laggauss(40)


@lru_cache(None)
def _hermite():
    return np.polynomial.hermite.hermgauss(24)


def hufnagel_valley_cn2(height_m, *, wind_m_s=21., ground_cn2=1.7e-14):
    """HV 5/7 profile, h in metres, Cn2 in m^(-2/3); input profile, not weather."""
    if min(height_m,wind_m_s,ground_cn2)<0:
        raise ValueError("HV inputs must be nonnegative")
    h=height_m
    return (.00594*(wind_m_s/27)**2*(1e-5*h)**10*math.exp(-h/1000)
            +2.7e-16*math.exp(-h/1500)+ground_cn2*math.exp(-h/100))


def fried_parameter(wavelength_m, elevation_deg, *, wind_m_s=21.,ground_cn2=1.7e-14):
    """Plane-wave r0 with zenith angle = 90deg-elevation, 0-30 km HV integral."""
    from scipy.integrate import quad
    if wavelength_m<=0 or not 0 < elevation_deg <= 90:
        raise ValueError("invalid wavelength or elevation")
    integral=quad(lambda h:hufnagel_valley_cn2(h,wind_m_s=wind_m_s,ground_cn2=ground_cn2),0,30000)[0]
    return (.423*(2*math.pi/wavelength_m)**2*integral/math.sin(math.radians(elevation_deg)))**(-3/5)


def fiber_transport(length_km, attenuation_db_km=.18, insertion_db=.5):
    """Passive fiber loss only; no implicit source/detector factor."""
    if min(length_km,attenuation_db_km,insertion_db)<0:
        raise ValueError("fiber length and losses must be nonnegative")
    return 10**(-(attenuation_db_km*length_km+insertion_db)/10)
