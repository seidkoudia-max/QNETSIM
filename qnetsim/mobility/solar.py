"""Approximate geometric solar elevation for map illumination, not sky radiance.

NOAA General Solar Position Calculations:
https://gml.noaa.gov/grad/solcalc/solareqns.PDF
No atmospheric refraction, terrain horizon, twilight radiance or satellite eclipse.
"""
import calendar
from datetime import timezone
import math

import numpy as np


def subsolar_point(when):
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("solar calculation requires a timezone-aware timestamp")
    when = when.astimezone(timezone.utc)
    hour = when.hour + when.minute/60. + (when.second+when.microsecond/1e6)/3600.
    days = 366 if calendar.isleap(when.year) else 365
    gamma = 2*math.pi/days*(when.timetuple().tm_yday-1+(hour-12)/24.)
    eqtime = 229.18*(.000075 + .001868*math.cos(gamma) - .032077*math.sin(gamma)
                    - .014615*math.cos(2*gamma) - .040849*math.sin(2*gamma))
    decl = (.006918 - .399912*math.cos(gamma) + .070257*math.sin(gamma)
            - .006758*math.cos(2*gamma) + .000907*math.sin(2*gamma)
            - .002697*math.cos(3*gamma) + .00148*math.sin(3*gamma))
    longitude = (180.-(hour*60.+eqtime)/4.+180.) % 360.-180.
    return math.degrees(decl), longitude


def solar_elevation_deg(latitude_deg, longitude_deg, when):
    decl, sun_lon = subsolar_point(when)
    lat, dec = np.radians(latitude_deg), math.radians(decl)
    hour_angle = np.radians(np.asarray(longitude_deg)-sun_lon)
    cosine = np.sin(lat)*math.sin(dec)+np.cos(lat)*math.cos(dec)*np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(cosine, -1., 1.)))
