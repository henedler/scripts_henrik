#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import numpy as np
# import astropy.tables as astrotab
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, Angle


def distanceOnSphere(RAs1, Decs1, RAs2, Decs2, rad=False):
    """
    Return the distances on the sphere from the set of points '(RAs1, Decs1)' to the
    set of points '(RAs2, Decs2)' using the spherical law of cosines.

    Using 'numpy.clip(..., -1, 1)' is necessary to counteract the effect of numerical errors, that can sometimes
    incorrectly cause '...' to be slightly larger than 1 or slightly smaller than -1. This leads to NaNs in the arccosine.
    """
    if rad:  # rad in rad out
        return np.radians(np.arccos(np.clip(
            np.sin(Decs1) * np.sin(Decs2) +
            np.cos(Decs1) * np.cos(Decs2) *
            np.cos(RAs1 - RAs2), -1, 1)))
    else:  # deg in deg out
        return np.degrees(np.arccos(np.clip(
            np.sin(np.radians(Decs1)) * np.sin(np.radians(Decs2)) +
            np.cos(np.radians(Decs1)) * np.cos(np.radians(Decs2)) *
            np.cos(np.radians(RAs1 - RAs2)), -1, 1)))


calibratorRAs           = np.array([123.4001379,212.835495, 277.3824204]) # in degrees
calibratorDecs          = np.array([48.2173778, 52.202770,  48.7461556])  # in degrees
calibratorNames         = np.array(["3C196",    "3C295",    "3C380"])

dwingeloo = EarthLocation.from_geodetic(52.8344444, 6.37055555)


cals = SkyCoord(calibratorRAs, calibratorDecs, unit='deg', frame='fk5')


def get_mid_hms(early, late):
    return (early + (late - early)/2).hms

def get_altitude_3c196(lts):
    return 90 - distanceOnSphere(123.4001379, 48.2173778, lts*360/24, 52.8344444)

def get_altitude_3c295(lts):
    return 90 - distanceOnSphere(212.835495, 52.202770, lts*360/24, 52.8344444)

def get_altitude_3c380(lts):
    return 90 - distanceOnSphere(277.3824204, 48.7461556, lts*360/24, 52.8344444)


lts_ranges = np.linspace(0,24,144*60)
a3c196, a3c295, a3c380 = [], [], []
for lts in lts_ranges:
    a3c196.append(get_altitude_3c196(lts))
    a3c295.append(get_altitude_3c295(lts))
    a3c380.append(get_altitude_3c380(lts))
a3c196, a3c295, a3c380 = np.array(a3c196), np.array(a3c295), np.array(a3c380)
n3c196 = np.max([a3c295, a3c380], axis=0)
n3c295 = np.max([a3c380, a3c196], axis=0)
n3c380 = np.max([a3c196, a3c295], axis=0)
print(lts_ranges[a3c295 > n3c295])
plt.plot(lts_ranges, a3c196, label='3c196')
plt.plot(lts_ranges, a3c295, label='3c295')
plt.plot(lts_ranges, a3c380, label='3c380')
plt.vlines([1.389, 11.118, 16.402],0,90)
plt.legend()
plt.xlabel('lts')
plt.ylabel('altitude [deg]')
plt.ylim(30,90)
plt.xlim(0,24)
plt.show()



def get_calib(lst):
    if   lst>1.389  and lst<=11.118: return '3c196'
    elif lst>11.118 and lst<=16.402: return '3c295'
    elif lst>16.402 or  lst<=1.389: return '3c380'
