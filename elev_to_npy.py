#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henrik Edler
"""
import numpy as np
import argparse
import sys
from casacore import tables
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ms', help='Input MS name.', type=str, default=None)

    args = parser.parse_args()
    ms = tables.table(args.ms, ack=False)
    times = np.unique(ms.TIME)
    d = ms.FIELD[0]['REFERENCE_DIR']
    c = SkyCoord(d, unit='rad')
    lofar = EarthLocation(lat=52.90889*u.deg, lon=6.86889*u.deg, height=0*u.m)
    time=Time(np.arange(times[0], times[-1], 60)/86400, format='mjd')
    elevation = c.transform_to(AltAz(obstime=time,location=lofar)).alt
    T = Time(time, format='iso', scale='utc')
    elevation = elevation.to_value('deg')
    if '.ms' in args.ms:
        prefix = args.ms.split('.ms')[0]
    else:
        prefix = args.ms.split('.MS')[0]
    np.savetxt(prefix + '-elev.txt', elevation)
