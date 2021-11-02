#!/usr/bin/env python

import matplotlib.pyplot as plt
from termcolor import colored
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import pickle
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

dwingeloo = EarthLocation.from_geodetic(52.8344444, 6.37055555)

def lst_str_to_float(lst):
    # turn lst interval string to list of two floats
    return [float(lst.split('-')[0]), float(lst.split('-')[1])]

def check_lst_coverage(opt_start, opt_end, allocated_lst):
    # Check the lst coverage of three allocated lst intervals against the lst interval where the pointing is in optimal elevation
    assert(len(allocated_lst) == 3)
    log = '' # log string for return
    if opt_end < opt_start: # check wrap
        opt_end += 24.
    allocated_lst = [lst_str_to_float(lst) for lst in allocated_lst]
    allocated_lst2 = [] # modified / unwrapped times
    for alloc_lst in allocated_lst:
        if alloc_lst[1] < alloc_lst[0]:
            alloc_lst[1] += 24.
        if not (((alloc_lst[0] > opt_start) and (alloc_lst[1] < opt_end)) or ((alloc_lst[0]+24 > opt_start) and (alloc_lst[1]+24 < opt_end))):
            print(colored(f'{alloc_lst[0]}-{alloc_lst[1]} not in {opt_start:.2f}-{opt_end:.2f}!!!', 'red'))
            log += '<oolstr>'
        if not (0.95 < (alloc_lst[1] - alloc_lst[0])< 8.05):
            print(colored('Time difference off...', alloc_lst[1] - alloc_lst[0], 'red'))
            sys.exit()
        allocated_lst2.append(alloc_lst)

    allocated_lst2 = np.array(allocated_lst2)
    if opt_end > 24:
        for i, w in enumerate(allocated_lst2):
            if w[1] < opt_start:
                allocated_lst2[i] += 24
    # sort according to start times:
    w1, w2, w3 = allocated_lst2[np.argsort(allocated_lst2[:,0])]
    if not ((w1[1] <= w2[0]) and (w2[1] <= w3[0])):
        print(colored('Overlap!', 'red'))
        log += '<overlap>'
    if not ((w2[1] - w1[0] > 2) and (w3[1] - w2[0] > 2)):
        print(colored('LST coverage insufficient!', 'red'))
        log += '<coverage>'
    print(f'Interval [0.00,{opt_end-opt_start:.2f}]\n', w1-opt_start, w2-opt_start, w3-opt_start)
    return log



def get_min_max_lst(ra, dec):
    min_elev = 55 if 40 < dec < 66 else 45
    lst = np.linspace(0., 24., 24*60)
    elevs = 90 - distanceOnSphere(ra, dec, lst * 360 / 24, 52.8344444)
    if np.sum([elevs > min_elev]) == len(elevs):
        print('Always good altitude.')
        return 0.0, 24.0
    elif np.sum(lst[elevs > min_elev]) == 0:
        print(colored('Never optimal...', 'red'))
        print(f'ra {ra}, dec {dec}')
        sys.exit(90)
    min_lst, max_lst = lst[elevs > min_elev][0], lst[elevs > min_elev][-1]
    if lst[elevs > min_elev][0] == 0.0:
        max_lst = lst[elevs < min_elev][0] - 1/60
        min_lst = lst[elevs < min_elev][-1] - 1/60
    return min_lst, max_lst

# cals = SkyCoord(calibratorRAs, calibratorDecs, unit='deg', frame='fk5')
tab = fits.open('allsky-grid.fits')[1]
df = pd.read_csv('output3.txt', delimiter=' ')
pointings = pd.DataFrame(index=np.unique(df[['beam1', 'beam2', 'beam3']]))
pointings['ra'] = np.NaN
pointings['dec'] = np.NaN
pointings['min_lst'] = np.NaN
pointings['max_lst'] = np.NaN
pointings['covered'] = 0.
pointings['error'] = ' '

lst_ranges = {}

for p_idx in pointings.index:
    covered = 0
    lst_ranges[p_idx] = []
    ra = tab.data[tab.data['name'] == p_idx]['ra'][0]
    dec = tab.data[tab.data['name'] == p_idx]['dec'][0]
    pointings.loc[p_idx, 'ra'] = ra
    pointings.loc[p_idx, 'dec'] = dec
    print(f'\n{p_idx} ra:{ra} dec:{dec}')
    pointings.loc[p_idx, 'min_lst'], pointings.loc[p_idx, 'max_lst'] = get_min_max_lst(ra, dec)
    print(f' lst from {pointings.loc[p_idx]["min_lst"]:.1f}--{pointings.loc[p_idx]["max_lst"]:.1f}')
    log = ' '
    for i, row in df.iterrows():
        if p_idx in [row['beam1'], row['beam2'], row['beam3']]:
            # print(row)
            lst_ranges[p_idx] = [row.loc['lst-range']] + lst_ranges[p_idx]
            pointings.loc[p_idx, 'covered'] += 1

    if pointings.loc[p_idx]['covered'] < 3:
        log += '<npoint>'
        print(colored('NOT ENOUGH POINTINGS', 'red'))
        print(pointings.loc[p_idx]['covered'])
        sys.exit()
    if pointings.loc[p_idx]['covered'] > 3:
        print(pointings.loc[p_idx]['covered'])
        print(colored('Too many pointings', 'yellow'))
        sys.exit()
    log += check_lst_coverage(pointings.loc[p_idx]['min_lst'], pointings.loc[p_idx]['max_lst'], lst_ranges[p_idx])
    pointings.loc[p_idx, 'error'] = log
pointings.to_csv('pointing_lst.csv', sep=',')
pickle.dump(lst_ranges, open('lst_ranges.pickle', 'wb') )

# df = pd.read_csv('pointing_lst.csv', delimiter=' ')
# lst_ranges = pickle.load(open('lst_ranges'))


