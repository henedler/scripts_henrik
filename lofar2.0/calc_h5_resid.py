#!/usr/bin/env python

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from losoto.h5parm import h5parm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot residuals for TEC-simulation')
    parser.add_argument('h5true', help='Input corruption h5parm', type=str)
    parser.add_argument('h5die', help='DIE cal solutions', type=str)
    parser.add_argument('h5dde', help='DDE cal solutions', type=str)
    args = parser.parse_args()

    # open h5parms
    h5true = h5parm(args.h5true)
    h5self = h5parm(args.h5die)
    h5dd = h5parm(args.h5dde)


    dirdict = {'Isl_patch_64': 'i57',
               'Isl_patch_110': 'i34',
               'Isl_patch_100': 'i37',
               'Isl_patch_36': 'i45',
               'Isl_patch_61': 'i36',
               'Isl_patch_91': 'i44'}

    dirs =['Isl_patch_64',
           'Isl_patch_110',
           'Isl_patch_100',
           'Isl_patch_36',
           'Isl_patch_61',
           'Isl_patch_91']


    ant = ['RS205', 'RS208', 'RS210', 'RS305', 'RS306', 'RS307', 'RS310', 'RS406', 'RS407', 'RS409', 'RS503', 'RS508', 'RS509']
    antlba = ['RS205LBA', 'RS208LBA', 'RS210LBA', 'RS305LBA', 'RS306LBA', 'RS307LBA', 'RS310LBA', 'RS406LBA', 'RS407LBA', 'RS409LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
    anthba = ['RS205HBA', 'RS208HBA', 'RS210HBA', 'RS305HBA', 'RS306HBA', 'RS307HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA', 'RS409HBA', 'RS503HBA', 'RS508HBA', 'RS509HBA']

    mode = 'LBA'
    if mode == 'LBA':
        refAnt = 'CS001LBA'
        antstat = antlba
    elif mode == 'HBA':
        refAnt = 'CS001HBA0'
        antstat = anthba
    else:
        refAnt = 'CS001'
        antstat = ant


    tabtrue = h5true.getSolset('sol000').getSoltab('tec000', sel={'ant':antlba})
    tec_true, coord_true= tabtrue.getValues(refAnt='CS001LBA')
    t_true, dir_true = coord_true['time'], coord_true['dir']
    tabdie = h5self.getSolset('sol000').getSoltab('tec000', sel={'ant':antlba})
    tec_self = interp1d(tabdie.getAxisValues('time'),tabdie.getValues(refAnt='CS001LBA')[0][...,0], axis=0, bounds_error=False, fill_value='extrapolate')(t_true)


    tabdde = h5dd.getSolset('sol000').getSoltab('tec000', sel={'ant':antstat, 'dir':dirs})
    tec_dde, coord_dde = tabdde.getValues(refAnt=refAnt)
    t_dde, dir_dde = coord_dde['time'], coord_dde['dir']
    if mode in ['LBA', 'HBA']:
        tec_dde = np.swapaxes(tec_dde, 0,1)
        tec_dde = np.swapaxes(tec_dde, 1,2)

    tec_dde = interp1d(tabdde.getAxisValues('time'),tec_dde, axis=0,  bounds_error=False, fill_value='extrapolate')(t_true)

    dir_true = list(dir_true)
    dir_map = [dir_true.index('['+dirdict[d]+']') for d in dir_dde]
    print(tec_dde.shape, tec_self.shape, tec_true.shape)
    # print('dir ',np.std(tec_true[...,dir_map] - tec_self - tec_dde, axis=(0,1)))
    print('ant ',np.std(tec_true[...,dir_map] - tec_self - tec_dde, axis=(0,2)))
    # print(np.median(np.std(tec_true[...,dir_map] - tec_self - tec_dde, axis=0).flatten()))
    # print(np.mean(np.std(tec_true[...,dir_map] - tec_self - tec_dde, axis=0).flatten()))
    rms = np.std(tec_true[...,dir_map] - tec_self - tec_dde)
    print(rms)

