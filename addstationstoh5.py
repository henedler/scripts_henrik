#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2018 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
add stations from MS to h5
"""

import argparse, logging
import casacore.tables as pt
import numpy as np
import logging
from losoto.h5parm import h5parm

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('h5',help='Input soltab as h5parm:solset:soltab [no default].')
parser.add_argument('ms',help='Input MS [no default].')
args  = parser.parse_args()

ant = pt.table(f'{args.ms}::ANTENNA')

stations = dict()
for d in ant:
    stations[d['NAME']] = d['POSITION']
h5, solset, soltab = args.h5.split(':')
h5 = h5parm(h5, readonly=False)
stations_h5 = h5.getSolset(solset).getAnt()
soltab = h5.getSolset(solset).getSoltab(soltab)


shape_old = soltab.val.shape
axes = soltab.getAxesNames()
shape_new = list(shape_old)
shape_new[1] = len(stations)
v_new = np.zeros(shape_new)
w_new = np.ones(shape_new)

axes_vals = [soltab.getAxisValues(ax) for ax in axes]

for i,s in enumerate(stations):
    if s in axes_vals[1]:
        logging.info(f'{s} found in h5parm')
        j = np.argwhere([si == s for si in axes_vals[1]])[0,0]
        v_new[:,i] = np.take(soltab.val, j, 1)
        w_new[:,i] = np.take(soltab.weight, j,  1)
    else:
        logging.info(f'{s} NOT found in h5parm')

axes_vals[1] = list(stations.keys())

soltab_old = soltab
# sourceTable = h5parm.getSolset(solset).obj._f_get_child('source').copy()
solset = h5.getSolset('sol000')
soltab.delete()
st = solset.makeSoltab('tec', 'tec000', axesNames=axes, axesVals=axes_vals, vals=v_new, weights=w_new)
# st = h5parm.getSolset(args.solset).obj._f_get_child('source')
# st = sourceTable
antennaTable = solset.obj._f_get_child('antenna')
antennaTable.append(list(zip(*(stations.keys(), stations.values()))))
# sp = [[xyz] for xyz in stations.values()]
# antennaTable.append(list(zip(*(source_names, vals))))
