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
transfer solutions from LBA to HBA or vice versa
"""

import argparse, logging
import casacore.tables as pt
import numpy as np
import logging
from losoto.h5parm import h5parm

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('h5in',help='Input soltab as h5parm:solset:soltab [no default].')
parser.add_argument('h5out',help='Output soltab as h5parm:solset:soltab [no default].')
parser.add_argument('ms',help='Input MS [no default].')
args  = parser.parse_args()

ant = pt.table(f'{args.ms}::ANTENNA')

stations = dict()
for d in ant:
    stations[d['NAME']] = d['POSITION']
h5in, solsetin, soltabin = args.h5in.split(':')
h5out, solsetout, soltabout = args.h5out.split(':')
h5in = h5parm(h5in, readonly=False)
soltabin = h5in.getSolset(solsetin).getSoltab(soltabin)


shape_old = soltabin.val.shape
axes = soltabin.getAxesNames()
shape_new = list(shape_old)
shape_new[1] = len(stations)
v_new = np.zeros(shape_new)
w_new = np.ones(shape_new)
axes_vals = [soltabin.getAxisValues(ax) for ax in axes]

stations_h5 = soltabin.getAxisValues('ant')

for i,s_ms in enumerate(stations):
    for j, s_h5 in enumerate(stations_h5):
        if s_ms[0:5] == s_h5[0:5]:
            logging.info(f'Use values of {s_h5} for {s_ms}  ')
            v_new[:, i] = np.take(soltabin.val, j, 1)
            w_new[:, i] = np.take(soltabin.weight, j, 1)


axes_vals[1] = list(stations.keys())

h5out = h5parm(h5out, readonly=False)
if solsetout in h5out.getSolsetNames():
    solset = h5out.getSolset(solsetout)
else:
    solset = h5out.makeSolset(solsetName=solsetout)

if soltabout in solset.getSoltabNames():
    logging.info('''Solution-table is already present in
             {}. It will be overwritten.''')
    solset.getSoltab(soltabout).delete()

st = solset.makeSoltab(soltabin.getType(), soltabout, axesNames=axes, axesVals=axes_vals, vals=v_new, weights=w_new)
# sourceTableIn = h5in.getSolset(solsetin).obj._f_get_child('source')
# sourceTable = st.obj._f_get_child('source')
# sourceTable = sourceTableIn

antennaTable = solset.obj._f_get_child('antenna')
antennaTable.append(list(zip(*(stations.keys(), stations.values()))))