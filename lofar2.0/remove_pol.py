#!/usr/bin/env python
import sys
import argparse
import shutil
import numpy as np
from scipy.interpolate import interp1d
from losoto.operations import tec
from losoto.h5parm import h5parm


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Remove pol axis from phase solutions, keep polXX')
    parser.add_argument('h5parm', help='H5parm', type=str)
    args = parser.parse_args()

    h5 = h5parm(args.h5parm)
    st = h5.getSolset('sol000').getSoltab('phase000')

    h5out = h5parm(args.h5parm[:-3]+'nopol'+args.h5parm[-3:], readonly=False)

    if 'sol000' in h5out.getSolsetNames():
        solset = h5out.getSolset('sol000')
    else:
        solset = h5out.makeSolset(solsetName='sol000')

    if 'phase000' in solset.getSoltabNames():
        solset.getSoltab('phase000').delete()

    phases = st.val[...,0]
    weights = np.ones_like(phases)

    ras, decs = np.array([st.getSolset().getSou()[k] for k in st.getAxisValues('dir')]).T
    source_names = st.getAxisValues('dir')
    outaxes = st.getAxesNames()[:-1]

    antennaTable = solset.obj._f_get_child('antenna')
    sp = [st.getSolset().getAnt()[key] for key in st.getAxisValues('ant')]
    antennaTable.append(list(zip(*(st.getAxisValues('ant'), sp))))

    st = solset.makeSoltab('phase', 'phase000', axesNames=outaxes, axesVals=[st.getAxisValues(ax) for ax in outaxes],
                           vals=phases, weights=weights)

    sourceTable = solset.obj._f_get_child('source')
    vals = [[ra, dec] for ra, dec in zip(ras, decs)]
    sourceTable.append(list(zip(*(source_names, vals))))
    h5out.close()
