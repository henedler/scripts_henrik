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
    parser = argparse.ArgumentParser(description='Merge LBA & HBA soltabs into a new h5parm. Only for sol000::phase000.')
    parser.add_argument('h5tec', help='H5parm', type=str)
    parser.add_argument('h5out', help='H5parm', type=str)
    args = parser.parse_args()

    h5tec = h5parm(args.h5tec)
    h5out = h5parm(args.h5out, readonly=False)


    sttec = h5tec.getSolset('sol000').getSoltab('tec000')
    stout = h5out.getSolset('sol000').getSoltab('phase000')
    # if not stlba.getAxesNames() == sthba.getAxesNames():
    #     raise ValueError(f'Soltab 1 axes: {stlba.getAxesNames()} does not match Soltab 2 axes: {sthba.getAxesNames()}')


    antsel = ['CS001','CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013','CS017','CS021','CS024',
              'CS026','CS028','CS030','CS031','CS032','CS101','CS103','CS201','CS301','CS302','CS401','CS501',
              'RS205','RS208','RS210','RS305','RS306','RS307','RS310','RS406','RS407','RS409','RS503','RS508','RS509']

    soltabOut = 'tec002'

    anttec = sttec.getAxisValues('ant')
    dirtec = sttec.getAxisValues('dir')
    tecvals = sttec.getValues(refAnt='CS001LBA')[0]
    print(tecvals.shape)
    tecvals = np.moveaxis(tecvals, [0,1,2], [2,0,1])
    tecvals = tecvals[:,-13:,dirtec == 'Isl_patch_80']
    tecweights = sttec.val
    tecweights = np.moveaxis(tecweights, [0,1,2], [2,0,1])
    tecweights = tecweights[:,-13:,dirtec == 'Isl_patch_80']
    tecvals = np.concatenate([np.zeros((len(tecvals), 24, 1)), tecvals], axis = 1)
    tecweights = np.concatenate([np.ones((len(tecvals), 24, 1)), tecweights], axis = 1)
    print(tecvals.shape)
    antout = stout.getAxisValues('ant')
    times = stout.getAxisValues('time')
    source_names = stout.getAxisValues('dir')

    print(anttec[-13:], antout)

    if 'sol000' in h5out.getSolsetNames():
        solset = h5out.getSolset('sol000')
    else:
        solset = h5out.makeSolset(solsetName='sol000')

    if soltabOut in solset.getSoltabNames():
        solset.getSoltab(soltabOut).delete()

    st = solset.makeSoltab('tec', soltabOut, axesNames=['time','ant','dir'],
                           axesVals=[times,antout,source_names], vals=tecvals,
                           weights=tecweights)

    h5out.close()
