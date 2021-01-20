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
    parser.add_argument('h5lba', help='H5parm of LBA', type=str)
    parser.add_argument('h5hba', help='H5parm of HBA', type=str)
    parser.add_argument('out', help='H5parm for output. Will be overwritten if it already exists!', type=str)
    args = parser.parse_args()

    h5lba = h5parm(args.h5lba)
    h5hba = h5parm(args.h5hba)

    h5out = h5parm(args.out, readonly=False)


    stlba = h5lba.getSolset('sol000').getSoltab('phase000')
    sthba = h5hba.getSolset('sol000').getSoltab('phase000')
    # if not stlba.getAxesNames() == sthba.getAxesNames():
    #     raise ValueError(f'Soltab 1 axes: {stlba.getAxesNames()} does not match Soltab 2 axes: {sthba.getAxesNames()}')

    times =  stlba.getAxisValues('time')
    times2 =  sthba.getAxisValues('time')
    assert all(times == times2), 'times do not match'
    freq = np.concatenate((stlba.getAxisValues('freq'),sthba.getAxisValues('freq')))
    dirlba = stlba.getAxisValues('dir')
    dirhba = sthba.getAxisValues('dir')
    dirorderlba, dirorderhba = np.argsort(dirlba), np.argsort(dirhba)

    assert len(set(dirlba).difference(set(dirhba))) == 0, 'directions wrong'

    antsel = ['CS001','CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013','CS017','CS021','CS024',
              'CS026','CS028','CS030','CS031','CS032','CS101','CS103','CS201','CS301','CS302','CS401','CS501',
              'RS205','RS208','RS210','RS305','RS306','RS307','RS310','RS406','RS407','RS409','RS503','RS508','RS509']
    antlba = stlba.getAxisValues('ant')
    anthba = sthba.getAxisValues('ant')

    # commonrs = []
    # for anti in antlba:
    #     for antj in anthba:
    #         if anti[:5] == antj[:5]:
    #             if antj[-1] != '1' and (antj[:2] == 'RS' or antj[:5] == 'CS001'):
    #                 commonrs.appenmerge_soltab_lba_hba.pyd(anti[:5])
    # inlba = [ant1[:5] in commonrs for ant1 in antlba]
    # inhba = []
    # for anti in anthba:
    #     if anti[-1] != '1' and anti[:5] in commonrs:
    #         inhba.append(True)
    #     else:
    #         inhba.append(False)

    slc_lba = [ant[0:5] in antsel for ant in stlba.getAxisValues('ant')]
    slc_hba = [ant[0:5] in antsel and ant[-1] != '1' for ant in sthba.getAxisValues('ant')]
    print('Ant slices: \n', stlba.getAxisValues('ant')[slc_lba])
    print(sthba.getAxisValues('ant')[slc_hba])
    print(stlba.val.shape)
    phlba = stlba.val[dirorderlba]
    phlba = phlba[...,slc_lba]
    phhba = sthba.val[dirorderhba]
    phhba = phhba[...,slc_hba,0]
    wlba = stlba.weight[dirorderlba]
    wlba = wlba[...,slc_lba]
    whba = sthba.weight[dirorderhba]
    whba = whba[...,slc_hba,0]

    print('concatenate along freq axis...')
    phases = np.concatenate((phlba,phhba), axis=2)
    weights = np.concatenate((wlba,whba), axis=2)

    if 'sol000' in h5out.getSolsetNames():
        solset = h5out.getSolset('sol000')
    else:
        solset = h5out.makeSolset(solsetName='sol000')

    if 'phase000' in solset.getSoltabNames():
        solset.getSoltab('phase000').delete()

    ras, decs = np.array([stlba.getSolset().getSou()[k] for k in dirlba[dirorderlba]]).T
    source_names = dirlba[dirorderlba]
    print(phases.shape, weights.shape, len(antsel), len(source_names))
    print(np.moveaxis(phases,[0,1,2,3],[3,0,1,2]).shape)
    st = solset.makeSoltab('phase', 'phase000', axesNames=['time','freq','ant','dir'],
                           axesVals=[times,freq,antsel,source_names], vals=np.moveaxis(phases,[0,1,2,3],[3,0,1,2]),
                           weights=np.moveaxis(weights,[0,1,2,3],[3,0,1,2]))

    antennaTable = solset.obj._f_get_child('antenna')
    sp = [stlba.getSolset().getAnt()[key+'LBA'] for key in antsel]
    antennaTable.append(list(zip(*(antsel, sp))))
    sourceTable = solset.obj._f_get_child('source')
    vals = [[ra, dec] for ra, dec in zip(ras, decs)]
    sourceTable.append(list(zip(*(source_names, vals))))
    h5out.close()
