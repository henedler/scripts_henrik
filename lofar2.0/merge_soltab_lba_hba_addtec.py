#!/usr/bin/env python
import sys
import argparse
import shutil
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romberg
from losoto.h5parm import h5parm

def selftec(soltab1, soltab2, sant, newtimes):
    """Self-cal tec solutions"""
    if 'LBA' in soltab1.getAxisValues('ant')[0]:
        refAnt = 'CS001LBA'
        sant += 'LBA'
    else:
        sant += 'HBA'
        refAnt = 'CS001HBA0'
    if ant[0:5] in ['RS208','RS210','RS307','RS310','RS406','RS407','RS409','RS508','RS509']:
        soltab = soltab2
    else:
        soltab = soltab1
    soltab.setSelection(ant = sant)
    # yvals = soltab.getValues(refAnt=refAnt)[0].flatten()
    # flag_outlier = np.ones_like(yvals, dtype=int) #~(np.abs(yvals) > 0.5)

    # f = interp1d(soltab.getAxisValues('time')[flag_outlier], yvals[flag_outlier], kind='nearest',
    #              bounds_error=False, fill_value=(yvals[flag_outlier][0],yvals[flag_outlier][-1]))
    f = interp1d(soltab.getAxisValues('time'), soltab.getValues(refAnt=refAnt)[0].flatten(), kind='nearest',
                 bounds_error=False, fill_value=(soltab.getValues(refAnt=refAnt)[0].flatten()[0],soltab.getValues(refAnt=refAnt)[0].flatten()[-1]))
    # delta_t = np.mean(np.diff(newtimes))
    # resampled_values = np.zeros_like(newtimes)
    # for i,t in enumerate(newtimes):
    #     resampled_values[i] = romberg(f,t-delta_t/2, t+delta_t/2, tol = 1.e-2, rtol=1.e-2, divmax=10)/delta_t
    # print(np.mean(np.abs(resampled_values -  f(newtimes))))
    # return resampled_values
    return f(newtimes)
if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Merge LBA & HBA soltabs into a new h5parm. Only for sol000::phase000. Also, add/subtract TEC000 DIE soltabs')
    parser.add_argument('h5lba', help='H5parm of LBA', type=str)
    parser.add_argument('h5hba', help='H5parm of HBA', type=str)
    parser.add_argument('h5dielba1', help='H5parm of DIE TEC LBA', type=str)
    parser.add_argument('h5dielba2', help='H5parm of DIE TEC LBA', type=str)
    parser.add_argument('h5diehba1', help='H5parm of DIE TEC HBA', type=str)
    parser.add_argument('h5diehba2', help='H5parm of DIE TEC HBA', type=str)
    parser.add_argument('out', help='H5parm for output. Will be overwritten if it already exists!', type=str)
    args = parser.parse_args()

    h5lba = h5parm(args.h5lba)
    h5hba = h5parm(args.h5hba)
    h5lbatec1 = h5parm(args.h5dielba1)
    h5lbatec2 = h5parm(args.h5dielba2)
    h5hbatec1 = h5parm(args.h5diehba1)
    h5hbatec2 = h5parm(args.h5diehba2)

    h5out = h5parm(args.out, readonly=False)


    stlba = h5lba.getSolset('sol000').getSoltab('phase000')
    sthba = h5hba.getSolset('sol000').getSoltab('phase000')
    stlbatec1 = h5lbatec1.getSolset('sol000').getSoltab('tec000')
    stlbatec2 = h5lbatec2.getSolset('sol000').getSoltab('tec000')
    sthbatec1 = h5hbatec1.getSolset('sol000').getSoltab('tec000')
    sthbatec2 = h5hbatec2.getSolset('sol000').getSoltab('tec000')
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
              'RS205','RS208','RS210','RS305','RS306','RS307','RS310','RS406','RS407','RS409','RS503','RS508','RS509'] #,'CS021
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
    # slc_hba = [ant[0:5] in antsel and ant[-1] != '1' for ant in sthba.getAxisValues('ant')]
    print('Ant slices: \n', stlba.getAxisValues('ant')[slc_lba])
    print(stlba.val.shape)
    phlba = stlba.val[dirorderlba]
    phlba = phlba[...,slc_lba]
    phlba_add = np.zeros_like(phlba)

    for i, ant in enumerate(stlba.getAxisValues('ant')[slc_lba]):
        if 'CS' in ant: continue
        teclba = selftec(stlbatec1, stlbatec2, ant[0:5], times)
        techba = selftec(sthbatec1, sthbatec2, ant[0:5], times)
        freqlba = stlba.getAxisValues('freq')
        thisant_ph = -8.44797245e9 * (teclba - techba)[:,np.newaxis] / freqlba[np.newaxis]
        thisant_ph = np.tile(thisant_ph, (phlba.shape[0],1,1))
        phlba_add[...,i] = thisant_ph
    phlba += phlba_add

    phhba = sthba.val[dirorderhba]
    phhba = phhba[...,slc_hba]
    wlba = stlba.weight[dirorderlba]
    wlba = wlba[...,slc_lba]
    whba = sthba.weight[dirorderhba]
    whba = whba[...,slc_hba]
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
                           axesVals=[times,freq,[ant+'LBA' for ant in antsel] ,source_names], vals=np.moveaxis(phases,[0,1,2,3],[3,0,1,2]),
                           weights=np.moveaxis(weights,[0,1,2,3],[3,0,1,2]))

    antennaTable = solset.obj._f_get_child('antenna')
    sp = [stlba.getSolset().getAnt()[key+'LBA'] for key in antsel]
    antennaTable.append(list(zip(*(antsel, sp))))
    sourceTable = solset.obj._f_get_child('source')
    vals = [[ra, dec] for ra, dec in zip(ras, decs)]
    sourceTable.append(list(zip(*(source_names, vals))))
    h5out.close()
