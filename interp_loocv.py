#!/usr/bin/env python

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.neighbors import KernelDensity
from losoto import h5parm
from losoto.lib_operations import normalize_phase
from losoto.operations import interpolatedirections, duplicate, residuals, plot


def _haversine(s1, s2):
    """
    Calculate the great circle distance between two points
    (specified in rad)
    """
    return 2*np.arcsin(np.sqrt(np.sin((s2[1]-s1[1])/2.0)**2 + np.cos(s1[1]) * np.cos(s2[1]) * np.sin((s2[0]-s1[0])/2.0)**2))

def make_kde_plot(kde, dirs, h5, res=1000):
    dirs = np.rad2deg(dirs)
    offst = 0.0
    grid = np.array(np.meshgrid(np.linspace(np.min(dirs[:,0])-offst,np.max(dirs[:,0])+offst,res),
                       np.linspace(np.max(dirs[:,1])-offst,np.min(dirs[:,1])+offst,res)))
    grid = np.deg2rad(np.resize(grid, (2,res**2)))
    samples = kde.score_samples(grid[::-1].T)
    plt.imshow(np.exp(np.resize(samples,(res,res))), extent = [np.min(dirs[:,0]),np.max(dirs[:,0]),np.min(dirs[:,1]),np.max(dirs[:,1])], vmin=0, zorder=0, aspect=1/np.cos(np.deg2rad(np.mean(dirs[:,1]))))
    plt.scatter(dirs[:,0], dirs[:,1], c='red', marker='x')
    plt.xlabel('RA', labelpad=13, fontsize=9.)
    plt.ylabel('Dec', labelpad=13, fontsize=9.)
    plt.savefig(h5+'_kde.png',dpi=200,bbox_inchs='tight')

parser = argparse.ArgumentParser(description = "Run leave-one-out cross-validation for direction-interpolation")
parser.add_argument("h5parm", help="inout h5parm", type=str)
parser.add_argument("--soltab", type=str, default='phase000')
parser.add_argument("--soltabTemp", type=str, default='phase001')
parser.add_argument("--soltabInterp", type=str, default='phaseInterp')
parser.add_argument("--soltabInterpRes", type=str, default='phaseInterpRes')
parser.add_argument("--soltabNN", type=str, default='phaseNN')
parser.add_argument("--soltabNNRes", type=str, default='phaseNNRes')
args = parser.parse_args()
h5, soltabname, soltabInterp, soltabTemp, soltabInterpRes, soltabNN, soltabNNRes = args.h5parm, args.soltab, args.soltabInterp, args.soltabTemp, args.soltabInterpRes, args.soltabNN, args.soltabNNRes

soltab = h5parm.openSoltab(h5, 'sol000', soltabname, readonly=False)
solset = soltab.getSolset()
dir_dict = dict(zip(soltab.dir, [solset.getSou()[k] for k in soltab.dir]))
dirs = np.array(list(dir_dict.values()))
centerra, centerdec = (np.max(dirs[:,0])+np.min(dirs[:,0]))/2, (np.max(dirs[:,1])+np.min(dirs[:,1]))/2
# KDE    - bandwidth is 20 arcmin
kde = KernelDensity(bandwidth=0.01744/3, metric='haversine').fit(dirs[:,::-1]) # bw in rad
make_kde_plot(kde, dirs, h5=h5)


vals, weights = soltab.val, soltab.weight

# prepare DataFrame
df = pd.DataFrame(columns=['d','ra','dec','center_dist','local_density','NN_dist','mean_delta_interp','rms_delta_interp','mean_delta_NN','rms_delta_NN'])
df.set_index('d', drop=True, inplace=True)

#### Make leave-one-out interpolated soltab
valuesInterp = np.zeros_like(vals)
weightsInterp = np.ones_like(weights)
valuesNN = np.zeros_like(vals)
weightsNN = np.ones_like(weights)

for i, (dir, radec) in enumerate(dir_dict.items()):
    radec = np.rad2deg(radec)
    print(i,dir)
    dirnames = list(soltab.dir)
    dirnames.remove(dir)
    soltab.setSelection(dir=dirnames)
    interpolatedirections.run(soltab, radec, soltabTemp, ncpu=64)
    stTemp = solset.getSoltab(soltabTemp, sel={'dir': ['interp_000']})
    valuesInterp[i] = stTemp.val[0]
    weightsInterp[i] = stTemp.weight[0]
    center_dist = _haversine(np.array([centerra,centerdec]), np.deg2rad(radec))

    other_dirs = dirs[[n != dir for n in dir_dict.keys()]]
    kde = KernelDensity(bandwidth=0.01744 / 3, metric='haversine').fit(other_dirs[:, ::-1])  # bw in rad
    kde2 = KernelDensity(bandwidth=0.01744 / 3, metric='haversine').fit(dirs[:, ::-1])  # bw in rad
    print(np.exp(kde.score_samples([np.deg2rad(radec)[::-1]]))[0],np.exp(kde2.score_samples([np.deg2rad(radec)[::-1]]))[0])
    local_dens = np.exp(kde.score_samples([np.deg2rad(radec)[::-1]]))[0]
    mean_delta = np.nanmean(normalize_phase(vals[i] - stTemp.val))
    rms_delta = np.nanstd(normalize_phase(vals[i] - stTemp.val))

    dists = []
    for j, (dirj, radecj) in enumerate(dir_dict.items()):
        if dir == dirj:
            dists.append(360.)
        else:
            d = _haversine(np.deg2rad(radec), radecj)
            dists.append(d)
    nn, nndist = np.argmin(dists), np.amin(dists)
    valuesNN[i] = vals[nn]
    weightsNN[i] = weights[nn]
    mean_deltaNN = np.nanmean(normalize_phase(vals[i]-valuesNN[i]))
    rms_deltaNN = np.nanstd(normalize_phase(vals[i]-valuesNN[i]))
    df.loc[h5.split('.')[0]+"_"+dir] = [radec[0],radec[1],center_dist,local_dens,nndist,mean_delta,rms_delta,mean_deltaNN,rms_deltaNN]
    print(f'mean: {mean_delta}, rms: {rms_delta}')
    print(f'mean: {mean_deltaNN}, rms: {rms_deltaNN}')
    soltab.clearSelection()

if soltabInterp in solset.getSoltabNames():
    print('Soltab {} exists. Overwriting...'.format(soltabInterp))
    solset.getSoltab(soltabInterp).delete()

soltab.clearSelection()
stInterp = solset.makeSoltab(soltype='phase', soltabName=soltabInterp, axesNames=soltab.getAxesNames(), \
                             axesVals=[soltab.getAxisValues(ax) for ax in soltab.getAxesNames()],
                             vals=valuesInterp, weights=weightsInterp)

# Make next-neighbor soltab
if soltabNN in solset.getSoltabNames():
    print('Soltab {} exists. Overwriting...'.format(soltabNN))
    solset.getSoltab(soltabNN).delete()

stNN = solset.makeSoltab(soltype='phase', soltabName=soltabNN, axesNames=soltab.getAxesNames(), \
                              axesVals=[soltab.getAxisValues(ax) for ax in soltab.getAxesNames()],
                              vals=valuesNN, weights=weightsNN)

#### Get residual soltabs
duplicate.run(soltab, soltabOut=soltabInterpRes, overwrite=True)
duplicate.run(soltab, soltabOut=soltabNNRes, overwrite=True)
print(soltabInterpRes, soltabInterp)
residuals.run(solset.getSoltab(soltabInterpRes), [soltabInterp])
residuals.run(solset.getSoltab(soltabNNRes), [soltabNN])

#### Plot stuff
# print('Plotting phases...')
# plot.run(soltab, ['time', 'freq'], 'ant', minmax=[-3.14,3.14], prefix='plots_{}/ph_'.format(h5.split('.')[0]))
# print('Plotting phase interpolations...')
# plot.run(stInterp, ['time', 'freq'], 'ant', minmax=[-3.14,3.14], prefix='plots_{}/phInterp_'.format(h5.split('.')[0]))
# plot.run(stNN, ['time', 'freq'], 'ant', minmax=[-3.14,3.14], prefix='plots_{}/phNN_'.format(h5.split('.')[0]))
# print('Plotting phase interpolation residuals...')
# plot.run(solset.getSoltab(soltabInterpRes), ['time', 'freq'], 'ant', minmax=[-3.14,3.14], prefix='plots_{}/phInterpRes_'.format(h5.split('.')[0]))
# plot.run(solset.getSoltab(soltabNNRes), ['time', 'freq'], 'ant', minmax=[-3.14,3.14], prefix='plots_{}/phNNRes_'.format(h5.split('.')[0]))

df['h5parm'] = h5
df['center_ra'] = centerra
df['center_dec'] = centerdec
df['n_dist'] = len(dir_dict)
df['tobs'] = (soltab.time[1]-soltab.time[0])*len(soltab.time)/3600
print(df['tobs'])

# df.drop(['d'], axis=1)

if os.path.exists('loocv.csv'):
     df_glob = pd.read_csv('loocv.csv')
     df_glob.set_index('d', drop=True, inplace=True)
     df_glob = df_glob.append(df)
else:
    df_glob = df
df_glob.to_csv('loocv.csv')
