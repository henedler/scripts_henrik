#!/usr/bin/env python
#
# Script to trace the flux-density evolution of a path in one or more fits-files.
# Required input: a ds9 region file which contains an ordered sequence of points, fits image(s).

import os, sys, argparse, pickle, copy
import logging as log
import numpy as np
import astropy.units as u
# from astropy.convolution import Gaussian2DKernel
# from radio_beam import Beams
# from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate#, integrate, optimize
import pandas as pd
import pyregion

import lib_fits, lib_aging
from path_analysis import  fit_path_to_regions
from lib_linearfit import linear_fit_bootstrap, linsq_spidx
log.root.setLevel(log.INFO)


parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('region', help='region to restrict the analysis to')
parser.add_argument('region2', help='region to restrict the analysis to')
parser.add_argument('stokesi', nargs=3, default=[], help='List of fits images of Stokes I.')
parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
parser.add_argument('--B', default = 5e-10, type=float, help='Magnetic field. Defaults to 5e-10Tesla.')
parser.add_argument('--injectindex', default = 0.65, type=float, help='Injection photoon index.')
parser.add_argument('--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
parser.add_argument('--regionpath', type=str, help='Name of the region defining a path on the sky.')
parser.add_argument('--fluxerr', default = None, type=float, help='Flux scale error of all images. Provide a fraction, e.g. 0.1.')
parser.add_argument('-o', '--out', default='colorcolor', type=str, help='Name of the output image and csv file.')
parser.add_argument('--circbeam', dest='circbeam', action='store_true', help='Force final beam to be circular (default: False, use minimum common beam area)')
parser.add_argument('-r', '--reuse', action='store_true', help='Reuse intermediate steps.')
args = parser.parse_args()

log.root.setLevel(log.INFO)
if len(args.stokesi) != 3:
    log.error('Need three images for color-color analysis')
df_list = []
z = args.z
# sort frequency
all_images = lib_fits.AllImages(args.stokesi)
freqs = all_images.freqs
# convolve
if all_images.suffix_exists('cc-conv') and args.reuse:
    log.info('Reuse si-conv images.')
    all_images = lib_fits.AllImages([name.replace('.fits', '-cc-conv.fits') for name in all_images.filenames])
else:
    # all_images.align_catalogue()
    if args.beam:
        all_images.convolve_to(args.beam, args.circbeam)
    else:
        all_images.convolve_to(circbeam=args.circbeam)
        all_images.write('cc-conv')

# all_images1 = lib_fits.AllImages([name.replace('.fits', '-debug1.fits') for name in all_images.filenames])
# all_images2 = lib_fits.AllImages([name.replace('.fits', '-debug2.fits') for name in all_images.filenames])
# all_images = all_images1
# calc noise -> regrid -> blank 3 sig -> apply reg
[image.calc_noise(force_recalc=True) for image in all_images]
[image.blank_noisy(3) for image in all_images]  # blank 3 sigma
all_images.regrid_common(pixscale=10)
all_images2 = copy.deepcopy(all_images)
[image.apply_region(args.region, invert=True) for image in all_images]
all_images.write('debug1')
[image.apply_region(args.region2, invert=True) for image in all_images2]
all_images2.write('debug2')
# # calc noise -> regrid -> blank 3 sig -> apply reg
# [image.calc_noise(force_recalc=True) for image in all_images]
# all_images.regrid_common(pixscale=5)
# [image.blank_noisy(3) for image in all_images]  # blank 3 sigma
# all_images2 = copy.deepcopy(all_images)
# [image.apply_region(args.region, invert=True) for image in all_images]
# all_images.write('debug1')
# [image.apply_region(args.region2, invert=True) for image in all_images2]
# all_images2.write('debug2')

mask = np.ones_like(all_images[0].img_data)
mask2 = np.ones_like(all_images2[0].img_data)
for image in all_images:
    image.apply_region(args.region, invert=True)
    isnan = np.isnan(image.img_data)
    mask[isnan] = 0
analysis_pixels = np.nonzero(mask)
log.info(f"{np.sum(mask):.2} pixel remaining for analysis.")

spidx = np.zeros((len(analysis_pixels[0]), 2))  # spidx lo spidx hi
spidx_err = np.zeros((len(analysis_pixels[0]), 2))  # spidx lo spidx hi
for i, (x, y) in enumerate(np.swapaxes(analysis_pixels,0,1)):
    print('.', end=' ')
    sys.stdout.flush()
    val4reglo = [image.img_data[x, y] for image in all_images[0:2]]
    val4reghi = [image.img_data[x, y] for image in all_images[1:]]
    noise = [image.noise for image in all_images]
    spidx[i,0], spidx_err[i,0] = linsq_spidx(freqs[0:2], val4reglo, noise[0:2])
    spidx[i,1], spidx_err[i,1] = linsq_spidx(freqs[1::], val4reghi, noise[1::])

if args.regionpath:
    path_xy, l = fit_path_to_regions(args.regionpath, all_images[0], args.z, 5/3600)
    distance = np.zeros(len(analysis_pixels[0]))
    for i, pix in enumerate(np.swapaxes(analysis_pixels,0,1)):
        idx_closest = np.argmin(np.linalg.norm(pix[np.newaxis,::-1] - path_xy, axis=1))
        distance[i] = l[idx_closest]
        # log.info(f'Closest point on path is a distance {distance[i]}')

for image in all_images2:
    isnan = np.isnan(image.img_data)
    mask2[isnan] = 0
analysis_pixels2 = np.nonzero(mask2)
log.info(f"{np.sum(mask2):.2} pixel remaining for analysis.")

spidx2 = np.zeros((len(analysis_pixels2[0]), 2))  # spidx lo spidx hi
spidx_err2 = np.zeros((len(analysis_pixels2[0]), 2))  # spidx lo spidx hi
for i, (x, y) in enumerate(np.swapaxes(analysis_pixels2,0,1)):
    print('.', end=' ')
    sys.stdout.flush()
    val4reglo = [image.img_data[x, y] for image in all_images2[0:2]]
    val4reghi = [image.img_data[x, y] for image in all_images2[1:]]
    noise = [image.noise for image in all_images2]
    spidx2[i,0], spidx_err2[i,0] = linsq_spidx(freqs[0:2], val4reglo, noise[0:2])
    spidx2[i,1], spidx_err2[i,1] = linsq_spidx(freqs[1::], val4reghi, noise[1::])

# Get Jaffe-Perola ageing spectral indices
si_times = np.linspace(0,500,7)
jplow = lib_aging.get_aging_si(freqs[0], freqs[1], args.B, args.injectindex, si_times, z)
jphigh = lib_aging.get_aging_si(freqs[1], freqs[2], args.B, args.injectindex, si_times, z)
# do the plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

orange = [1., 0.33333333, 0., 1.]
blue = [0., 0.23529412, 1., 1.]
plt.plot(jplow, jphigh, label=f'Jaffe-Perola, B$={args.B*1e10:.1f}\mu$G', color='green', zorder=5, ls='dashed')

freqs *= 1e-6
plt.xlabel(r'$\alpha_{' + f'{freqs[0]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + r'}$')
plt.ylabel(r'$\alpha_{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[2]:.0f}\mathrm{{MHz}}' + r'}$')
_min, _max = np.min(spidx), np.max(spidx)
plt.scatter(spidx[:,0], spidx[:,1], s=17, marker='s', edgecolors=orange, cmap='magma_r', c=distance, alpha=0.5, linewidths=0.5, label='WAT-GReET', zorder=4)
cbar = plt.colorbar()
plt.scatter(spidx2[:,0], spidx2[:,1], s=17, marker='s', c=blue, alpha=0.5, linewidths=0.5, label='Phoenix', zorder=3)
spidx = np.loadtxt('halo1.txt')
plt.scatter(spidx[:,0], spidx[:,1], s=17, marker='s', c='green', alpha=0.5, linewidths=0.5, label='halo1', zorder=4)
spidx2 = np.loadtxt('halo2.txt')
plt.scatter(spidx2[:,0], spidx2[:,1], s=17, marker='s', c='purple', alpha=0.5, linewidths=0.5, label='halo2', zorder=4)


plt.plot([_min-0.1, _max+0.1], [_min-0.1, _max+0.1], label='PL injection', zorder=1, c='grey', ls='dotted')
cbar.set_label('distance [kpc]',labelpad=12, rotation = 270)
plt.legend()

plt.xlim([_min-0.1, _max+0.1])
plt.ylim([_min-0.1, _max+0.1])
plt.minorticks_on()

log.info(f'Save plot to {args.out}.pdf...')
plt.tight_layout()
plt.savefig(args.out+'.png', bbox_inches='tight')
# plt.savefig(args.out+'.pdf', bbox_inches='tight')
plt.close()



# print(analysis_pixels)
# plt.scatter(analysis_pixels[0], analysis_pixels[1], c=distance, cmap='plasma')
# plt.scatter(path_xy[:,0], path_xy[:,1])
# plt.savefig('debug_colorcolor.png')