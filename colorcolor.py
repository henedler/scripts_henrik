#!/usr/bin/env python
#
# Script to trace the flux-density evolution of a path in one or more fits-files.
# Required input: a ds9 region file which contains an ordered sequence of points, fits image(s).

import os, sys, argparse, pickle
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

from lib_linearfit import linear_fit_bootstrap, fit_path_to_regions, linsq_spidx
log.root.setLevel(log.INFO)



parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('region', help='region to restrict the analysis to')
parser.add_argument('bg', help='Path to ds9 region for background estimation.')
parser.add_argument('stokesi', nargs=3, default=[], help='List of fits images of Stokes I.')
parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
parser.add_argument('--B', default = 5e-10, type=float, help='Magnetic field. Defaults to 5e-10Tesla.')
parser.add_argument('--injectindex', default = 0.65, type=float, help='Injection photoon index.')
parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
parser.add_argument('--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
parser.add_argument('--regionpath', type=str, help='Name of the region defining a path on the sky.')
parser.add_argument('--fluxerr', default = None, type=float, help='Flux scale error of all images. Provide a fraction, e.g. 0.1.')
parser.add_argument('-o', '--out', default='colorcolor', type=str, help='Name of the output image and csv file.')
parser.add_argument('-d', '--debug', action='store_true', help='Debug output.')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity.')
parser.add_argument('-r', '--reuse', action='store_true', help='Reuse intermediate steps.')
args = parser.parse_args()

if args.verbose:
    log.root.setLevel(log.INFO)
if len(args.stokesi) != 3:
    log.error('Need three images for color-color analysis')
df_list = []
z = args.z
# sort frequency
freqs = [lib_fits.Image(filepath).freq for filepath in args.stokesi]
all_images = lib_fits.AllImages([args.stokesi[i] for i in np.argsort(freqs)])
freqs = np.sort(freqs)
# convolve images to the same beam (for now force circ beam)
if args.reuse and np.all([os.path.exists(name.replace('.fits', '-recenter-convolve-regrid.fits')) for name in args.stokesi]):
    log.info('Reuse prepared images.')
    all_images = lib_fits.AllImages([name.replace('.fits', '-recenter-convolve-regrid.fits') for name in args.stokesi])
else:
    log.info('Recenter, convolve and regrid all images')
    # if args.radec
    #     all_images.center_at(*args.radec)
    # else: # recenter at first image
    #     all_images.center_at(all_images[0].img_hdr['CRVAL1'], all_images[0].img_hdr['CRVAL2'])
    # if args.debug: all_images.write('recenter')
    all_images.convolve_to(circbeam=True) # elliptical beam seems buggy in some cases. Also, circ beam is nice to treat covariance matrix of pixels
    if args.debug: all_images.write('recenter-convolve')
    all_images.regrid_common(pixscale=4.,radec=args.radec)
    all_images.write('recenter-convolve-regrid')

mask = np.ones_like(all_images[0].img_data)
for image in all_images:
    image.calc_noise() # update. TODO: which is best way? BG region??
    image.blank_noisy(3) # blank 3 sigma
    image.apply_region(args.region, invert=True)
    isnan = np.isnan(image.img_data)
    mask[isnan] = 0
analysis_pixels = np.nonzero(mask)
log.debug(f"{np.sum(mask):.2%} pixel remaining for analysis.")
if args.debug: all_images.write('blank')


if os.path.exists('tmp_cc'+args.out+'.pickle') and args.reuse:
    with open( 'tmp_cc'+args.out+'.pickle', "rb" ) as f:
        spidx, spidx_err = pickle.load(f)
else:
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
        # (ahi, bhi, sahi, sbhi) = linear_fit_bootstrap(x=freqs[1:], y=val4reghi, yerr=noise[1:], tolog=True)
        # spidx[i] = alo, ahi
        # spidx_err[i] = salo, sahi
    with open( 'temp_colorcolor.pickle', "wb" ) as f:
        pickle.dump([spidx, spidx_err], f)

if args.regionpath:
    path_xy, l = fit_path_to_regions(args.regionpath, all_images[0], z, 100)
    distance = np.zeros(len(analysis_pixels[0]))
    for i, pix in enumerate(np.swapaxes(analysis_pixels,0,1)):
        idx_closest = np.argmin(np.linalg.norm(pix[np.newaxis,::-1] - path_xy, axis=1))
        distance[i] = l[idx_closest]
        log.debug(f'Closest point on path is a distance {distance[i]}')

# Get Jaffe-Perola ageing spectral indices
si_times = np.linspace(0,500,7)
jplow = lib_aging.get_aging_si(freqs[0], freqs[1], args.B, args.injectindex, z, si_times)
jphigh = lib_aging.get_aging_si(freqs[1], freqs[2], args.B, args.injectindex, z, si_times)
# do the plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.plot(jplow, jphigh, label=f'Jaffe-Perola, B$={args.B*1e10:.1f}\mu$G', color='red', zorder=2, ls='dashed')

freqs *= 1e-6
plt.xlabel(r'$\alpha_{' + f'{freqs[0]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + r'}$')
plt.ylabel(r'$\alpha_{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[2]:.0f}\mathrm{{MHz}}' + r'}$')
_min, _max = np.min(spidx), np.max(spidx)
plt.scatter(spidx[:,0], spidx[:,1], s=17, marker='s', edgecolors='black', cmap='plasma_r', c=distance, alpha=0.6, linewidths=0.5)
plt.plot([_min-0.1, _max+0.1], [_min-0.1, _max+0.1], label='PL injection', zorder=1, c='grey', ls='dotted')
cbar = plt.colorbar()
cbar.set_label('distance [kpc]',labelpad=12, rotation = 270)
plt.legend()

plt.xlim([_min-0.1, _max+0.1])
plt.ylim([_min-0.1, _max+0.1])
plt.minorticks_on()

log.info(f'Save plot to {args.out}.pdf...')
plt.tight_layout()
plt.savefig(args.out+'.pdf', bbox_inches='tight')
plt.close()



# print(analysis_pixels)
# plt.scatter(analysis_pixels[0], analysis_pixels[1], c=distance, cmap='plasma')
# plt.scatter(path_xy[:,0], path_xy[:,1])
# plt.savefig('debug_colorcolor.png')