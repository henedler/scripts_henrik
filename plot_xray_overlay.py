#!/usr/bin/python
import urllib
import urllib.error

import argparse, os
import sys
import logging
import PIL
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import scipy.ndimage
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
from reproject import reproject_exact
import legacystamps

from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from lib_plot import addRegion, addCbar, addBeam, addScalebar, setSize
from lib_fits import flatten, Image

# TODO use frits Legacy survey script


parser = argparse.ArgumentParser(description='Plotting script to overlay fits contours on SDSS image')
parser.add_argument('image', help='fits image to plot.')
parser.add_argument('--ra', type=float, help='Ra in deg')
parser.add_argument('--dec', type=float, help='Dec in deg')
parser.add_argument('--titlename', help='Title name of target')
parser.add_argument('-s', '--size', type=float, default=8., help='size in arcmin')
parser.add_argument('-z', '--redshift', type=float, help='redshift.')
parser.add_argument('-d', '--distance', type=float, help='distance in Mpc.')
parser.add_argument('-u', '--upsample', type=int, default=2, help='Upsample the background image by this factor compared to the radio map.')
parser.add_argument('--no_axes', default=False, action='store_true', help='Show no axes.')
parser.add_argument('--ctr_start', type=float, help='Start contours at X.')
parser.add_argument('--ctr_end', type=float, help='End contours at X.')
parser.add_argument('--skip', action='store_true', help='Skip existing plots?')
parser.add_argument('--layer', type=str, default='ls-dr9', help='ls-dr9 / sdss')
parser.add_argument('--transparent', default=False, action='store_true', help='Transparent background (png).')
parser.add_argument('-o', '--outfile', default=None, help='prefix of output image')

args = parser.parse_args()
if args.image == None:
    logging.error('No input image found.')
    sys.exit()

# Usage:
fontsize = 20
name = args.target  # NGC ... and M.. names definitely work
titlename = args.titlename
if os.path.exists(titlename + '.png') and args.skip:
    print(f'{titlename}.png exists - exiting.')
    sys.exit(0)
size = args.size * u.arcmin  # Size of the image in arcmin (so 10'x10')
# Load FITS file
header, data = flatten(args.image)
data = data * 1000
in_wcs = WCS(header)
if args.ra and args.dec: # use ra dec
    coord = SkyCoord(ra=args.ra*u.degree, dec = args.dec*u.degree)
else:
    if '+' in name:
        names = name.split('+')
        coords = [SkyCoord.from_name(n) for n in names]
        mean_ra = np.mean([coord.ra.to_value('deg') for coord in coords])
        mean_dec = np.mean([coord.dec.to_value('deg') for coord in coords])
        coord = SkyCoord(ra=mean_ra*u.degree, dec = mean_dec*u.degree)
    else:
        coord = SkyCoord.from_name(name)
# Cutout central region
cutout = Cutout2D(data, coord, size=size, wcs=in_wcs)

# create upsampled wcs
uhdr = cutout.wcs.to_header()
# make sure upsample factor leads to the correct integer value for the new image size
factor = args.upsample
uhdr['CDELT1'] /= factor
uhdr['CDELT2'] /= factor
uhdr['CRPIX1'] *= factor
uhdr['CRPIX2'] *= factor
uhdr['NAXIS'] = 2
uhdr['NAXIS1'] = np.array(np.shape(cutout.data)[0], dtype=int)*factor
uhdr['NAXIS2'] = np.array(np.shape(cutout.data)[1], dtype=int)*factor
sample_data, __footprint = reproject_exact((cutout.data, cutout.wcs), uhdr, parallel=False)
# smooth sample data...
sample_data = scipy.ndimage.filters.gaussian_filter(sample_data, factor/3)

# get background image from hips
uwcs = WCS(uhdr)
fname = legacystamps.download(coord.ra.deg, coord.dec.deg, bands='grz', mode='jpeg', layers='', size=size.to_value('deg'),
                      pixscale=np.abs(3600 * header['CDELT1'] / factor), autoscale=True, )
image = plt.imread(fname)
image = image[::-1]
os.system(f"rm {fname}")

# Plot image with e.g. matplotlib
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=uwcs, slices=('x', 'y'))
lon = ax.coords['ra']
lat = ax.coords['dec']
ax.imshow(image, origin="lower", interpolation='kaiser')
if args.ctr_start:
    ctr_start = args.ctr_start
else:
    ctr_start = np.percentile(sample_data, 60)
if args.ctr_end:
    ctr_end = args.ctr_end
else:
    ctr_end = np.percentile(sample_data, 99.9)
contour_limits = np.logspace(np.log10(ctr_start), np.log10(ctr_end), 10)
vmin, vmax = contour_limits[0], np.nanmax(sample_data)
norm = ImageNormalize(sample_data, vmin=vmin, vmax=vmax, stretch=SqrtStretch())
ctr1 = ax.contour(sample_data, levels=contour_limits, cmap='cool', norm=norm, alpha=1, linewidths=1)
ctr2 = ax.contour(sample_data, levels=-contour_limits[::-1], cmap='cool', alpha=1, linewidths=1, linestyles='dashed', norm=norm)

h1,_ = ctr1.legend_elements()
h2,_ = ctr2.legend_elements()
if args.ctr_start in [1.0,2.0,3.0,4.0,5.0]:
    ax.legend([h1[0], h2[0]], [f'[{args.ctr_start:.0f}, {2*args.ctr_start:.0f}, {4*args.ctr_start:.0f}...]$\,\sigma$'+'$_\\mathrm{rms}$',
                           f'-{args.ctr_start:.0f}$\,\sigma$'+'$_\\mathrm{rms}$'], fontsize=fontsize)
else:
    ax.legend([h1[0], h2[0]], [f'[{args.ctr_start:.1f}, {2*args.ctr_start:.1f}, {4*args.ctr_start:.1f}...]$\,\sigma$'+'$_\\mathrm{rms}$',
                               f'-{args.ctr_start:.1f}$\,\sigma$'+'$_\\mathrm{rms}$'], fontsize=fontsize)

# plot a nice arrow pointing towards the cluster center
if args.arrow:
    coord_m87 = SkyCoord.from_name('M87')
    pix_coord_center = uwcs.wcs_world2pix([[coord.ra.value, coord.dec.value]], 1)[0]
    pix_coord_m87 = uwcs.wcs_world2pix([[coord_m87.ra.value, coord_m87.dec.value]], 1)[0]
    sep = coord_m87.separation(coord)
    delta_pix = pix_coord_m87 - pix_coord_center
    delta_pix /= np.linalg.norm(delta_pix)
    scale = np.min(np.shape(sample_data) * factor)
    arr_origin = pix_coord_center + 0.3 * scale * delta_pix
    ax.arrow(*arr_origin, *(0.07 * scale * delta_pix), color='#f64fff', width=2.5)
    if delta_pix[1] < 0:
        va = 'top'
    else:
        va = 'bottom'
    if delta_pix[0] < 0:
        ha = 'left'
    else:
        ha = 'right'
    ax.annotate(f'{sep.to_value("deg"):.2f}' + '$^\circ$', xy=arr_origin, color='#f64fff', ha=ha, fontsize=fontsize,
                 va=va)

redshift = args.redshift if args.redshift else (70*args.distance/3e5) # assume ~nearby
addScalebar(ax, uwcs, redshift, 10, fontsize, color='white')

# labels
if not args.no_axes:
    lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize)
    lat.set_axislabel('Declination (J2000)', fontsize=fontsize)
    lon.set_ticklabel(size=fontsize)
    lat.set_ticklabel(size=fontsize)

    # small img
    lon.set_major_formatter('hh:mm:ss')
    lat.set_major_formatter('dd:mm')
    lat.set_ticklabel(rotation=90) # to turn dec vertical
else:
    ax.axis('off')

#ax.annotate(f'{args.sigma_ctr:.1f}' + '$^\circ$', xy=arr_origin, color='#f64fff', ha=ha, fontsize=fontsize,
#            va=va)

# plt.title(f"{titlename}, "+r"$\sigma_{rms}=$"+f"{noise:.3f}mJy/beam", size=fontsize+2)
fn = args.outfile if args.outfile else titlename
plt.title(f"{titlename}", size=fontsize+2)
plt.savefig(fn + '.png', bbox_inches='tight', transparent=args.transparent)
