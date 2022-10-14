#!/usr/bin/python

import argparse, os
import sys
import logging
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import regions
from astropy.coordinates import SkyCoord
import scipy.ndimage
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from reproject import reproject_exact
from mpl_toolkits.axes_grid1 import make_axes_locatable
import legacystamps

from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from lib_plot import addRegion, addCbar, addBeam, addScalebar, setSize
from lib_fits import flatten

parser = argparse.ArgumentParser(description='Plotting script to overlay fits contours on SDSS image')
parser.add_argument('target', help='Name of target (Messier, NGC, VCC, IC...). Can also be multiple targets provided like M60+NGC4647.')
parser.add_argument('-s', '--size', type=float, default=8., help='size in arcmin')
# parser.add_argument('-z', '--redshift', type=float, default=None, help='redshift.')
parser.add_argument('-d', '--distance', type=float, default=17, help='distance in Mpc.')
parser.add_argument('-u', '--upsample', type=int, default=4, help='Upsample the image by this factor.')
parser.add_argument('--skip', action='store_true', help='Skip existing plots?')
parser.add_argument('--transparent', default=False, action='store_true', help='Transparent background (png).')
parser.add_argument('--fluxes', nargs=4, help='flux high / error high / flux low / error low in mJy', type=float)
parser.add_argument('--rms', nargs=2, help='hardcodede rms high, low', type=float)
parser.add_argument('-o', '--outfile', default=None, help='prefix of output image')

args = parser.parse_args()
base = '/beegfs/p1uy068/virgo/mosaics/2022_08/'

img =        base+'high/mosaic-restored.fits'
img_res =    base+'high/mosaic-residual.fits'
# img = '/Users/henrikedler/virgo/images/mosaic-restored.fits'
# img_res = '/Users/henrikedler/virgo/images/mosaic-residual.fits'
# img_lo = '/Users/henrikedler/virgo/images/low-mosaic-restored.fits'
# img_lo_res = '/Users/henrikedler/virgo/images/low-mosaic-residual.fits'
img_lo = base + 'low/low-mosaic-restored.fits'
img_lo_res = base + 'low/low-mosaic-residual.fits'
regionfile = 'all_0608.reg'

# Usage:
fontsize = 12
name = args.target  # NGC ... and M.. names definitely work
titlename = args.target
if os.path.exists(titlename + '.png') and args.skip:
    print(f'{titlename}.png exists - exiting.')
    sys.exit(0)
size = args.size * u.arcmin  # Size of the image in arcmin (so 10'x10')

# Load FITS file
img_hdr, img_data = flatten(img)
img_lo_hdr, img_lo_data = flatten(img_lo)
img_data *= 1000
img_lo_data *= 1000
in_wcs = WCS(img_hdr)
in_wcs_lo = WCS(img_lo_hdr)
if '+' in name:
    names = name.split('+')
    coords = [SkyCoord.from_name(n) for n in names]
    mean_ra = np.mean([coord.ra.to_value('deg') for coord in coords])
    mean_dec = np.mean([coord.dec.to_value('deg') for coord in coords])
    coord = SkyCoord(ra=mean_ra*u.degree, dec = mean_dec*u.degree)
else:
    coord = SkyCoord.from_name(name)

# Cutout central region
cutout_lo = Cutout2D(img_lo_data, coord, size=size, wcs=in_wcs_lo)
uhdr = cutout_lo.wcs.to_header()
uwcs = WCS(uhdr)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131, projection=in_wcs, slices=('x', 'y'))
ax2 = fig.add_subplot(132, projection=in_wcs_lo, slices=('x', 'y'))
ax3 = fig.add_subplot(133, projection=uwcs, slices=('x', 'y'))

cbar_kwargs = dict(label=r'$S$ [mJy$\,$beam$^{-1}$]', orientation='horizontal')

for data, ax, wcs, hdr in zip([img_data, img_lo_data], [ax1, ax2], [in_wcs, in_wcs_lo], [img_hdr, img_lo_hdr]):
    xrange, yrange = setSize(ax, wcs, coord.ra.deg, coord.dec.deg, size.to_value('deg'),  size.to_value('deg'))
    logging.info('Plotting  {}-{}, {}-{} from {}x{}.'.format(xrange[1], xrange[0], yrange[0], yrange[1], len(data[0]),
                                                             len(data[:, 0])))
    data_visible = Cutout2D(data, coord, size=size, wcs=wcs).data
    if data_visible.ndim < 2:
        raise ValueError('Selected coordinates out of image.')
    interval = AsymmetricPercentileInterval(20, 99.95)  # 99.99)  # 80 - 99.99 percentile
    stretch = SqrtStretch()
    int_min, int_max = interval.get_limits(data_visible)
    logging.info('min: {},  max: {}'.format(int_min, int_max))
    norm = ImageNormalize(data, vmin=float(int_min), vmax=float(int_max), stretch=stretch)
    im = ax.imshow(data, origin="lower", cmap='magma', interpolation='kaiser', norm=norm)
    pixcoord = wcs.wcs_world2pix([[coord.ra.deg, coord.dec.deg]], 0)
    ax.scatter(pixcoord[0][0], pixcoord[0][1], c='grey', marker='x', zorder=5)
    # divider = make_axes_locatable(ax)
    # cax = divider.new_horizontal(size="5%", pad=0.05, pack_start=True)
    # cax = divider.append_axes("top", size="5%", pad=0.05)
    # cax.xaxis.set_ticks_position("top")
    # cax.xaxis.tick_top()
    # cax.xaxis.set_label_position('top')
    # cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
    # cax.xaxis.set_ticks_position("top")
    # cax.xaxis.tick_top()
    # cax.xaxis.set_label_position('top')

if args.rms:
    noise = 1000*np.array(args.rms)
    print(f"Using input background rms: {noise[0]:.3f}mJy/beam.")
    print(f"Using input background rms: {noise[1]:.3f}mJy/beam.")
else:
    noise= []
    for i,noisemap in enumerate([img_res, img_lo_res]):
        print(f"Calculate noise... {noisemap}")
        hdr_n, data_n = flatten(noisemap)
        data_n *= 1000
        cutout_n = Cutout2D(data_n, coord, size=size, wcs=WCS(hdr_n))
        noise.append(np.nanstd(cutout_n.data))
        print(f"Found background rms: {noise[i]:.3f}mJy/beam.")

ax1.annotate(r'$\sigma_\mathrm{rms}=' + f'{int(1000*noise[0])}' + r'\,\mathrm{\frac{{\mu}Jy}{beam}}$',
             xy=(0.5,0.05),ha='center', xycoords='axes fraction', fontsize=fontsize, c='white', verticalalignment='bottom')
ax2.annotate(r'$\sigma_\mathrm{rms}=' + f'{int(1000*noise[1])}' + r'\,\mathrm{\frac{{\mu}Jy}{beam}}$',
            xy = (0.5,0.05),ha='center', xycoords = 'axes fraction', fontsize=fontsize, c='white', verticalalignment = 'bottom')

if args.fluxes:
    ax1.annotate(rf'$S={args.fluxes[0]:.1f}\pm{args.fluxes[1]:.1f}\,$mJy',
                 xy=(0.05, 0.95), ha='left', va='top', xycoords='axes fraction', fontsize=fontsize, c='white')
    ax2.annotate(rf'$S={args.fluxes[2]:.1f}\pm{args.fluxes[3]:.1f}\,$mJy',
                 xy=(0.05, 0.95), ha='left', va='top', xycoords='axes fraction', fontsize=fontsize, c='white')

# create upsampled wcs
# make sure upsample factor leads to the correct integer value for the new image size
factor = args.upsample
uhdr['CDELT1'], uhdr['CDELT2'] = uhdr['CDELT1']/factor, uhdr['CDELT2']/factor
uhdr['CRPIX1'], uhdr['CRPIX2'] = uhdr['CRPIX1']*factor, uhdr['CRPIX2']*factor
uhdr['NAXIS'] = 2
uhdr['NAXIS1'] = np.array(np.shape(cutout_lo.data)[0], dtype=int)*factor
uhdr['NAXIS2'] = np.array(np.shape(cutout_lo.data)[1], dtype=int)*factor
uwcs = WCS(uhdr)
sample_data, __footprint = reproject_exact((cutout_lo.data, cutout_lo.wcs), uhdr, parallel=False)
# smooth sample data...
sample_data = scipy.ndimage.filters.gaussian_filter(sample_data, factor/3)

# get background image from hips
fname = legacystamps.download(coord.ra.deg, coord.dec.deg, bands='grz', mode='jpeg', size=size.to_value('deg'),
                      pixscale=np.abs(3600 * img_lo_hdr['CDELT1'] / factor), autoscale=True, )
print(fname)
image = plt.imread(fname)
image = image[::-1]
# os.system(f"rm {fname}")

ax3.imshow(image, origin="lower", interpolation='kaiser')
contour_limits = 3 * 2 ** np.arange(20) * noise[1]
vmin, vmax = contour_limits[0], np.nanmax(sample_data)
norm = ImageNormalize(sample_data, vmin=vmin, vmax=vmax, stretch=SqrtStretch())
cntr = ax3.contour(sample_data, levels=contour_limits, cmap='cool', norm=norm, alpha=1, linewidths=1)
ax3.contour(sample_data, levels=-contour_limits[::-1], cmap='cool', alpha=1, linewidths=1, linestyles='dashed', norm=norm)
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("top", size="5%", pad=0.05)
# cbar = fig.colorbar(cntr, cax=cax, **cbar_kwargs)
# cax.xaxis.set_ticks_position("top")
pixcoord = uwcs.wcs_world2pix([[coord.ra.deg, coord.dec.deg]], 0)
ax3.scatter(pixcoord[0][0], pixcoord[0][1], c='grey', marker='x')

# plot a nice arrow pointing towards the cluster center
coord_m87 = SkyCoord.from_name('M87')
pix_coord_center = uwcs.wcs_world2pix([[coord.ra.value, coord.dec.value]],1)[0]
pix_coord_m87 = uwcs.wcs_world2pix([[coord_m87.ra.value, coord_m87.dec.value]],1)[0]
sep = coord_m87.separation(coord)
delta_pix = pix_coord_m87 - pix_coord_center
delta_pix /= np.linalg.norm(delta_pix)
scale = np.min(np.shape(sample_data)*factor)
arr_origin = pix_coord_center + 0.3*scale*delta_pix
ax3.arrow(*arr_origin, *(0.07*scale*delta_pix),  color='#f64fff', width=2.5)
if delta_pix[1] < 0 :
    va = 'top'
else:
    va = 'bottom'
if delta_pix[0] < 0:
    ha = 'left'
else:
    ha = 'right'
ax3.annotate(f'{sep.to_value("deg"):.2f}'+'$^\circ$', xy=arr_origin,  color='#f64fff',ha=ha, fontsize=fontsize, va=va)

scbarkpc = 10 if size.to_value('arcmin') < 10 else 20


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo.luminosity_distance(0.0043)
redshift = (cosmo.H0*args.distance*u.Mpc/const.c).decompose() # assume ~nearby
addScalebar(ax3, uwcs, redshift, scbarkpc, fontsize, color='white')
addBeam(ax1, img_hdr, edgecolor='white')
addBeam(ax2, img_lo_hdr, edgecolor='white')

matches = 0
this_reg = None
for r_split in regions.Regions.read(regionfile, format='ds9'):
    if r_split.meta['text'] == args.target:
        matches += 1
        this_reg = r_split
if matches == 0:
    raise ValueError(f'Found no bg region corresponding to {r_split.meta["text"]}')
elif matches > 1:
    raise ValueError(f'Found non-unique bg region corresponding to {r_split.meta["text"]}')
this_reg.write(f'temp-{titlename}.reg', format='ds9', overwrite=True)
addRegion(f'temp-{titlename}.reg', ax2, img_lo_hdr, color='#03fc30', text=False)
os.system(f'rm -r temp-{titlename}.reg')

for ax in [ax1, ax2, ax3]:
    # labels
    lon = ax.coords['ra']
    lat = ax.coords['dec']
    lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize)
    lat.set_axislabel('Declination (J2000)', fontsize=fontsize)
    lon.set_ticklabel(size=fontsize)
    lat.set_ticklabel(size=fontsize)
    # small img
    lon.set_major_formatter('hh:mm:ss')
    lat.set_major_formatter('dd:mm')
    lat.set_ticklabel(rotation=90) # to turn dec vertical

# plt.title(f"{titlename}, "+r"$\sigma_{rms}=$"+f"{noise:.3f}mJy/beam", size=fontsize+2)
fig.suptitle(f"{titlename}", size=fontsize+2)
plt.savefig(titlename + '.png', bbox_inches='tight', transparent=args.transparent)
