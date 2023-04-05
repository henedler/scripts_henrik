#!/usr/bin/python3

import sys, os
import numpy as np
import argparse
import logging
from lib_fits import flatten, Image
from astropy.io import fits
import regions
import matplotlib
matplotlib.use('Agg') # aplpy api suggestion
import matplotlib.pyplot as plt
import matplotlib.hatch
from astropy.wcs import WCS
import colormaps
from astropy.table import Table
from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from lib_plot import addRegion, addCbar, addBeam, addScalebar, setSize, ArrowHatch
logging.root.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Basic plotting script for fits images')
parser.add_argument('image', nargs='+', help='fits image to plot.')
parser.add_argument('--region', nargs='+', help='ds9 region files to plot (optional).')
parser.add_argument('--type', default='stokes', help='stokes / si / sierr / si+err / curvature / curverr ')
parser.add_argument('-c', '--center', nargs=2, type=float, default=None, help='Center ra/dec in deg') # [157.945, 35.049] A1033
parser.add_argument('-s', '--size', nargs=2, type=float, default=[7.8, 7.8], help='size in arcmin')
parser.add_argument('-z', '--redshift', type=float, default=None, help='redshift.')
parser.add_argument('-n', '--noise', type=float, default=None, help='Hardcode noise level in mJy/beam.')
parser.add_argument('-o', '--outfile', default=None, help='prefix of output image')
parser.add_argument('--interval', default=None, nargs=2, type=float, help='Provide min/max interval.')
parser.add_argument('--no_cbar', default=False, action='store_true', help='Show no cbar.')
parser.add_argument('--cbar_vertical', default=False, action='store_true', help='Show cbar vertical.')
parser.add_argument('--no_sbar', default=False, action='store_true', help='Show no scalebar.')
parser.add_argument('--sbar_kpc', default=100, type=float, help='Show how many kpc of scalebar?.')
parser.add_argument('--stretch', default='sqrt', type=str, help='Use sqrt for normal, log for very extended.')
parser.add_argument('--show_grid', action='store_true', help='Show grid.')
parser.add_argument('--no_axes', default=False, action='store_true', help='Show no axes.')
parser.add_argument('--png', default=False, action='store_true', help='Save as .png (default: pdf).')
parser.add_argument('--transparent', default=False, action='store_true', help='Transparent background (png).')
parser.add_argument('--cat', default=None, type=str, help='Plot catalogue.')
parser.add_argument('--dpi', default=200, type=int)
parser.add_argument('--show_contours', action='store_true', help='Show contours.')

args = parser.parse_args()
if args.image == None:
    logging.error('No input image found.')
    sys.exit()

filename = args.image[0]
regions = args.region
plottype = args.type
if args.outfile:
    outfile = args.outfile+'.pdf'
else:
    outfile = args.image[0].replace('fits','pdf')

# stretch type (only stokes) 'log' (for extended) or 'sqrt' (for compact)
stretch_type = args.stretch
# Style
fontsize = 14
# Scalebar
show_scalebar = not args.no_sbar
if show_scalebar:
    z = args.redshift
kpc = args.sbar_kpc # how many kpc is the scalebar?
accentcolor = 'black'
# plt.style.use('dark_background')
# accentcolor = 'white' if args.type == 'stokes' else 'black'

show_cbar = not args.no_cbar
show_grid = args.show_grid
show_axes = not args.no_axes
show_contours = args.show_contours
contout_base_sigma = 3 # will be this times [1,2,4,8,16]
n_contour = 9

logging.info('Setting up...')
if plottype != 'stokes':
    with fits.open(filename) as fitsimage:
        header = fitsimage[0].header
        data = fitsimage[0].data
else:
    header, data = flatten(filename)

img = Image(filename)

# Plot extensions
if args.center is None:
    center = [img.ra, img.dec]
else:
    center = args.center
if args.size is None:
    size = [np.abs(header['NAXIS1']*header['CDELT1'])*60,np.abs(header['NAXIS2']*header['CDELT2'])*60]
else:
    size = args.size

if plottype in ['stokes']:
    data *= 1e3 # to mJy
    if args.noise:
        sigma = args.noise
    else:
        sigma = img.calc_noise() * 1e3
    logging.info(f'Noise is {sigma:.2e} mJy/beam')

wcs = WCS(header)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=wcs, slices=('x', 'y'))
lon = ax.coords['ra']
lat = ax.coords['dec']


# zoom in in pixel
print(center, size)
xrange, yrange = setSize(ax, wcs, center[0], center[1], *np.array(size)/60)
logging.info('Plotting  {}-{}, {}-{} from {}x{}.'.format(xrange[1], xrange[0], yrange[0], yrange[1], len(data[0]), len(data[:,0])))
data_visible = data[yrange[0]:yrange[1],xrange[1]:xrange[0]] # select only data that is visible in plot
if data_visible.ndim < 2:
    raise ValueError('Selected coordinates out of image.')
# if we want to scale SI map transparency with error
if plottype == 'si+err':
    _, data_alpha = flatten(args.image[1])

# normalizer
if plottype == 'stokes':
    if stretch_type == 'sqrt':
        interval = AsymmetricPercentileInterval(20, 99.9)  # 99.99)  # 80 - 99.99 percentile
        stretch = SqrtStretch()
    elif stretch_type == 'log':
        interval = AsymmetricPercentileInterval(80, 99.999)#99.99)  # 80 - 99.99 percentile
        stretch = LogStretch()
    elif stretch_type == 'linear':
        interval = AsymmetricPercentileInterval(60, 99.9)  # 99.99)  # 80 - 99.99 percentile
        stretch = LinearStretch()
    else:
        print('Stretch type unknown.')
        sys.exit(0)
    if args.interval:
        int_min, int_max = args.interval
        # int_min = 1 * sigma
    else:
        int_min, int_max = interval.get_limits(data_visible)
elif plottype == 'curvature':
    interval = PercentileInterval(99)
    stretch = LinearStretch()
    rang = np.max(interval.get_limits(data_visible))
    int_min, int_max = -rang, rang
else:
    interval = PercentileInterval(99)
    stretch = LinearStretch()
    if args.interval:
        int_min, int_max = args.interval
    else:
        int_min, int_max = interval.get_limits(data_visible)

logging.info('min: {},  max: {}'.format(int_min,int_max))
norm = ImageNormalize(data, vmin=float(int_min), vmax=float(int_max), stretch=stretch)

# bkgr image
logging.info("Image...")
if plottype in ['si','si+err']:
    from colormaps import *
    ul_mask = np.ones_like(data, dtype=int)
    ll_mask = np.ones_like(data, dtype=int)
    all_limit_mask = np.ones_like(data, dtype=int)
    for k in header.keys(): # find upper limits from header...
        if k[0:2] == 'UL':
            i, j = np.array(header[k].replace(' ', '').split(',')).astype(int)
            ul_mask[i,j] = 0
            all_limit_mask[i,j] = 0
        elif k[0:2] == 'LL':
            i, j = np.array(header[k].replace(' ', '').split(',')).astype(int)
            ll_mask[i, j] = 0
            all_limit_mask[i, j] = 0
    ul_mask[np.isnan(data)] = -1
    ll_mask[np.isnan(data)] = -1
    all_limit_mask[np.isnan(data)] = -1
    if plottype == 'si':
        im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='turbo', norm=norm)
    elif plottype == 'si+err':
        low_cut = np.percentile(data_alpha[~np.isnan(data_alpha)], 15)
        data_alpha[data_alpha < low_cut] = low_cut
        alpha = 0.7 * (np.nanmin(data_alpha)/data_alpha)**2 + 0.3
        alpha[np.isnan(alpha)] = 0.3
        print(alpha.shape, data.shape)
        im = ax.imshow(data, alpha=alpha, origin='lower', interpolation='nearest', cmap='turbo', norm=norm)
        ax.contour(all_limit_mask, levels=[0.5], linewidths=0.5, colors=('black',), antialiased=True)
    # ax.contourf(all_limit_mask, alpha=0.5, color='white', colors=('white',), levels=[-0.5,0.5], antialiased=True) # this was used to shade UL and LL
    if np.any(ul_mask == 0): # only if we have any upper limits
        matplotlib.hatch._hatch_types.append(ArrowHatch)
        ax.contourf(ul_mask, alpha=0.0, levels=[-0.5,0.5], hatches=['arr{270}{7}{2}',''], antialiased=True)
    if np.any(ll_mask == 0): # only if we have any upper limits
        matplotlib.hatch._hatch_types.append(ArrowHatch)
        ax.contourf(ll_mask, alpha=0.0, levels=[-0.5,0.5], hatches=['arr{90}{7}{2}',''], antialiased=True)
elif plottype == 'sierr':
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='RdPu', norm=norm)
elif plottype == 'curvature':
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='coolwarm', norm=norm)
    ax.contour(np.isnan(data), levels=[0.5], linewidths=0.5, colors=('black',), antialiased=True)
elif plottype == 'curverr':
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='RdPu', norm=norm)
else:
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='Greys_r', norm=norm)
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='Oranges_r', norm=norm)
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='uhh_b', norm=norm)
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='YlOrRd_r', norm=norm)
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='cubehelix', norm=norm) # Try YlOrRed,
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='Blues_r', norm=norm) # Try YlOrRed,
    # im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='magma', norm=norm) # Try YlOrRed,
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='magma', norm=norm) # Try YlOrRed,

# contours
if show_contours:
    print("Contour...")
    contour_limits = contout_base_sigma * 2**np.arange(n_contour) * sigma
    ax.contour(data, transform=ax.get_transform(WCS(header)),linewdiths=0.1, levels=contour_limits, colors='white', alpha=0.7) #grey
    ax.contour(data, transform=ax.get_transform(WCS(header)),linewdiths=0.1, levels=-contour_limits[::-1], colors='white', alpha=0.7, linestyles='dashed') # grey

# EVCC catalogue
if args.cat:
    print("Catalogue...")
    evcc = Table.read(args.cat)
    try:
        cra, cdec = [evcc['RAJ2000'], evcc['DEJ2000']]
    except KeyError:
        try:
            cra, cdec = [evcc['RA'], evcc['DE']]
        except KeyError:
            cra, cdec = [evcc['_RAJ2000'], evcc['_DEJ2000']]
    ax.scatter(cra, cdec, marker='x', c='red', lw=1, transform=ax.get_transform('world'))

# add beam
addBeam(ax, header, edgecolor=accentcolor)

logging.info("Refinements...")
# grid - BUG with ndim images?
if show_grid:
    ax.coords.grid(color=accentcolor, ls='dotted', alpha=0.2)

# colorbar
if show_cbar:
    if args.cbar_vertical:
        addCbar(fig, plottype, im, header, float(int_min), float(int_max), fontsize=fontsize+1,cbanchor=[0.772, 0.11, 0.03, 0.77], orientation='vertical')
    else:
        addCbar(fig, plottype, im, header, float(int_min), float(int_max), fontsize=fontsize+1)

# scalebar
if show_scalebar:
    addScalebar(ax, wcs, z, kpc, fontsize, color=accentcolor)

# regions
if regions is not None:
    if isinstance(regions, str):
        regions = [regions]
    for region in regions:
        logging.info('Adding region: '+ str(region))
        addRegion(region, ax, header)

# # markers
# from matplotlib.patches import Rectangle, Circle # note this is stretched as ra is squeezed in angles
# r = Rectangle((1500, 1500), 3000, 3000, edgecolor='red', facecolor='none') # LR corner + width and height
# from astropy import units as u
# from astropy.visualization.wcsaxes import SphericalCircle # this is a real circle
# r = SphericalCircle((268.083333 * u.deg, 44.703333 * u.deg), 3.75/2. * u.arcmin,
#                      edgecolor='yellow', facecolor='none',
#                      transform=ax.get_transform('fk5'))
# ax.add_patch(r)
#
# ax.scatter([90.8345611,90.8275587], [42.1889537,42.2431142], edgecolor='red', facecolor=(1, 0, 0, 0.5), transform=ax.get_transform('world'))

# labels
if show_axes:
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

try:
    if np.any(all_limit_mask == 0):  # only if we have any upper limits
        import matplotlib.patches as mpatches
        handles = []
        if np.any(ul_mask == 0):  # only if we have any upper limits
            handles.append(mpatches.Patch( facecolor='k', hatch=r'arr{270}{9}{2}', label='Upper limits', fill=False))
        if np.any(ll_mask == 0):  # only if we have any upper limits
            handles.append(mpatches.Patch( facecolor='k', hatch=r'arr{90}{9}{2}', label='Lower limits', fill=False))
        legend = ax.legend(handles = handles, loc=2, fontsize=fontsize, handleheight=1.5)
except NameError:
    pass

if args.transparent or args.png:
    outfile = outfile.replace('pdf', 'png')
logging.info("Saving..."+outfile)
fig.savefig(outfile, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
