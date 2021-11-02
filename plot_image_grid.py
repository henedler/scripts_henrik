#!/usr/bin/python
import sys, os
import numpy as np
import argparse
import logging
from lib_fits import flatten, Image

import matplotlib
matplotlib.use('Agg') # aplpy api suggestion
import matplotlib.pyplot as plt
import matplotlib.hatch
from astropy.wcs import WCS
from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from lib_plot import addRegion, addCbar, addBeam, addScalebar, setSize
logging.root.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Plotting for multiple sub-images from a single fits file.')
parser.add_argument('image', help='fits image to plot.')
parser.add_argument('--region', nargs='+', help='ds9 region files to plot (optional).')
parser.add_argument('-o', '--outfile', default=None, help='prefix of output image')
parser.add_argument('--interval', default=None, nargs=2, help='Provide min/max interval.')

args = parser.parse_args()
if args.image == None:
    logging.error('No input image found.')
    sys.exit()

filename = args.image
regions = args.region
if args.outfile:
    outfile = args.outfile+'.pdf'
else:
    outfile = args.image.replace('fits','pdf')

# Plot extensions
centers = [[159.351370,35.804950],
          [159.714410,36.020990],
          [158.646110,33.825020],
          [159.494540,34.996100],
          [157.956570,35.042770],
          [158.532120,35.741120]]

size = [0.2, 0.2] # deg
# stretch type (only stokes) 'log' (for extended) or 'sqrt' (for compact)
stretch_type = 'sqrt'
# Style
fontsize = 13

# Scalebar
plot_scalebar = False
z = 0.1259 # redshift
kpc = 200 # how many kpc is the scalebar?

# Colorbar
show_cbar = False

# Contours
show_contours = False
contout_base_sigma = 3 # will be this times [1,2,4,8,16]
n_contour = 9


logging.info('Setting up...')
header, data = flatten(filename)
img = Image(filename)
data *= 1e3 # to mJy
sigma = 1.0 #img.calc_noise(betaSigma=True)*1e3
logging.info(f'Noise is {sigma:.2e} mJy/beam')

wcs = WCS(header)

# fig, axs = plt.subplots(3, 2, gridspec_kw={'hspace': -0.69, 'wspace': 0.15}, subplot_kw={'projection':wcs},
#                         figsize=(3 * 5, 2 * 5))
fig, axs = plt.subplots(3, 2, gridspec_kw={'hspace': 0.11, 'wspace': -0.71}, subplot_kw={'projection':wcs},
                        figsize=(3 * 5, 2 * 5))

# normalizer
interval = AsymmetricPercentileInterval(80, 99.9997)#99.99)  # 80 - 99.99 percentile
if stretch_type == 'sqrt':
    stretch = SqrtStretch()
elif stretch_type == 'log':
    stretch = LogStretch()
else:
    print('Stretch type unknown.')
    sys.exit(0)
if args.interval:
    int_min, int_max = args.interval
else:
    int_min, int_max = interval.get_limits(data)
    int_min = 1 * sigma

logging.info('min: {},  max: {}'.format(int_min,int_max))
norm = ImageNormalize(data, vmin=float(int_min), vmax=float(int_max), stretch=stretch)

# iterate over tile images
for i, center in enumerate(centers):
    j = 0
    stop = False
    for row in range(len(axs)):
        for col in range(len(axs[0])):
            if j == i:
                ax = axs[row,col]
                stop = True
                break
            j += 1
        if stop: break

    # zoom in in pixel
    xrange, yrange = setSize(ax, wcs, center[0], center[1], *size)
    logging.info('Plotting  {}-{}, {}-{} from {}x{}.'.format(xrange[1], xrange[0], yrange[0], yrange[1], len(data[0]), len(data[:,0])))
    data_visible = data[xrange[1]:xrange[0],yrange[0]:yrange[1]] # select only data that is visible in plot


    # bkgr image
    logging.info("Image...")
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='cubehelix', norm=norm)

    # contours
    if show_contours:
        print("Contour...")
        contour_limits = contout_base_sigma * 2**np.arange(n_contour) * sigma
        print(contour_limits)
        ax.contour(data, transform=ax.get_transform(WCS(header)), levels=contour_limits, colors='grey', alpha=0.7)
        ax.contour(data, transform=ax.get_transform(WCS(header)), levels=-contour_limits[::-1], colors='grey', alpha=0.7, linestyles='dashed')

    # add beam
    accentcolor = 'white'
    addBeam(ax, header, edgecolor=accentcolor)

    logging.info("Refinements...")
    # grid - BUG with ndim images?
    # ax.coords.grid(color=accentcolor, ls='dotted', alpha=0.2)

    # scalebar
    if plot_scalebar:
        addScalebar(ax, wcs, z, kpc, fontsize, color=accentcolor)

    # regions
    if regions is not None:
        if isinstance(regions, str):
            regions = [regions]
        for region in regions:
            logging.info('Adding region: '+ str(region))
            addRegion(region, ax, header)

    # labels
    lon = ax.coords['ra']
    lat = ax.coords['dec']

    if row == len(axs)-1:
        lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize-1)
    else:
        lon.set_axislabel('  ')
    if col == 0:
        lat.set_axislabel('Declination (J2000)', fontsize=fontsize-1)
    else:
        lat.set_axislabel('  ')
    lon.set_ticklabel(size=fontsize-4)
    lat.set_ticklabel(size=fontsize-4)

    # small img
    lon.set_major_formatter('hh:mm:ss')
    lat.set_major_formatter('dd:mm')
    lat.set_ticklabel(rotation=90) # to turn dec vertical

# colorbar
if show_cbar:
    addCbar(fig, 'stokes', im, header, int_max, fontsize=fontsize, cbanchor=[0.127, 0.69, 0.772, 0.02])

logging.info("Saving..."+outfile)
fig.savefig(outfile, bbox_inches='tight')

# small image
#os.system('pdf_reducer.sh toothLBA.pdf')
