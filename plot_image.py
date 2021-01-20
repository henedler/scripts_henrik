#!/usr/bin/python

import sys, os
import numpy as np
import argparse
import logging
from lib_fits import flatten

import matplotlib
matplotlib.use('Agg') # aplpy api suggestion
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import (LogStretch, SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from astropy import units as u

logging.root.setLevel(logging.INFO)

def add_scalebar(ax, wcs, z, kpc, color='black'):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    print("-- Redshift: %f" % z)
    degperpixel = np.abs(wcs.all_pix2world(0,0,0)[1] - wcs.all_pix2world(0,1,0)[1]) # delta deg for 1 pixel
    degperkpc = cosmo.arcsec_per_kpc_proper(z).value/3600.
    pixelperkpc = degperkpc/degperpixel
    scalebar = AnchoredSizeBar(ax.transData, kpc*pixelperkpc, '%i kpc' % kpc, 'lower right', pad=0.5, color=color, frameon=False, sep=5, label_top=True, size_vertical=2)
    ax.add_artist(scalebar)

def setSize(wcs, ra, dec, size_ra, size_dec):
    """
    Properly set bottom left and top right pixel assuming a center and a size in deg
    """
    # bottom
    dec_b = dec - size_dec/2.
    # top
    dec_t = dec + size_dec/2.
    # bottom left
    ra_l = ra-size_ra/np.cos(dec_b*np.pi/180)/2.
    # top right
    ra_r = ra+size_ra/np.cos(dec_t*np.pi/180)/2.

    x,y = wcs.wcs_world2pix([ra_l,ra_r]*u.deg, [dec_b,dec_t]*u.deg, 1, ra_dec_order=True)
    ax.set_xlim(x[1], x[0])
    ax.set_ylim(y[0], y[1])
    return x.astype(int), y.astype(int)

def addBeam(ax, hdr, edgecolor='black'):
    """
    hdr: fits header of the file
    """
    from radio_beam import Beam

    bmaj = hdr['BMAJ']
    bmin = hdr['BMIN']
    bpa = hdr['BPA']
    beam = Beam(bmaj*u.deg,bmin*u.deg,bpa*u.deg)

    assert np.abs(hdr['CDELT1']) == np.abs(hdr['CDELT2'])
    pixscale = np.abs(hdr['CDELT1'])
    posx = ax.get_xlim()[0]+bmaj/pixscale
    posy = ax.get_ylim()[0]+bmaj/pixscale
    r = beam.ellipse_to_plot(posx, posy, pixscale *u.deg)
    r.set_edgecolor(edgecolor)
    r.set_facecolor('white')
    ax.add_patch(r)

def addRegion(regionfile, ax):
    import pyregion
    reg = pyregion.open(regionfile)
    reg = reg.as_imagecoord(header)
    patch_list, artist_list = reg.get_mpl_patches_texts()
    for p in patch_list:
        ax.add_patch(p)
    for a in artist_list:
        ax.add_artist(a)

parser = argparse.ArgumentParser(description='Basic plotting script for fits images')
parser.add_argument('image', help='fits image to plot.')
parser.add_argument('--region', nargs='+', help='ds9 region files to plot (optional).')
parser.add_argument('--si', help='Make SI plot?', action='store_true')

args = parser.parse_args()
if args.image == None:
    logging.error('No input image found.')
    sys.exit()

filename = args.image
regions = args.region

# Plot extensions
center = [157.945, 35.049] # deg
size = [0.14, 0.14] # deg

# Style
is_spidx = args.si
fontsize = 16

# Scalebar
plot_scalebar = True
z = 0.1259 # redshift
kpc = 150 # how many kpc is the scalebar?
# Colorbar
show_cbar = True

logging.info('Setting up...')
header, data = flatten(filename)
if not is_spidx:
    data *= 1e3 # to mJy
wcs = WCS(header)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=wcs, slices=('x', 'y'))
lon = ax.coords['ra']
lat = ax.coords['dec']


# zoom in in pixel
xrange, yrange = setSize(wcs, center[0], center[1], *size)
print(xrange, yrange)
logging.info('Plotting  {}-{}, {}-{} from {}x{}.'.format(xrange[1], xrange[0], yrange[0], yrange[1], len(data[0]), len(data[:,0])))
data_visible = data[xrange[1]:xrange[0],yrange[0]:yrange[1]] # select only data that is visible in plot

# normalizer
if is_spidx:
    interval = PercentileInterval(99)
    stretch = LinearStretch()
    int_min, int_max = interval.get_limits(data_visible)
else:
    interval = AsymmetricPercentileInterval(90,99.99) # 80 - 99.99 percentile
    stretch = SqrtStretch()
    int_min, int_max= interval.get_limits(data_visible)
    int_min = -0.5 * np.nanstd(data_visible[np.abs(data_visible) < 5 * np.nanstd(data_visible)])  # possibly use 1 sigma for min

logging.info('min: {},  max: {}'.format(int_min,int_max))
norm = ImageNormalize(data, vmin=int_min, vmax=int_max, stretch=stretch)

# bkgr image
logging.info("Image...")
if is_spidx:
    im = ax.imshow(data, origin='lower',  interpolation='nearest', cmap='jet', norm=norm)
else:
    im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='cubehelix', norm=norm)

# contours
# print("Contour...")
header_c, data_c = flatten(filename)
# ax.contour(data_c, transform=ax.get_transform(WCS(header_c)), levels=np.logspace(-2, 0., 5), colors='red', alpha=0.5)

# add beam
accentcolor = 'black' if is_spidx else 'white'
addBeam(ax, header_c, edgecolor=accentcolor)

logging.info("Refinements...")
# grid - BUG with ndim images?
ax.coords.grid(color=accentcolor, ls='dotted', alpha=0.2)

# colorbar
if show_cbar:
    cbaxes = fig.add_axes([0.127, 0.89, 0.772, 0.02])
    fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.tick_top()
    if is_spidx:
        cbaxes.xaxis.set_label_text('Spectral Index',fontsize=fontsize)
    else:
        cbaxes.xaxis.set_label_text(r'Flux density (mJy beam$^{-1}$)', fontsize=fontsize)
    cbaxes.xaxis.set_label_position('top')

# scalebar
if plot_scalebar:
    add_scalebar(ax, wcs, z, kpc, color='white')

# regions
if regions is not None:
    if isinstance(regions, str):
        regions = [regions]
    for region in regions:
        logging.info('Adding region: '+ str(region))
        addRegion(region, ax)

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
lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize)
lat.set_axislabel('Declination (J2000)', fontsize=fontsize)
lon.set_ticklabel(size=fontsize)
lat.set_ticklabel(size=fontsize)

# small img
lon.set_major_formatter('hh:mm:ss')
lat.set_major_formatter('dd:mm')
lat.set_ticklabel(rotation=90) # to turn dec vertical

logging.info("Saving..."+filename.replace('fits','pdf'))
fig.savefig(filename.replace('fits','pdf'), bbox_inches='tight')

# small image
#os.system('pdf_reducer.sh toothLBA.pdf')
