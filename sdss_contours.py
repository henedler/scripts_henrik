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
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
from reproject import reproject_exact

from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from lib_plot import addRegion, addCbar, addBeam, addScalebar
from lib_fits import flatten


def get_overlay_image(coord: SkyCoord, size, wcs: WCS, window='optical') -> [np.array]:
    """
    Returns an image from the SDSS image query in bytes format. Can be used
    to set as RGB background for FITSimages in aplpy.
    We check if the img exists in astroquery (https://astroquery.readthedocs.io/en/latest/skyview/skyview.html) cat
    and then get it from hips2fits query (https://aladin.u-strasbg.fr/hips/list)

    :param coord: Coordinate object for the center
    :param size: float, size of the cutout in deg
    :param wcs: WCS of the fits file
    :return: img, io.BytesIO object, visual image of the chosen region
    """
    def get_img_from_hips(survey):
        # survey: list of len two: [survey name to check for image existance, hips sever path]
        print(f"Trying {survey}")
        # Load the Image
        return PIL.Image.fromarray(hips2fits.query_with_wcs(survey, wcs=wcs, format="png"))

    image = None
    survey_list_optical = [ 'CDS/P/SDSS9/color', 'CDS/P/DSS2/color'] # 'CDS/P/PanSTARRS/DR1/color-i-r-g',
    survey_list_uv = ['CDS/P/GALEXGR6/AIS/color']
    survey_list_ir = ['ov-gso/P/HeViCS/350', 'ESAVO/P/HERSCHEL/SPIRE-350']

    if window.lower() == 'optical':
        survey_list = survey_list_optical
    elif window.lower() == 'ir':
        survey_list = survey_list_ir
    elif window.lower() == 'uv':
        survey_list = survey_list_uv
    else:
        raise ValueError(f'Spectral window {window} unknow, use optical / ir / uv.')

    for survey in survey_list:
        try:
            image = get_img_from_hips(survey)
            if image is not None:
                break
        except urllib.error.HTTPError as e:
            print(f"Not found in {survey[0]}")
    if image is None:
        raise ValueError('Could not find image in databases...')
    return image


parser = argparse.ArgumentParser(description='Plotting script to overlay fits contours on SDSS image')
parser.add_argument('image', help='fits image to plot.')
parser.add_argument('target', help='Name of target (Messier, NGC, VCC, IC...)')
parser.add_argument('--titlename', help='Title name of target')
parser.add_argument('-s', '--size', type=float, default=8., help='size in arcmin')
parser.add_argument('-z', '--redshift', type=float, default=0.0043, help='redshift.')
parser.add_argument('-n', '--noise', type=float, default=0.2, help='Hardcode noise level in mJy/beam.')
parser.add_argument('-u', '--upsample', type=int, default=2, help='Upsample the image by this factor.')
parser.add_argument('--noisemap', type=str, help='Noise map in mJy/beam.')
parser.add_argument('--skip', action='store_true', help='Skip existing plots?')
parser.add_argument('--window', type=str, help='optical (default) / IR / UV ')
parser.add_argument('-o', '--outfile', default=None, help='prefix of output image')

args = parser.parse_args()
if args.image == None:
    logging.error('No input image found.')
    sys.exit()

# Usage:
fontsize = 12
name = args.target  # NGC ... and M.. names definitely work
titlename = args.titlename if args.titlename else args.target
if os.path.exists(titlename + '.png') and args.skip:
    print(f'{titlename}.png exists - exiting.')
    sys.exit(0)
size = args.size * u.arcmin  # Size of the image in arcmin (so 10'x10')
# Load FITS file
header, data = flatten(args.image)
data = data * 1000
in_wcs = WCS(header)
coord = SkyCoord.from_name(name)
# Cutout central region
cutout = Cutout2D(data, coord, size=size, wcs=in_wcs)
noise = args.noise
if args.noisemap:
    print(f"Using {args.noisemap} to calculate noise...")
    # Load FITS file
    wcs_n, data_n = flatten(args.noisemap)
    data_n = data_n * 1000
    cutout_n = Cutout2D(data_n, coord, size=size, wcs=wcs_n)
    noise = np.nanstd(cutout_n.data)
    print(f"Found background rms: {noise:.3f}mJy/beam.")

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
# sample_data = scipy.ndimage.filters.gaussian_filter(sample_data, factor/3)

# get background image from hips
uwcs = WCS(uhdr)
image = get_overlay_image(coord, size=[uhdr['NAXIS1'], uhdr['NAXIS2']], wcs=uwcs, window=args.window)

# Plot image with e.g. matplotlib
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=uwcs, slices=('x', 'y'))
lon = ax.coords['ra']
lat = ax.coords['dec']
ax.imshow(image, origin="lower", cmap="gray", interpolation='kaiser')
contour_limits = 3 * 2 ** np.arange(20) * noise
vmin, vmax = contour_limits[0], np.nanmax(sample_data)
norm = ImageNormalize(sample_data, vmin=vmin, vmax=vmax, stretch=SqrtStretch())
ax.contour(sample_data, levels=contour_limits, cmap='Reds', norm=norm, alpha=1, linewidths=1)
ax.contour(sample_data, levels=-contour_limits[::-1], cmap='Reds', alpha=1, linewidths=1, linestyles='dashed', norm=norm)

addScalebar(ax, uwcs, args.redshift, 10, fontsize, color='white')
addBeam(ax, header, edgecolor='white')

lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize)
lat.set_axislabel('Declination (J2000)', fontsize=fontsize)
lon.set_ticklabel(size=fontsize)
lat.set_ticklabel(size=fontsize)

# small img
lon.set_major_formatter('hh:mm:ss')
lat.set_major_formatter('dd:mm')
lat.set_ticklabel(rotation=90)  # to turn dec vertical

plt.title(f"{titlename}, "+r"$\sigma_{rms}=$"+f"{noise:.3f}mJy/beam", size=fontsize+2)
plt.savefig(titlename + '.png', bbox_inches='tight')
