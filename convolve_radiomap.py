#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import os, sys, argparse, logging
import numpy as np
from lib_linearfit import linear_fit_bootstrap
from lib_fits import flatten
from astropy.io import fits as pyfits
from astropy.wcs import WCS as pywcs
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import SkyCoord
import astropy.units as u
import pyregion
# https://github.com/astrofrog/reproject
from reproject import reproject_interp, reproject_exact

reproj = reproject_exact
logging.root.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description='Convolve radiomap to larger beam')
parser.add_argument('image')
parser.add_argument('--beam', dest='beam', nargs='+', type=float,
                    help='3 parameters final beam to convolve all images (BMAJ (arcsec), BMIN (arcsec), BPA (deg))')
args = parser.parse_args()

from lib_fits import Image, find_freq

# find the smallest common beam
target_beam = [args.beam[0] / 3600., args.beam[1] / 3600., args.beam[2]]

logging.info('Final beam: %.1f" %.1f" (pa %.1f deg)' \
             % (target_beam[0] * 3600., target_beam[1] * 3600., target_beam[2]))

image = Image(args.image)

# Generate regrid headers
rwcs = pywcs(naxis=2)
rwcs.wcs.ctype = image.get_wcs().wcs.ctype
cdelt = target_beam[1] / 5.  # 1/5 of minor axes (deg)
logging.info('Pixel scale: %f"' % (cdelt * 3600.))
rwcs.wcs.cdelt = [-cdelt, cdelt]
mra = image.img_hdr['CRVAL1']
mdec = image.img_hdr['CRVAL2']
rwcs.wcs.crval = [mra, mdec]

xsize = int(np.rint(image.img_hdr['NAXIS1']*np.abs(image.img_hdr['CDELT1']/cdelt) ))
ysize =int(np.rint(image.img_hdr['NAXIS1']*np.abs(image.img_hdr['CDELT1']/cdelt) ))
if xsize % 2 != 0: xsize += 1
if ysize % 2 != 0: ysize += 1
rwcs.wcs.crpix = [xsize / 2, ysize / 2]

regrid_hdr = rwcs.to_header()
regrid_hdr['NAXIS'] = 2
regrid_hdr['NAXIS1'] = xsize
regrid_hdr['NAXIS2'] = ysize
regrid_hdr['CRPIX1'] = int(np.rint(image.img_hdr['CRPIX1']*np.abs(image.img_hdr['CDELT1']/cdelt) ))
regrid_hdr['CRPIX2'] = int(np.rint(image.img_hdr['CRPIX2']*np.abs(image.img_hdr['CDELT2']/cdelt) ))
regrid_hdr['BMAJ'], regrid_hdr['BMIN'], regrid_hdr['BPA'] = image.get_beam()

image.convolve(target_beam)
image.regrid(regrid_hdr)
image.write(args.image.replace('.fits', f'-{int(args.beam[0])}as.fits'))