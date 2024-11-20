#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 - Francesco de Gasperin
# 2021 - Modified by Henrik Edler
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
from lib_linearfit import linear_fit_bootstrap, linsq_spidx
import lib_fits
from lib_fits import AllImages
from astropy.io import fits as pyfits
from astropy.wcs import WCS as pywcs
# https://github.com/astrofrog/reproject
from reproject import reproject_interp, reproject_exact
reproj = reproject_exact
logging.root.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Make spectral index maps, e.g. spidxmap.py --region ds9.reg --noise --sigma 5 --save *fits')
parser.add_argument('images', nargs='+', help='List of images to use for spidx')
parser.add_argument('--beam', dest='beam', nargs='+', type=float, help='3 parameters final beam to convolve all images (BMAJ (arcsec), BMIN (arcsec), BPA (deg))')
parser.add_argument('--bgreg', dest='bgreg', help='DS9 region file for background estimation.')
parser.add_argument('--region', dest='region', help='Ds9 region to restrict analysis')
parser.add_argument('--size', dest='size', type=float, help='Size (horizontal and vertical) of final image in degree')
parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
parser.add_argument('--shift', dest='shift', action='store_true', help='Shift images before calculating spidx')
parser.add_argument('--noise', dest='noise', action='store_true', help='Calculate noise of each image')
parser.add_argument('--save', dest='save', action='store_true', help='Save intermediate results')
parser.add_argument('--force', dest='force', action='store_true', help='Force remake intermediate results')
parser.add_argument('--sigma', dest='sigma', type=float, help='Restrict to pixels above this sigma in all images')
parser.add_argument('--fluxscaleerr', nargs='*', dest='fluxscaleerr', type=float, default=0.0, help='Systematic error of flux scale. One value for all images.')
parser.add_argument('--upperlimit', dest='upperlimit', type=float, help='Place upper limits below this value if not detected at highest frequency at sigma. Float, e.g. -1.0')
parser.add_argument('--lowerlimit', dest='lowerlimit', type=float, help='Place lower limits below this value if not detected at lowest frequency at sigma. Float, e.g. -0.6')
parser.add_argument('--circbeam', dest='circbeam', action='store_true', help='Force final beam to be circular (default: False, use minimum common beam area)')
parser.add_argument('--output', dest='output', default='spidx.fits', help='Name of output mosaic (default: spidx.fits)')

args = parser.parse_args()

# check input
if len(args.images) < 2:
    logging.error('Requires at lest 2 images.')
    sys.exit(1)
elif len(args.images) > 2 and (args.upperlimit or args.lowerlimit):
    logging.error('Upper-limit currently only supported for two frequencies')
    sys.exit()

if args.beam is not None and len(args.beam) != 3:
    logging.error('Beam must be in the form of "BMAJ BMIN BPA" (3 floats).')
    sys.exit(1)

if args.radec is not None and len(args.radec) != 2:
    logging.error('--radec must be in the form of "RA DEC" (2 floats).')
    sys.exit(1)

if args.sigma and not args.noise:
    logging.error('Cannot use --sigma flag without calculating noise. Provide also --noise.')
    sys.exit(1)

if len(args.fluxscaleerr) == 1:
    fluxscaleerr = args.fluxscaleerr[0]*np.ones(len(args.images))
else:
    fluxscaleerr = np.array(args.fluxscaleerr)
    if not (len(args.images) == len(args.fluxscaleerr)):
        logging.error(f'Either provide one fluxscaleerr for all images or excatly one per image.')
        sys.exit(1)

if __name__ == '__main__':
    ########################################################
    # prepare images and make catalogues if necessary
    all_images = AllImages([imagefile for imagefile in args.images])

    #####################################################
    # find+apply shift w.r.t. first image
    if args.shift:
        if all_images.suffix_exists('si-shift') and not args.force:
            logging.info('Reuse si-shift images.')
            all_images = AllImages([name.replace('.fits', '-si-shift.fits') for name in all_images.filenames])
        else:
            all_images.align_catalogue()
            if args.save: all_images.write('si-shift')

    #########################################################
    # convolve
    if all_images.suffix_exists('si-conv') and not args.force:
        logging.info('Reuse si-conv images.')
        all_images = lib_fits.AllImages([name.replace('.fits', '-si-conv.fits') for name in all_images.filenames])
    else:
        if args.beam:
            all_images.convolve_to(args.beam, args.circbeam)
        else:
            all_images.convolve_to(circbeam=args.circbeam)
        if args.save: all_images.write('si-conv')
    # regrid
    if all_images.suffix_exists('si-conv-regr') and not args.force:
        logging.info('Reuse si-regr images.')
        all_images = lib_fits.AllImages([name.replace('.fits', '-si-conv-regr.fits') for name in all_images.filenames])
    else:
        all_images.regrid_common(size=args.size, radec=args.radec)
        if args.save: all_images.write('si-conv-regr')

    for i, image in enumerate(all_images):
        if args.noise:
            if args.sigma:
                image.calc_noise(sigma=args.sigma, bg_reg=args.bgreg)  # after mask?/convolution
                print(image.noise)
                image.blank_noisy(args.sigma)
            else:
                image.calc_noise() # after mask?/convolution
        if args.region is not None:
            image.apply_region(args.region, invert=True) # after convolution to minimise bad pixels


    #########################################################
    # do spdix and write output
    rwcs = pywcs(naxis=2)
    rwcs.wcs.ctype = all_images[0].get_wcs().wcs.ctype
    rwcs.wcs.cdelt = all_images[0].get_wcs().wcs.cdelt
    rwcs.wcs.crval = all_images[0].get_wcs().wcs.crval
    rwcs.wcs.crpix = all_images[0].get_wcs().wcs.crpix
    xsize, ysize = all_images[0].img_data.shape # might be swapped
    regrid_hdr = rwcs.to_header()
    regrid_hdr['NAXIS'] = 2
    regrid_hdr['NAXIS1'] = xsize
    regrid_hdr['NAXIS2'] = ysize
    frequencies = [ image.freq for image in all_images ]
    regrid_hdr['FREQLO'] = np.min(frequencies)
    regrid_hdr['FREQHI'] = np.max(frequencies)
    b = all_images[0].get_beam()
    assert np.all([image.get_beam() == b for image in all_images])
    regrid_hdr['BMAJ'] = b[0]
    regrid_hdr['BMIN'] = b[1]
    regrid_hdr['BPA'] = b[2]
    if args.noise: yerr = np.array([ image.noise for image in all_images ])
    else: yerr = None
    spidx_data = np.empty(shape=(xsize, ysize))
    spidx_data[:] = np.nan
    spidx_err_data = np.empty(shape=(xsize, ysize))
    spidx_err_data[:] = np.nan

    ul, ll = 0, 0
    for i in range(xsize):
        print('.', end=' ')
        sys.stdout.flush()
        for j in range(ysize):
            val4reg = np.array([ image.img_data[i,j] for image in all_images ])
            # if args.upperlimit:
            #     if np.isnan(val4reg[:-1]).any(): continue # all but last for UL
            # if args.lowerlimit:
            #     if np.isnan(val4reg[1:]).any(): continue  # all but first for LL
            # else:
            #     if np.isnan(val4reg).any(): continue
            if np.isnan(val4reg).all(): continue
            if len(frequencies) == 2:
                if args.upperlimit and np.isnan(val4reg[-1]):
                    this_err = np.sqrt(yerr[0] ** 2 + (fluxscaleerr[0] * val4reg[0]) ** 2)
                    spidx_data[i,j], _ = linsq_spidx(frequencies, [val4reg[0], args.sigma*yerr[-1]], [this_err, 0.])
                    if spidx_data[i,j] < args.upperlimit: # only show upper limits lower than this
                        regrid_hdr[f'UL{ul}'] = f'{i},{j}'
                        ul += 1
                    else:
                        spidx_data[i, j] = np.nan
                        spidx_err_data[i, j] = np.nan
                elif args.lowerlimit and np.isnan(val4reg[0]):
                    this_err = np.sqrt(yerr[-1] ** 2 + (fluxscaleerr[1] * val4reg[1]) ** 2)
                    spidx_data[i, j], _ = linsq_spidx(frequencies, [args.sigma * yerr[0], val4reg[1]],  [0., this_err])
                    if spidx_data[i, j] > args.lowerlimit:  # only show upper limits lower than this
                        regrid_hdr[f'LL{ll}'] = f'{i},{j}'
                        ll += 1
                    else:
                        spidx_data[i,j] = np.nan
                        spidx_err_data[i,j] = np.nan
                else:
                    this_err = np.sqrt(yerr ** 2 + (fluxscaleerr * val4reg) ** 2)
                    spidx_data[i,j], spidx_err_data[i,j] = linsq_spidx(frequencies, val4reg, this_err)
            else:
                (a, b, sa, sb) = linear_fit_bootstrap(x=frequencies, y=val4reg, yerr=yerr, tolog=True)
                spidx_data[i,j] = a
                spidx_err_data[i,j] = sa


    spidx = pyfits.PrimaryHDU(spidx_data, regrid_hdr)
    spidx_err = pyfits.PrimaryHDU(spidx_err_data, regrid_hdr)
    outname = args.output
    if not outname[-5::] == '.fits':
        outname += '.fits'
    spidx.writeto(outname, overwrite=True)
    spidx_err.writeto(outname.replace('.fits','-err.fits'), overwrite=True)

