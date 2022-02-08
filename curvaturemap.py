#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 - Henrik Edler, Francesco de Gasperin
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

parser = argparse.ArgumentParser(description='Make spectral index maps, e.g. curvaturemap.py --region ds9.reg --save *fits')
parser.add_argument('images', nargs=2, help='List of spidxmaps to use for curvature')
parser.add_argument('--beam', dest='beam', nargs='+', type=float, help='3 parameters final beam to convolve all images (BMAJ (arcsec), BMIN (arcsec), BPA (deg))')
parser.add_argument('--region', dest='region', help='Ds9 region to restrict analysis')
parser.add_argument('--size', dest='size', type=float, help='Size (horizontal and vertical) of final image in degree')
parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
parser.add_argument('--save', dest='save', action='store_true', help='Save intermediate results')
parser.add_argument('--force', dest='force', action='store_true', help='Force remake intermediate results')
parser.add_argument('--circbeam', dest='circbeam', action='store_true', help='Force final beam to be circular (default: False, use minimum common beam area)')
parser.add_argument('--output', dest='output', default='survature.fits', help='Name of output mosaic (default: curvature.fits)')

args = parser.parse_args()

# check input
if len(args.images) != 2:
    logging.error('Requires 2 images.')
    sys.exit(1)

if args.beam is not None and len(args.beam) != 3:
    logging.error('Beam must be in the form of "BMAJ BMIN BPA" (3 floats).')
    sys.exit(1)

if args.radec is not None and len(args.radec) != 2:
    logging.error('--radec must be in the form of "RA DEC" (2 floats).')
    sys.exit(1)

from lib_fits import Image, find_freq


if __name__ == '__main__':
    ########################################################
    # prepare images and make catalogues if necessary
    all_images = []
    all_errors = []
    all_beams = []
    for imagefile in args.images:
        image = Image(imagefile)
        all_beams.append(image.get_beam())
        all_images.append(image)
        si_err_image = imagefile.replace('.fits', '-err.fits')
        if os.path.isfile(si_err_image):
            err_image = Image(si_err_image)
            assert err_image.get_beam() == image.get_beam()
            all_errors.append(err_image)
        else:
            raise FileNotFoundError(f'SI error image {si_err_image} does not exist.')
    #####################################################
    # find the smallest common beam
    if args.beam is None:
        if args.circbeam:
            maxmaj = np.max([b[0] for b in all_beams])
            target_beam = [maxmaj*1.01, maxmaj*1.01, 0.] # add 1% to prevent crash in convolution
        else:
            from radio_beam import Beams
            my_beams = Beams([b[0] for b in all_beams] * u.deg, [b[1] for b in all_beams] * u.deg, [b[2] for b in all_beams] * u.deg)
            common_beam = my_beams.common_beam()
            target_beam = [common_beam.major.value, common_beam.minor.value, common_beam.pa.value]
    else:
        target_beam = [args.beam[0]/3600., args.beam[1]/3600., args.beam[2]]

    logging.info('Final beam: %.1f" %.1f" (pa %.1f deg)' \
        % (target_beam[0]*3600., target_beam[1]*3600., target_beam[2]))

    #####################################################
    # Generate regrid headers
    rwcs = pywcs(naxis=2)
    rwcs.wcs.ctype = all_images[0].get_wcs().wcs.ctype
    cdelt = target_beam[1]/5. # 1/5 of minor axes (deg)
    logging.info('Pixel scale: %f"' % (cdelt*3600.))
    rwcs.wcs.cdelt = [-cdelt, cdelt]
    if args.radec is not None:
        mra = args.radec[0] #*np.pi/180
        mdec = args.radec[1] #*np.pi/180
    else:
        mra = all_images[0].img_hdr['CRVAL1']
        mdec = all_images[0].img_hdr['CRVAL2']
    rwcs.wcs.crval = [mra,mdec]
    # Align image centers:
    # for image in all_images:
    #     image.apply_recenter_cutout(mra, mdec)

    # if size is not give is taken from the mask
    if args.size is None:
        if args.region is not None:
            r = pyregion.open(args.region)
            mask = r.get_mask(header=all_images[0].img_hdr, shape=all_images[0].img_data.shape)
            intermediate = pyfits.PrimaryHDU(mask.astype(float), all_images[0].img_hdr)
            intermediate.writeto('mask.fits', overwrite=True)
            w = all_images[0].get_wcs()
            y, x = mask.nonzero()
            ra_max, dec_max = w.all_pix2world(np.max(x), np.max(y), 0, ra_dec_order=True)
            ra_min, dec_min = w.all_pix2world(np.min(x), np.min(y), 0, ra_dec_order=True)
            args.size = 2*np.max( [ np.max([np.abs(ra_max-mra),np.abs(ra_min-mra)]), np.max([np.abs(dec_max-mdec),np.abs(dec_min-mdec)]) ] )
        else:
            logging.warning('No size or region provided, use entire size of first image.')
            sys.exit('not implemented')

    xsize = int(np.rint(args.size/cdelt))
    ysize = int(np.rint(args.size/cdelt))
    if xsize % 2 != 0: xsize += 1
    if ysize % 2 != 0: ysize += 1
    rwcs.wcs.crpix = [xsize/2,ysize/2]

    regrid_hdr = rwcs.to_header()
    regrid_hdr['NAXIS'] = 2
    regrid_hdr['NAXIS1'] = xsize
    regrid_hdr['NAXIS2'] = ysize
    regrid_hdr['BMAJ'], regrid_hdr['BMIN'], regrid_hdr['BPA'] = image.get_beam()
    logging.info('Image size: %f deg (%i %i pixels)' % (args.size,xsize,ysize))

    intermediate = pyfits.PrimaryHDU(all_images[1].img_data, all_images[1].img_hdr)
    intermediate.writeto('test.fits', overwrite=True)
    #########################################################
    # regrid, convolve and only after apply mask
    for image in all_images + all_errors:

        if os.path.exists(image.imagefile+'-conv.fits') and not args.force:
            data, hdr = pyfits.getdata(image.imagefile+'-conv.fits', 0, header=True)
            image.img_data = data
            image.img_hdr = hdr
            image.set_beam([hdr['BMAJ'], hdr['BMIN'], hdr['BPA']])
        else:
            image.convolve(target_beam, stokes=False)
            if args.save:
                image.write(image.imagefile+'-conv.fits', inflate=True)

        if os.path.exists(image.imagefile+'-regrid-conv.fits') and not args.force:
            data, hdr = pyfits.getdata(image.imagefile+'-regrid-conv.fits', 0, header=True)
            image.img_data = data
            image.img_hdr = hdr
            image.set_beam([hdr['BMAJ'], hdr['BMIN'], hdr['BPA']])
        else:
            image.regrid(regrid_hdr)
            if args.save:
                image.write(image.imagefile+'-regrid-conv.fits', inflate=True)

        if args.region is not None:
            image.apply_region(args.region, invert=True) # after convolution to minimise bad pixels

    #########################################################
    # do curvature and write output
    curv_data = (all_images[0].img_data - all_images[1].img_data)
    # not 100% sure if this is correct, probably the two SI errors can be correlated...
    curv_err_data = (all_errors[0].img_data**2 + all_errors[1].img_data**2)**0.5
    # SI data might contain upper limits! Set to nan where error is not defined!
    curv_data[np.isnan(curv_err_data)] = np.nan

    curv = pyfits.PrimaryHDU(curv_data, regrid_hdr)
    curv_err = pyfits.PrimaryHDU(curv_err_data, regrid_hdr)
    curv.writeto(args.output, overwrite=True)
    curv_err.writeto(args.output.replace('.fits','-err.fits'), overwrite=True)

