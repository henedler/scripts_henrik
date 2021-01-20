#!/usr/bin/env python
#
# This script can be used to tune the parameters of WSCLEAN to a specific source
# (experimental)
#
# Copyright (C) 2020  Henrik Edler, contributed code from Francesco de Gasperin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import argparse
import logging
import glob
import numpy as np
from scipy.optimize import minimize
import astropy.io.fits as pyfits
import pyregion

# TODO python 3.6 test
def flatten(f, channel = 0, freqaxis = 0):
    """
    Flatten a fits file so that it becomes a 2D image. Return new header and data
    """
    from astropy import wcs

    naxis=f[0].header['NAXIS']
    if (naxis == 2):
        return f[0].header,f[0].data

    w               = wcs.WCS(f[0].header)
    wn              = wcs.WCS(naxis = 2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt    = w.wcs.cdelt[0:2]
    wn.wcs.crval    = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    header["NAXIS1"] = f[0].header['NAXIS1']
    header["NAXIS2"] = f[0].header['NAXIS2']
    copy=('EQUINOX','EPOCH')
    for k in copy:
        r = f[0].header.get(k)
        if (r):
            header[k] = r

    slicing = []
    for i in range(naxis,0,-1):
        if (i <= 2):
            slicing.append(np.s_[:],)
        elif (i == freqaxis):
            slicing.append(channel)
        else:
            slicing.append(0)

    # slice=(0,)*(naxis-2)+(np.s_[:],)*2
    return header, f[0].data[tuple(slicing)]


def target_region_rms(imgname, region):
    """
    Calculate the
    Parameters
    ----------
    filename
    region

    Returns
    -------

    """
    # open fits
    with pyfits.open(imgname) as fits:
        header, data = flatten(fits)
        r = pyregion.open(region)
        mask = r.get_mask(header=header, shape=data.shape)
        rms = np.std(data[mask])
    return rms


def clean(msfiles, imgprefix="paramsearch", multiscales="0,10,20,40",
          scalebias = 0.67, weight=-0.3, thresh=1, mask_thresh=3, pxscale=3.8,
          imsize="800 800"):
    """ Call WSCLEAN using the specified arguments """
    # performance parameters
    ncpu = os.cpu_count()
    mgain = 0.85

    # clean
    call = (f"wsclean " \
            f"-name {imgprefix} " \
            f"-size {imsize} " \
            f"-scale {str(pxscale)+'arcsec'} " \
            f"-weight briggs {weight} " \
            f"-niter 10000 " \
            f"-minuv-l 30 " \
            f"-mgain {mgain} " \
            f"-baseline-averaging 5 " \
            f"-no-update-model-required " \
            f"-j {ncpu} " \
            f"-parallel-deconvolution {ncpu*16} " \
            f"-parallel-reordering {ncpu} " \
            f"-temp-dir paramsearch " \
            f"-auto-threshold {thresh} " \
            f"-auto-mask {mask_thresh} " \
            f"-join-channels " \
            f"-fit-spectral-pol 3 " \
            f"-channels-out 9 " \
            f"-deconvolution-channels 3 " \
            f"-multiscale  " \
            f"-multiscale-scales {multiscales}  " \
            f"-multiscale-scale-bias {scalebias}  " \
            f"-multiscale-gain 0.15 ")
            # f"-use-idg " \
            # f"-idg-mode hybrid " \
            # f"-mem 10 " \
    # only compute psf and dirty image for the very first iteration
    global iteration
    if iteration > 0:
        call += f"-reuse-psf {imgprefix} "
        call += f"-reuse-dirty {imgprefix} "
    iteration += 1

    call += f"{msfiles} "
    call += f">> paramsearch/out_clean_params.log"

    logging.debug(f"Start cleaning iteration {iteration}")
    os.system(call)
    return 0


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
logging.info('Find Clean Parameters -- Henrik Edler')

# Check python version..
if sys.version_info < (3,6):
    logging.error("You need atleast python3.6")
    sys.exit(1)

parser = argparse.ArgumentParser(description = "Estimate adequate WSCLEAN hyperparameters for one source.")
parser.add_argument("msfiles", nargs="*", help="Input ms files")
parser.add_argument("-r", "--region", help='target region', type=str)
args = parser.parse_args()

msfiles = args.msfiles
msfiles = " ".join(msfiles)
region = args.region


iteration = 1
def residual_merit(multiscales):
    if not os.path.isdir("paramsearch"):
        os.mkdir("paramsearch")
    imgname = "paramsearch/paramsearch"
    # clean(msfiles, imgprefix=imgname, multiscales=f"{multiscales[0]},{multiscales[1]},{multiscales[2]},{multiscales[3]}")
    clean(msfiles, imgprefix=imgname, multiscales=f"0,{multiscales[0]},20,46.25")
    target_rms = target_region_rms(imgname+'-MFS-residual.fits', region)
    logging.info(f"Multiscale-max {multiscales} || target RMS: {target_rms}")
    return target_rms

res = minimize(residual_merit, 5, method="Nelder-Mead", options={'maxiter':50,
                                                                    'xatol':0.1,
                                                                    'fatol':0.00005})
logging.info(f"Success: {res.success}, X_final: {res.x}, Target_RMS_final: {res.fun}, nit: {res.nit}")
