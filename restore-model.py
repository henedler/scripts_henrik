#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henrik Edler
"""



from astropy.io import fits
import numpy as np
import argparse
import lib_fits
import sys
from reproj_test import reproject_interp_chunk_2d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='First input fits file.', type=str, default=None)
    parser.add_argument("image", help="Second input fits file", type=str, default=None)

    args = parser.parse_args()
    model = lib_fits.Image(args.model)
    image = lib_fits.Image(args.image)

    ### convolve model to beam of image
    print("WARNING: Assume beam of input image for model image")
    model.set_beam((1e-10,1e-10,0)) # zero does not work
    total_flux = np.nansum(model.img_data)
    print(f'Total flux = {total_flux}')

    model.convolve(image.get_beam(), stokes=False)
    pix_area = model.get_beam_area('pix')
    model.img_data *=  pix_area # norm again
    model.write(args.model.split('.fits')[0] + '-convolve.fits')

    arr, fp = reproject_interp_chunk_2d((model.img_data, model.img_hdr), image.img_hdr, hdu_in=0, order='bilinear', blocks=(1000, 1000), parallel=False)
    model.write(args.model.split('.fits')[0] + '-convolve-regrid.fits')
    arr[np.isnan(arr)] = 0.0

    image.img_data = image.img_data + arr

    image.write(args.image.split('.fits')[0] + '-added.fits')



