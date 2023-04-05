#!/usr/bin/env python
import  os, argparse
import numpy as np
from astropy.io import fits
import lib_fits
from lib_beamdeconv import deconvolve_ell, EllipticalGaussian2DKernel
from reproject import reproject_interp, reproject_exact
reproj = reproject_exact
from astropy import convolution

parser = argparse.ArgumentParser(description='')
parser.add_argument('radio', type=str,  help='Radio image.')
parser.add_argument('sfr', type=str,  help='SFR map.')
parser.add_argument('logmstar', type=float, help='log m star [Msun].')
#parser.add_argument('--size', nargs=2, help='Size of square image.')
#parser.add_argument('--scale', help='Scale of pixel in asec.')
#parser.add_argument('--ms', help='Input measurement set.')
#parser.add_argument('--shift', nargs=2, help='::h::m::.::s :d::m::.:s if not imaging at phase center')
#parser.add_argument('--region', help='Region to merge with mask.')
#parser.add_argument('--outname', help='Name of output mosaic (default: input_name_regrid.fits)')

args = parser.parse_args()

sfr = fits.open(args.sfr)[0]
radio = lib_fits.Image(args.radio)
radio_noise = radio.calc_noise()
target_beam = [1/180, 1/180, 0.]
beam = [1/240, 1/240, 0]
convolve_beam = deconvolve_ell(target_beam[0], target_beam[1], target_beam[2], beam[0], beam[1], beam[2])
print('Convolve beam: %.3f" %.3f" (pa %.1f deg)' % (convolve_beam[0] * 3600, convolve_beam[1] * 3600, convolve_beam[2]))
# do convolution on data
bmaj, bmin, bpa = convolve_beam
pixsize = abs(sfr.header['PC1_1'])
fwhm2sigma = 1. / np.sqrt(8. * np.log(2.))
gauss_kern = EllipticalGaussian2DKernel((bmaj * fwhm2sigma) / pixsize, (bmin * fwhm2sigma) / pixsize,
                                        (90 + bpa) * np.pi / 180.)  # bmaj and bmin are in pixels
sfr.data[sfr.data == 0.0] = np.nan
sfr.data = convolution.convolve(sfr.data, gauss_kern, boundary=None, preserve_nan=True)

# regrid_hdr = radio.img_hdr
# print(dir(sfr.header), dir(regrid_hdr))
# print(sfr.header.cards, regrid_hdr.cards)
# sfr.header['CDELT1'] = sfr.header['PC1_1']
# sfr.header['CDELT2'] = sfr.header['PC2_2']
# del sfr.header['PC1_1']
# del sfr.header['PC2_2']
# sfr.data, __footprint = reproj((sfr.data, sfr.header), regrid_hdr, parallel=True)
# sfr.header = regrid_hdr
# sfr.header['PC1_1'] = regrid_hdr['CDELT1']
# sfr.header['PC2_2'] = regrid_hdr['CDELT2']
regrid_hdr = sfr.header
del regrid_hdr['FILTERS'], regrid_hdr['BUNIT']
regrid_hdr['BMIN'], regrid_hdr['BMAJ'], regrid_hdr['BPA'] = radio.img_hdr['BMIN'], radio.img_hdr['BMAJ'], radio.img_hdr['BPA']
radio_data_regrid, __footprint = reproj((radio.img_data, radio.img_hdr), regrid_hdr, parallel=True)
radio_regrid = fits.PrimaryHDU(radio_data_regrid,regrid_hdr)
radio_regrid.writeto(args.radio.replace('.fits', '-regrid.fits'), overwrite=True)
# sfr.writeto(args.sfr.replace('.fits', '-conv-regrid.fits'), overwrite=True)

N0, gamma = 10**21.7144, 0.208
lum_jy = 1e-26 * 4*np.pi * (16.5 * 1e6 * 3.08567758e16)**2
gfactor=2.0*np.sqrt(2.0*np.log(2.0))
beam_area = 2.0*np.pi*(target_beam[0]*target_beam[1]*3600**2)/(gfactor*gfactor) # arcsec^2
kpcsq_per_beam = beam_area / (0.08**2)
ratio = (N0 * lum_jy**-1 * radio_regrid.data * args.logmstar**gamma * kpcsq_per_beam) / sfr.data

ratio_map = fits.PrimaryHDU(ratio, regrid_hdr)
ratio_map.writeto(args.radio.replace('.fits', '-regrid-ratio.fits'), overwrite=True)
