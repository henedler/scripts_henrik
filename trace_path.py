#!/usr/bin/env python
#
# Script to trace a path in a fits-file.
# Required input: a ds9 region file which contains an ordered sequence of points.

import os, sys, argparse
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.convolution import Gaussian2DKernel
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate, integrate, optimize
import pyregion

import lib_fits


def beam_ellipse(ra, dec, image):
    b = lib_fits.Image(image).get_beam()
    ra = Angle(str(ra)+'d', unit=u.deg).to_string(sep = ':', unit=u.hour)
    dec = Angle(dec, unit=u.deg).to_string(sep = ':')
    ell_str = f"ellipse({ra}, {dec}, {b[0]*3600}\", {b[1]*3600}\", {b[2]})"
    return pyregion.parse(ell_str)

parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('image', help='fits image.')
parser.add_argument('region', help='ds9 region.')
parser.add_argument('--z', default = 0.1259, help='Source redshift.')
parser.add_argument('--si', default='', action='store_true', help='Spectral index.')
args = parser.parse_args()

hdu = pyfits.open(args.image)[0]
img_hdr, img_data = hdu.header, hdu.data
wcs = WCS(img_hdr)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
degperpixel = np.abs(wcs.all_pix2world(0, 0, 0)[1] - wcs.all_pix2world(0, 1, 0)[1])  # delta deg for 1 pixel
degperkpc = cosmo.arcsec_per_kpc_proper(args.z).value / 3600.
pixelperkpc = degperkpc / degperpixel

# Sort according to decreasing RA:
region = pyregion.open(args.region)
ra_order = np.argsort([r.coord_list[0] for r in region])
reg_sort = pyregion.ShapeList([region[i] for i in ra_order])
sorted_filename = args.region.split('.')[::-1][0]+'_sorted.reg'
reg_sort.write(sorted_filename)

trace = pyregion.open(sorted_filename)
trace = np.array([p.coord_list for p in trace.as_imagecoord(img_hdr)])

print(f"#### Trace consists of {len(trace)} points.")
t, c, k = interpolate.splrep(trace[:,0], trace[:,1], s=0, k=3)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

def dl(x):
    """ Length element at point x"""
    return (1+spline.derivative()(x)**2)**0.5

def curve_length(x):
    """ Length of the curve from start of trace to x."""
    return integrate.quad(dl,trace[0,0], x, epsabs=0.01, epsrel=0.001)[0]

total_length = curve_length(trace[-1,0])
print(f"#### Total length of curve: {total_length} pixels")

def xy_at_length(l):
    """ Get the x,y values at a path length L"""
    x_l = optimize.brentq(lambda x: curve_length(x)-l, trace[0,0], trace[-1,0], xtol=0.01)
    return np.array([x_l, spline(x_l)])

print("#### Find evenly spaced points...")
points = np.array([xy_at_length(l) for l in np.linspace(0,total_length,int(total_length/0.5))])
print(f'#### Check points: First diff - {trace[0]-points[0]}; Second diff = {trace[-1]-points[-1]}')

data = img_data
p_world = wcs.all_pix2world(points, 0)
trace_data_px = np.zeros(len(points))

data[np.isnan(data)] = 0.
interp_data = interpolate.RectBivariateSpline(np.arange(data.shape[1]), np.arange(data.shape[1]), data.T)
trace_data_px = interp_data(points[:,0], points[:,1], grid=False)
trace_data_psf = interp_data(points[:,0], points[:,1], grid=False)

for i,p in enumerate(p_world):
    beam = beam_ellipse(p[0], p[1], args.image).as_imagecoord(img_hdr)
    beam_mask = beam.get_mask(hdu=hdu, header=img_hdr)
    b = lib_fits.Image(args.image).get_beam()
    where0, where1 = np.argwhere(np.any(beam_mask, axis=0)), np.argwhere(np.any(beam_mask, axis=1))
    n0 = (where0[-1] - where0[0])[0]
    n1 = (where1[-1] - where1[0])[0]
    psf_weight = Gaussian2DKernel(b[0]/degperpixel, b[1]/degperpixel, theta=np.deg2rad(b[2]), x_size=n0, y_size=n1).array
    psf_weight/np.max(psf_weight)
    print(psf_weight)
    print(psf_weight.shape, data[beam_mask].shape)
    print(data[beam_mask])
    trace_data_psf[i] = np.nanmean(data[beam_mask])

assert img_hdr['CDELT1'], img_hdr['CDELT2']
pix_to_deg = img_hdr['CDELT1']
import matplotlib.pyplot as plt

plt.plot(np.arange(len(points))/pixelperkpc, -trace_data_px, label='pixel')
plt.plot(np.arange(len(points))/pixelperkpc, -trace_data_psf, label='psf')
plt.legend()
plt.xlabel("distance [kpc]")
plt.ylabel("spectral index")
# plt.plot(points[:,0], points[:,1])
plt.savefig('test.pdf')

#pyregion.get_mask(beam_ellipse(ra, rec, img), pyfits.open(args.image))

