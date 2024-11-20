import glob
import sys
import regions
import glob
from astropy.io import fits
import lib_fits
import argparse
from astropy.wcs import WCS
import numpy as np

parser = argparse.ArgumentParser(description='For each region in region_bkp, subtract the mean of this region from the region - leave out areas within region_src.')
parser.add_argument('imagename', help='fits image')
parser.add_argument('region_bkg', help='region(s) to subtract the mean from')
parser.add_argument('region_src', help='region to ignore for mean calbulation')
args = parser.parse_args()

img = lib_fits.Image(args.imagename)
wcs = img.get_wcs()

rsbkg = regions.Regions.read(args.region_bkg)
rssrc = regions.Regions.read(args.region_src)

src_mask = np.zeros(img.img_data.shape)
for r in rssrc:
    src_mask += r.to_pixel(wcs).to_mask().to_image(img.img_data.shape)
src_mask[src_mask>1] = 1
src_mask = src_mask.astype(bool)

for i, r in enumerate(rsbkg):
    img_mask = r.to_pixel(wcs).to_mask().to_image(img.img_data.shape)
    img_mask = img_mask.astype(bool)
    try:
        img_mask[(rsbkg[i+1].to_pixel(wcs).to_mask().to_image(img.img_data.shape)).astype(bool)] = 0
    except IndexError:
        pass
    try:
        img_mask[(rsbkg[i+1].to_pixel(wcs).to_mask().to_image(img.img_data.shape)).astype(bool)] = 0
    except IndexError:
        pass
    img_mask = img_mask.astype(bool)
    img.img_data[img_mask] -= np.mean(img.img_data[img_mask&~src_mask])

img.write(args.imagename.replace('.fits','-meansub.fits'))

