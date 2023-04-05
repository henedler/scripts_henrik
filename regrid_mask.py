#!/usr/bin/env python
import  os, argparse
from astropy.io import fits
import lib_fits

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inmask',  help='List of masks to merge/regrid.')
# parser.add_argument('--hdr_from', help='Name of fits file to use header for regrid.')
parser.add_argument('--size', nargs=2, help='Size of square image.')
parser.add_argument('--scale', help='Scale of pixel in asec.')
parser.add_argument('--ms', help='Input measurement set.')
parser.add_argument('--shift', nargs=2, help='::h::m::.::s :d::m::.:s if not imaging at phase center')
parser.add_argument('--region', help='Region to merge with mask.')
parser.add_argument('--outname', help='Name of output mosaic (default: input_name_regrid.fits)')

args = parser.parse_args()

# if args.hdr_from:
#     if args.size or args.scale:
#         raise ValueError('Ambigious information, provide either --hdr_from or --scale and --size')
if args.size and args.scale:
    pass
else:
    raise ValueError('Provide --size and --scale or --hdr_from.')

inmask = args.inmask
if args.outname:
    outname = args.outname
else:
    outname = inmask.split('.fits')[0] +'_regrid.fits'

image = lib_fits.Image(inmask)

# make template fits file with new header
if args.shift:
    os.system(f'wsclean -niter 0 -nmiter 0  -channel-range 0 1 -no-reorder -interval 0 10 -name template '
              f'-scale {args.scale}asec -size {int(args.size[0])} {int(args.size[1])} -shift {args.shift[0]} {args.shift[1]} {args.ms}')
else:
    os.system(f'wsclean -niter 0 -nmiter 0  -channel-range 0 1 -no-reorder -interval 0 10 -name template '
              f'-scale {args.scale}asec -size {int(args.size[0])} {int(args.size[1])} {args.ms}')
os.system('rm template-dirty.fits')
# os.system(f'mv template-image.fits')

regrid_hdr = lib_fits.flatten('template-image.fits')[0]
os.system('rm template-image.fits')

image.regrid(regrid_hdr)
img_data = image.img_data
img_data[img_data >= 0.5] = 1
img_data[img_data < 0.5] = 0
if args.region:
    image.apply_region(args.region, blankvalue=1)
image.write(outname)

