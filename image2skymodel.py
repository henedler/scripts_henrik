#!/usr/bin/env python3

import argparse
import bdsf

parser = argparse.ArgumentParser(description='Extract sources from fits images')
parser.add_argument('image', help='fits image')
#parser.add_argument('--region', nargs='+', help='ds9 region files to plot (optional).')

args = parser.parse_args()
name = args.image
img = bdsf.process_image(name, rms_box=(70,10), frequency=54e6, atrous_do=True, atrous_jmax=3, adaptive_thresh=True, adaptive_rms_box=True )

img.write_catalog(catalog_type='gaul', clobber=True, format='bbs')