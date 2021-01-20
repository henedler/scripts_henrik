#!/usr/bin/env python
# -*- coding: utf-8 -*-

# small script to copy the final minor iteration image of all directions of the dd-serial pipeline


import sys, os, glob, pickle
from shutil import copy2 as copy
import numpy as np

str_to_match = 'ddcalM-c00-{}-cdd{}-MFS-image.fits'

matches = glob.glob('img/'+str_to_match.format('*',"*"))
dirs = np.unique([m.split('-') for m in matches])
if not os.path.exists('fnal_img_ddcalM'):
    os.mkdir('fnal_img_ddcalM')
for d in dirs:
    matches_thisdir = [m for m in matches if m.split('-')[2] == d]
    cdd = [m.split('-')[3][3:] for m in matches_thisdir]
    if len(cdd) > 0:
        print('copy: img/'+str_to_match.format(d,max(cdd))+' -> fnal_img_ddcalM/'+str_to_match.format(d,max(cdd)))
        copy('img/'+str_to_match.format(d,max(cdd)), 'fnal_img_ddcalM/'+str_to_match.format(d,max(cdd)))

