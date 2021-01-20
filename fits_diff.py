#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:21:09 2019

@author: Henrik Edler
"""



from astropy.io import fits
import numpy as np
import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_1', help='First input fits file.', type=str, default=None)
    parser.add_argument("file_2", help="Second input fits file", type=str, default=None)
    parser.add_argument("--output", help="Output fits file", type=str, default='output.fits')
    parser.add_argument("--operation", help="specify operation to perform", type=str, default='subtract')

    args = parser.parse_args()    
    f1 = fits.open(args.file_1)
    f2 = fits.open(args.file_2)
    print('Image 1 RMS: ', np.std(f1[0].data))
    print('Image 2 RMS: ', np.std(f2[0].data))
    if args.operation == 'subtract':
        difference = f1[0].data - f2[0].data

        if abs(np.min(difference)) > np.max(difference):
            difference = - difference
        f1[0].data = difference
    elif args.operation == 'divide':
        ratio = f1[0].data / f2[0].data
        f1[0].data = ratio
    else: 
        print('Error')
    print('Residual RMS: ', (np.mean(f1[0].data**2))**0.5)
        
        
    f1.writeto(args.output, overwrite = True)
