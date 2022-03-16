#!/usr/bin/python3

import os, sys
import numpy as np
from astropy.table import Table

gridfile = 'allsky-grid-lofar2survey.fits'

grid = Table.read(gridfile)

grid['hrs'] = 3
grid['hrs'][(np.abs(grid['GAL_LAT'])>23) & (grid['dec']>20)] = 18
grid.write(gridfile, overwrite=True)
