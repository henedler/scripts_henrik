#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# blank_fits.py reg img.fits

from lib_fits import Image
from astropy.io import fits as pyfits
import os, sys, shutil

regionfile = sys.argv[1]

shutil.copy2(sys.argv[2], sys.argv[2].replace('dirty','mask'))
im = Image(sys.argv[2].replace('dirty','mask'))

im.write(sys.argv[2].replace('dirty','mask'))
mask_name = image_name + '.newmask'
    if os.path.exists(mask_name): os.system('rm -r ' + mask_name)
    print('Making mask:', mask_name)
    img.export_image(img_type='island_mask', img_format='fits', outfile=mask_name)
        with pyfits.open(mask_name) as fits:
            data = fits[0].data
            fits[0].data = data
            fits.writeto(mask_name, clobber=True)



