#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 - Francesco de Gasperin
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

#./fitscutout.py fitsfile [set position and size below]
position = (4776, 3266) # pixel
size = (700, 700) # pixel

import os, sys, re
import numpy as np
from astropy.io import fits
import astroquery
from astropy.nddata import Cutout2D
from astroquery.ned import Ned
import astropy.units as u
import argparse
from astropy.coordinates import SkyCoord
from regions import EllipseSkyRegion, EllipsePixelRegion, CircleAnnulusSkyRegion, write_ds9
from astropy.wcs import WCS
from lib_catalog import Cat
from lib_fits import flatten, Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input fits file", type=str, default=None)
    parser.add_argument("--racut", nargs='*', type=float, default=[182.7,185.,187.,189.,191.,193.8], help="Cut at these RAs")
    parser.add_argument("--deccut", nargs='*', type=float, default=[2.87,6.5,10.,13.5,18.13], help="Cut at these DECs")
    parser.add_argument("--cat", default=None, type=str, help="Use this cat for base regions.")
    args = parser.parse_args()

    if args.cat:
        cat = Cat(args.cat, 'evcc')
        cat['VCC'] = np.array(cat['VCC'], dtype=int)
        # cat['NGC'] = np.array([n.replace(' ','0') for n in cat['NGC']], dtype=int)
        print(type(cat[cat['VCC'] > 0]))
        cat = cat[(cat['VCC'] > 0) | (cat['NGC'] != ' ')]
        print('restrict to VCC and NGC: ', len(cat))

    img = Image(args.image)
    beam = img.get_beam()

    racut = np.sort(args.racut)[::-1]
    deccut = np.sort(args.deccut)[::-1]
    i = 1
    for decmin, decmax in zip(deccut[1:], deccut[:-1]):
        for ramin, ramax in zip(racut[1:], racut[:-1]):
            if not os.path.exists(f'templates'): os.mkdir(f'templates')
            pix = img.get_wcs().wcs_world2pix([[ramin, decmin],[ramin, decmax], [ramax, decmin], [ramax, decmax]],0)
            xmin, xmax = np.min(pix[:,0]), np.max(pix[:,0])
            ymin, ymax = np.min(pix[:,1]), np.max(pix[:,1])
            position = np.array([xmin+xmax, ymin+ymax]) / 2
            size = 1.05*np.array([xmax-xmin, ymax-ymin], dtype=int)
            # Make the cutout, including the WCS
            cutout = Cutout2D(img.img_data, position=position, size=[size[1],size[0]], wcs=img.get_wcs(), mode='trim')
            # Update the FITS header with the cutout WCS
            # Write the cutout to a new FITS file
            filename = args.image
            filename = filename.split('/')[-1]
            filename = filename.replace('.fits', f'-{i:02}.fits')
            print(f'writing cutout {i:02}')
            fits.writeto(filename, cutout.data, cutout.wcs.to_header(), overwrite=True)
            cut = Image(filename)
            cut.set_beam(beam)
            cut.write(filename)

            # now do region
            if args.cat: # get a cat for only this grid cell
                regions = []
                bg_regions = []
                t = Cat(cat.cat, 'evcc')
                t.filter(rectangle=[ramin,ramax,decmin,decmax])
                for row in t.cat: # iterate sources in this grid cell
                    # TODO left here -> query for average position angle and use as starting region.
                    if row['VCC'] > 0:
                        name = f'VCC{row["VCC"]}'
                    elif row['NGC'] != ' ':
                        name = f'NGC{row["NGC"]}'
                    else: raise ValueError('Neither VCC not NGC found...')
                    try:
                        t = Ned.get_table(name, table='diameters')
                        d = t['NED Position Angle'][~np.isnan(t['NED Position Angle'])]
                    except astroquery.exceptions.RemoteServiceError: pass
                    pa = np.rad2deg(np.arctan2(np.sin(d * np.pi / 180).sum(), np.cos(d * np.pi / 180).sum()))
                    center = SkyCoord(row['RA'], row['DEC'], unit='deg', frame='fk5')
                    rad = row['Rad']*u.arcsec  if row['Rad']*u.arcsec > 30*u.arcsec else 30*u.arcsec
                    region = EllipseSkyRegion(center=center, height=rad, width=rad, angle=pa*u.deg ,meta={'text':name}, visual={'color':'red'})
                    bg = CircleAnnulusSkyRegion(center=center, inner_radius=3*u.arcmin, outer_radius=6*u.arcmin, meta={'text':name}, visual={'color':'blue'})
                    pix_region = region.to_pixel(cutout.wcs)
                    region_data = pix_region.to_mask().get_values(cutout.data)
                    if np.all(np.isnan(region_data)):
                        print(name, ' out of mosaic.')
                        continue
                    region.serialize(format='ds9')
                    regions.append(region)
                    bg.serialize(format='ds9')
                    bg_regions.append(bg)

                write_ds9(regions, f'templates/vccngc-{i:02}.reg',  overwrite=True)
                write_ds9(bg_regions, f'templates/bg-vccngc-{i:02}.reg',  overwrite=True)
            i+=1
