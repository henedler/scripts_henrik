# Copyright (C) 2022 - Henrik Edler
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

import numpy as np
import logging
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

log = logging.getLogger('libcat.logger')
log.setLevel('INFO')

def separation(c_ra,c_dec,ra,dec):
    # all values in degrees
    return np.sqrt((np.cos(c_dec*np.pi/180.0)*(ra-c_ra))**2.0+(dec-c_dec)**2.0)

class RadioCat():
    """ Wrapper class to include filtering and cross-matching utilities for astropy tables"""
    def __init__(self, cat, catname, col_tflux='Total_flux', col_pflux='Peak_flux', col_maj='Maj', ra='RA', dec='DEC', wcs=None, resolution=None, scale=1.0):
        if isinstance(cat, str):
            cat = Table.read(cat)
        elif isinstance(cat, Table):
            pass
        else:
            raise TypeError(f"cat must be a filename or astropy Table, not {type(cat)}")
        # fix column names and units
        if col_tflux:
            cat['Total_flux'] = scale*cat[col_tflux].to('Jy')
            try:
                cat['E_Total_flux'] = scale*cat['e_' + col_tflux].to('Jy')
            except KeyError:
                cat['E_Total_flux'] = scale*cat['E_' + col_tflux].to('Jy')
        if col_pflux:
            cat['Peak_flux'] = scale*cat[col_pflux].to('Jy/beam')
            try:
                cat['E_Peak_flux'] = scale*cat['e_' + col_pflux].to('Jy/beam')
            except KeyError:
                cat['E_Peak_flux'] = scale*cat['E_' + col_pflux].to('Jy/beam')
        if col_maj:
            cat['Maj'] = cat[col_maj].to('degree')
        if ra:
            cat['RA'] = cat[ra].to('degree')
        if dec:
            cat['DEC'] = cat[dec].to('degree')
        self.cat = cat
        self.wcs = wcs
        self.catname = catname
        self.resolution = resolution

    def __len__(self):
        return len(self.cat)

    def __str__(self):
        return self.cat.__str__()

    def __iter__(self):
        self.index = 0
        return self.cat

    def __next__(self):
        try:
            nextrow = self.cat[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return nextrow

    def __getitem__(self, key):
        return self.cat[key]

    def __setitem__(self, key, value):
        self.cat[key] = value

    def keys(self):
        return self.cat.keys()

    def filter(self, isolation=None, rectangle=None, circle=None, ellipse=None, sigma=5, size=None, peak_to_total=None, minflux=None):
        """
        Filter the catalogue on a variety of properties.

        Parameters
        ----------
        isolation: float, optional.
            Source isolation in arcsec.
        rectangle: (4,) array-like.
            Filter for sources in a rectangle defined by [minra, maxra, mindec, maxdec]
        circle: (3,) array-like.
            [RA, DEC, radius] in deg for circle filtering.
        ellipse: (5,) array-like
            [RA, DEC, height, width, angle] of filter ellipse in degree.
        sigma: float.
            Signal to noise.
        size: float.
            major axis filtering in arcsec.
        peak_to_total: float.
            Minimal peak to total flux factor
        """
        cat = self.cat
        logstr = f'Filter {self.catname}. Initial:{len(cat)}'
        if rectangle: # filter in rectangle
            i = (cat['RA'] > rectangle[0]) & (cat['RA'] < rectangle[1]) & (cat['DEC'] > rectangle[2]) & (cat['DEC'] < rectangle[3])
            cat =  cat[i]
            logstr += f'->rectangle:{len(cat)}'
        if circle: # filter in circle
            r = separation(circle[0], circle[1], cat['RA'], cat['DEC'])
            cat['center_dist'] = r
            cat = cat[cat['center_dist'] < circle[2]]
            logstr += f'->circle:{len(cat)}'
        if ellipse: # filter in ellipse (e.g. for primary beam)
            from regions import EllipseSkyRegion
            if not self.wcs:
                raise ValueError('Need catalog with wcs to apply ellipse filter.')
            sc = SkyCoord(ellipse[0], ellipse[1], unit='degree')
            region_sky = EllipseSkyRegion(center=sc, height=2 * ellipse[2] * u.deg, width=2 * ellipse[3] * u.deg, angle= ellipse[4] * u.deg)
            in_beam = region_sky.contains(SkyCoord(cat['RA'], cat['DEC'], unit='degree'), self.wcs)
            cat = cat[in_beam]
            logstr += f'->ellipse:{len(cat)}'
        if isolation:
            cat['NN_dist']=np.nan
            for row in cat:
                dist=3600.0*separation(row['RA'],row['DEC'],cat['RA'],cat['DEC'])
                dist.sort()
                row['NN_dist']=dist[1]
            cat=cat[cat['NN_dist']> isolation]
            logstr += f'->isolation:{len(cat)}'
        if size:
            cat = cat[cat['Maj'] < size*u.arcsec]
            logstr += f'->size:{len(cat)}'
        if minflux:
            cat = cat[cat['Total_flux'] > minflux*u.Jy]
            logstr += f'->fluxcut:{len(cat)}'
        if sigma:
            cat = cat[cat['Total_flux'] > sigma*cat['E_Total_flux']]
            logstr += f'->sigma:{len(cat)}'
        if peak_to_total:
            cat = cat[cat['Peak_flux'] / sigma*cat['Total_flux'] < peak_to_total]
            logstr += f'->peak_to_total:{len(cat)}'
        # log.info(logstr)
        print(logstr)
        self.cat = cat

    def match(self, cat2, radius, cols=[]):
        """
        Match this catalogue with another.
        Parameters
        ----------
        cat2: RadioCat object, catalogue to match with.
        radius: float, match radius in arcsec
        cols: list, non-standard columns of matchcat to transfer to new cat.

        Returns
        -------
        n_match: int, number of matches
        """
        cat = self.cat
        label = cat2.catname
        oldv = ['Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'RA', 'DEC']
        newv = ['_' + s for s in oldv]
        blankv = ['_separation', '_dRA', '_dDEC']
        for s in newv + blankv + cols:
            cat[label + s] = None
        cat[label + '_match'] = False
        rdeg = radius / 3600.0

        minra = np.min(cat['RA'] - rdeg)
        maxra = np.max(cat['RA'] + rdeg)
        mindec = np.min(cat['DEC'] - rdeg)
        maxdec = np.max(cat['DEC'] + rdeg)
        # pre-filter tab, which may be all-sky
        cat2 = cat2[(cat2['RA'] > minra) & (cat2['RA'] < maxra) & (cat2['DEC'] > mindec) & (cat2['DEC'] < maxdec)]
        matches = 0
        for r in cat:
            dist = 3600.0 * separation(np.array(r['RA']), np.array(r['DEC']), np.array(cat2['RA']), np.array(cat2['DEC']))
            stab = cat2[dist < radius]
            df = dist[dist < radius]
            if len(stab) == 1:
                # got an unique match
                matches += 1
                for i in range(len(oldv)):
                    if oldv[i] in stab.colnames:
                        r[label + newv[i]] = stab[0][oldv[i]]

                r[label + '_separation'] = df[0]
                r[label + '_match'] = True
                r[label + '_dRA'] = 3600.0 * np.cos(np.pi * r['DEC'] / 180.0) * (r['RA'] - stab[0]['RA'])
                r[label + '_dDEC'] = 3600.0 * (r['DEC'] - stab[0]['DEC'])
        print(f'{label}: matched {matches} sources')
        return matches

    def get_matches(self, labels):
        keys = [label+'_match' for label in labels]
        for i, key in enumerate(keys):
            if i == 0:
                matched = self[key].copy()
            else:
                matched &= self[key]
        print(f'Found {np.sum(matched)} common matches')
        return self[matched]

    def print_ratios(self, labels):
        t = self.get_matches(labels)
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            ratios = t['Total_flux'] / t[label + '_Total_flux']
            print(f"{self.catname}/{label}")
            median = np.median(ratios)
            print(f"median: {median}")
            print(f"percentiles/median: 16%:{np.percentile(ratios, 16)/median}, 84%:{np.percentile(ratios, 84)/median}")


