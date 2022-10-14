#!/usr/bin/env python3

import regions
import scipy.stats
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.table import Table
import numpy as np
import sys
import pandas as pd
import argparse
import warnings
from lib_linearfit import *

def flatten(f,channel=0,freqaxis=0):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
    if naxis==2:
        return f[0].header,f[0].data

    w = wcs.WCS(f[0].header)
    wn=wcs.WCS(naxis=2)
    
    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]
    
    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH')
    for k in copy:
        r=f[0].header.get(k)
        if r:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        elif i==freqaxis:
            slice.append(channel)
        else:
            slice.append(0)
        
    # slice=(0,)*(naxis-2)+(np.s_[:],)*2
    return header,f[0].data[slice]

class RadioError(Exception):
    """Base class for exceptions in this module."""
    pass

class radiomap:
    """ Process a fits file as though it were a radio map, calculating beam areas etc """
    def __init__(self, filename, verbose=False):
        self.filename = filename
        self.fitsfile=fits.open(filename)
        # Catch warnings to avoid datfix errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gfactor=2.0*np.sqrt(2.0*np.log(2.0))
            self.f=self.fitsfile[0]
            self.prhd=self.fitsfile[0].header

            # Get units and resolution
            self.units=self.prhd.get('BUNIT')
            if self.units is None:
                self.units=self.prhd.get('UNIT')
            if self.units!='JY/BEAM' and self.units!='Jy/beam':
                print('Warning: units are',self.units,'but code expects JY/BEAM')
            self.bmaj=self.prhd.get('BMAJ')
            self.bmin=self.prhd.get('BMIN')
            if self.bmaj is None:
                # Try RESOL1 and RESOL2
                self.bmaj=self.prhd.get('RESOL1')
                self.bmin=self.prhd.get('RESOL2')
            if self.bmaj is None:
                if verbose:
                    print('Can\'t find BMAJ in headers, checking history')
                try:
                    history=self.prhd['HISTORY']
                except KeyError:
                    history=None
                if history is not None:
                    for line in history:
                        if 'HISTORY' in line:
                            continue # stops it finding nested history
                        if 'CLEAN BMAJ' in line:
                            bits=line.split()
                            self.bmaj=float(bits[3])
                            self.bmin=float(bits[5])
                                
            if self.bmaj is None:
                raise RadioError('No beam information found')

            w=wcs.WCS(self.prhd)
            self.wcs = w
            cd1=-w.wcs.cdelt[0]
            cd2=w.wcs.cdelt[1]
            if ((cd1-cd2)/cd1)>1.0001 and ((self.bmaj-self.bmin)/self.bmin)>1.0001:
                raise RadioError('Pixels are not square (%g, %g) and beam is elliptical' % (cd1, cd2))

            self.bmaj/=cd1
            self.bmin/=cd2
            if verbose:
                print('beam is',self.bmaj,'by',self.bmin,'pixels')

            self.area=2.0*np.pi*(self.bmaj*self.bmin)/(gfactor*gfactor)
            if verbose:
                print('beam area is',self.area,'pixels')

            # Remove any PC... keywords we may have, they confuse the pyregion WCS
            for i in range(1,5):
                for j in range(1,5):
                    self.quiet_remove('PC0%i_0%i' % (i,j))
                
            # Now check what sort of a map we have
            naxis=len(self.fitsfile[0].data.shape)
            if verbose: print('We have',naxis,'axes')
            self.cube=False
            if naxis<2 or naxis>4:
                raise RadioError('Too many or too few axes to proceed (%i)' % naxis)
            if naxis>2:
                self.nchans=1
                # a cube, what sort?
                frequency=0
                self.cube=True
                freqaxis=-1
                stokesaxis=-1
                for i in range(3,naxis+1):
                    ctype=self.prhd.get('CTYPE%i' % i)
                    if 'FREQ' in ctype:
                        freqaxis=i
                    elif 'STOKES' in ctype:
                        stokesaxis=i
                    elif 'VOPT' in ctype:
                        pass
                    else:
                        print('Warning: unknown CTYPE %i = %s' % (i,ctype))
                if verbose:
                    print('This is a cube with freq axis %i and Stokes axis %i' % (freqaxis, stokesaxis))
                if stokesaxis>0:
                    nstokes=self.prhd.get('NAXIS%i' % stokesaxis)
                    if nstokes>1:
                        raise RadioError('Multiple Stokes parameters present, not handled')
                if freqaxis>0:
                    nchans=self.prhd.get('NAXIS%i' % freqaxis)
                    if verbose:
                        print('There are %i channel(s)' % nchans)
                    self.nchans=nchans
            else:
                self.nchans=1
                    
            # that a bad (zero) value will be present, so keep
            # checking if one is found.

            if not(self.cube) or freqaxis<0:

                # frequency, if present, must be in another keyword
                frequency=self.prhd.get('RESTFRQ')
                if frequency is None or frequency==0:
                    frequency=self.prhd.get('RESTFREQ')
                if frequency is None or frequency==0:
                    frequency=self.prhd.get('FREQ')
                if frequency is None or frequency==0:
                    # It seems some maps present with a FREQ ctype
                    # even if they don't have the appropriate axes!
                    # The mind boggles.
                    for i in range(5):
                        type_s=self.prhd.get('CTYPE%i' % i)
                        if type_s is not None and type_s[0:4]=='FREQ':
                            frequency=self.prhd.get('CRVAL%i' % i)
                self.frq=[frequency]
                flathdr,flatd=flatten(self.fitsfile)
                self.d=[flatd]
                self.headers=[flathdr]

            else:
                # if this is a cube, frequency/ies should be in freq header
                basefreq=self.prhd.get('CRVAL%i' % freqaxis)
                deltafreq=self.prhd.get('CDELT%i' % freqaxis)
                self.frq=[basefreq+deltafreq*i for i in range(nchans)]
                self.d=[]
                self.headers=[]
                for i in range(nchans):
                    header,data=flatten(self.fitsfile,freqaxis=freqaxis,channel=i)
                    self.d.append(data)
                    self.headers.append(header)
            for i,f in enumerate(self.frq):
                if f is None:
                    print(('Warning, can\'t get frequency %i -- set to zero' % i))
                    self.frq[i]=0
            if verbose:
                print('Frequencies are',self.frq,'Hz')
            #self.fitsfile.close()

    def quiet_remove(self,keyname):
        if self.prhd.get(keyname,None) is not None:
            self.prhd.remove(keyname)

class applyregion:
    """ apply a region from pyregion to a radiomap """
    def __init__(self,rm,region,offsource=None,robustrms=3):
        """
        provides:
        rms -- the rms in the aperture
        robustrms -- the rms for pixels below robustrms * the normal rms (it should cut sources)
        flux -- the flux of the aperture
        mean -- the mean in the apertur
        error -- error on the flux given the rms in offsource
        """
        self.rm = rm
        self.rms=[]
        self.max=[]
        self.min=[]
        self.flux=[]
        self.error=[]
        self.mean=[]
        self.robustrms=[]
        self.mean_error=[]

        for i,d in enumerate(rm.d):
            mask_r=region.to_mask()
            pixels=np.sum(mask_r)
            data = mask_r.cutout(d) #np.extract(mask_r,d)
            self.rms.append(np.nanstd(data))
            self.max.append(np.max(data[np.logical_not(np.isnan(data))]))
            self.min.append(np.min(data[np.logical_not(np.isnan(data))]))
            self.robustrms.append(np.nanstd(data[np.where(data < robustrms * self.rms[-1])]))
            self.flux.append(data[np.logical_not(np.isnan(data))].sum()/rm.area)
            self.mean.append(np.nanmean(data))
            self.mean_error.append(np.sqrt(np.nanmean(data)/np.sqrt(np.count_nonzero(~np.isnan(data)))))

            # calc noise
            if offsource is not None:
                self.error.append(offsource[i]*np.sqrt(pixels/rm.area))
            else:
                self.error.append(0.)

def printflux(fgss,fluxerr=None):
    """
    fgss -- region to work on, 2d array [ radiomeasure x region ]
    fluxerr -- percentage of flux error for spidx maps (only spidx)
    """
    # cycle on region
    for n, fgs in enumerate(fgss):
        # cycle on rm
        for fg in fgs:
            for i in range(fg.rm.nchans):
                freq = fg.rm.frq[i]
                print(n,fg.rm.filename,'%8.4g %10.6g %10.6g' % (freq,fg.flux[i],fg.error[i]))
 
def printmean(fgss,fluxerr=None):
    # cycle on region
    for n, fgs in enumerate(fgss):
        # cycle on rm
        for fg in fgs:
            for i in range(fg.rm.nchans):
                freq = fg.rm.frq[i]
                print(n,fg.rm.filename,'%8.4g %10.6g +/- %10.6g' % (freq,fg.mean[i],fg.mean_error[i]))

def printspidx(fgss,fluxerr=None):

    # cycle on region
    for n, fgs in enumerate(fgss):
        freqs = []
        fluxes = []
        errors = []

        # cycle on rm
        for fg in fgs:
            for i in range(fg.rm.nchans):
                freqs += fg.rm.frq
                fluxes += fg.flux
                errors += fg.error

        # lin reg
        if not all(e == 0 for e in errors):
            yerr = 0.434*np.sqrt(np.array(errors)**2+(np.array(fluxerr)*np.array(fluxes)/100)**2)/np.array(fluxes)
        else:
            yerr = None
        #print freqs, fluxes, yerr

        (a, b, sa, sb) = linear_fit_bootstrap(x=np.log10(freqs), y=np.log10(fluxes), yerr=yerr)
        print(n, '%8.4g %8.4g' % (a, sa))

def radioflux(file,bgfile,fgr,bgr=None,individual=False,action='Flux',fluxerr=0,nsigma=0,verbose=False,output=None, evcc=None):
    """Determine the flux in a region file for a set of files. This is the
    default action for the code called on the command line, but
    may be useful to other code as well.

    Keyword arguments:
    files -- list of files (mandatory)
    bgfiles -- list of background files (mandatory)
    fdr -- foreground region name (mandatory)
    bgr -- background region name (optional)
    individual -- separate region into individual sub-regions
    action -- what to do once fluxes are measured: allows a user-defined action
              which must be a drop-in replacement for printflux
    fluxerr -- flux error in % for spidxmap
    nsigma -- keep only pixels above these sigma level in ALL maps (bgr must be specified)
    """
    action = {'flux':printflux, 'mean':printmean, 'spidx':printspidx}[action]

    rm = radiomap(file,verbose=verbose)
    bgrm = radiomap(bgfile,verbose=verbose)

    assert rm.d[0].size == bgrm.d[0].size
    # initial mask
    mask = (np.zeros_like(rm.d) == 0)

    measurement = []
    fg_ir=regions.Regions.read(fgr, format='ds9')
    bg_ir=regions.Regions.read(bgr, format='ds9')
    evcc = Table.read(evcc)
    evcc.convert_bytestring_to_unicode()

    for fg_ir_split in fg_ir:
        matches = 0
        for bg_ir_split in bg_ir:
            if bg_ir_split.meta['text'] == fg_ir_split.meta['text']:
                matches += 1
                bg_apply = applyregion(bgrm,bg_ir_split.to_pixel(wcs=rm.wcs))
        if matches == 0:
            raise ValueError(f'Found no bg region corresponding to {fg_ir_split.meta["text"]}')
        elif matches > 1:
                raise ValueError(f'Found non-unique bg region corresponding to {fg_ir_split.meta["text"]}')
        fg_ir_split_pix =  fg_ir_split.to_pixel(wcs=rm.wcs)
        gf_area_asecsq = fg_ir_split_pix.area*np.abs(rm.wcs.wcs.cdelt[0]*rm.wcs.wcs.cdelt[1])*3600**2
        fg_apply = applyregion(rm, fg_ir_split_pix, offsource=bg_apply.rms)
        if isinstance(fg_ir_split, regions.EllipseSkyRegion):
            lls = np.max([fg_ir_split.height.to_value('arcmin'), fg_ir_split.width.to_value('arcmin')])
            reg_type = 'e'
        elif isinstance(fg_ir_split, regions.PolygonSkyRegion):
            vert = fg_ir_split.vertices
            lls = 0
            for v in vert:
                sep = np.max(v.separation(vert).to_value('arcmin'))
                if sep > lls:
                    lls = sep
            reg_type = 'p'
        else:
            raise TypeError(f'region should be Ellipse or Polygon, not {type(fg_ir_split)}.')
        name = fg_ir_split.meta["text"]
        if 'VCC' in name:
            src = evcc[evcc['VCC'] == int(name[3:])]
        elif 'NGC' in name:
            src = evcc[evcc['NGC'] == name[3:]]
        ra, dec, _evcc, vcc, ngc, MmI, MTyp1 = src[['RAJ2000','DEJ2000','EVCC','VCC','NGC','MmI','MTyp1']].as_array()[0]
        measurement.append([ra, dec, name, _evcc, vcc, ngc, fg_apply.flux[0], fg_apply.error[0], fg_apply.max[0], bg_apply.rms[0], gf_area_asecsq, reg_type, lls, MmI, MTyp1 ])
    df = pd.DataFrame(measurement, columns=['RA','DEC','Name','EVCC','VCC','NGC', 'Total_flux', 'E_stat_Total_flux', 'Peak_flux', 'E_stat_Peak_flux', 'Area', 'Region', 'LLS','MmI','MTyp1'])

    df.to_csv(output)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Measure fluxes from FITS files.')
    parser.add_argument('file', metavar='FILE', help='FITS file to process')
    parser.add_argument('bgfile', help='BG FITS file to process')
    parser.add_argument('-f','--foreground', dest='fgr', action='store',default='ds9.reg',help='Foreground region file to use.')
    parser.add_argument('-b','--background', dest='bgr', action='store',default='',help='Background region file to use.')
    parser.add_argument('-i','--individual', dest='indiv', action='store_true',default=False,help='Break composite region file into individual regions.')
    parser.add_argument('-e','--fluxerr', dest='fluxerr', action='store',default=0, type=float, help='Flux error in %% for spidx maps only.')
    parser.add_argument('-s','--sigma', dest='nsigma', action='store',default=0, type=float, help='Try to cut all the images above a certain sigma. Only pixel over that sigma in ALL the images are considered. Valid only for spidx.')
    parser.add_argument('-a','--action', dest='action', action='store',default='flux',help='Action to perform: flux, mean, spidx.')
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true',default=False,help='Be verbose.')
    parser.add_argument('--evcc', default=None)
    parser.add_argument('-o', '--outname', default='measurement.csv')

    args = parser.parse_args()

    radioflux(args.file, args.bgfile, args.fgr,args.bgr,args.indiv,args.action,args.fluxerr,args.nsigma,verbose=args.verbose, output=args.outname, evcc=args.evcc)
