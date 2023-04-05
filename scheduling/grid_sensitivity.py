#!/usr/bin/env python3

import os, sys
import healpy as hp
import numpy as np
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
import k3match
import scipy

def process_command_line():
    import argparse
    parser = argparse.ArgumentParser(description="Generate set of pointings for sky surveys.")
    parser.add_argument('gridfile',help="Grid file [fits format].",type=str)
    parser.add_argument('-o','--output',dest='output',help="Output prefix for files. Default: grid_sens.", default='grid_sens')
    parser.add_argument('-m','--mode',dest='mode',help="Station mode: LBA_INNER, LBA_OUTER, LBA_SPARSE, LBA_ALL, HBA_DUAL, HBA_INNER [default: LBA_ALL].", default='LBA_ALL')
    parser.add_argument('-f','--freq',dest='freq', type=float, help="Frequency (MHz) [default: 54].", default=54.)
    parser.add_argument('-b','--band',dest='band', type=float, help="Bandwidth (MHz) [default: 24].", default=24.)
    parser.add_argument('-t','--time',dest='time', type=float, help="Integration time (hrs) [default: 0]. If 0 taken from 'hrs' column in the gridfile.", default=0.)
    parser.add_argument('-n','--nstations',dest='nstations', type=int, help="Number of stations (Dutch: 38, with IS: 52) [default: 52].", default=52)
    parser.add_argument('-c','--beamcut',dest='beamcut', type=float, help="Cut the beam to 0 at a certain distance in deg, otherwise all beam down to 30 per cent is used (0 is no cut) [default: 0].", default=0)
    parser.add_argument('-r','--recenter', dest='recenter', type=float, nargs=5, help="Recenter the image: ra (deg), dec (deg), rot (deg), xsize (arcmin), ysize (arcmin). Default: mollview of the full sky.")
    parser.add_argument('--plotcenters', dest='plot_centers', help="Plot the pointing centres. Default: do not plot.", action='store_true')
    parser.add_argument('-s', '--smearing', dest='smearing', type=float, nargs=3, help="Add smearing attenuation. Arguments: time resolution [s] freq resolution [MHz] angular resolution [arcsec]. Default: do not add.")
    args = parser.parse_args()

    return args

class Beam():
    def __init__(self, mode, freq, cut, do_smearing=False, delta_t=None, delta_f=None, resolution=None):
        """
        fwhm: is the fwhm of the beam [float]
        """

        if 'LBA' in mode:
            beam_freqs = np.array([15,30,45,60,75])
        else:
            beam_freqs = np.array([120,150,180])

        if mode == 'LBA_INNER':
            beam_fwhm = np.array([39.08, 19.55, 13.02, 9.77, 7.82])
        elif mode == 'LBA_OUTER':
            beam_fwhm = np.array([15.49, 7.75, 5.16, 3.88, 3.10])
        elif mode == 'LBA_SPARSE':
            # beam_fwhm = np.mean([[39.08, 19.55, 13.02, 9.77, 7.82], [15.49, 7.75, 5.16, 3.88, 3.10]], axis=0)
            beam_fwhm = np.array([19.39, 9.70, 6.46, 4.58, 3.88])
            beam_fwhm = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
        elif mode == 'LBA_ALL':
            beam_fwhm = np.array([19.39, 9.70, 6.46, 4.58, 3.88])
        elif mode == 'HBA_INNER':
            beam_fwhm = np.array([4.75, 3.80, 3.17])
        elif mode == 'HBA_DUAL':
            print('Only HBA_INNER is implemented.')
            sys.exit()

        # interpolation
        fun = interp1d(beam_freqs, beam_fwhm, fill_value='extrapolate', kind='linear')
        self.FWHM = fun(freq)
        print('Estimated FWHM: %f deg (freq: %f MHz)' % (self.FWHM, freq))
        #self.FWHM = beam_fwhm[(np.abs(beam_freqs - freq)).argmin()]
        self.freq = freq
        self.cut = cut

        # smearing
        self.do_smearing = do_smearing
        if do_smearing:
            self.delta_t = delta_t
            self.delta_f = delta_f
            self.resolution = resolution/3600 # [to deg]

        self.beam_attenuation = self.gauss

    def gauss(self, sep):
        gval = np.exp(-2.773*(sep/self.FWHM)**2.) # the coeff in front converts FWHM into sigma
        gval[gval <= 0.3] = 0 # cut at 30%
        if self.do_smearing:
            gval *= self.time_smearing(sep)
            gval *= self.bandwidth_smearing(sep)
        if self.cut != 0: gval[sep >= self.cut] = 0 # artificial cut
        return gval

    def time_smearing(self, sep):
        """
        sep: distance from phase centre [deg]
        delta_t: time resolution [seconds]
        resolution: interferometer angular resolution [deg]
        """
        #Same as above but provides the flux loss for a given time averaging
        Reduction = 1-1.22E-9*(sep/self.resolution)**2.0 * self.delta_t**2.0
        Reduction[Reduction < 0.] = 0
        #print('At radius %s deg and resolution %sdeg the source will have %s percent of its flux if data smoothed to %s sec'%(sep,self.resolution,Reduction,self.delta_t))
        return Reduction

    def bandwidth_smearing(self, sep):
        """
        delta_Theta: distance from phase centre [deg]
        delta_freq: freq resolution [MHz]
        freq: freq [MHz]
        resolution: interferometer angular resolution [deg]
        """
        # Same as above but gives the flux loss for a given frequency averaging.
        beta = (self.delta_f/self.freq) * (sep/self.resolution)
        gamma = 2*(np.log(2)**0.5)
        Reduction = ((np.pi**0.5)/(gamma * beta)) * (scipy.special.erf(beta*gamma/2.0))
        #print('at radius %s deg and resolution %sdeg at frequency of %s the source will have %s percent of its flux if data smoothed in freq to %s'%(sep,self.resolution,self.freq,Reduction,self.delta_f))
        return Reduction



class Noise():
    def __init__(self, freq, mode, BW, nant):
        """
        freq [MHz]
        dec [deg]
        BW [MHZ]
        """
        self.freq = freq * 1e6
        self.mode = mode
        self.BW = BW * 1e6
        self.nant = nant
        self.eta = 0.95 # correlator efficiency
        self.fudge = 1.8 # make theoretical prediction closer to reality

        # if large BW calculate the effective SEFD as the mean of all the SEFDs across the band
        SEFDs = []
        for nu in np.arange(self.freq-self.BW/2, self.freq+self.BW/2, 1e6):
            SEFDs.append(self.interp_SEFD(nu, self.mode))
        self.SEFD = np.average(SEFDs, weights=SEFDs) # weighted mean of SEFD as radio data are also weighted by that
        print('Effective SEFD: %f Jy (SEFD at mid freq: %f Jy)' % (self.SEFD, self.interp_SEFD(self.freq, self.mode)))

    def get_noise(self, dec, time=1):
        """
        the declination and time per pointing are likely the only things that change many times in the survey, so
        we keep it as a parameter that is not fixed and can be passed as a list

        dec : list of declinations
        time : list of times in hrs [default 1 hr]
        """
        time *= 3600 # time in hrs
        self.dec_factor = self.declination_sensivity_factor(dec)
        self.noise = self.SEFD/(self.eta * np.sqrt(2*self.BW*time))
        return self.fudge * self.dec_factor * self.noise / np.sqrt(self.nant*(self.nant-1))

    def interp_SEFD(self, freq, mode):
        # Load antennatype datapoints and interpolate, then reevaluate at freq.
        # If SPARSE, get the mean(OUTER,INNER) - if ALL divide it by 2 to account for the double in A_eff
        if mode == 'LBA_SPARSE' or mode == 'LBA_ALL':
            sefd_pth1 = os.path.dirname(os.path.abspath(__file__)) + '/SEFDs/SEFD_LBA_INNER.csv'
            points1 = np.loadtxt(sefd_pth1, dtype=float, delimiter=',')
            poli1 = np.polyfit(points1[:, 0], points1[:, 1], 2)
            fun1 = np.poly1d(poli1)
            sefd_pth2 = os.path.dirname(os.path.abspath(__file__)) + '/SEFDs/SEFD_LBA_OUTER.csv'
            points2 = np.loadtxt(sefd_pth2, dtype=float, delimiter=',')
            poli2 = np.polyfit(points1[:, 0], points1[:, 1], 2)
            fun2 = np.poly1d(poli2)
            if mode == 'LBA_SPARSE':
                return np.mean([fun1(freq),fun2(freq)])
            elif mode == 'LBA_ALL':
                return np.mean([fun1(freq),fun2(freq)])/2.
        else:
            sefd_pth = 'SEFDs/SEFD_%s.csv' % (mode)
            points = np.loadtxt(sefd_pth, dtype=float, delimiter=',')
            poli = np.polyfit(points[:, 0], points[:, 1], 2)
            fun = np.poly1d(poli)
        return fun(freq)

    def declination_sensivity_factor(self, declination):
        '''
        compute sensitivy factor lofar data, reduced by delclination, eq. from G. Heald.
        input declination is units of degrees
        '''
        factor = 1./(np.cos(2.*np.pi*(declination - 52.9)/360.)**2)
        return factor


if __name__ == "__main__":

    args = process_command_line()
    
    if ('HBA' in args.mode and args.freq < 100) or ('LBA' in args.mode and args.freq > 100):
        print('Error, frequency %f not good for mode %s' % (args.freq,args.mode))
        sys.exit(1)
    
    savename = args.output
    
    grid = Table.read(args.gridfile, format='ascii')
    
    sky_ra = grid['ra']
    sky_dec = grid['dec']
    
    NSIDE = 128
    NPIX = hp.nside2npix(NSIDE)
    m = np.zeros(NPIX)
    Npnt = np.zeros(NPIX)
    MinD = np.zeros(NPIX)
    print('Angular resolution of HEALPIX map is',hp.nside2resol(NSIDE,arcmin=True),'arcmin')
    
    print('Calculating beam...')
    if args.smearing:
        B = Beam(args.mode, args.freq, args.beamcut, True, args.smearing[0], args.smearing[1], args.smearing[2])
    else:
        B = Beam(args.mode, args.freq, args.beamcut)
    print('Calculating noise...')
    N = Noise(args.freq, args.mode, args.band, args.nstations)
    
    # plot rms map including beam circles
    dtorad = np.pi/180.
    decmap, ramap = hp.pix2ang(NSIDE, range(NPIX))
    ramap = ramap / dtorad
    decmap = -1*(decmap/dtorad -90.)
    
    # match to the pointings wirh *radius* = FWHM, so I find matches quite far out in the beam
    (ipnt, imap, d) = k3match.celestial(sky_ra, sky_dec, ramap, decmap, B.FWHM)
    pix_with_pnts = np.unique(imap)
    print("Npix={N:d} covered={n:d}".format(N=NPIX ,n=len(pix_with_pnts)))
    
    # calculate hrs
    if args.time > 0:
        print('Assuming all pointings with integration time %f hrs.' % args.time)
        grid['hrs'] = args.time
    else:
        print('Getting integration time from the grid file.')
    
    # for each pixel that is in a pointing
    for ni in pix_with_pnts:
        seps = d[imap == ni] # distances to pointings for this pixel
        g = B.beam_attenuation(seps)**2 # beam attenuation
        t = np.array( grid['hrs'][ipnt[imap == ni]] ) # hrs contributed to each pixel by each pointing
        m[ni] += np.sum(g*t)
        Npnt[ni] = len(seps[g>0]) # number of pointings covered
        MinD[ni] = np.min(d[imap == ni])
    
    print('Producing noise map...')
    # here the get_noise is calculated assuming 1 hr but the m includes the time in hrs
    rmsmap = N.get_noise(decmap)/np.sqrt(m)  # divide the noise by the attenuation of the beam and by the multiple hrs per pointing
    # move to mJy
    rmsmap *= 1e3
    print(2,np.sum(rmsmap<2), np.mean(rmsmap[rmsmap<2]), np.sum(rmsmap<2)*(27.5/60)**2)
    print(2.5,np.sum(rmsmap<2.5), np.mean(rmsmap[rmsmap<2.5]), np.sum(rmsmap<2.5)*(27.5/60)**2)
    print(3.0,np.sum(rmsmap<3.0), np.mean(rmsmap[rmsmap<3.0]), np.sum(rmsmap<3.0)*(27.5/60)**2)
    print(3.5,np.sum(rmsmap<3.5), np.mean(rmsmap[rmsmap<3.5]), np.sum(rmsmap<3.5)*(27.5/60)**2)
    print(4.0,np.sum(rmsmap<4.0), np.mean(rmsmap[rmsmap<4.0]), np.sum(rmsmap<4.0)*(27.5/60)**2)
    print(4.3,np.sum(rmsmap<4.3), np.mean(rmsmap[rmsmap<4.3]), np.median(rmsmap[rmsmap<4.3]), np.sum(rmsmap<4.3)*(27.5/60)**2)
    print(4.5,np.sum(rmsmap<4.5), np.mean(rmsmap[rmsmap<4.5]), np.sum(rmsmap<4.5)*(27.5/60)**2)
    print(4.8,np.sum(rmsmap<4.8), np.mean(rmsmap[rmsmap<4.8]), np.sum(rmsmap<4.8)*(27.5/60)**2)
    print(5.0,np.sum(rmsmap<5.0), np.mean(rmsmap[rmsmap<5.0]), np.sum(rmsmap<5.0)*(27.5/60)**2)
    print(10.0,np.sum(rmsmap<14.0), np.mean(rmsmap[rmsmap<14.0]),  np.median(rmsmap[rmsmap<14.0]), np.sum(rmsmap<14.0)*(27.5/60)**2)

    print('Plotting...')
    # map statistics
    Srmsmap = rmsmap[np.isfinite(rmsmap)]
    mn = Srmsmap.min()
    mx = Srmsmap.max()
    av = np.mean(Srmsmap)
    st = np.std(Srmsmap)
    
    # figure 1 - pointing Numer
    if args.recenter:
        rot = args.recenter[0:3]
        xsize = args.recenter[3]
        ysize = args.recenter[4]
        hp.gnomview(Npnt, rot=rot, xsize=xsize, ysize=ysize, reso=1, title="Pointing number per pixel ({n:d} pointings) - FWHM: {fwhm:.1f}deg".format(n=len(sky_ra), fwhm=B.FWHM), unit="Npointings", cmap='cubehelix')
    else:
        hp.mollview(Npnt, title="Pointing number per pixel ({n:d} pointings) - FWHM: {fwhm:.1f}deg".format(n=len(sky_ra), fwhm=B.FWHM), unit="Npointings", rot=180., cmap='cubehelix')
    hp.graticule()
    if args.plot_centers:
        hp.visufunc.projscatter(sky_ra,sky_dec,marker='.',color='None',edgecolors='y',s=(B.FWHM*60/2)**2,lonlat=True,zorder=999)
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_pointingnumber.png".format(n=savename))
    
    # figure 2 - RMS map
    if args.recenter:
        rot = args.recenter[0:3]
        xsize = args.recenter[3]
        ysize = args.recenter[4]
        hp.gnomview(rmsmap, rot=rot, xsize=xsize, ysize=ysize, reso=1, title="RMS map mJy/b ({n:d} pointings) - FWHM: {fwhm:.1f} deg".format(n=len(sky_ra), fwhm=B.FWHM), unit="RMS (mJy/b)", min=mn, max=av+2*st, cmap='cubehelix')
    else:
        hp.mollview(rmsmap, title="RMS map mJy/b ({n:d} pointings) - FWHM: {fwhm:.1f} deg".format(n=len(sky_ra), fwhm=B.FWHM), unit="RMS", rot=180., min=0.7*mn, max=mn+2*st, cmap='cubehelix')
    hp.graticule()
    if args.plot_centers:
        hp.visufunc.projscatter(sky_ra,sky_dec,marker='.',color='None',edgecolors='y',s=(B.FWHM*60/2)**2,lonlat=True,zorder=999)
    #hp.projplot(hetdex_extent_ra, hetdex_extent_dec, lonlat=True)#, coord='G')
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_pointingrms.png".format(n=savename))
    
    # hist 1 - histogram of rms per healpix pixel
    pl.figure()
    ax = pl.subplot()
    bins=np.arange(0.1,5.0,0.025)
    pl.hist(rmsmap[rmsmap<5.0],bins=bins, histtype='bar', ec='black', alpha=.8)
    pl.xlabel("rms")
    pl.ylabel("N")
    pl.xlim(0.1,5.0)
    pl.grid(alpha=.2)
    pl.title("RMS hist ({n:d} pointings) : {fwhm:.3f} beam FWHM".format(n=len(sky_ra), fwhm=B.FWHM))
    s = "min: {mn:.2f}\nmax: {mx:.2f}\navg: {av:.2f}\nstd: {st:.2f}".format(mn=mn, mx=mx, av=av, st=st)
    pl.text(0.82, 0.98, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    pl.savefig("{n:s}_pointingrms_hist.png".format(n=savename))
    
    # hist 2 - histogram number of pointings covering each pixel
    pl.figure()
    ax = pl.subplot()
    bins=np.arange(1,15,1)
    pl.hist(Npnt,bins=bins, histtype='bar', ec='black', alpha=.8)
    pl.xlabel("Number of pointing covering each pixel")
    pl.ylabel("N")
    pl.xlim(1,15)
    pl.grid(alpha=.2)
    pl.title("Npnt ({n:d} pointings) : {fwhm:.3f} beam FWHM".format(n=len(sky_ra), fwhm=B.FWHM))
    pl.savefig("{n:s}_pointingnum_hist.png".format(n=savename))
    
    # hist 3 - histogram min dist from closest phase centre
    pl.figure()
    ax = pl.subplot()
    bins=np.arange(0,B.FWHM/2.,0.05)
    pl.hist(MinD[MinD>0],bins=bins, histtype='bar', ec='black', alpha=.8)
    pl.xlabel("Minimum distance [deg]")
    pl.ylabel("N")
    pl.xlim(0,B.FWHM/2.)
    pl.grid(alpha=.2)
    pl.title("Min dist ({n:d} pointings) : {fwhm:.3f} beam FWHM".format(n=len(sky_ra), fwhm=B.FWHM))
    pl.savefig("{n:s}_pointingdist_hist.png".format(n=savename))
