#!/usr/bin/python
# Distribute num_points (default 20) on a sphere using the algorithm from
# "Distributing many points on a sphere" by E.B. Saff and
# A.B.J. Kuijlaars, Mathematical Intelligencer 19.1 (1997) 5-11

import sys
from pyslalib import slalib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from mpl_toolkits.basemap import *

def y_rotate(x,y,z,angle_rad):
   new_x = (x + 0*y + 0*z)
   new_y = (0*x + np.cos(angle_rad)*y - np.sin(angle_rad)*z)
   new_z = (0*x + np.sin(angle_rad)*y + np.cos(angle_rad)*z)
   return new_x,new_y,new_z

def z_rotate(x,y,z,angle_rad):
   new_x = (np.cos(angle_rad)*x - np.sin(angle_rad)*y + 0*z)
   new_y = (np.sin(angle_rad)*x + np.cos(angle_rad)*y - 0*z)
   new_z = (0*x + 0*y + z)
   return new_x,new_y,new_z

def process_command_line():
    import argparse
    parser = argparse.ArgumentParser(description="Generate set of pointings for sky surveys.")
    parser.add_argument('n',help="Number of pointings.", nargs=1,type=int)
    parser.add_argument('-o','--output',dest='o',help="Output prefix for files. Default: grid.", default='grid')
    parser.add_argument('--ra',dest='cutra',help="Cut in ra (min,max in deg). Default: 0,360", default="0,360")
    parser.add_argument('--dec',dest='cutdec',help="Cut in dec (min,max in deg). Default: -90,90", default="-90,90")
    parser.add_argument('-f','--fwhm',dest='fwhm',help="FWHM in degree. Default: 4.58 (LOFAR2.0)", default=4.85)
    args=parser.parse_args()
    
    args.n = args.n[0]
    if not type(args.n) is int or args.n < 10:
        sys.stderr.write("error: number of points must be >= 10\n")
        sys.exit(1)

    return args
      
def calc_points(N):
   points = []
   old_phi = 0
   for k in range(1,N+1):
      h = -1 + 2*(k-1)/float(N-1)
      theta = np.arccos(h)
      if k==1 or k==N:
         phi = 0
      else:
         phi = (old_phi + 3.6/np.sqrt(N*(1-h*h))) % (2*np.pi)

      points.append((np.sin(phi)*np.sin(theta), np.cos(theta), np.cos(phi)*np.sin(theta) ))
      old_phi = phi
         
   return points

class Beam():
    def __init__(self, fwhm):
        """
        typ: is a fwhm float or a name among: uGMRT300
        """
        try:
            self.FWHM = float(fwhm)
            self.shape = self.gauss
        except:
            print "Wrong format for beam FWHM: %s." % str(fwhm)
            sys.exit(1)

    def gauss(self, sep):
        gval = np.exp(-2.773*(sep/self.FWHM)**2.)
        gval[gval <= 0.3] = 0 # cut at 30%
        return gval

def circ(ra0, dec0, d, N=100.):
    """
    Coord of a circle with center ra0,dec0 and radius d
    """
    theta = np.arange(0, 2.*np.pi+0.001, 2.*np.pi/N)
    ra = ra0+d*np.cos(theta)/np.cos(dec0*np.pi/180.)
    dec = dec0+d*np.sin(theta)
    return ra, dec

def plot_pointings_sky_region(plotdatas, FWHM, imagename="pointings_sky_region.png", title="tiled", wholesky=True):
    import matplotlib.pyplot as pl

    # plot all sky
    f1 = pl.figure(figsize=(10,7))
    # define base map class.
    #width = 28000000; lon_0 = 180; lat_0 = 40
    if wholesky:
        bax = Basemap(projection='moll',lat_0=0,lon_0=180, resolution=None)
    else:
        bax = Basemap(width=20000000,height=10000000,projection='aeqd',lat_0=52,lon_0=187)
 
    bax.drawmapboundary()
    pl.title(title, fontsize=12)
 
    # draw and label ra/dec grid lines every 30 degrees.
    degtoralabel = lambda deg : r"%dh" % int(deg/15)
    degtodeclabel = lambda deg : "%d$^\circ$" % deg
    bax.drawparallels(np.arange(-30, 90, 15 ) )#, fmt=degtodeclabel )#, labels=[1,0,0,0])
    bax.drawmeridians(np.arange(0, 360, 15) )#, fmt=degtoralabel) 
    for h in range(6,20,2):
        x,y = bax(h*15,20)
        pl.text(x,y, degtoralabel(h*15), horizontalalignment='center', verticalalignment='center', backgroundcolor='w', fontsize=6)
    for d in range(0,90,30):
        x,y = bax(105, d)
        pl.text(x,y, degtodeclabel(d), horizontalalignment='center', verticalalignment='center', backgroundcolor='w', fontsize=6)
 
    for plotdata in plotdatas:
        print 'plotting', plotdata['label']
        for hi in range(len(plotdata['ra'])):
            rai = plotdata['ra'][hi]
            deci = plotdata['dec'][hi]
            Cra, Cdec  = circ(rai, deci, FWHM/2.)
            px, py = bax(Cra, Cdec)
            ppx, ppy = bax(rai, deci)
            if hi != 0:
                bax.plot(ppx, ppy, plotdata['color'], latlon=False,alpha=0.5, label=None, marker='o')
            else:
                bax.plot(ppx, ppy, plotdata['color'], latlon=False,alpha=0.5, marker='o', label= plotdata['label'])
 
    pl.legend(loc='lower center', labelspacing=0.1, ncol=3, bbox_to_anchor=(0.5, -0.15))
         
    pl.savefig(imagename)
     
    return

def plot_sky_pointingcount(sky_ra, sky_dec, B, savename, Nside=2**8, doall=True):
    """
    FWHM -- FWHM in degree
    """
    import matplotlib.pyplot as pl
    import healpy as hp
    import k3match

    # set Nside bigger for better resolution - but very slow!!
    # get the sky pixels
    dtorad = np.pi/180.
    Npix = hp.nside2npix(Nside)
    Npnt = np.zeros(Npix)
    T = np.zeros(Npix)
    decmap,ramap=hp.pix2ang(Nside, range(Npix))
    ramap = ramap /dtorad
    decmap = -1*(decmap/dtorad -90.)
     
    plotdatas = [ {'ra': sky_ra, 'dec': sky_dec, 'color':'gray', 'label':'{savename} ({n})'.format(savename=savename, n=len(sky_ra))} ]
    plot_pointings_sky_region(plotdatas, B.FWHM, imagename="{savename}_tilesky.png".format(savename=savename))
 
    if not doall:
        return
     
    # match to the pointings wirh *radius* = FWHM, so I find matches quite far out in the beam
    (ipnt, imap, d) = k3match.celestial(sky_ra, sky_dec, ramap, decmap, B.FWHM) 
    pix_with_pnts = np.unique(imap)
    print "Npix={N:d} covered={n:d}".format(N=Npix ,n=len(pix_with_pnts))
     
    # for each pixel that is in a pointing
    for ni in pix_with_pnts:
        seps = d[imap == ni] # distances to pointings for this pixel
        Npnt[ni] = len(seps) # number of pointings covered1
        g = B.shape(seps)**2
        T[ni] += np.sum(g)
             
    rmsmap = 1./np.sqrt(T)  # convert time to rms
     
    # figure 1 - pointing Numer
    hp.mollview(Npnt, title="Pointing number per pixel ({n:d} pointings) : {fwhm:.1f} beam".format(n=len(sky_ra), fwhm=B.FWHM), unit="Npointings", rot=180.,min=0, max=6)
    hp.graticule()
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_tilesky_pointingnumber.png".format(n=savename))
    # map statistics
    Srmsmap = rmsmap[np.isfinite(rmsmap)]
    mn = Srmsmap.min()
    mx = Srmsmap.max()
    av = np.mean(Srmsmap)
    st = np.std(Srmsmap)
     
    # figure 2 - RMS map
    hp.mollview(rmsmap, title="RMS map ({n:d} pointings) : {fwhm:.1f} beam".format(n=len(sky_ra), fwhm=B.FWHM), unit="RMS", rot=180., max=0.5, min=2)
    hp.graticule()
    #hp.projplot(hetdex_extent_ra, hetdex_extent_dec, lonlat=True)#, coord='G')
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        #hp.projtext(h*15,0, degtoralabel(h*15), lonlat=True, coord='G')
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_tilesky_pointingrms.png".format(n=savename))
     
    # figure 2 - RMS map
    hp.mollview(rmsmap, title="RMS map ({n:d} pointings) : {fwhm:.1f} beam".format(n=len(sky_ra), fwhm=B.FWHM), unit="RMS", rot=180., max=1.2, min=0.6)
    hp.graticule()
    #hp.projplot(hetdex_extent_ra, hetdex_extent_dec, lonlat=True)#, coord='G')
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        #hp.projtext(h*15,0, degtoralabel(h*15), lonlat=True, coord='G')
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_tilesky_pointingrms_fixedscale.png".format(n=savename))
 
    hp.mollview(rmsmap, title="Polar view: RMS map ({n:d} pointings) : {fwhm:.1f} beam".format(n=len(sky_ra), fwhm=B.FWHM), unit="RMS", rot=[180.,90.],  max=1.2, min=0.6)
    hp.graticule()
    #hp.projplot(hetdex_extent_ra, hetdex_extent_dec, lonlat=True)#, coord='G')
    degtoralabel = lambda deg : "%+d$^h$" % int(deg/15)
    degtodeclabel = lambda deg : "%+d$^\circ$" % deg
    for h in [6,12,18]:
        #hp.projtext(h*15,0, degtoralabel(h*15), lonlat=True, coord='G')
        hp.projtext(h*15,0, degtodeclabel(h*15), lonlat=True)#, coord='G')
    pl.savefig("{n:s}_tilesky_pointingrms_polar_fixedscale.png".format(n=savename))
 
    pl.figure()
    ax = pl.subplot()
    bins=np.arange(0.8,2.0,0.01)
    pl.hist(rmsmap[rmsmap<2.],bins=bins)
    pl.xlabel("rms")
    pl.ylabel("N(rms)")
    pl.xlim(0.8,2.0)
    pl.title("RMS hist ({n:d} pointings) : {fwhm:.3f} beam FWHM".format(n=len(sky_ra), fwhm=B.FWHM))
    s = "min: {mn:.2f}\nmax: {mx:.2f}\navg: {av:.2f}\nstd: {st:.2f}".format(mn=mn, mx=mx, av=av, st=st)
    pl.text(0.02, 0.98, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    pl.savefig("{n:s}_tilesky_pointingrms_hist.png".format(n=savename))
    
    return

if __name__ == "__main__":  
    args = process_command_line()
    B = Beam(args.fwhm)
    points = calc_points(args.n)
    
    minra, maxra = map(int, args.cutra.split(","))
    mindec, maxdec = map(int, args.cutdec.split(","))
    print "Cut in RA: %i -- %i" % (minra, maxra)
    print "Cut in Dec: %i -- %i" % (mindec, maxdec)

    ofile = open(args.o+'-pointings.txt', "w")
    
    zrotangle = 0.0
    yrotangle = 0.0
    
    ras = []
    decs = []
    for point in points:
       newx,newy,newz= y_rotate(point[0],point[1],point[2],np.pi/2.0) #To put the spiral top at the zenith
       newx,newy,newz= z_rotate(newx,newy,newz,zrotangle)
       newx,newy,newz= y_rotate(newx,newy,newz,yrotangle)
       raj2000, decj2000 = slalib.sla_cc2s([newx,newy,newz])
       raj2000 = slalib.sla_dranrm(raj2000)
       raj2000 *= 180./np.pi
       decj2000 *= 180./np.pi
       # fix rounding errors
       if decj2000 > 90: decj2000 = 90
       if decj2000 < -90: decj2000 = -90
       
       if decj2000 < mindec or decj2000 > maxdec or raj2000 < minra or raj2000 > maxra:
             continue
    
       ras.append(raj2000)
       decs.append(decj2000)
    
       name = 'P%03.1i_%+02.1i' % (raj2000, decj2000)
       ofile.write("%s %f %f \n" % (name, raj2000, decj2000))
    
    print "Total number of valid pointings: %i over %i." % (len(ras), len(points))
    
    ofile.close()
    
    # find distances to the 6 closest (excluding itself)
    catalog = SkyCoord(ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree)
    __idx, d2d1, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=2)
    __idx, d2d2, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=3)
    __idx, d2d3, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=4)
    __idx, d2d4, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=5)
    __idx, d2d5, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=6)
    __idx, d2d6, __d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=7)
    
    with open(args.o+'-dist.txt', "w") as dfile:
        tot1 = 0
        tot2 = 0
        tot3 = 0
        tot4 = 0
        tot5 = 0
        tot6 = 0
        for d in d2d1:
            dfile.write("%f \n" % (d.degree))
            tot1+=d.degree
        for d in d2d2:
            dfile.write("%f \n" % (d.degree))
            tot2+=d.degree
        for d in d2d3:
            dfile.write("%f \n" % (d.degree))
            tot3+=d.degree
        for d in d2d4:
            dfile.write("%f \n" % (d.degree))
            tot4+=d.degree
        for d in d2d5:
            dfile.write("%f \n" % (d.degree))
            tot5+=d.degree
        for d in d2d6:
            dfile.write("%f \n" % (d.degree))
            tot6+=d.degree
    print "Average distances:"
    print "1 close:", tot1/len(d2d1)
    print "2 close:", tot2/len(d2d2)
    print "3 close:", tot3/len(d2d3)
    print "4 close:", tot4/len(d2d4)
    print "5 close:", tot5/len(d2d5)
    print "6 close:", tot6/len(d2d6)
    print "1/sqrt(3) is", B.FWHM/np.sqrt(3)
    print "1/sqrt(2) is", B.FWHM/np.sqrt(2)
    print "1/1.2 is", B.FWHM/1.2
    
    plot_sky_pointingcount(ras, decs, B, args.o, Nside=2**7, doall=True)
