#!/usr/bin/env python
#
# Script to trace the flux-density evolution along a path at different frequencies.
# Required input: a ds9 region file which contains an ordered sequence of points, fits image(s).
# Author: Henrik Edler

import os, sys, argparse, itertools, multiprocessing, pickle
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.use('Agg')
from astropy import units as u
from astropy.coordinates import Angle
from scipy import interpolate
from scipy.optimize import curve_fit, minimize, brute
import pyregion
import lmfit

import lib_fits, lib_aging
from lib_linearfit import linsq_spidx
log.root.setLevel(log.INFO)

def beam_ellipse(ra, dec, image):
    """
    Return pyregion ellpse regon coresponding to the image beam at ra, dec
    Parameters
    ----------
    ra: float, ra in degrees
    dec: float, dec in degrees
    image: obj, lib_fits.Image

    Returns
    -------
    beam_ellipse: obj, pyregion region
    """
    b = image.get_beam()
    ra = Angle(str(ra)+'d', unit=u.deg).to_string(sep = ':', unit=u.hour)
    dec = Angle(dec, unit=u.deg).to_string(sep = ':')
    fact = 1/np.sqrt(4*np.log(2)) # this factor is needed to make the pixels in the beam area match the beam_area from the lib_fits function. Not sure why...
    ell_str = f"fk5;ellipse({ra}, {dec}, {b[0]*3600*fact}\", {b[1]*3600*fact}\", {b[2]})"
    reg_obj = pyregion.parse(ell_str)
    return reg_obj

def convert_segment_region_to_lines(file):
    # convert a segment region to lines -> aux function since pyregion cannot deal with ds9 segments.
    newfile = '._temp_' + file
    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    with open(file, 'r') as file_i:
        if os.path.exists(newfile): os.remove(newfile)
        with open(newfile, "w") as file_o:
            for line in file_i:
                if 'segment' in line:
                    line = line[line.find('segment')+8::]
                    line, suffix = line.split(')')
                    if '#' not in suffix:
                        suffix += ' # '
                    line = line.replace('\w','') # remove all whitespaces
                    coords = line.split(',')
                    assert len(coords) % 2 == 0
                    out_str = ''
                    last_ra, last_dec = '', ''
                    for i, coord in enumerate(pairwise(coords)):
                        ra, dec = coord
                        if i > 0:
                            out_str += f'line({last_ra},{last_dec},{ra},{dec}){suffix}'
                        last_ra, last_dec = ra, dec

                    file_o.writelines(out_str)
                else:
                    file_o.writelines(line)

def fit_path_to_regions(region, image, z, spacing='beam'):
    """
    Fit a path to a number of ds9 point regions. Then, return points on this path at a certain spacing.
    The region file must be ordered!

    Parameters
    ----------
    region: string, filename of ds9 region file
    image: object, lib_fits.Image()
    z: float, redshift
    spacing: string or float, optional. Default = 'beam'
        distance between sample points on path in deg. If 'beam', use primary beam FWHM.

    Returns
    -------
    xy : (n,2) array of type int
        Interpolated path in pixel coordinates
    l: (n,) array of floats
        Length along the path in kpc
    """
    approxres = 100000 # this is ~infinite for "integration" in length. The higher, the slower but more accurate.
    # Load region, region must be sorted!
    trace = pyregion.open(region)
    trace = np.array([p.coord_list for p in trace.as_imagecoord(image.img_hdr)])

    # Linear interpolation
    distance_lin = np.cumsum(np.linalg.norm(np.diff(trace, axis=0), axis=1))
    distance_lin = np.insert(distance_lin,0,0)
    tck, u = interpolate.splprep([trace[:,0], trace[:,1]], u=distance_lin, s=0)

    # use PB FWHM as spacing
    if spacing == 'beam':
        beam = image.get_beam()
        if not beam[0] == beam[1]:
            raise ValueError(f'Circular beam required!')
        spacing =  beam[0] / image.get_degperpixel()
        log.info(f'Using beam FWHM spacing {spacing:.2f} pix / {image.get_beam()[0]*3600:.0f}arcsec')
    else:
        spacing = spacing / image.get_degperpixel()
        log.info(f'Using spacing {spacing:.2f} pix / {spacing*image.get_degperpixel()*3600:.0f}arcsec')


    # Cubic spline interpolation of linear interpolated data to get correct distances
    # Calculate a lot of points on the spline and then use the accumulated distance from points n to point n+1 as the integrated path
    xy = interpolate.splev(np.linspace(0,u[-1],approxres), tck, ext=2)
    distance_cube = np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=0))
    distance_cube = np.insert(distance_cube,0,0)
    tck, u = interpolate.splprep([xy[0], xy[1]], s=0)
    n_pts = int(distance_cube[-1] / spacing)
    length = np.linspace(0, spacing * n_pts / distance_cube[-1], n_pts)
    xy = np.array(interpolate.splev(length, tck, ext=2)).T # points where we sample in image coords.
    length = length * distance_cube[-1] / image.get_pixelperkpc(z) # length at point i in kpc
    log.info(f"Trace consists of {len(trace)} points. Linear interpolation length: {distance_lin[-1]/image.get_pixelperkpc(z):.2f} kpc,  cubic interpolation length: {distance_cube[-1]/image.get_pixelperkpc(z):.2f} kpc")
    return xy, length

def get_path_regions(region, image, z, mode, offset=0.0):
    # Aux function to get region files for plotting. This will do the path interpolation, and either write exactly the
    # regions used for flux extraction or write the full interpolated path with 100kpc segments.
    if mode == 'fluxregions':
        spacing = 'beam'
        suffix = '-fluxregions.reg'
    elif mode == 'pathlength':
        spacing = 1/3600
        suffix = '-interpolated.reg'
    else: raise ValueError("mode must be fluxregions or pathlength.")
    df = pd.DataFrame()
    xy, l = fit_path_to_regions(region, image, z, spacing)
    df['l'] = l + offset
    radec = image.get_wcs().all_pix2world(xy, 0)
    df['ra'], df['dec'] = radec.T
    line = 'global color=#ababab dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nfk5'
    if offset == 0.0:
        n_pt, last_pt = 0, -1
    else:
        n_pt = int(offset // 100) + 1
        last_pt = n_pt - 1
    for i, p in df.iterrows():
        try:
            _ra = Angle(str(p['ra']) + 'd', unit=u.deg).to_string(sep=':', unit=u.hour)
            _dec = Angle(p['dec'], unit=u.deg).to_string(sep=':')
            _rad = 0.5 * Angle(image.get_beam()[0], unit=u.arcsec).value * 3600
            line += f'\ncircle({_ra}, {_dec}, {_rad:.5f}")'
        except IndexError:
            pass
        if p['l'] // 100 == n_pt and last_pt != n_pt and mode=='pathlength':
            print('draw pt at ', p['l'])
            line += f'\npoint({str(p["ra"])},{str(p["dec"])}) # point = circle 5 text=' + '{' + f'{int(n_pt * 100)}kpc' + '}'
            last_pt = n_pt
            n_pt += 1
    interp_region_name = region.split('.')
    interp_region_name[-1] = suffix
    interp_region_name = ''.join(interp_region_name)
    if os.path.exists(interp_region_name): os.remove(interp_region_name)
    with open(''.join(interp_region_name), 'w') as f:
        f.write(line)

def interpolate_path(region, image, z, offset=0.0, fluxerr=0.0):
    """
    Interpolate a path defined by ordered ds9 points and calculate beam-spaced points on this path.
    Slide a psf-sized region along this path and calculate the mean image value along the path.
    Parameters
    ----------
    region: string
        ds9 region filename - MUST be ordered from start to end.
    image: obj, lib_fits.Image
    n: int, number of points to space on the path
    z: float, redshift

    Returns
    -------
    df: pandas DataFrame :
        trace_data: array, shape(n,)
            Values of sliding psf region means.
        xy: array, shape (n,2)
            Evenly spaced points on the path
        path_length: array, shape (n,)
            Length of the path from first point to point n in kpc
        offset: float, optional. Default = 0.0
            Offset of path start from injection point in kpc
    """
    get_path_regions(region, image, z, 'pathlength', offset) # get the region files as output
    get_path_regions(region, image, z, 'fluxregions', offset)
    df = pd.DataFrame()
    xy, l = fit_path_to_regions(region, image, z)
    df['l'] = l + offset
    radec = image.get_wcs().all_pix2world(xy,0) #TODO check origin
    df['ra'], df['dec'] = radec.T

    path_data_psf = np.zeros(len(xy))
    path_data_error = np.zeros(len(xy))
    norm_pix = None
    for i,p in enumerate(radec):
        beam = beam_ellipse(p[0], p[1], image).as_imagecoord(image.img_hdr)
        beam_mask = beam.get_mask(hdu=image.img_hdu, header=image.img_hdr, shape=image.img_data.shape)
        # b = lib_fits.Image(args.image).get_beam()
        # where0, where1 = np.argwhere(np.any(beam_mask, axis=0)), np.argwhere(np.any(beam_mask, axis=1))
        # n0 = (where0[-1] - where0[0])[0]
        # n1 = (where1[-1] - where1[0])[0]
        # fwhm2sig = 1. / np.sqrt(8. * np.log(2.))
        # psf_weight = Gaussian2DKernel(fwhm2sig*b[0]/degperpixel, fwhm2sig*b[1]/degperpixel, theta=np.deg2rad(b[2]), x_size=n0, y_size=n1).array
        # psf_weight/np.max(psf_weight)
        # print(psf_weight)
        # print(psf_weight.shape, data[beam_mask].shape)
        # print(data[beam_mask])
        npix = np.sum(beam_mask)
        if i == 0: norm_pix = npix
        data = image.img_data[beam_mask]
        nndata = data[~np.isnan(data)]
        path_data_psf[i] = np.sum(nndata) / image.get_beam_area(unit='pixel')
        path_data_psf[i] *= norm_pix / npix # normalize to npix of first iteration, this is because depending on the center, the number of pixels covered by a region can vay
        path_data_error[i] = image.noise * np.sqrt(npix / image.get_beam_area(unit='pixel'))
        path_data_error[i] *= norm_pix / npix
        print(f'flux: {path_data_psf[i]/path_data_error[i]:.2f} sigma')

    df[f'F_{image.mhz}'] = path_data_psf
    df[f'F_err_stat_{image.mhz}'] = path_data_error
    df[f'F_err_{image.mhz}'] = np.sqrt(path_data_error**2 + (path_data_psf*fluxerr)**2) # stat and sys
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    path_analysis.py <fits image> <ds9 region>')
    parser.add_argument('region', help='ds9 point regions defining the path, must be ordered from start to end!')
    parser.add_argument('bg', help='Path to ds9 region for background estimation.')
    parser.add_argument('stokesi', nargs='+', default=[], help='List of fits images of Stokes I.')
    parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
    parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
    parser.add_argument('-b', '--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
    parser.add_argument('--iidx', default = 0.65, type=float, help='Injection index.')
    parser.add_argument('--injection-offset', default = 0.0, type=float, help='Offset of injection point from beginning of path in kpc.')
    parser.add_argument('--max-fit-length', default = np.inf, type=float,  help='Distance from the first region until which data points should be taken into account for the fitting in kpc.')
    parser.add_argument('--fluxerr', default = 0.0, type=float, help='Flux scale error of all images. Provide a fraction, e.g. 0.1 for a 10%% error. If set to a value greater zero, the flux scale uncertainty will be included in the fit as systematic offset! Make sure this is what you want to do. Otherwise, specify --ignore_flux_corr.')
    parser.add_argument('--ignore_fluxerr_corr', action='store_true', help='Ignore the correlation of the flux scale uncertainties.')
    parser.add_argument('-o', '--out', default='path_analysis', type=str, help='Name of the output image and csv file.')
    parser.add_argument('--align', action='store_true', help='Align the images.')
    parser.add_argument('--reuse-shift', action='store_true', help='Resue catalogue shifted images if available.')
    parser.add_argument('--reuse-regrid', action='store_true', help='Resue regrid images if availalbe.')
    parser.add_argument('--reuse-df', action='store_true', help='Resue data frame if availalbe.')
    parser.add_argument('--reuse-fit', action='store_true', help='Reuse fit results if availalbe.')
    args = parser.parse_args()

    stokesi = []
    all_images = lib_fits.AllImages(args.stokesi)
    nimg = len(all_images)

    # find+apply shift w.r.t. first image
    if args.align:
        if args.reuse_shift and all_images.suffix_exists('shifted'):
            log.info('Reuse cat shifted images.')
            all_images = lib_fits.AllImages([name.replace('.fits', '-shifted.fits') for name in args.stokesi])
        else:
            log.info('Align images to catalogue matches')
            all_images.align_catalogue()
            all_images.write('shifted')

    # convolve images to the same beam (for now force circ beam)
    if args.reuse_regrid and all_images.suffix_exists('convolve-regrid'):
        log.info('Reuse prepared images.')
        all_images = lib_fits.AllImages([name.replace('.fits', '-convolve-regrid.fits') for name in all_images.filenames])
    else:
        log.info('Recenter, convolve and regrid all images')
        all_images.convolve_to(circbeam=True)
        all_images.regrid_common(pixscale=1)
        all_images.write('convolve-regrid')

    for image in all_images:
        image.calc_noise() # update noise in all images

    if args.reuse_df and os.path.exists(f'{args.out}.csv'):
        log.info('Reuse data frame.')
        df = pd.read_csv(f'{args.out}.csv')
    else:
        df_list = []
        for image in all_images:
            df_thisimg = interpolate_path(args.region, image, args.z, args.injection_offset, args.fluxerr)
            df_list.append(df_thisimg)

        df = pd.concat(df_list, axis=1)
        df = df.reindex(df_list[0].index)
        df = df.loc[:,~df.columns.duplicated()]
        # calc spectral index from sampled fluxes.
        # we copy column F_nuXX -> F_mod_nuXX just to make sure we do not accidentally use the modified flux values!
        for i, image in enumerate(all_images):
            df[f'F_temp_{image.mhz}'] = df[f'F_{image.mhz}'].copy()

        for image1, image2 in zip(all_images[:-1], all_images[1:]):
                # find masks for lower limits / upper limits / not even a limit
                is_ll = np.logical_and(df[f'F_{image1.mhz}'] < 3 * df[f'F_err_stat_{image1.mhz}'],
                                       df[f'F_{image2.mhz}'] > 3 * df[f'F_err_stat_{image2.mhz}'])
                is_ul = np.logical_and(df[f'F_{image1.mhz}'] > 3 * df[f'F_err_stat_{image1.mhz}'],
                                       df[f'F_{image2.mhz}'] < 3 * df[f'F_err_stat_{image2.mhz}'])
                is_none = np.logical_and(df[f'F_{image1.mhz}'] < 3 * df[f'F_err_stat_{image1.mhz}'],
                                         df[f'F_{image2.mhz}'] < 3 * df[f'F_err_stat_{image2.mhz}'])

                # set temp flux to 3 sigma where we have a limit
                df.loc[is_ll, f'F_temp_{image1.mhz}'] = 3 * df.loc[is_ll, f'F_err_stat_{image1.mhz}']
                df.loc[is_ul, f'F_temp_{image2.mhz}'] = 3 * df.loc[is_ul, f'F_err_stat_{image2.mhz}']

                si, sierr_stat = linsq_spidx([image1.freq, image2.freq],
                                        [df[f'F_temp_{image1.mhz}'], df[f'F_temp_{image2.mhz}']],
                                        [df[f'F_err_stat_{image1.mhz}'],  df[f'F_err_stat_{image2.mhz}']])
                si, sierr = linsq_spidx([image1.freq, image2.freq],
                                             [df[f'F_temp_{image1.mhz}'], df[f'F_temp_{image2.mhz}']],
                                             [df[f'F_err_{image1.mhz}'],  df[f'F_err_{image2.mhz}']])
                si[is_none], sierr[is_none], sierr_stat[is_none] = np.nan, np.nan, np.nan

                df[f'SI_{image1.mhz}-{image2.mhz}'] = si
                df[f'SI_err_stat_{image1.mhz}-{image2.mhz}'] = sierr_stat
                df[f'SI_err_{image1.mhz}-{image2.mhz}'] = sierr
                df[f'SI_ul_{image1.mhz}-{image2.mhz}'] = is_ul
                df[f'SI_ll_{image1.mhz}-{image2.mhz}'] = is_ll

        log.info(f'Save DataFrame to {args.out}.csv')
        df.to_csv(f'{args.out}.csv')

    ####################################################################################################################
    # do the fitting
    S = lib_aging.S_model(epsrel=1e-4) #1.5e-2)
    B_min = 3.18e-10 * (1+args.z)**2 * 3**-0.5 # This is B_eq  / sqrt(3)
    log.info(f"Using minimum loss magnetic field B={B_min:.4e}T")

    kmpers_to_kpc_per_Myr = float((u.Myr / u.s * u.km / u.kpc).decompose().to_string())
    l_sel = (0 <= df['l'].values) & (df['l'].values - args.injection_offset <= args.max_fit_length)  # length selection

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    # Define X and Y data for the fit!
    nus = np.array([im.freq for im in all_images])  # image frequencies
    si_pairs = [f'{image1.mhz}-{image2.mhz}' for image1, image2 in zip(all_images[:-1], all_images[1:])]
    # construct X values -> [[d1, nu1, nu2, .., nuN], [d2, nu1, nu2, ..., nuN],..., [dN..]]
    X = np.array([[d_i, *nus] for d_i in df['l'][l_sel]])
    # construct Y values as arrays
    Y = df[[f'SI_{pair}' for pair in si_pairs]][l_sel].to_numpy()
    if args.ignore_fluxerr_corr: # if not fitting flux scale error use full error including statistical and systematic
        Yerr = df[[f'SI_err_{pair}' for pair in si_pairs]][l_sel].to_numpy()
    else: # use only statistical uncertainty and not systematic if fitting sys uncertainty
        Yerr = df[[f'SI_err_stat_{pair}' for pair in si_pairs]][l_sel].to_numpy()

    Yul = df[[f'SI_ul_{pair}' for pair in si_pairs]][l_sel].to_numpy()
    Yll = df[[f'SI_ll_{pair}' for pair in si_pairs]][l_sel].to_numpy()
    # set data to NaN where we have UL, LL for fit.
    Y[np.logical_or(Yul, Yll)] = np.nan
    Yerr[np.logical_or(Yul, Yll)] = np.nan

    # define function for fit -> give spectral index at location and frequency depending on B field and velocity (age)
    def residual_SI_aging_path(param, ignore_fluxerr_corr=False):
        """
        Function for fitting the flux density along a path (e.g. a RG tail).
        https://www.desy.de/~sschmitt/blobel/apltalk.pdf
        https://www.desy.de/~sschmitt/blobel/apltalk.pdf
        Uses from above:
        X: (n_pt, 1 + n_nu) array. Contains input points, the two cols are [number of points, [distance, nu1, nu2,...nui]].
        B_min: minimum aging mag field
        Parameters
        ----------
        p: object, input parameters [v, b1, ..., bn-1] where v velocity in km/s and bi are the relative errors in the flux scales - b1 = s1/s2 etc.
        ignore_fluxerr_corr: bool, whether to take into account correlation of flux scale uncert. by fitting them (default) or to ignore the correlation.

        Returns
        -------
        merit: float, residual cost function, weighted by (Y/Yerr)**2
        """
        global X, Y, Yerr
        if len(param) > 1 and not ignore_fluxerr_corr:
            v, bs = param[0], param[1:]
        else:
            v, bs = param[0], np.zeros(nimg-1)
        if np.ndim(X) == 1:
            X = [X]
        map_args = []
        for id_d, d in enumerate(X[:, 0]):
            for id_nu in range(len(X[0]) - 2):
                id_nu += 1
                map_args.append(
                    [X[id_d, id_nu], X[id_d, id_nu + 1], B_min, args.iidx, d / (v * kmpers_to_kpc_per_Myr), args.z, S])
        model = np.array(list(p.starmap(lib_aging.get_aging_si, map_args)))

        # This is the +1 sigma shift for an error in the flux scale ratio -> these are the additive systematics in our problem
        # ln(expectation_fluxratio + 1sigm_error_fluxratio) / ln(nu1/nu2)
        shift = np.array([np.log(1 + np.sqrt(2)*args.fluxerr) / np.log(nus[i]/nus[i+1]) for i in range(nimg-1)]) # for a 1 sigma change of systematic
        # this is simply the scale of the additive error, 0 +/- 1
        bs_shift = np.tile(bs*shift, (len(X),1)) # shape: (n_pts * n_imgs-1)
        residual = np.sum(((Y + bs_shift - model.reshape((len(X), len(X[0]) - 2)) )/Yerr)**2) + np.sum(bs**2)
        print(param, residual)
        return residual

    if args.reuse_fit and os.path.exists(f'{args.out}-fit.pickle') and os.path.exists(f'{args.out}-fit-norms.pickle'):
        # reuse fit results for SI-fit and for normalization fit
        log.info('Reuse fit results.')
        with open(f'{args.out}-fit.pickle', 'rb') as f:
            v, bs = pickle.load(f)

        with open(f'{args.out}-fit-norms.pickle', 'rb') as f:
            norm_results = pickle.load(f)
    else:
        # Grid search for starting values...
        log.info('Perform grid search to find suited starting value (range: 500km/s to 2000km/s)')
        gridsearch = brute(residual_SI_aging_path, [[500,4000]], Ns=20, full_output=True, finish=None)
        log.info(f'Best grid point: {gridsearch[0]} km/s')

        if args.fluxerr > 0. and not args.ignore_fluxerr_corr: # take into account that the fluxscale uncertainties of each image are correlated by fitting them explicitly
            x0 = np.array([gridsearch[0], *np.zeros(nimg-1)])
            # x0 = np.array([gridsearch[0][0], *np.zeros(nimg-1)])
            bounds = tuple([[10,20000]] +  [[-2,3] for i in range(nimg-1)])
        else: # here either no fluxscale uncertainties or they are assumed to be uncorrelated.
            x0 = gridsearch[0]
            bounds = ([[10,20000]])
        log.info('Start the spectral age model fitting (this may take a while)...')
        mini = minimize(residual_SI_aging_path, x0, args=(args.ignore_fluxerr_corr), bounds=bounds) # Test is tol makes sense
        # mini = minimize(residual_SI_aging_path, x0, args=(args.ignore_fluxerr_corr), method='Nelder-Mead') # Test is tol makes sense

        dof = np.product(np.shape(Y)) - 1 # degrees of freedom, number of points minus 1 parameter (velocity)
        if args.fluxerr > 0 and not args.ignore_fluxerr_corr: # if fitting flux scale uncertainty, this is also degree of freedom.
            dof -= nimg - 1 # (if we also fit the flux-scale uncertainties)
        print(f"Fit chi-sq={mini['fun']}, d.o.f.={dof}")
        result = mini['x']
        v = result[0]
        if args.fluxerr > 0 and not args.ignore_fluxerr_corr:
            bs = result[1:]
        else:
            bs = np.zeros(nimg-1)

        # mini = lmfit.Minimizer(residual_SI_aging_path, params, nan_policy='propagate', reduce_fcn=red_fct) # identity function to allow Minimizer to accept scalar return
        # first solve with grid search
        # out1 = mini.minimize(method='brute')
        # log.info('Finished initial grid search')
        # lmfit.report_fit(out1.params, min_correl=0.5)
        # with open(f'{args.out}-grid.pickle', 'wb') as f:
        #     pickle.dump([out1.params, out1.brute_grid, out1.brute_Jout], f)
        # log_Jout = np.log(out1.brute_Jout)
        # plt.contourf(*out1.brute_grid, log_Jout, vmin=np.min(log_Jout), vmax=np.median(log_Jout), levels=np.linspace(np.min(log_Jout), np.median(log_Jout), 20) )
        # plt.colorbar()
        # plt.xlabel('B [T]')
        # plt.ylabel('v [km/s]')
        # plt.savefig('fit-grid.png')

        # print(Yerr)
        # log.info('Leastsq fit')
        # result = mini.minimize(method='leastsq', params=params) # out1.params
        # lmfit.report_fit(result.params, min_correl=0.5)

        # log.info('Confidence intervals')
        # cx, cy, grid = lmfit.conf_interval2d(mini, result, 'B', 'v', 30, 20, ((2.0e-10, 7e-10), (600, 1300)))
        # save to pickle
        with open(f'{args.out}-fit.pickle', 'wb') as f:
            pickle.dump([v, bs], f)

    si_offsets = np.array([np.log(1 + np.sqrt(2)*args.fluxerr) / np.log(nus[i]/nus[i+1]) for i in range(nimg-1)]) # for a 1 sigma change of systematic
    if args.ignore_fluxerr_corr:
        log.info(f"Fit results: v={v:.5f} km/s")
    else:
        log.info(f"Fit results: v={v:.5f} km/s:{np.array(bs)*si_offsets} ({bs} sigma)")

    ## now fit normalization (in log space)
    # define function to fit the normalizations
    def fit_N0(x, N0):
        """
        Model function to fit normalizations after deriving everything else from the SI fit.
        Fitting in log10-space!
        Parameters
        ----------
        x: array_like, shape(len(nu), 5); independent vars like [[nu_0, B, iidx, t, z], [nu_1, B, iidx, t, z], ...]
        N0: float, dependent variable - Normalization to fit

        Returns
        -------
        y: array_like, shape len(nu),
        """
        # map_args = np.array([[*x_i, N0] for x_i in x])
        # res =  np.array(list(p.starmap(S.evaluate, map_args)))
        res = np.empty(len(x))
        for i, x in enumerate(x):
            seval = S.evaluate(*x, N0)
            res[i] = np.log10(seval)
            print(seval)
        return res

    S_model = lmfit.Model(fit_N0)
    S_model.make_params(N0=1.5e21) # initial guess
    S_model.set_param_hint('N0', min=0.0) # initial guess
    # params = S_model.param

    norm_results = []
    for i, row in df.iterrows():
        if not l_sel[i]: continue # only fit normalization where we also fitted the SI
        x = np.array([[nu_i, B_min, args.iidx, row['l'] / (kmpers_to_kpc_per_Myr * v), args.z] for nu_i in nus], dtype=float)
        # print(row[[f'F_{im.mhz}' for im in all_images]])
        y = row[[f'F_{im.mhz}' for im in all_images]].to_numpy(dtype=float)
        yerr = row[[f'F_err_{im.mhz}' for im in all_images]].to_numpy(dtype=float)
        sigma_logy = np.abs(yerr/y)
        norm_results.append(S_model.fit(np.log10(y), N0=1.5e21, x=x, weights=sigma_logy**-2))  # For non-log fit: (y/yerr)**2)
    print('Normalization factors: ', [norm_result.params['N0'] for norm_result in norm_results])
    # save to pickle
    with open(f'{args.out}-fit-norms.pickle', 'wb') as f:
        pickle.dump(norm_results, f)

    B =  B_min # result.best_values['B']
    velocity = v*kmpers_to_kpc_per_Myr #result.params['v']

    ####################################################################################################################
    # do the plotting
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
    ax[1].set_xlabel('distance [kpc]')

    fs = 7 # fontsize
    lw = 0.6
    ebarconf = {'capsize': 1,
                'elinewidth': lw,
                'linestyle': 'None',
                'markersize': 1.5,
                'marker': 'o',
                'xerr': np.mean(np.diff(df['l']))/2}

    cs = ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C7', 'C8'] # loop colors
    ### Do flux density part
    # loop through images / frequencies and get the flux densities of the best fittin model, plot measured and fitted fluxes
    for j, im in enumerate(all_images):
        df[f'F_model_{im.mhz}'] = np.nan
        df[f'F_model_hi_{im.mhz}'] = np.nan
        df[f'F_model_lo_{im.mhz}'] = np.nan
        # 1. Get the model fluxes and uncertainties for the range of distances (l_sel) that we used in the fit.
        #    Here we have a fit for the normalization N0 and we want to plot them.
        for i, (l_i, norm_result) in enumerate(zip(df['l'].values[l_sel], norm_results)):
            df.loc[np.argwhere(l_sel)[i], f'F_model_{im.mhz}'] = S.evaluate(im.freq, B, args.iidx, l_i / velocity, args.z, norm_result.params['N0'])
            # df.loc[np.argwhere(l_sel)[i], f'F_model_hi_{im.mhz}'] = S.evaluate(im.freq, B, args.iidx, (l_i - results[2]) / velocity, args.z, N0_i)
        # TODO plot fit uncertainty band
        # scale data to 144MHz using injectindex
        scale = (144e6/im.freq)**(-args.iidx)
        # plot best-fitting model - only where we fitted the model since there we have a value for the normalization TODO plot errors
        ax[0].plot(df['l'][l_sel], df[f'F_model_{im.mhz}'][l_sel]*scale, color=cs[j], label=f'model {im.mhz} MHz', alpha=0.7, linewidth=lw)
        # plot measured fluxes and errors
        errs = df[f'F_err_{im.mhz}'] # include systematic error
        ax[0].errorbar(df['l'], df[f'F_{im.mhz}']*scale, errs*scale,
                       uplims=df[f'F_{im.mhz}'] < 3*errs, color=cs[j],
                       label=f'{im.mhz} MHz', **ebarconf)
        ax[0].set_ylabel(r'$S_{144}$ [Jy]')

    # if args.velocity
    times = np.linspace(0,df['l'].values[-1]/velocity,10) #8

    ax[0].set_xlim([0,df['l'].max()])
    ax[0].set_yscale('log')
    ax[0].set_ylim(top=1.e-1, bottom=2.e-4)
    ax[0].yaxis.set_tick_params(right='on', labelright=False, which='both')
    ax0_top = ax[0].twiny()
    ax0_top.set_xlim([0,df['l'].values[-1]/velocity])
    # ax0_top.set_xticks(velocity*times)
    # ax0_top.set_xticklabels(times)
    ax0_top.xaxis.set_major_formatter(FormatStrFormatter('%.i'))
    ax0_top.set_xlabel('age [Myr]')
    ax0_top.xaxis.set_tick_params(bottom='off', top='on', labeltop=True, direction='out', labelbottom=False)

    ax[0].xaxis.set_tick_params(bottom='on', top='off',  labeltop=False, labelbottom=False, direction='in')
    ax[0].legend(loc='best', fontsize=fs, ncol=2)

    ### Do spectral index part
    si_model = []
    # we copy column F_nuXX -> F_mod_nuXX just to make sure we do not accidentally use the modified flux values!
    for i, (image1, image2) in enumerate(zip(all_images[:-1], all_images[1:])):
        i += 1
        ax[1].errorbar(df['l'], df[f'SI_{image1.mhz}-{image2.mhz}'], df[f'SI_err_{image1.mhz}-{image2.mhz}'], uplims=df[f'SI_ul_{image1.mhz}-{image2.mhz}'], lolims=df[f'SI_ll_{image1.mhz}-{image2.mhz}'], color=cs[i+j], label=r'$\alpha_{' + f'{image1.freq*1e-6:.0f}' + '}^{' + f'{image2.freq*1e-6:.0f}' + r'}$', **ebarconf)
        delta_si = bs[i-1] * np.log(1 + np.sqrt(2) * args.fluxerr) / np.log(nus[i-1] / nus[i])
        df[f'SI_model_{image1.mhz}-{image2.mhz}'] = lib_aging.get_aging_si(image1.freq, image2.freq, B, args.iidx, df['l'] / velocity, args.z, S) - delta_si

        ax[1].plot(df['l'][l_sel], df[f'SI_model_{image1.mhz}-{image2.mhz}'][l_sel], color=cs[i+j], marker='None', label=r'$\alpha_{' + f'{image1.mhz}' + '}^{' + f'{image2.mhz}' + r'}$'+f' JP', linewidth=lw)
        # plot SI where we do not fit but extrapolate as dotted line. 1st mask entry needs to be changed to not get a gap.
        extrapol = ~l_sel
        try:
            extrapol[np.argwhere(extrapol)[0] - 1] = True
        except: IndexError
        ax[1].plot(df['l'][extrapol], df[f'SI_model_{image1.mhz}-{image2.mhz}'][extrapol], color=cs[i+j], marker='None', linestyle='dotted', linewidth=lw)
        # as soon as we have fit uncertainty band, use fill between...
        # ax[1].fill_between(df['l'], si-sierr, si+sierr, color=cs[i+j], alpha=0.3)

    ax1_top = ax[1].twiny()
    # ax1_top.set_xticks(velocity*times)
    # ax1_top.set_xticklabels(velocity*times)
    ax1_top.set_xlim([0,df['l'].values[-1]/velocity])
    ax1_top.xaxis.set_tick_params(top='on', bottom='off', labeltop=False, labelbottom=False, direction='in')


    ax[1].set_xlim([0,df['l'].max()])
    ax[1].yaxis.set_tick_params(right='on', labelright=False)
    # ax[1].set_xticks(velocity*times)
    # ax[1].set_xticklabels(velocity*times)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.i'))
    ax[1].xaxis.set_tick_params(bottom='on',  top='off', labeltop=False, labelbottom=True)
    ax[1].set_xlabel('distance [kpc]')
    ax[1].set_ylabel('spectral index')
    ax[1].set_ylim(top=-0.5, bottom=-4.5)
    ax[1].legend(fontsize=fs, ncol=2,loc='lower left')

    log.info(f'Save plot to {args.out}.pdf...')
    plt.savefig(args.out+'.pdf')
    plt.close()
