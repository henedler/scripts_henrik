#!/usr/bin/env python
import numpy as np
import scipy.optimize
from astropy.io import fits
import lib_fits
from lib_beamdeconv import deconvolve_ell, EllipticalGaussian2DKernel
from reproject import reproject_exact

reproj = reproject_exact
import pyregion
from astropy import convolution
from scipy import odr
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.stats import median_absolute_deviation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from astropy.visualization import (SqrtStretch, PercentileInterval,
                                   LinearStretch, LogStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)
from scipy.optimize import minimize
from lib_plot import addRegion, addCbar, addBeam, addScalebar, setSize, ArrowHatch


def calc_noise(data):
    eps = 1e-3
    data = data[~np.isnan(data) & (data != 0)]  # remove nans and 0s
    initial_len = len(data)
    mad_old = 0.
    for i in range(1000):
        mad = median_absolute_deviation(data)
        # print('MAD noise: %.3e on %f%% data' % (mad, 100*len(data)/initial_len))
        if np.isnan(mad): break
        if np.abs(mad_old - mad) / mad < eps:
            rms = np.nanstd(data)
            print('Noise: %.3e (data len: %i -> %i - %.3e%%)' % (
                rms, initial_len, len(data), 100 * len(data) / initial_len))
            return rms
        data = data[np.abs(data) < (5 * mad)]
        mad_old = mad
    raise Exception('Noise estimation failed to converge.')


# Linear model for ODR-fit
def fct_odr(b, val_x):
    return b[0] * val_x + b[1]


def fit_odr(val_x, val_y):
    # First sort the data into different arrays based on their spectral index
    val_log_x = np.log10(val_x)
    val_log_y = np.log10(val_y)
    # Define the odr fitting function and then start the fit
    linear = odr.Model(fct_odr)
    mydata = odr.Data(val_log_x, val_log_y, )
    myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    outodr = myodr.run()

    def fc(x, a, b):
        return x * a + b

    # # Print results to screen and then return all the relevant data (values, std errors and chi squared)
    # print('########## FIT RESULTS ###########')
    # print('ODR:    a =\t', '%0.3f' % outodr.beta[0], '+/-\t', '%0.3f' % outodr.sd_beta[0])
    # print('        b =\t', '%0.3f' % outodr.beta[1], '+/-\t', '%0.3f' % outodr.sd_beta[1])
    # print('Chi squared = \t', '%0.3f' % outodr.sum_square)
    return [outodr.beta[0], outodr.sd_beta[0]], [outodr.beta[1], outodr.sd_beta[1]], outodr.sum_square  # , linreg


def estimate_lcre(radio, sfr, radio_noise, sfr_noise, pixsize, sigma=4, roi_mask=None, name=None):
    def get_plcoeff_smoothed(lcre, doplot=False):
        lcre = np.abs(lcre)
        kernelsize = ((lcre / 1.177) / 3600) / (pixsize)
        kernel_extent = int(kernelsize) * 5
        if kernel_extent % 2 == 0:
            kernel_extent += 1
        gauss_kern = EllipticalGaussian2DKernel(kernelsize, kernelsize, (90 + bpa) * np.pi / 180., x_size=kernel_extent,
                                                y_size=kernel_extent)
        sfr_conv = convolution.convolve(sfr, gauss_kern, boundary='extend', preserve_nan=True)
        mask = (radio > sigma * radio_noise) & (sfr > sigma * sfr_noise) & (~np.isnan(sfr_conv)) & (
            ~np.isnan(radio)) & ~roi_mask
        radio_masked = radio[mask]
        sfr_masked = sfr_conv[mask]
        # m, b = np.polyfit(np.log10(sfr_masked), np.log10(radio_masked), 1)
        m, b, sumsq = fit_odr(sfr_masked, radio_masked)
        m = m[0]
        b = b[0]
        sfr_conv[~mask] = 0
        radio_plot = radio
        radio_plot[~mask] = 0
        if doplot:
            print('Corr:', np.corrcoef(sfr_masked, radio_masked))
            fig, axs = plt.subplots(2, 2)
            axs[0,0].imshow(radio_plot)
            axs[0,1].imshow(sfr_conv)
            axs[1,0].imshow(radio_plot / sfr_conv)
            axs[1,1].scatter(sfr_masked, radio_masked, s=1)
            axs[0,0].set_title(f'gal={name.split("/")[-1]}, m={m:.2f}, sumsq={sumsq:.2f}')
            xs = np.linspace(np.min(sfr_masked), np.max(sfr_masked), 10)
            axs[1,1].set_xlabel('B_model')
            axs[1,1].set_ylabel('B_obs')
            axs[1,1].plot(xs, xs**m*10**b, c='C1')
            axs[1,1].plot(xs, xs**m, c='C2')
            axs[1,1].set_xscale('log')
            axs[1,1].set_yscale('log')
            plt.savefig(f'{name}-fit.png')
        print(lcre, m, sumsq)  # , np.corrcoef(np.log10(sfr_masked), np.log10(radio_masked)))
        return np.abs(m - 1)
    grid = np.linspace(10, 65, 10)
    gridsearch = [get_plcoeff_smoothed(g) for g in grid]
    res = minimize(get_plcoeff_smoothed, grid[np.argmin(gridsearch)], tol=0.05, method='Nelder-mead')  # arcsec
    get_plcoeff_smoothed(res['x'][0], True)
    print(res)
    diffus = (res['x'][0] / 1.177) / 3600
    kernelsize = (diffus * 1.177) / pixsize
    kernel_extent = int(kernelsize) * 5
    if kernel_extent % 2 == 0:
        kernel_extent += 1
    gauss_kern = EllipticalGaussian2DKernel(kernelsize, kernelsize, (90 + bpa) * np.pi / 180., x_size=kernel_extent,
                                            y_size=kernel_extent)
    sfr_conv = convolution.convolve(sfr, gauss_kern, boundary='extend', preserve_nan=True)
    mask = (radio > sigma * radio_noise) & (sfr > sigma * sfr_noise) & (~np.isnan(sfr_conv)) & (~np.isnan(radio))
    radio_masked = radio[mask]
    sfr_masked = sfr_conv[mask]
    plt.scatter(np.log10(sfr_masked), np.log10(radio_masked))
    xs = np.linspace(np.min(np.log10(sfr_masked)), np.max(np.log10(sfr_masked)), 10)
    plt.xlabel('')
    # plt.plot(xs, xs+b, c='C1')
    return res['x'][0]


fwhm2sigma = 1. / np.sqrt(8. * np.log(2.))
reg = pyregion.open('mask.reg') # masked stuff in radio
sfr_reg = pyregion.open('sfr_mask.reg') # masked stuff in SFR
tails = pyregion.open('tails.reg') # masked stuff in SFR
t = Table.read('/home/p1uy068/data/virgo/sfr_maps/LOFAR_HRS_sample.csv')
lvcs = Table.read('/home/p1uy068/data/virgo/sfr_maps/lvcs_230313_dist.csv')

imf_conversion =  1.0 # 1.56  factor to get to Salpeter IMF
galsic = [800, 3105, 3258, 3476]
sfrnoiseic = imf_conversion * np.array([0.00038, 0.0004, 0.00035, 0.00035])
galsngc = [4254, 4302, 4330, 4388, 4396, 4402, 4424, 4438, 4501, 4522, 4548, 4569, 4607, 4634, 4654]
sfrnoises = [0.00047, 0.00039, 0.00043, 0.0006, 0.00038, 0.0004, 0.000394, 0.00037, 0.00044, 0.00046, None, 0.00048,
             0.00037, None, 0.00034]
sfrnoises = Table([galsngc, np.array(sfrnoises)], names=['NGC', 'sfrnoise'])
for row in sfrnoises:
    if row['sfrnoise']:
        row['sfrnoise'] = imf_conversion * row['sfrnoise']

colors = [(1, 1, 1), (0.65, 0.65, 0.65)]  # R -> G -> B
n_bins = [0, .1]  # Discretizes the interpolation into bins
cmap_name = 'grey'
grey_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
t['r25'][t['IC'] == 3476] = 1.6 * 60
t['r25'][t['IC'] == 3258] = 1.2 * 60
t['r25'][t['IC'] == 3105] = 1.2 * 60
t['r25'][t['IC'] == 800] = 1.1 * 60

hres = True
ngc = False

if hres:
    # results hres
    lcrengc = {4254: 64.8984375, 4302: 55.251736111111114, 4330: 24.027777777777775, 4388: None, 4396: 28.8203125, 4402: 17.495659722222225, 4424: 23.020833333333336, 4438: None, 4501: 16.74045138888889, 4522: 12.1875, 4548: 93.412109375, 4569: 31.565104166666668, 4607: 19.999999999999993, 4634: 25.069444444444443, 4654: 37.45833333333333}
    lcreic = {800: 25.069444444444443, 3105: None, 3258: 3.0, 3476: 21.21527777777778}
else:
    # transport length - of not None, skip computation
    lcreic = {800: 20.625, 3105: 15.09375, 3258: 23.0625, 3476: 21.84375}
    lcreic = {800: 28.111979166666668, 3105: 28.333333333333332, 3258: 23.09027777777778, 3476: 22.951388888888893}

    # lcreic =  {800:None,     3105:6.25,   3258:6.26,    3476:6.26}
    lcrengc = {4254: 67.99609375, 4302: 49.07291666666667, 4330: 23.055555555555557, 4388: None, 4396: 15.934895833333332,
               4402: 20.069444444444436, 4424: 18.024305555555557, 4438: None, 4501: 10.9375, 4522: 21.21527777777778,
               4548: 68.40234375, 4569: 24.375, 4607: 18.276041666666664, 4634: 22.15277777777778, 4654: 38.020833333333336}
    lcrengc = {4254: 67.99609375, 4302: 49.07291666666667, 4330: None, 4388: None, 4396: 15.934895833333332, 4402: 20.069444444444436, 4424: 18.024305555555557, 4438: None, 4501: 10.9375, 4522: 20.069444444444436, 4548: 68.40234375, 4569: 24.375, 4607: 18.276041666666664, 4634: 22.15277777777778, 4654: 37.51215277777777}

# for key in lcrengc.keys():
#     lcrengc[key] = None
#
# for key in lcreic.keys():
#     lcreic[key] = None

galslist = galsngc if ngc else galsic
lcrelist = lcrengc if ngc else lcreic
sfrnoises = sfrnoises if ngc else sfrnoiseic
# lcrelist[4548] = None

if __name__ == '__main__':
    for gal in galslist[0:]:
        if hres and gal == 3105:
            continue
        if ngc:
            if f'{gal}' in t['NGC']:
                row = t[t['NGC'] == f'{gal}']
            else:
                print(f'skip {gal}')
                continue
        else:
            if gal in t['IC']:
                row = t[t['IC'] == gal]
            else:
                print(f'skip {gal}')
                continue
        print(f'Running {gal}')
        row_lvcs = lvcs[lvcs['VCC'] == row['VCC'].data[0]]
        if hres:
            radio_noise = row_lvcs['rms_high'].data[0] # 9as noise level
        else:
            radio_noise = row_lvcs['rms_low'].data[0] # 20as noise level
        rad, logmstar, center = row_lvcs['Rad'].data[0], row['logMstar'].data[0], [row['RAJ2000'].data[0],
                                                                                   row['DEJ2000'].data[0]]
        # determine plot size from galaxy radius
        if gal in [4254]:
            size = [12 * rad / 200, 12 * rad / 200]
        elif gal in [3476]:
            size = [10 * rad / 200, 10 * rad / 200]
        else:
            size = [7 * rad / 200, 7 * rad / 200]
        # Load data
        if ngc:
            sfrmap = f'VICTORIA_SFR_maps/NGC{gal}_NUV+100_9as.fits'
            if hres:
                radiomap = f'radiomaps/NGC{gal}-VCC{row["VCC"].data[0]}-high-9as.fits'
            else:
                radiomap = f'radiomaps/NGC{gal}-VCC{row["VCC"].data[0]}-low.fits'
        else:
            sfrmap = f'VICTORIA_SFR_maps/IC{gal}_NUV+100_9as.fits'
            if hres:
                radiomap = f'radiomaps/VCC{row["VCC"].data[0]}-high-9as.fits'
            else:
                radiomap = f'radiomaps/VCC{row["VCC"].data[0]}-low.fits'
        sfr = fits.open(sfrmap)[0]
        sfr.data = imf_conversion * sfr.data # correct for IMF
        sfr_bgreg = pyregion.open(sfrmap.replace('.fits', '.reg'))
        radio = lib_fits.Image(radiomap)

        if gal in [3105]:
            sigma_map = 2
        else:
            sigma_map = 3
        # regrid radio data to SFR header
        regrid_hdr = sfr.header
        del regrid_hdr['FILTERS'], regrid_hdr['BUNIT']
        regrid_hdr['BMIN'], regrid_hdr['BMAJ'], regrid_hdr['BPA'] = radio.img_hdr['BMIN'], radio.img_hdr['BMAJ'], \
                                                                    radio.img_hdr['BPA']
        radio_data_regrid, __footprint = reproj((radio.img_data, radio.img_hdr), regrid_hdr, parallel=True)
        radio_data_regrid[radio_data_regrid < sigma_map * radio_noise] = np.nan
        radmask = reg.as_imagecoord(regrid_hdr).get_mask(sfr)  # radio mask
        if gal in [4254]:
            fac = 1.5
        elif gal in [4330, 4522, 4654]:
            fac = 1.2
        else:
            fac = 1.0
        roi_reg = pyregion.parse(
            f'fk5;circle({row["RAJ2000"].data[0]},{row["DEJ2000"].data[0]},{row["r25"].data[0] * fac}")')
        roi_reg.write('roi.reg')
        roi_mask = ~roi_reg.get_mask(hdu=sfr)
        radio_data_regrid[radmask | roi_mask] = np.nan # blank radio image where masked or where not in ROI
        rad_dat_ctr = radio_data_regrid.copy()

        radio_regrid = fits.PrimaryHDU(radio_data_regrid, regrid_hdr)
        radio_regrid.writeto(radiomap.replace('.fits', '-regrid.fits'), overwrite=True)

        # Do first convolution of SFR data to match radio data
        target_beam = [9.01/3600, 9.01/3600, 0.] if hres else [1 / 180, 1 / 180, 0.]
        N0, gamma = 7.81e21, 0.21
        lum_jy = 1e-26 * 4 * np.pi * (16.5 * 1e6 * 3.08567758e16) ** 2
        gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        beam_area = 2.0 * np.pi * (target_beam[0] * target_beam[1] * 3600 ** 2) / (gfactor * gfactor)  # arcsec^2
        kpcsq_per_beam = beam_area / (0.08 ** 2)

        beam = [1 / 400, 1 / 400, 0]
        convolve_beam = deconvolve_ell(target_beam[0], target_beam[1], target_beam[2], beam[0], beam[1], beam[2])
        print('Convolve beam: %.3f" %.3f" (pa %.1f deg)' % (
            convolve_beam[0] * 3600, convolve_beam[1] * 3600, convolve_beam[2]))
        bmaj, bmin, bpa = convolve_beam
        pixsize = abs(sfr.header['CDELT1'])
        gauss_kern = EllipticalGaussian2DKernel((bmaj * fwhm2sigma) / pixsize, (bmin * fwhm2sigma) / pixsize,
                                                (90 + bpa) * np.pi / 180.)  # bmaj and bmin are in pixels
        sfr_regimg = sfr_reg.as_imagecoord(regrid_hdr)
        stars_mask = (sfr.data == 0.0) | sfr_regimg.get_mask(sfr)
        sfr.data[stars_mask] = np.nan
        sfr.data = convolution.convolve(sfr.data, gauss_kern, boundary='extend', preserve_nan=True)
        sfr_dat_ctr = sfr.data.copy() # so we can NaN where outside of RoI
        sfr_dat_ctr[roi_mask | radmask] = np.nan
        sfr_ctr_noise = np.nanstd(sfr_dat_ctr[sfr_bgreg.as_imagecoord(regrid_hdr).get_mask(sfr) & (sfr.data != 0)])
        sfr.writeto(sfrmap.replace('.fits', '-conv.fits'), overwrite=True)
        # Estimate diffusion length of CRE
        radio_ff = ((radio_data_regrid / (beam_area / ((3600 * regrid_hdr['CDELT1']) ** 2))) * (
                1e-26 * 4 * np.pi * (16.5 * 1e6 * 3.08567758e16) ** 2))
        radio_noise_ff = ((radio_noise / (beam_area / ((3600 * regrid_hdr['CDELT1']) ** 2))) * (
                1e-26 * 4 * np.pi * (16.5 * 1e6 * 3.08567758e16) ** 2))
        sfr_dat_ff = (N0 * (sfr.data * (regrid_hdr['CDELT1'] * 3600 * 0.08) ** 2) * (10 ** (logmstar - 10)) ** gamma)
        sfr_noise_ff = (
                    N0 * (sfr_ctr_noise * (regrid_hdr['CDELT1'] * 3600 * 0.08) ** 2) * (10 ** (logmstar - 10)) ** gamma)
        if gal in [800, 3105]:
            sigma = 3
        elif gal in [4548]:
            sigma = 4
        else:
            sigma = 5
        if lcrelist[gal]:
            lcre = lcrelist[gal]
        else:
            lcre = np.abs(
                estimate_lcre(radio_ff, sfr_dat_ff, radio_noise_ff, sfr_noise_ff, pixsize, sigma=sigma,
                              roi_mask=roi_mask, name=radiomap.split('-')[0]))
        lcrelist[gal] = lcre
        # 1 kpc == 12.5
        print(lcre)
        print(f'lcre {lcre / 12.5}kpc')
        diffus = lcre / 3600

        # Do second convolution to take into account diffusion
        gauss_kern = EllipticalGaussian2DKernel((diffus / 1.177) / pixsize, (diffus / 1.177) / pixsize,
                                                (90 + bpa) * np.pi / 180.)  # bmaj and bmin are in pixels
        sfr.data = convolution.convolve(sfr.data, gauss_kern, boundary='extend', preserve_nan=True)
        sfr_ctr_conv2_noise = np.nanstd(sfr.data[sfr_bgreg.as_imagecoord(regrid_hdr).get_mask(sfr) & (sfr.data != 0)])
        sfr.data[sfr.data < sigma_map * sfr_ctr_conv2_noise] = np.nan
        sfr.writeto(sfrmap.replace('.fits', '-conv2.fits'), overwrite=True)

        ratio = ((radio_regrid.data / (beam_area / ((3600 * regrid_hdr['CDELT1']) ** 2))) * (
                1e-26 * 4 * np.pi * (16.5 * 1e6 * 3.08567758e16) ** 2)) / (
                        N0 * (sfr.data * (regrid_hdr['CDELT1'] * 3600 * 0.08) ** 2) * (10 ** (logmstar - 10)) ** gamma)
        ratio_map = fits.PrimaryHDU(ratio, regrid_hdr)
        ratio_map.writeto(radiomap.replace('.fits', '-regrid-ratio.fits'), overwrite=True)
        wcs = WCS(regrid_hdr)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=wcs, slices=('x', 'y'))

        xrange, yrange = setSize(ax, wcs, center[0], center[1], *np.array(size) / 60)
        data_visible = ratio[xrange[1]:xrange[0], yrange[0]:yrange[1]]
        vmin, vmax = -0.8, 0.8
        log_ratio = np.log10(ratio)
        log_ratio[log_ratio > vmax] = vmax
        log_ratio[log_ratio < vmin] = vmin
        norm = ImageNormalize(log_ratio, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        stars_mask[radmask] = 1
        stars_mask = np.array(stars_mask, dtype=float)
        ax.imshow(stars_mask, origin='lower', interpolation='nearest', cmap=grey_cmap, label='mask')
        ax.imshow(radmask, origin='lower', interpolation='nearest', cmap=grey_cmap)
        # im = ax.imshow(np.log10(ratio), origin='lower', interpolation='nearest', cmap='coolwarm', norm=norm)
        im = ax.contourf(log_ratio, levels=np.linspace(-0.8,0.8,17),  origin='lower', cmap='coolwarm', norm=norm, antialiased=False)

        # rad_dat_ctr[np.isnan(rad_dat_ctr)] = 0
        ctr_sig = 5
        ax.contour(rad_dat_ctr, [ctr_sig * radio_noise], colors=['#ee6afc'], linewidths=3)
        ax.contour(sfr_dat_ctr, [ctr_sig * sfr_ctr_noise], colors=['#63fa05'], linewidths=3) # #63fa05

        # coord_m87 = SkyCoord.from_name('M87')
        # coord = SkyCoord(ra=center[0] * u.degree, dec=center[1] * u.degree)
        # pix_coord_center = wcs.wcs_world2pix([[coord.ra.value, coord.dec.value]], 1)[0]
        # pix_coord_m87 = wcs.wcs_world2pix([[coord_m87.ra.value, coord_m87.dec.value]], 1)[0]
        # sep = coord_m87.separation(coord)
        # delta_pix = pix_coord_m87 - pix_coord_center
        # delta_pix /= np.linalg.norm(delta_pix)
        # # TODO scale
        # scale = np.min(np.shape(sfr.data))
        # arr_origin = pix_coord_center + 0.07 * scale * delta_pix
        # ax.arrow(*arr_origin, *(0.03 * delta_pix * scale), color='black', width=(scale / 250) * 2.5, zorder=2)
        # if delta_pix[1] < 0:
        #     va = 'top'
        # else:
        #     va = 'bottom'
        # if delta_pix[0] < 0:
        #     ha = 'left'
        # else:
        #     ha = 'right'
        fontsize = 22
        # ax.annotate(f'{sep.to_value("deg"):.2f}' + '$^\circ$', xy=arr_origin, color='black', ha=ha,
        #             fontsize=fontsize + 2,
        #             va=va)
        if ngc:
            addScalebar(ax, wcs, (70 * 16.5 / 3e5), 5, fontsize)
        else:
            addScalebar(ax, wcs, (70 * 16.5 / 3e5), 3, fontsize)
        # addCbar(fig, 'ratio', im, regrid_hdr, float(vmin), float(vmax), fontsize=fontsize)
        if False:
            cbaxes = fig.add_axes([0.127, 0.89, 0.772, 0.02])
            cbar = fig.colorbar(im, cax=cbaxes, norm=im.norm, orientation='horizontal',label=r'$\mathrm{log_{10}(Radio/SFR)}$')
            cbaxes.xaxis.set_label_text(r'$\mathrm{log_{10}(Radio/SFR)}$', fontsize=fontsize)
            cbaxes.xaxis.tick_top()
            cbaxes.xaxis.set_tick_params(labelsize=fontsize-4)
            cbaxes.xaxis.set_label_position('top')

            custom_lines = [Line2D([0], [0], color='#63fa05', lw=4),
                            Line2D([0], [0], color='#ee6afc', lw=4),
                            Circle([0, 0], 0.1, color='#a6a6a6')]
        titlename = f'NGC {gal}' if ngc else f'IC {gal}'
        ax.annotate(titlename, xy=[0.45, 0.1], fontsize=fontsize + 2, xycoords='figure fraction')
        # add beam
        #print(np.isclose(np.abs(regrid_hdr['CDELT1']), np.abs(regrid_hdr['CDELT2']), rtol=1e-3))
        addBeam(ax, regrid_hdr, edgecolor='black')
        addRegion('tails.reg', ax, regrid_hdr, color='k') # RoI to take into account for fitting
        # lon = ax.coords['ra']
        # lat = ax.coords['dec']
        # lon.set_axislabel('Right Ascension (J2000)', fontsize=fontsize)
        # lat.set_axislabel('Declination (J2000)', fontsize=fontsize)
        # lon.set_ticklabel(size=fontsize)
        # lat.set_ticklabel(size=fontsize)
        # lon.set_major_formatter('hh:mm:ss')
        # lat.set_major_formatter('dd:mm')
        # lat.set_ticklabel(rotation=90)  # to turn dec vertical
        ax.axis('off')
        # ax.legend(custom_lines, [f'${ctr_sig}\sigma$ SFR', f'${ctr_sig}\sigma$ radio', 'masked'], fontsize=fontsize)
        if hres:
            plt.savefig(radiomap.split('-')[0] + '-ratio-hres.png', bbox_inches='tight')
        else:
            plt.savefig(radiomap.split('-')[0] + '-ratio.png', bbox_inches='tight')
print(lcrelist)
for k, v in lcrelist.items():
    try:
        print(k, v/12.5)
    except TypeError:
        print(k, v)