#!/usr/bin/env python
# Subtract tool: subtract hres model of sources within input region
#                and image subtracted data at lowres
#                This tool requires LiLF to be installed.

import argparse, os, sys
import numpy as np
from LiLF import lib_util, lib_img, lib_ms, lib_log

logger_obj = lib_log.Logger('script-subtract.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
w = lib_util.Walker('script-subtract.walker')

parser = argparse.ArgumentParser(description = "Create point-source subtracted image.")
parser.add_argument("msfiles", nargs="*", help="Input ms files")
parser.add_argument("region",  help='subtract region', type=str)
parser.add_argument("-n","--name",  help='imagename prefix', default='subtract', type=str)
parser.add_argument("-s","--size",  help='size in deg, e.g. 1.1,2.1', default='0.17,0.17', type=str)
parser.add_argument("-k","--kernelsize", help='kernel size in arcsec', default=20, type=float)
args = parser.parse_args()

### run stuff
imagepre, region, size, kernelsize = args.name, args.region, args.size, args.kernelsize
size = np.array(size.split(','), dtype=float)
MSs = lib_ms.AllMSs(args.msfiles, s, check_flags=False)
ch_out = MSs.getChout(4e6)  # chout from dd-serial
pixscale = float('%.1f' % (MSs.resolution / 3.5))
imsize = [int(size[0] * 1.5 / (pixscale / 3600.)), int(size[1] * 1.5 / (pixscale / 3600.))]  # add 50%
imsize[0] += imsize[0] % 2
imsize[1] += imsize[1] % 2
if imsize[0] < 256: imsize[0] = 256
if imsize[1] < 256: imsize[1] = 256

logger.debug('Image size: ' + str(imsize) + ' - Pixel scale: ' + str(pixscale))

#### Make hres image
imagename = imagepre + '-hres'
maskname = imagepre + '-mask.fits'
with w.if_todo('make_fits_mask'):
    # make dirty image to get fits file
    lib_util.run_wsclean(s, f'wsclean-{imagename}-mask.log',
                         MSs.getStrWsclean(),
                         name=imagename,
                         size=imsize,
                         scale=str(pixscale) + 'arcsec',
                         no_update_model_required=''
                         )
    os.system(f'mv {imagename}-image.fits {maskname}')
    # blank mask
    im = lib_img.Image(maskname)
    lib_img.blank_image_reg(maskname, region, inverse=True, blankval=0., )
    lib_img.blank_image_reg(maskname, region, inverse=False, blankval=1., )

with w.if_todo('image_hres'):
    # clean CORRECTED_DATA --> fill MODEL_DATA
    logger.info('Cleaning ' + str(imagename) + '...')
    lib_util.run_wsclean(s, f'wsclean-{imagename}-hres.log',
                         MSs.getStrWsclean(),
                         name=imagename,
                         do_predict=True,
                         # data_column='CORRECTED_DATA',
                         size=imsize,
                         temp_dir='.',
                         scale=str(pixscale) + 'arcsec',
                         weight='briggs -1.0',
                         niter=1000000,
                         no_update_model_required='',
                         minuv_l=100,
                         mgain=0.85,
                         # multiscale='',  # do not use multiscale to get no emission outside of regon
                         # multiscale_scales='0,7,14', #20
                         baseline_averaging='',
                         auto_threshold=0.3,
                         auto_mask=1.5,
                         parallel_deconvolution=512,
                         fits_mask=maskname,
                         join_channels='',
                         channels_out=ch_out,
                         fit_spectral_pol=3
                         )
    os.system(f'cat logs/wsclean-{imagename}-hres.log | grep "background noise"')


with w.if_todo('subtract'):
    MSs.run('addcol2ms.py -m $pathMS -c SUBTRACTED_DATA -i DATA', log='$nameMS_addcol.log', commandType='python')
    MSs.run('addcol2ms.py -m $pathMS -c CORRECTED_DATA -i DATA', log='$nameMS_addcol.log', commandType='python') # if DATA not present
    logger.info('Set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA...')
    MSs.run('taql "update $pathMS set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA"', log='$nameMS_subtract.log',
            commandType='general')


imagename = imagepre + '-lowres-subtracted'
with w.if_todo('image_subtracted_lowres'):
    # revert mask
    lib_img.blank_image_reg(maskname, region, inverse=False, blankval=0.)
    lib_img.blank_image_reg(maskname, region, inverse=True, blankval=1.)

    # clean CORRECTED_DATA --> fill MODEL_DATA
    logger.info('Cleaning ' + str(imagename) + '...')
    lib_util.run_wsclean(s, f'wsclean-{imagename}.log',
                         MSs.getStrWsclean(),
                         taper_gaussian=kernelsize,
                         name=imagename,
                         data_column='SUBTRACTED_DATA',
                         size=imsize,
                         temp_dir='.',
                         scale=str(pixscale) + 'arcsec',
                         weight='briggs -0.6',
                         niter=1000000,
                         no_update_model_required='',
                         maxuv_l=20000,
                         minuv_l=100,
                         mgain=0.80,
                         multiscale='',
                         multiscale_gain=0.15,
                         baseline_averaging='',
                         auto_threshold=0.7,
                         auto_mask=1.5,
                         # fits_mask=maskname, # maybe comment
                         join_channels='',
                         channels_out=ch_out,
                         fit_spectral_pol=3
                         )
    os.system(f'cat logs/wsclean-{imagename}.log | grep "background noise"')
