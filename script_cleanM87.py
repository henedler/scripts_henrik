#!/usr/bin/env python3
import sys, os, glob, re
from shutil import copy2, copytree, move
import numpy as np
import casacore.tables as pt
import lsmtool

########################################################
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-cleanM87.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
parset_dir = '/home/p1uy068/opt/src/LiLF/parsets/LOFAR_virgo/'

MSs = lib_ms.AllMSs( glob.glob('mss-avg/TC*[0-9].MS'), s )

basemask = 'self/masks/basemask.fits'
basemaskC = 'self/masks/basemaskC.fits'
baseregion = 'self/masks/baseregion.reg'
baseregionC = 'self/masks/baseregionC.reg'
uvlambdamin = 30
# imagename = 'img/img-n31'
# ddf_parms = {
#     'Data_MS': MSs.getStrDDF(),
#     'Data_ColName': 'DATA',
#     'Data_Sort': 1,
#     'Weight_ColName': 'WEIGHT_SPECTRUM',
#     'Weight_Robust': -0.5,
#     'Image_Cell': 1.2,
#     'Image_NPix': 1200,
#     'Deconv_CycleFactor': 0.0,
#     'Deconv_PeakFactor': 0.00, # recommended cyril 730
#     'Deconv_RMSFactor': 0.0, # recommended cyril 730
#     'Deconv_MaxMinorIter': 1000000,
#     'Deconv_MaxMajorIter': 20,
#     'Deconv_FluxThreshold': 0.0,
#     'Deconv_Mode': 'SSD2',
#     'Deconv_PSFBox': 'full',
#     'CF_wmax': 50000, # maximum w coordinate
#     'Freq_NDegridBand': 0, # higher is better, limited by ram
#     'Freq_NBand': 5,
#     'Mask_Auto': 1,
#     'Mask_SigTh': 3.0,
#     'GAClean_MinSizeInit': 10,
#     'GAClean_MaxMinorIterInitHMP': 100000,
#     'GAClean_RMSFactorInitHMP': 0.3, # Clean to 0.3 sig
#     'GAClean_AllowNegativeInitHMP': 1, # try allow negative
#     'SSD2_PolyFreqOrder':3,
#     'Facets_NFacets': 1,
#     'Facets_PSFOversize': 2.0,
#     'Output_Name': imagename,
#     'Output_Mode': 'Clean',
#     'Output_Cubes': 'i',
#     'Output_Also': 'onNeds',
#     #'Predict_ColName': 'MODEL_DATA',
#     # 'Mask_External': basemask
# }
#
# lib_util.run_DDF(s, 'ddfacet.log', **ddf_parms)

imagename = 'img/img'
modelimg = '/beegfs/p1uy068/virgo/models/m87/m87'

wsclean_params = {
    'scale': '0.5arcsec',
    'data_column': 'MODEL_DATA',
    'size': 2400,
    'weight': 'briggs -1.5',
    'join_channels': '',
    # 'deconvolution_channels': 32,
    'fit_spectral_pol': 8,  # 3 worked fine, let's see if the central residual improves with 5
    'channels_out': len(MSs.getFreqs()) // 12,
    'minuv_l': uvlambdamin,
    'multiscale': '',
    'name': imagename,
    'no_update_model_required': '',
    # 'do_predict': True,
    'baseline_averaging': 10
}

# clean CORRECTED_DATA

if not os.path.exists(basemask) or not os.path.exists(basemaskC):
    logger.info('Create masks...')
    # dummy clean to get image -> mask
    lib_util.run_wsclean(s, 'wsclean.log', MSs.getStrWsclean(), niter=0, channel_range='0 1',
                         interval='0 10', name=imagename, scale=wsclean_params['scale'],
                         size=wsclean_params['size'], nmiter=0)
    # create basemask
    copy2(f'{parset_dir}/masks/VirAhbaEllipse.reg', f'{baseregion}')
    copy2(f'{imagename}-image.fits', f'{basemask}')
    lib_img.blank_image_reg(basemask, baseregion, inverse=True, blankval=0.)
    lib_img.blank_image_reg(basemask, baseregion, inverse=False, blankval=1.)

logger.info('Predict...')
s.add(f'wsclean -predict -fits-mask {basemaskC} -name {modelimg} -j {s.max_processors} -channels-out {wsclean_params["channels_out"]} {MSs.getStrWsclean()}',
      log='wscleanPRE-field.log', commandType='wsclean', processors='max')
s.run(check=True)

logger.info('Cleaning...')
lib_util.run_wsclean(s, f'wsclean.log', MSs.getStrWsclean(), niter=1500000,
                     fits_mask=basemask, multiscale_scales='0,20,30,45,66,99,150', nmiter=30, mgain=0.6, gain=0.08,
                     multiscale_gain=0.12, threshold=0.0001,
                     auto_threshold=1.2, auto_mask=3.0, **wsclean_params)
os.system(f'cat logs/wsclean.log | grep "background noise"')

logger.info("Done.")
