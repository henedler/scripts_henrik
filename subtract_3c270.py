#!/usr/bin/env python

# usage: runMS.py data_dir COMMAND

import sys, glob
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-run.logger')
logger = lib_log.logger

def get_ddf_parms_from_header(img):
    """
    Parse the HISTORY header of a DDFacet image and return a dict containing the options used to create the image.
    Will replace '-' by '_'.
    Parameters
    ----------
    img: str, filename of image

    Returns
    -------
    params_dict: dict,
    """
    params_dict = dict()
    hdr = fits.open(img)[0].header['HISTORY']
    for line in hdr:
        if line.count('=') == 1 and line.count('-') == 1:
            _key, _value = line.replace('-','_').replace(' ', '').split('=')
            params_dict[_key] = _value
        else:
            continue
    return params_dict

s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
MSs = lib_ms.AllMSs( glob.glob(sys.argv[1] + '/*.MS'), s, check_flags=False )

# s.add('MaskDicoModel.py --MaskName=%s --InDicoModel=%s --OutDicoModel=%s' % (outmask, 'image_full_ampphase_di_m.NS.DicoModel','3C273.DicoModel'),
#       log='MaskDicoModel.log', commandType='DDFacet', processors='max')

# DDF predict+corrupt in MODEL_DATA of everything BUT the calibrator
indico = wideDD_image.root + '.DicoModel'
outdico = indico + '-' + target_reg_file.split('.')[0] # use prefix of target reg
inmask = sorted(glob.glob(wideDD_image.root + '*_mask-ddcal.fits'))[-1]
outmask = outdico + '.mask'
lib_img.blank_image_reg(inmask, target_reg_file, outfile=outmask, inverse=False, blankval=0.)
s.add('MaskDicoModel.py --MaskName=%s --InDicoModel=%s --OutDicoModel=%s' % (outmask, indico, outdico),
      log='MaskDicoModel.log', commandType='python', processors='max')
s.run(check=True)

# get DDF parameters used to create the image/model
ddf_parms = get_ddf_parms_from_header(wideDD_image.imagename)
# change for PREDICT
ddf_parms['Data_MS'] = MSs.getStrDDF()
ddf_parms['Data_ColName'] = 'CORRECTED_DATA'
ddf_parms['Output_Mode'] = 'Predict'
ddf_parms['Predict_InitDicoModel'] = outdico
ddf_parms['Output_Mode'] = 'Predict'
ddf_parms['Beam_Smooth'] = 1
ddf_parms['DDESolutions_DDSols'] = dde_h5parm + ':sol000/phase000+amplitude000'

logger.info('Predict corrupted rest-of-the-sky...')
lib_util.run_DDF(s, 'ddfacet-pre.log', **ddf_parms, Cache_Reset=1)
