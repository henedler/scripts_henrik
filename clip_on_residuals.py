#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Clip data on the residuals after subtracting the best model

import  argparse
import numpy as np
import casacore.tables as pt

########################################################
from LiLF import lib_ms, lib_util, lib_log

parser = argparse.ArgumentParser(description='Clip on RESIDUAL_DATA')
parser.add_argument('ms', nargs='*', help='input ms files')
parser.add_argument('-s', '--sigma', type=float, default=5., help='sigma value to clip at.')
parser.add_argument('--skip_subtract', action='store_true', help='Skip setting RESIDUAL=CORRECTED-MODEL.')
args = parser.parse_args()
logger_obj = lib_log.Logger('script-clip.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)

MSs = lib_ms.AllMSs(args.ms, s, check_flags=False )

if not args.skip_subtract:
    logger.info('Addcol RESIDUAL_DATA')
    MSs.addcol('RESIDUAL_DATA', 'CORRECTED_DATA')
    logger.info('Setting RESIDUAL_DATA = CORRECTED_DATA - MODEL_DATA')
    MSs.run('taql "UPDATE $pathMS SET RESIDUAL_DATA = CORRECTED_DATA-MODEL_DATA"', log=f'$nameMS_residual.log')

logger.info('Clipping on RESIDUAL_DATA')
for MS in MSs.getListObj():
    with pt.table(MS.pathMS, readonly=False) as t:
        residuals = t.getcol('RESIDUAL_DATA')
        flags = t.getcol('FLAG')
        ant1 = t.getcol('ANTENNA1')
        ant2 = t.getcol('ANTENNA2')
        sigma = np.nanstd(residuals[~flags])
        newflags = (np.abs(residuals) > args.sigma * sigma) | flags
        logger.info(f'({MS.nameMS}) Using sigma {sigma:2e}. Flagged data: before {np.sum(flags)/flags.size:.3%}; after {np.sum(newflags)/flags.size:.3%}')
        t.putcol('FLAG', newflags)