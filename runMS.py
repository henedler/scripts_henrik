#!/usr/bin/env python

# usage: runMS.py data_dir COMMAND

import sys, glob
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-run.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
MSs = lib_ms.AllMSs(glob.glob(sys.argv[1] + '/*.MS') , s, check_flags=False )
if len(MSs.getListStr()) == 0:
        MSs = lib_ms.AllMSs(glob.glob(sys.argv[1] + '/*.MS'), s, check_flags=False)

commandType = 'DP3' if 'DP3' in sys.argv[2] else 'general'
MSs.run(sys.argv[2],
        log='$nameMS_run.log', commandType=commandType)