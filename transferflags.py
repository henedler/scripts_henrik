#!/usr/bin/env python

# usage: transferflags.py data_dir 51,52,53

import sys, glob
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-transferflags.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
MSs = lib_ms.AllMSs( glob.glob(sys.argv[1] + '/*.MS'), s, check_flags=False )
MSs.run(f"taql 'UPDATE $pathMS SET FLAG=T,FLAG_ROW=T WHERE TIME IN [SELECT DISTINCT TIME FROM $pathMS where FLAG_ROW==T AND ANTENNA1 IN [{sys.argv[2]}]]'",
        log='$nameMS_taql.log', commandType='general')