#!/usr/bin/env python

# usage: deletecol.py data_dir col_name

import sys, glob
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-deletecol.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
MSs = lib_ms.AllMSs( glob.glob(sys.argv[1] + '/*.MS'), s, check_flags=False )
MSs.run(f'taql "ALTER TABLE $pathMS DELETE COLUMN {sys.argv[2]}"', log='$nameMS_taql_delcol.log', commandType='general')