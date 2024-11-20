#!/usr/bin/env python

# usage: runMS.py data_dir COMMAND
import os
import sys, glob

import numpy as np
from LiLF import lib_ms, lib_img, lib_util, lib_log
logger_obj = lib_log.Logger('script-renameTC.logger')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
print(glob.glob(sys.argv[1] + '/*.MS'))


globglob = glob.glob(sys.argv[1])
logger.info(f'Query returned the following files: {globglob}')

MSs = lib_ms.AllMSs(globglob, s, check_flags=False)
times = np.ones(len(MSs.getListObj()))
for i, MS in enumerate(MSs.getListObj()):
        start_time = MS.getTimeRange()[0]
        times[i] = start_time

print(np.argsort(times), times[np.argsort(times)])

os.system('mkdir mss-ordered')

for i in np.argsort(times):
    MS = MSs.getListObj()[i]
    MS.move(f'mss-ordered/TC{i:02}.MS', overwrite=False, keepOrig=True)

logger.info('done.')
# if len(MSs.getListStr()) == 0:
#         MSs = lib_ms.AllMSs(glob.glob(sys.argv[1] + '/*.MS'), s, check_flags=False)
#
# commandType = 'DP3' if 'DP3' in sys.argv[2] else 'general'
# MSs.run(sys.argv[2],
#         log='$nameMS_run.log', commandType=commandType)