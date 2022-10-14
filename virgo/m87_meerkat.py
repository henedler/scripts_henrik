#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from  casatasks import gaincal, applycal

ms = 'data-avg-flag.ms'
ms = 'casa.ms'
ms = 'dp3.ms'
ms = 'small.ms'
ms = '1561723022.ms'
ms = '1561822555_sdp_l0-3C274-corr_avg.ms'
ms = '1561723022_sdp_l0-3C274-corr_avg.ms'

def run_wsc(name, ms, col): #-fits-mask
    cmd = f'''wsclean -j 64 -data-column {col} -minuv-l 100 -padding 1.3  -reorder -parallel-reordering 4 -parallel-gridding 4 -clean-border 1 -name  {name} -size 1000 1000 -scale 1.5arcsec -weight briggs 0.0 -no-mfs-weighting -taper-gaussian 6 -niter 1000000 -update-model-required -mgain 0.95 -multiscale -multiscale-scales 0,10,20,40,80 -auto-threshold 0.5  -auto-mask 2.5 -fits-mask manual2_3C274.fits -join-channels -channels-out 10 {ms}'''
    os.system(cmd)

use_casa = True
do_amps = False

# flagmanager(vis="/stimela_mount/msdir/1561822555_sdp_l0-3C274-corr_avg.ms",mode="restore",versionname="M87a_s
# elfcal_before",oldname="",comment="",
# # 2022-09-27 13:30:21   INFO    flagmanager::::+                merge="replace")
for i in range(4):
    print(f'clean-{i:02}')
    if i==0:
        os.system(f"wsclean -predict -name Lband -size 4000 4000 -scale 0.5asec {ms}")
        # run_wsc(f'img/M87_{i:02}',ms, 'DATA')
    else:
        run_wsc(f'img/M87_{i:02}', ms, 'CORRECTED_DATA')

    if use_casa:
        # refant='m000'
        if i == 0: # phase-only
            gaincal(vis=ms, caltable=f'diag-ph-{i:02}.tb', solint='int', uvrange='> 100lambda', calmode='p',
                    gaintype='G', minsnr=0)
            applycal(vis=ms, gaintable=[f'diag-ph-{i:02}.tb'])
        else:
            gaincal(vis=ms, caltable=f'diag-ph-{i:02}.tb', solint='int',  uvrange='> 100lambda', calmode='p', gaintype='G', minsnr=0)
            gaincal(vis=ms, caltable=f'diag-amp-{i:02}.tb', solint='256s',  uvrange='> 100lambda', calmode='a', gaintype='G',
                    minsnr=0, gaintable=f'diag-ph-{i:02}.tb')
            applycal(vis=ms, gaintable=[f'diag-ph-{i:02}.tb',f'diag-amp-{i:02}.tb'])
    else:
        print(f'solve-ph-{i:02}')
        os.system(f'DP3 DP3-sol.parset msin={ms} sol.h5parm=diag-{i:02}.h5 sol.mode=diagonalphase sol.smoothnessconstraint=5e6')
        os.system(f'losoto diag-{i:02}.h5 losoto-plot-ph.parset')
        print(f'apply-ph-{i:02}')
        os.system(f'DP3 DP3-cor.parset msin={ms} cor.parmdb=diag-{i:02}.h5')
        if do_amps:
            print(f'solve-amp-{i:02}') # solint 16 for full ms
            os.system(f'DP3 DP3-sol.parset msin={ms} msin.datacolumn=CORRECTED_DATA sol.solint=32 sol.smoothnessconstraint=10e6 sol.h5parm=diag-amp-{i:02}.h5')
            if False:
                os.system(f'losoto diag-amp-{i:02}.h5 losoto-flag.parset')
            else:
                os.system(f'losoto diag-amp-{i:02}.h5 losoto-plot-amp.parset')
            print(f'apply-amp-{i:02}')
            os.system(f'DP3 DP3-cor.parset msin={ms} msin.datacolumn=CORRECTED_DATA cor.parmdb=diag-amp-{i:02}.h5 cor.correction=amplitude000')
            # os.system(f'DP3 DP3-cor2.parset msin={ms} cor1.parmdb=diag-{i:02}.h5 cor2.parmdb=diag-{i:02}.h5')
        os.system(f'mv plots plots-{i:02}')
