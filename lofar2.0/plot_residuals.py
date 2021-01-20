#!/usr/bin/env python

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from losoto.h5parm import h5parm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot residuals for TEC-simulation')
    parser.add_argument('h5true', help='Input corruption h5parm', type=str)
    parser.add_argument('h5die', help='DIE cal solutions', type=str)
    parser.add_argument('h5dde', help='DDE cal solutions', type=str)
    args = parser.parse_args()

    # open h5parms
    h5true = h5parm(args.h5true)
    h5self = h5parm(args.h5die)
    h5dd = h5parm(args.h5dde)

    sttrue = h5true.getSolset('sol000').getSoltab('tec000')
    stself = h5self.getSolset('sol000').getSoltab('tec000')
    stdd = h5dd.getSolset('sol000').getSoltab('tec000')

    dirdict = {'Isl_patch_47': 'i57',
               'Isl_patch_93': 'i34',
               'Isl_patch_80': 'i37',
               'Isl_patch_72': 'i61',
               'Isl_patch_98': 'i79',
               'Isl_patch_45': 'i36',
               'Isl_patch_38': 'i42',
               'Isl_patch_19': 'i45'}

    # antcol = {'CS001': 'C0',
    #              'RS205': 'C1',
    #              'RS208': 'C2',
    #              'RS210': 'C4',
    #              'RS305': 'C5',
    #              'RS306': 'C6',
    #              'RS307': 'C8',
    #              'RS310': 'C9',
    #              'RS406': 'C10',
    #              'RS407': 'C11',
    #              'RS409': 'C12',
    #              'RS503': 'C14',
    #              'RS508': 'C15',
    #              'RS509': 'C16'}

    antcol = {'RS208': 'C0',
              'RS210': 'C1',
              'RS307': 'C2',
              'RS310': 'C4',
              'RS409': 'C5',
              'RS509': 'C6'}

    def truetec(time,ant,dir):
        """Input corruption"""
        ant += 'LBA'
        tab = h5true.getSolset('sol000').getSoltab('tec000', sel={'ant':ant,'dir':dirdict[dir]})
        f = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt='CS001LBA')[0].flatten(), kind='nearest', bounds_error=False, fill_value='extrapolate')
        return f(time)

    def selftec(time,ant):
        """Self-cal tec solutions"""
        ant += 'LBA'
        tab = h5self.getSolset('sol000').getSoltab('tec000', sel={'ant':ant})
        f = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt='CS001LBA')[0].flatten(), kind='nearest',bounds_error=False, fill_value='extrapolate')
        return f(time)

    def ddetec(time,ant,dir):
        """DDE tec solution"""
        refAnt = 'CS001'
        if 'HBA' in h5dd.getSolset('sol000').getSoltab('tec000').getAxisValues('ant')[0]:
            ant += 'HBA'
            refAnt = 'CS001HBA0'
            if ant == 'CS001HBA':
                ant += '0'
        elif 'LBA' in h5dd.getSolset('sol000').getSoltab('tec000').getAxisValues('ant')[0]:
            ant += 'LBA'
            refAnt = 'CS001LBA'
        tab = h5dd.getSolset('sol000').getSoltab('tec000', sel={'ant':ant,'dir':dir})
        f = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt=refAnt)[0].flatten(), bounds_error=False,kind='nearest', fill_value='extrapolate')
        return f(time)

    # time arrays
    t_true = sttrue.getAxisValues('time')
    t_dd = stdd.getAxisValues('time')


    def fill_ax_dir(ax, ant, dir):
        ax[0].scatter((t_true-t_true[0])/3600, truetec(t_true,ant,dir)-selftec(t_true,ant), s=.1, c='grey', label=ant, alpha=0.5)
        ax[0].scatter((t_true-t_true[0])/3600, ddetec(t_true,ant,dir), s=.1, c=antcol[ant])
        ax[1].axhline(0, color='black',zorder=-100, lw=1)
        ax[1].scatter((t_true-t_true[0])/3600, ddetec(t_true,ant,dir)-truetec(t_true,ant,dir)+selftec(t_true,ant), s=.1, c='C3')

        bbox_props = dict(boxstyle="round, pad=0.3", fc="white", ec="grey", lw=1)
        ax[1].annotate(ant, [2.6, 0.075], fontsize=10, bbox=bbox_props)

        ax[0].set_yticks([-0.1, -0.05, 0., 0.05, 0.1])
        ax[0].set_ylim(-0.15, 0.15)
        ax[0].set_yticklabels([])

        ax[1].set_yticks([-0.05, 0., 0.05])
        ax[1].set_ylim(-0.075, 0.075)
        ax[1].set_yticklabels([])

    for vals, weights, coord, selection in stdd.getValuesIter(returnAxes=['time','ant'], weight=True, reference="CS001"):
        if coord['dir'] not in ['Isl_patch_45', 'Isl_patch_72', 'Isl_patch_98']: continue
        fig, axs = plt.subplots(4, 3, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': (2,1,2,1)},
                                figsize=(6*1.25, 4*1.25))

        fill_ax_dir(axs[0:2,0], 'RS208', coord['dir'])
        fill_ax_dir(axs[0:2,1], 'RS210', coord['dir'])
        fill_ax_dir(axs[0:2,2], 'RS307', coord['dir'])
        fill_ax_dir(axs[2:,0],  'RS310', coord['dir'])
        fill_ax_dir(axs[2:,1],  'RS409', coord['dir'])
        fill_ax_dir(axs[2:,2],  'RS509', coord['dir'])

        # fill_ax_dir(axs[0:2,0], 'CS001', coord['dir'])
        # fill_ax_dir(axs[0:2,1], 'RS205', coord['dir'])
        # fill_ax_dir(axs[0:2,2], 'RS208', coord['dir'])
        # fill_ax_dir(axs[0:2,3], 'RS210', coord['dir'])
        # fill_ax_dir(axs[0:2,4], 'RS305', coord['dir'])
        # fill_ax_dir(axs[0:2,5], 'RS306', coord['dir'])
        # fill_ax_dir(axs[0:2,6], 'RS307', coord['dir'])
        # fill_ax_dir(axs[2:,0],  'RS310', coord['dir'])
        # fill_ax_dir(axs[2:,1],  'RS406', coord['dir'])
        # fill_ax_dir(axs[2:,2],  'RS407', coord['dir'])
        # fill_ax_dir(axs[2:,3],  'RS409', coord['dir'])
        # fill_ax_dir(axs[2:,4],  'RS503', coord['dir'])
        # fill_ax_dir(axs[2:,5],  'RS508', coord['dir'])
        # fill_ax_dir(axs[2:,6],  'RS509', coord['dir'])

        axs[3, 1].set_xlabel('time [h]', fontsize=10)
        axs[3, 2].xaxis.set_label_coords(1.5, -0.38)

        axs[0, 0].set_xlim(0, 8)
        axs[0, 0].set_xticks([0, 2, 4, 6])

        axs[0,0].set_ylabel('dTEC\n[TECU]', fontsize=10)
        axs[1,0].set_ylabel('residuals\n[TECU]', fontsize=10)
        axs[2,0].set_ylabel('dTEC\n[TECU]', fontsize=10)
        axs[3,0].set_ylabel('residuals\n[TECU]', fontsize=10)

        bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1)
        axs[0,0].annotate(coord['dir'].replace('Isl_patch_', 'direction '), [0,0.125], fontsize=12, weight='bold', bbox=bbox_props)

        axs[0,0].set_yticklabels([-0.1, -0.05, 0., 0.05, 0.1])
        axs[1,0].set_yticklabels([-0.05, 0., 0.05])
        axs[2,0].set_yticklabels([-0.1, -0.05, 0., 0.05, 0.1])
        axs[3,0].set_yticklabels([-0.05, 0., 0.05])
        [ax.grid() for ax in axs.flatten()]
        [ax.set_axisbelow(True) for ax in axs.flatten()]

        fig.tight_layout()
        fig.savefig(f"hba_tec_residuals_{coord['dir']}.png", dip=1000, bbox_to_anchor='tight')