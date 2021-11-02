#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from losoto.h5parm import h5parm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot residuals for TEC-simulation')
    parser.add_argument('h5true', help='Input corruption h5parm', type=str)
    parser.add_argument('h5die1', help='DIE cal solutions', type=str)
    parser.add_argument('h5die2', help='DIE cal solutions', type=str)
    parser.add_argument('h5dde', help='DDE cal solutions', type=str)
    parser.add_argument('-o', '--output', default='residual', type=str)
    args = parser.parse_args()

    # open h5parms
    h5true = h5parm(args.h5true)
    h5self1 = h5parm(args.h5die1)
    h5self2 = h5parm(args.h5die2)
    h5dd = h5parm(args.h5dde)

    sttrue = h5true.getSolset('sol000').getSoltab('tec000')
    stself1 = h5self1.getSolset('sol000').getSoltab('tec000')
    stself2 = h5self2.getSolset('sol000').getSoltab('tec000')
    stdd = h5dd.getSolset('sol000').getSoltab('tec000')

    dirdict = {'Isl_patch_64': 'i57',
               'Isl_patch_110': 'i34',
               'Isl_patch_100': 'i37',
               'Isl_patch_36': 'i45',
               'Isl_patch_61': 'i36',
               'Isl_patch_91': 'i44'}

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
        refAnt = 'CS001LBA'
        ant += 'LBA'
        print(dir)
        tab = h5true.getSolset('sol000').getSoltab('tec000', sel={'ant':ant,'dir':dirdict[dir]})
        f = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt=refAnt)[0].flatten(), kind='nearest', bounds_error=False, fill_value='extrapolate')
        return f(time)

    def selftec(time,ant):
        """Self-cal tec solutions"""
        if ant[0:5] in ['RS208', 'RS210', 'RS307', 'RS310', 'RS406', 'RS407', 'RS409', 'RS508', 'RS509']:
            h5self = h5self2
        else:
            h5self = h5self1
        if 'LBA' in list(h5self.getSolset('sol000').getAnt().keys())[0]:
            refAnt = 'CS001LBA'
            ant += 'LBA'
        else:
            refAnt = 'CS001HBA0'
            ant += 'HBA'
            if ant == 'CS001HBA':
                ant += '0'
        tab = h5self.getSolset('sol000').getSoltab('tec000', sel={'ant':ant})
        f = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt=refAnt)[0].flatten(), kind='nearest',bounds_error=False, fill_value='extrapolate')
        w = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt=refAnt, weight=True)[0].flatten(), kind='nearest',bounds_error=False, fill_value='extrapolate')
        return f(time), w(time)

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
        w = interp1d(tab.getAxisValues('time'),tab.getValues(refAnt=refAnt, weight=True)[0].flatten(), bounds_error=False,kind='nearest', fill_value='extrapolate')
        return f(time), w(time)

    # time arrays
    t_true = sttrue.getAxisValues('time')
    t_dd = stdd.getAxisValues('time')


    def fill_ax_dir(ax, ant, dir):
        ax[0].scatter((t_true-t_true[0])/3600, truetec(t_true,ant,dir)-selftec(t_true,ant)[0], s=.1, c='grey', label=ant, alpha=0.5)
        ax[0].scatter((t_true-t_true[0])/3600, ddetec(t_true,ant,dir)[0], s=.1, c=antcol[ant])
        ax[1].axhline(0, color='black',zorder=-100, lw=1)
        residual = ddetec(t_true, ant, dir)[0] - truetec(t_true, ant, dir) + selftec(t_true, ant)[0]
        above = residual > 0.05
        below = residual < -0.05
        ax[1].scatter((t_true-t_true[0])/3600, residual, s=.1, c='C3')
        ax[1].scatter((t_true-t_true[0])[below]/3600, [-0.0475]*sum(below), marker='v', s=.1, c='C3')
        ax[1].scatter((t_true-t_true[0])[above]/3600, [0.0475]*sum(above), marker='^', s=.1, c='C3')

        bbox_props = dict(boxstyle="round, pad=0.3", fc="white", ec="grey", lw=1)
        ax[1].annotate(ant, [2.6, 0.05], fontsize=10, bbox=bbox_props)

        ax[0].set_yticks([-0.05, 0., 0.05])
        ax[0].set_ylim(-0.1, 0.1)
        ax[0].set_yticklabels([])

        ax[1].set_yticks([-0.025, 0., 0.025])
        ax[1].set_ylim(-0.05, 0.05)
        ax[1].set_yticklabels([])

    def get_resid_ant_dir(ant, dir):
        diff = ddetec(t_true,ant,dir)[0] - (truetec(t_true, ant, dir) - selftec(t_true, ant)[0])
        flag_mask = np.logical_and(ddetec(t_true,ant,dir)[1], selftec(t_true, ant)[1])
        if ant == 'RS208' and dir == 'Isl_patch_100':
            for i in range(0, len(diff), 1000):
                print(np.std(diff[i:i+1000]), np.std(diff))
        return np.std(diff[flag_mask])


    # freq = h5parm('solutions/interp_lba.h5').getSolset('sol000').getSoltab('phase000').getAxisValues('freq')
    def mod(d):
        """ wrap phases to (-pi,pi)"""
        return np.mod(d + np.pi, 2. * np.pi) - np.pi
    def get_ph_resid_ant_dir(ant, dir):
        diff = ddetec(t_true,ant,dir)[0] - (truetec(t_true, ant, dir) - selftec(t_true, ant)[0])
        flag_mask = np.logical_and(ddetec(t_true,ant,dir)[1], selftec(t_true, ant)[1])
        print('diff',diff)
        print('freq',freq)
        diff_ph = -8.44797245e9*np.outer(diff, 1/freq)
        print('diff_ph',diff_ph)
        diff_ph = mod(diff_ph)
        # print('diff_ph',diff_ph)
        return np.std(diff_ph[flag_mask])

    # df = pd.DataFrame(columns = list(antcol)[1::])
    # for vals, weights, coord, selection in stdd.getValuesIter(returnAxes=['time', 'ant'], weight=True,
    #                                                           reference="CS001"):
    #     this_dir = []
    #     for ant in antcol.keys():
    #         if ant == 'CS001': continue
    #         this_dir.append(get_ph_resid_ant_dir(ant, coord['dir']))
    #     df = df.append(pd.Series(this_dir, index=df.columns, name=coord['dir']))
    # df.to_csv(args.output+'.csv')



    for vals, weights, coord, selection in stdd.getValuesIter(returnAxes=['time','ant'], weight=True, reference="CS001"):
        # fig, axs = plt.subplots(4, 7, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': (2,1,2,1,)},
        #                         figsize=(6*1.25, 4*1.25))
        fig, axs = plt.subplots(4, 3, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': (2,1,2,1)},
                                figsize=(6*1.25, 4*1.25))

        fill_ax_dir(axs[0:2,0], 'RS208', coord['dir'])
        fill_ax_dir(axs[0:2,1], 'RS210', coord['dir'])
        fill_ax_dir(axs[0:2,2], 'RS307', coord['dir'])
        fill_ax_dir(axs[2:,0],  'RS310', coord['dir'])
        fill_ax_dir(axs[2:,1],  'RS409', coord['dir'])
        fill_ax_dir(axs[2:,2],  'RS509', coord['dir'])
        #
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
        axs[0,0].annotate(coord['dir'].replace('Isl_patch_', 'direction '), [0.23,0.074], fontsize=12, weight='bold', bbox=bbox_props)

        axs[0,0].set_yticklabels([ -0.05, 0., 0.05])
        axs[1,0].set_yticklabels([-0.025, 0., 0.025])
        axs[2,0].set_yticklabels([-0.05, 0., 0.05])
        axs[3,0].set_yticklabels([-0.025, 0., 0.025])
        #[ax.grid() for ax in axs.flatten()]
        [ax.set_axisbelow(True) for ax in axs.flatten()]

        fig.tight_layout()
        fig.savefig(args.output+f"{coord['dir']}.png", dpi=300, bbox_to_anchor='tight')