#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:18:47 2019

@author: p1uy068
"""
import numpy as np
#import matplotlib.pyplot as plt
import casacore.tables as pt



#tab = pt.table('/storage/p1uy068/lgc_data/sim/uncorrupted/uncorrupted.MS')
unc = '~/data/sim/uncorrupted_xx0/uncorrupted.MS'
tec = '~/data/sim/tec_xx0/tec.MS'


def unflagged_re_data(ms_name, datacolumn = 'CORRECTED_DATA', ant1 = 0,
                      ant2 = 1):
    tab = pt.table(ms_name)    
    BL = pt.taql('SELECT FROM $tab WHERE ANTENNA1 = $ant1 AND ANTENNA2 = $ant2')    
    flag = ~np.all(BL.getcol('FLAG'), axis = (1,2))
    data = BL.getcol(datacolumn)[flag]
    return data
    
def get_ms_frequency(ms_name):
    freq = pt.table(ms_name+'/SPECTRAL_WINDOW')[0]['CHAN_FREQ']
    return freq
    
    
def get_phase(data):
    phase_xx = np.arctan(-np.imag(data[:,:,0])/np.real(data[:,:,0]))
    return phase_xx

dat_un = unflagged_re_data(unc)
phase_un = get_phase(dat_un)
freq_un = get_ms_frequency(unc)

dat_tec = unflagged_re_data(tec)
phase_tec = get_phase(dat_tec)
freq_tec = get_ms_frequency(tec)
print(freq_tec,freq_un)
assert np.all(freq_tec == freq_un)

phase_dif = phase_un - phase_tec

print(phase_dif)#np.average(phase_dif, axis = 0))
#flag = ~np.all(tab.getcol('FLAG'), axis = (1,2))
#SB_phase = np.arctan(np.real(SB[:,0])/np.real(SB[:,3]))


