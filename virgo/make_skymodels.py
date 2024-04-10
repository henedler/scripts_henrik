#!/usr/bin/env python3
# There are two places where we want a skymodel of the field:
# 1. For demix:
# The skymodel should be intrinsic, contain the bright sources within the FWHM. It should contain M87 if M87 is not
# demixed (for fields closer than 4 deg?).
# 2. For self:
# The field should be intrinsic, contain the bright sources within the FWHM.


import argparse
import lsmtool
from LiLF import lib_ms, lib_util

parser = argparse.ArgumentParser(description='Get initial skymodels for LOFAR Virgo field')
parser.add_argument('MS', help='MS file.')
#parser.add_argument('mode', help='Demix .')
# parser.add_argument('--apparent', action='store_true', help='Attenuate for average beam.')
# parser.add_argument('--m87', action='store_true', help='Add M87 to the skymodel.')
#parser.add_argument('-s', '--size', type=float, default=8., help='size in arcmin')
args = parser.parse_args()
prefix = '/beegfs/p1uy068/virgo/models/LBA'

MS = lib_ms.MS(args.MS)
fwhm = 6.9 # MS.getFWHM(freq='min')
ra, dec = MS.getPhaseCentre()


field_hba = lsmtool.load(f'{prefix}/LVCS_20as_gaul_filtered_freqscaled.skymodel', beamMS=args.MS)
dists = field_hba.getDistance(ra,dec)
field_hba.select(dists < fwhm/2) # remove distant sources
field_hba.select('I>1.0', applyBeam=True) # keep only reasonably bright sources

print(field_hba.info())
m87_dist = lib_util.distanceOnSphere(ra, dec, 187.705930, 12.391123)
if m87_dist > 4:
    print(f'm87_dist-{m87_dist}deg. Assume M87 demixed and not put it in model.')
else:
    m87 = lsmtool.load(f'{prefix}/VirA.skymodel', beamMS=args.MS)
    field_hba.concatenate(m87)

print(field_hba.info())
field_hba.group('single', root='target')

field_hba.write('tgts.skymodel', clobber=True, applyBeam=False)


# m87 = lsmtool.load(f'{prefix}/VirA.skymodel')


