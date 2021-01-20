#!/usr/bin/env python
# -*- coding: utf-8 -*-

# small script to transfer a list of existing direction objects of a dd-serial calibration to another observation
# E.g. LBA <-> HBA


import sys, os, glob, pickle
from shutil import copy2 as copy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import lsmtool
import lib_fits

from LiLF import lib_ms, lib_img, lib_dd, lib_util
w = lib_util.Walker('pipeline-dd-serial.walker')
with w.if_todo('cleaning'):
    pass


input_pth = '/home/p1uy068/node31/lofar2.0/HBA/tgts/ddcal'
output_pth = '/home/p1uy068/node31/lofar2.0/LBA/tgts'
removeExtendedCutoff = 0.0002

s = lib_util.Scheduler(dry = False)
MSs = lib_ms.AllMSs(glob.glob('mss-avg/TC*[0-9].MS'), s, check_flags=False)
detectability_dist = MSs.getListObj()[0].getFWHM(freq='max')*1.5/2.  # 1.8 to go to close to the null
phase_center = MSs.getListObj()[0].getPhaseCentre()

os.system('cd ' + output_pth)

print('make paths...')
if not os.path.exists('ddcal/'):
    os.makedirs('ddcal/')
if not os.path.exists('ddcal/c00/skymodels/'):
    os.makedirs('ddcal/c00/skymodels/')
if not os.path.exists('ddcal/init/'):
    os.makedirs('ddcal/init/')
for subdir in ['plots', 'images', 'solutions', 'skymodels']:
    if not os.path.exists(f'ddcal/c00/{subdir}'): os.makedirs(f'ddcal/c00/{subdir}')


# copy stuff
print('copy stuff...')
print(f'cp {input_pth}/directions-c00.pickle ddcal/')
os.system(f'cp {input_pth}/directions-c00.pickle ddcal/')
os.system(f'cp {input_pth}/c00/skymodels/Isl_patch*.reg ddcal/c00/skymodels/')
os.system('cp self/images/wideM-1* ddcal/init/')


full_image = lib_img.Image('ddcal/init/wideM-1-MFS-image.fits')
mask_cl = full_image.imagename.replace('.fits', '_mask-cl.fits')  # this is used to find calibrators
mask_ext = full_image.imagename.replace('.fits', '_mask-ext.fits')  # this is used for the initial subtract

print('make image masks... ', mask_cl,' ', mask_ext)
### make the masks
# if not os.path.exists(mask_cl):
#     full_image.makeMask(threshisl=7, atrous_do=False, remove_extended_cutoff=removeExtendedCutoff,
#                         only_beam=False, maskname=mask_cl, write_srl=True)
if not os.path.exists(mask_cl):
    full_image.makeMask(threshisl=4, atrous_do=False, remove_extended_cutoff=removeExtendedCutoff,
                        only_beam=False, maskname=mask_cl, write_srl=True)
if not os.path.exists(mask_ext):
    full_image.makeMask(threshisl=4, atrous_do=True, remove_extended_cutoff=0,
                        only_beam=False, maskname=mask_ext)

if not os.path.exists(full_image.skymodel_cut):
    full_image.selectCC(checkBeam=False, maskname=mask_cl)
# blank model
print('cleanup model images...')
for model_file in glob.glob(full_image.root + '*model.fits'):
    lib_img.blank_image_fits(model_file, mask_ext, model_file, inverse=True, blankval=0.)

# locating DD-calibrators
lsm = lsmtool.load(full_image.skymodel_cut)
lsm.select('%s == True' % mask_cl)
lsm.group(mask_cl, root='cl')
img_beam = full_image.getBeam()
# This regroup nearby sources
x = lsm.getColValues('RA', aggregate='wmean')
y = lsm.getColValues('Dec', aggregate='wmean')
flux = lsm.getColValues('I', aggregate='sum')
grouper = lib_dd.Grouper(list(zip(x,y)), flux, look_distance=0.1, kernel_size=0.07, grouping_distance=0.03)
grouper.run()
clusters = grouper.grouping()
grouper.plot()
os.system('mv grouping*png ddcal/c%02i/plots/' % 0)
patchNames = lsm.getPatchNames()

print('Merging nearby sources...')
for cluster in clusters:
    patches = patchNames[cluster]
    if len(patches) > 1:
        lsm.merge(patches.tolist())
lsm.setPatchPositions(method='mid')

# Iterate directions
with open('ddcal/directions-c00.pickle', "rb") as f:
    directions = pickle.load(f)
for no,d in enumerate(directions):
    print('Workig on ', d.name)
    assert d.name not in lsm.getPatchNames() # make sure the cluster name is not already present by chance
    model_root = 'ddcal/c00/skymodels/%s-init' % (d.name)
    for model_file in glob.glob(full_image.root + '*[0-9]-model.fits'):
        os.system('cp %s %s' % (model_file, model_file.replace(full_image.root, model_root)))
    d.set_model(model_root, typ='init', apply_region=True)
    # make fits mask
    print('make fits mask...')
    fitsmask = f'ddcal/c00/skymodels/{d.name}_mask.fits'
    model = d.get_model('init') + '-0000-model.fits'
    lib_img.blank_image_reg(model, d.get_region(), outfile=fitsmask, blankval=1, inverse=False)
    lib_img.blank_image_reg(fitsmask, d.get_region(), blankval=0, inverse=True)
    # apply region mask
    dlsm = lsm.copy()
    dlsm.select('%s == True' % fitsmask) # make new sky model containtin only sources in this cluister
    lsm.remove('%s == True' % fitsmask) # remove them from old skymodel
    dlsm.group('single', root=d.name)
    dlsm.setPatchPositions(method='mid')
    # add current cluster back
    lsm.concatenate(dlsm, inheritPatches=True)

    size = dlsm.getPatchSizes(units='deg')[0]
    fluxes = dlsm.getColValues('I')
    spidx_coeffs = dlsm.getColValues('SpectralIndex')
    ref_freq = dlsm.getColValues('ReferenceFrequency')
    gauss_area = (dlsm.getColValues('MajorAxis') * dlsm.getColValues('MinorAxis') / (
            img_beam[0] * img_beam[1]))  # in beams
    for i in range(len(dlsm)):
        if gauss_area[i] > 1:
            fluxes[i] /= gauss_area[i]  # reduce the fluxes for gaussians to the peak value

    d.set_flux(fluxes, spidx_coeffs, ref_freq)
    if size < 4 * img_beam[0] / 3600:
        size = 4 * img_beam[0] / 3600
    # for complex sources force a larger region
    if len(dlsm) > 1 and size < 10 * img_beam[0] / 3600:
        size = 10 * img_beam[0] / 3600
    print('DEBUG:',d.name,np.sum(fluxes),ref_freq.mean(),size,img_beam)
    ra, dec = np.array(dlsm.getPatchPositions(asArray=True)).flatten()
    d.set_position([ra, dec], distance_peeloff=detectability_dist, phase_center=phase_center)
    d.set_size(size * 1.2)  # size increased by 20%

    d.clean()
    d.converged = None
    directions[no] = d

# create a concat region for debugging
os.system('cat ddcal/c00/skymodels/Isl*reg > ddcal/c00/skymodels/all-c00.reg')

skymodel_cl = 'ddcal/c00/skymodels/cluster.skymodel'
lsm.setPatchPositions(method='mid')
lsm.write(skymodel_cl, format='makesourcedb', clobber=True)
lsm.setColValues('name', [x.split('_')[-1] for x in
                          lsm.getColValues('patch')])  # just for the region - this makes this lsm useless
lsm.write('ddcal/c00/skymodels/cluster-c00.reg', format='ds9', clobber=True)

with open('ddcal/directions-c00.pickle', "wb") as f:
    pickle.dump(directions, f)
