#!/usr/bin/env python
import sys
import astropy.cosmology

cosmo = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
print(sys.argv)
print(cosmo.arcsec_per_kpc_proper(float(sys.argv[1]))**-1)

# print(cosmo.arcsec_per_kpc_comoving(float(sys.argv[1]))**-1)