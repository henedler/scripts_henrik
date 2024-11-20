#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import numpy as np
import lsmtool as lsm


def spectrum(source, nu):
    sh_coeffs = {'3c196': [83.084, -0.699, -0.110],
                '3c295': [97.763, -0.582, -0.298, 0.583, -0.363],
                '3c380': [77.352, -0.767]}

    coeffs = sh_coeffs[source]

    for i in range(len(coeffs)):
        if i == 0:
            flux_density = coeffs[i]
        else:
            flux_density *= 10**(coeffs[i]*np.log10(nu/150e6)**i)

    return flux_density

#
# if 'Vir' in os.getcwd():
#     patch = 'VirA'
#     nouseblrange = ''
#     #f = lambda nu: 1226. * 10**(-0.79 * (np.log10(nu/150.e6))**1)
#     f = lambda nu: 10**(2.4466 - 0.8116 * ((np.log10(nu/1.e9))**1) - 0.0483 * ((np.log10(nu/1.e9))**2) ) # PB17
# elif 'Tau' in os.getcwd():
#     patch = 'TauA'
#     nouseblrange = '' #'[500..5000]' # below is a point, above 10 times is hopefully resolved out
#     #f = lambda nu: 1838. * 10**(-0.299 * (np.log10(nu/150.e6))**1)
#     f = lambda nu: 10**(2.9516 - 0.2173 * ((np.log10(nu/1.e9))**1) - 0.0473 * ((np.log10(nu/1.e9))**2) - 0.0674 * ((np.log10(nu/1.e9))**3)) # PB17
# elif 'Cas' in os.getcwd():
#     patch = 'CasA'
#     nouseblrange = '' #'[15000..1e30]'
#     #f = lambda nu: 11733. * 10**(-0.77 * (np.log10(nu/150.e6))**1)
#     f = lambda nu: 10**(3.3584 - 0.7518 * ((np.log10(nu/1.e9))**1) - 0.0347 * ((np.log10(nu/1.e9))**2) - 0.0705 * ((np.log10(nu/1.e9))**3)) # PB17
# elif 'Cyg' in os.getcwd():
#     patch = 'CygA'
#     nouseblrange = ''
#     #f = lambda nu: 10690. * 10**(-0.67 * (np.log10(nu/150.e6))**1) * 10**(-0.204 * (np.log10(nu/150.e6))**2) * 10**(-0.021 * (np.log10(nu/150.e6))**3)
#     f = lambda nu: 10**(3.3498 - 1.0022 * ((np.log10(nu/1.e9))**1) - 0.2246 * ((np.log10(nu/1.e9))**2) + 0.0227 * ((np.log10(nu/1.e9))**3) + 0.0425 * ((np.log10(nu/1.e9))**4)) # PB17


print(f'{spectrum(sys.argv[1],float(sys.argv[2])):.2f}Jy at {float(sys.argv[2]):.2f}MHz.')