import numpy as np
import astropy.constants as const
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def get_n_e_quenched(x, t, B=10e-6, tau=10e1, save=False):
    # this is the -dE/dt term for sync + IC losses in SI units. b = (4/3) * sigma_t * c * gamma^2 * (Urad + Umag)
    b = (4 / 3) * (const.sigma_T * const.c * (((B * u.G) ** 2) / (2 * const.mu0) + 6e5 * u.eV * u.m ** -3) * (
            (const.m_e * const.c ** 2) ** -2)).to_value('GeV^-1 Myr^-1')
    q = 1  # injection constant

    dt = t[1] - t[0]
    dx = x[1] - x[0]

    warned = False

    N = np.zeros((len(x), len(t)))  # grid for results
    # fill t0 = 0 with initial condition
    N[:, 0] = (q / (1.2 * b)) * 10 ** (-3.2 * x)  # slope of 3.2 as steady-state solution for injection with 2.2
    if ((dt * 1.2 * b) * (10 ** x[-1])) > 1e-2:
        print(f'Warning: T/N={((dt * 1.2 * b) * (10 ** x[-1]))} decrease dt')

    for j in tqdm(np.arange(0, len(t) - 1)):  # for each time
        for i in range(len(x)):  # loop over all energy
            # test dt size:
            if i == 0:
                N[0, j + 1] = N[0, 0]
            else: # use backward scheme in x, forward scheme in t
                N[i, j + 1] = N[i, j] + (b * dt * 10 ** (-x[i])) / (dx * np.log(10)) * (
                        N[i, j] * 10 ** (2 * x[i]) - N[i - 1, j] * 10 ** (2 * x[i - 1])) + dt * q * 10 ** (
                                      -2.2 * x[i]) * np.exp(-t[j] / tau)

        TperN = ((b * dt * 10 ** (-x[i])) / (dx * np.log(10)) * (
                N[i, j] * 10 ** (2 * x[i]) - N[i - 1, j] * 10 ** (2 * x[i - 1]))) / N[:, j]
        if (np.max(np.abs(TperN)) > 1e-2) and not warned:
            print(f'Warning2: after {t[j]}  Myr - T/N={np.max(np.abs(TperN))} - decrease dt')
            warned = True
    ## plotting
    for i in range(len(t)):
        if i == 1:
            plt.plot(10 ** x, N[:, i], label=f'{t[i]:.1f} Myr')

        elif i % int(len(t) / 10) == 0:
            plt.plot(10 ** x, N[:, i], label=f'{t[i]:.1f} Myr')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy/GeV')
    plt.ylabel('N')
    plt.legend()
    plt.title(f'd$x={dx:.2e}$, d$t={dt:.2e}$ Myr, $tau={tau:.0f}$ Myr')
    if save:
        plt.savefig(f'e-spec_dx{dx:.1e}_dt{dt:.1e}_tau{tau:.0f}.png')
    return N


parser = argparse.ArgumentParser()
parser.add_argument('--tmax', type=float, default=10, help='max time in Myr')
parser.add_argument('--dt',   type=float,default=1e-4, help='time steps in Myr')
parser.add_argument('--tau', type=float, default=100, help='quenching scale in Myr')
parser.add_argument('--B',  type=float, default=1e-5, help='B field in Gauss')
parser.add_argument('--save', action='store_true', help='save plot?')


args = parser.parse_args()

x = np.linspace(np.log10(3e-1), np.log10(20), 50) # Energy/P grid in GeV
t = np.linspace(0, args.tmax, int(args.tmax/args.dt)) # time grid in Myr

N = get_n_e_quenched(x, t, B=args.B, tau=args.tau, save=args.save)
