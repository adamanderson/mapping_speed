import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.integrate as sciint


def shot_noise(nu, power):
    return np.sqrt(2. * const.Planck * nu * power)

def correlation_noise(nu, power, delta_nu, correlation):
    return np.sqrt(2 * correlation * power**2. / delta_nu)

def tes_phonon_noise_P(Tbolo, G, gamma):
    return np.sqrt(4. * gamma * const.Boltzmann * G * Tbolo**2.)

def tes_johnson_noise_I(f, Tc, R0, L, tau=0):
    return np.sqrt(4. * const.Boltzmann * Tc * R0) * \
            np.sqrt(1 + (2*np.pi * f)**2. * tau**2.) / L

def tes_johnson_noise_P(f, Tc, Psat, L, Popt=0, tau=0):
    return np.sqrt(4. * const.Boltzmann * Tc * (Psat - Popt)) * \
            np.sqrt(1 + (2*np.pi * f)**2. * tau**2.) / L

def load_johnson_noise_I(T_L, R_L, R_bolo):
    return np.sqrt(4. * const.Boltzmann * R_L * T_L) / R_bolo

def dIdP(Vbias_rms, L=None):
    if L is not None:
        return np.sqrt(2.) / Vbias_rms * L / (1 + L)
    else:
        return np.sqrt(2.) / Vbias_rms

def dIdP_full(nu, T, Rfrac, k, Tc, Rn, R_L, L, alpha, beta, C, Popt=0):
    G = 3.*k*(Tc**2.)
    R_0 = Rn*Rfrac
    I_0 = np.sqrt(Psat(k, Tc, T, Popt) / R_0)
    P_J = I_0**2. * R_0
    loopgain = P_J * alpha / (G * Tc)
    G = 3.*k*(Tc**2.)
    tau = C / G
    tau_el = L / (R_L + R_0*(1 + beta))
    tau_I = tau / (1 - loopgain)

    S = (-np.sqrt(2.) / (I_0 * R_0)) * ( L / (tau_el * R_0 * loopgain) +
                                (1 - R_L / R_0) +
                                1j * 2.*np.pi*nu * (L*tau / (R_0*loopgain)) * (1./tau_I + 1./tau_el) -
                                (2.*np.pi*nu)**2. * tau * L / (loopgain * R_0))**-1
    return np.abs(S)

def Vbias_rms(Psat, Popt, Rbolo):
    PJ = Psat - Popt
    Vbias = np.sqrt(PJ * Rbolo)
    return Vbias

def Psat(k, Tc, T, Popt=0):
    return k*(Tc**3 - T**3) - Popt

def G(Tc, k=None, Psat=None, Popt=None, Tbath=None):
    # check arguments
    if k is None and (Psat is None or Popt is None or Tbath is None):
        raise ValueError('Either argument `k` must be set, or all of '
                         'arguments `Psat`, `Popt`, and `Tbath` must be set.')

    if k is not None:
        return 3.*k*(Tc**2.)
    else:
        return 3.*Tc**2. * (Psat + Popt) / (Tc**3 - Tbath**3)

def readout_noise_I(Ibase, Lsquid, fbias, Rbolo):
    return Ibase * np.abs(1 + 1j*2*np.pi*fbias*Lsquid / Rbolo)

def tau_natural(k, Tc, C):
    G = 3.*k*(Tc**2.)
    tau = C / G
    return tau

def planck_spectral_density(nu, temp):
    dPdnu = const.Planck * nu / (np.exp(const.Planck * nu / (const.Boltzmann * temp)) - 1) * 1e12
    return dPdnu

def planck_power(temp, nu_min, nu_max):
    power = sciint.quad(planck_spectral_density, a=nu_min, b=nu_max, args=(temp))
    return power
