import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.integrate as sciint

def Pplanck(f, T):
    dPdnu = const.Planck * f / (np.exp(const.Planck * f / (const.Boltzmann * T)) - 1)
    return dPdnu

class TES:
    def __init__(self, n, k, Tc, Rn, Popt, fcenter, fBW, optical_eff, nei_readout):
        self.n = n
        self.k = k
        self.Tc = Tc
        self.Rn = Rn
        self.Popt = Popt
        self.fBW = fBW
        self.fcenter = fcenter
        self.eff = optical_eff
        self.nei = nei_readout
        self.dPdT_eval = self.dPdT()

    def Psat(self, Tb):
        return self.k * (self.Tc**self.n - Tb**self.n)

    def nep_phonon(self, Tb):
        gamma = (self.n+1.0) / (2.0*self.n + 3.0) * \
                (1.0 - (Tb / self.Tc)**(2.0*self.n+3.0)) / (1.0 - (Tb / self.Tc)**(self.n+1.0))
        G = self.k * self.Tc**self.n
        return np.sqrt(4. * gamma * const.Boltzmann * G * self.Tc**2.)

    def nep_photon(self, correlation):
        nep_uncorr = np.sqrt(2. * const.Planck * self.fcenter * self.Popt)
        nep_corr = np.sqrt(correlation * self.Popt**2. / self.fBW)
        return np.sqrt(nep_uncorr**2. + nep_corr**2.)

    def nep_readout(self, Tb, Rfrac):
        Vbias = np.sqrt((self.Psat(Tb) - self.Popt) * self.Rn * Rfrac)
        return self.nei * Vbias

    def dPdT(self):
        def integrand(f, T):
            return f**2 * np.exp(const.Planck * f / (const.Boltzmann * T)) / \
                   (np.exp(const.Planck * f / (const.Boltzmann * T)) - 1.)**2.
        T = self.Popt / (self.eff * const.Boltzmann * self.fBW)
        integral = sciint.quad(integrand, a=self.fcenter - self.fBW/2,
                               b=self.fcenter + self.fBW/2,
                               args=(T))
        dPdT = self.eff * const.Planck**2 / (const.Boltzmann * T**2.) * integral[0]
        return dPdT

    def nep_total(self, Tb, correlation, Rfrac):
        return np.sqrt(self.nep_phonon(Tb)**2. +
                       self.nep_readout(Tb, Rfrac)**2. +
                       self.nep_photon(correlation)**2)

    def mapping_speed(self, Tb, correlation, Rfrac):
        return 1. / ((self.nep_total(Tb, correlation, Rfrac) / self.dPdT_eval)**2.)
