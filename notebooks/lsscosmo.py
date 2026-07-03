"""
lsscosmo.py
-----------
Small, self-contained cosmology helper library for the LSS problem sets.
Pure numpy/scipy. No special characters in code; ASCII only.

Contents
  - Cosmology: distances and expansion for flat w0-wa-CDM
  - growth_factor / growth_rate: linear growth D(z), f(z)
  - transfer_bbks: BBKS transfer function with Sugiyama shape parameter
  - power_spectrum_nowiggle / power_spectrum_bao: linear P(k), toy BAO template
  - sigma_R, normalize_sigma8: amplitude normalization
  - gaussian_pk_error: fractional error on a band power
  - lognormal_box, measure_pk: minimal lognormal mock and P(k) estimator

Conventions
  - Distances returned in Mpc unless a name ends in _hmpc (then h^-1 Mpc).
  - Wavenumbers k are in h/Mpc throughout the P(k) functions.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad, cumulative_trapezoid

C_KM_S = 299792.458  # speed of light in km/s


class Cosmology(object):
    """Flat w0-wa-CDM background (CPL dark energy)."""

    def __init__(self, Om=0.3153, Ob=0.0493, h=0.6736, ns=0.9649,
                 sigma8=0.8111, w0=-1.0, wa=0.0, Tcmb=2.7255):
        self.Om = float(Om)
        self.Ob = float(Ob)
        self.h = float(h)
        self.ns = float(ns)
        self.sigma8 = float(sigma8)
        self.w0 = float(w0)
        self.wa = float(wa)
        self.Tcmb = float(Tcmb)
        self.Ode = 1.0 - self.Om
        self.H0 = 100.0 * self.h  # km/s/Mpc

    # ----- expansion -----
    def de_density(self, z):
        a = 1.0 / (1.0 + z)
        return a ** (-3.0 * (1.0 + self.w0 + self.wa)) * np.exp(-3.0 * self.wa * (1.0 - a))

    def E(self, z):
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.Om * (1.0 + z) ** 3 + self.Ode * self.de_density(z))

    def H(self, z):
        return self.H0 * self.E(z)

    # ----- distances (Mpc) -----
    def comoving_distance(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.zeros_like(z)
        for i, zi in enumerate(z):
            val, _ = quad(lambda x: 1.0 / self.E(x), 0.0, zi)
            out[i] = (C_KM_S / self.H0) * val
        return out if out.size > 1 else out[0]

    def DM(self, z):
        return self.comoving_distance(z)  # flat universe: transverse = comoving

    def DH(self, z):
        return C_KM_S / self.H(z)

    def DV(self, z):
        z = np.asarray(z, dtype=float)
        dm = self.DM(z)
        dh = self.DH(z)
        return (z * dm * dm * dh) ** (1.0 / 3.0)

    # ----- linear growth -----
    def _growth_ode(self, lna, y):
        # y = [D, dD/dlna]; solve D'' + (2 + dlnH/dlna) D' - 1.5 Om(a) D = 0
        a = np.exp(lna)
        z = 1.0 / a - 1.0
        E = self.E(z)
        Om_a = self.Om * (1.0 + z) ** 3 / (E * E)
        # dlnE/dlna via finite difference
        dl = 1e-4
        Ep = self.E(1.0 / np.exp(lna + dl) - 1.0)
        Em = self.E(1.0 / np.exp(lna - dl) - 1.0)
        dlnE = (np.log(Ep) - np.log(Em)) / (2.0 * dl)
        D, Dp = y
        Dpp = -(2.0 + dlnE) * Dp + 1.5 * Om_a * D
        return [Dp, Dpp]

    def growth_unnormalized(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a_init = 1e-3
        lna0, lna1 = np.log(a_init), 0.0
        sol = solve_ivp(self._growth_ode, [lna0, lna1], [a_init, a_init],
                        dense_output=True, rtol=1e-8, atol=1e-10)
        a_eval = 1.0 / (1.0 + z)
        D = sol.sol(np.log(a_eval))[0]
        D0 = sol.sol(0.0)[0]
        return D, D0

    def growth_factor(self, z):
        D, D0 = self.growth_unnormalized(z)
        out = D / D0
        return out if out.size > 1 else out[0]

    def growth_rate(self, z):
        # f = dlnD/dlna, by finite difference in a
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + z)
        dl = 1e-3
        lna = np.log(a)
        Dp, _ = self.growth_unnormalized(1.0 / np.exp(lna + dl) - 1.0)
        Dm, _ = self.growth_unnormalized(1.0 / np.exp(lna - dl) - 1.0)
        f = (np.log(Dp) - np.log(Dm)) / (2.0 * dl)
        return f if f.size > 1 else f[0]

    # ----- transfer function (BBKS + Sugiyama shape) -----
    def transfer_bbks(self, k_hmpc):
        k = np.asarray(k_hmpc, dtype=float)  # h/Mpc
        # Sugiyama 1995 shape parameter
        gamma = self.Om * self.h * np.exp(-self.Ob - np.sqrt(2.0 * self.h) * self.Ob / self.Om)
        q = k / gamma  # k in h/Mpc, gamma in h/Mpc units effectively
        q = np.where(q <= 0, 1e-10, q)
        t = (np.log(1.0 + 2.34 * q) / (2.34 * q)) * (
            1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
        ) ** (-0.25)
        return t

    # ----- power spectra (h/Mpc units; P in (Mpc/h)^3) -----
    def _pk_shape(self, k_hmpc):
        k = np.asarray(k_hmpc, dtype=float)
        t = self.transfer_bbks(k)
        return k ** self.ns * t * t

    def sigma_R(self, R, pk_func):
        # rms of the field smoothed with a top-hat of radius R (Mpc/h)
        def integrand(lnk):
            k = np.exp(lnk)
            x = k * R
            w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
            return (k ** 3) * pk_func(k) * w * w / (2.0 * np.pi ** 2)
        val, _ = quad(integrand, np.log(1e-4), np.log(50.0), limit=200)
        return np.sqrt(val)

    def _norm(self):
        s8_shape = self.sigma_R(8.0, self._pk_shape)
        return (self.sigma8 / s8_shape) ** 2

    def power_spectrum_nowiggle(self, k_hmpc, z=0.0):
        A = self._norm()
        D = self.growth_factor(z) if np.ndim(z) == 0 else 1.0
        return A * self._pk_shape(k_hmpc) * D * D

    def bao_wiggle(self, k_hmpc, A_bao=0.05, s_bao=100.0, sigma_silk=7.0):
        # toy BAO template: oscillation of wavelength 2 pi / s_bao in k,
        # Silk-damped at high k. For teaching/visualization only.
        k = np.asarray(k_hmpc, dtype=float)
        return 1.0 + A_bao * np.sin(k * s_bao) * np.exp(-(k * sigma_silk) ** 2)

    def power_spectrum_bao(self, k_hmpc, z=0.0, **wig):
        return self.power_spectrum_nowiggle(k_hmpc, z) * self.bao_wiggle(k_hmpc, **wig)


# ---------- Gaussian errors ----------
def gaussian_pk_error(k, dk, Pk, nbar, volume):
    """Fractional 1-sigma error on the band power P(k).
    sigma_P/P = sqrt(2/Nk) (1 + 1/(nbar P)), Nk = V k^2 dk / (2 pi^2).
    volume in (Mpc/h)^3, nbar in (h/Mpc)^3, Pk in (Mpc/h)^3."""
    Nk = volume * k * k * dk / (2.0 * np.pi ** 2)
    return np.sqrt(2.0 / Nk) * (1.0 + 1.0 / (nbar * Pk))


# ---------- minimal lognormal mock ----------
def gaussian_box(pk_func, L, N, seed=0):
    """Generate one Gaussian density field delta(x) in a cubic box whose
    measured power spectrum recovers pk_func in expectation.
    L: box side (Mpc/h); N: grid per side; pk_func(k[h/Mpc]) -> P (Mpc/h)^3.
    Returns real delta with mean 0."""
    rng = np.random.default_rng(seed)
    d = L / N
    kx = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kmag[0, 0, 0] = 1.0
    Pk = pk_func(kmag)
    Pk[0, 0, 0] = 0.0
    Pk = np.where(Pk < 0, 0.0, Pk)
    white = rng.normal(size=(N, N, N))
    wk = np.fft.fftn(white)
    amp = np.sqrt(Pk / (d ** 3))  # discrete normalization (see notes)
    delta = np.fft.ifftn(wk * amp).real
    return delta


def measure_pk(delta, L, kbins):
    """Angle-averaged power spectrum of a real field delta on an NxNxN grid.
    Returns k_centers, Pk, Nmodes per bin."""
    N = delta.shape[0]
    d = L / N
    dk_field = np.fft.fftn(delta) * (d ** 3)
    pk3d = (np.abs(dk_field) ** 2) / (L ** 3)
    kx = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).ravel()
    p = pk3d.ravel()
    idx = np.digitize(kmag, kbins)
    kc, pkm, nm = [], [], []
    for b in range(1, len(kbins)):
        sel = idx == b
        n = np.count_nonzero(sel)
        if n > 0:
            kc.append(kmag[sel].mean())
            pkm.append(p[sel].mean())
            nm.append(n)
    return np.array(kc), np.array(pkm), np.array(nm)


if __name__ == "__main__":
    c = Cosmology()
    print("D(0)=", round(c.growth_factor(0.0), 4),
          " D(1)=", round(c.growth_factor(1.0), 4),
          " f(0)=", round(c.growth_rate(0.0), 4),
          " f(1)=", round(c.growth_rate(1.0), 4))
    print("DM(0.5)=", round(float(c.DM(0.5)), 2), "Mpc",
          " DH(0.5)=", round(float(c.DH(0.5)), 2), "Mpc")
    print("sigma8 check=", round(c.sigma_R(8.0, lambda k: c.power_spectrum_nowiggle(k, 0.0)), 4))
