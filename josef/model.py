"""Independent, unmodified numerical core for Josef's 2D shepherding slide.

Extracted from the seminar's ``part3_continuification.py`` (Torus2D, herder
reference, feedback and Euler--Maruyama loop). Kept here to avoid external
paths, style.py dependencies, and accidental changes in other team modules.
The finite-agent feedback uses a smoothed density and a speed cap; the
continuum convergence theorem is not a finite-N error guarantee.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def mcrit_2d(D, k, L):
    return 4 * np.pi * D * k * (1 + L**2) ** 1.5 / L**3


def n_herders(NT, Mc):
    return NT * Mc / (1 - Mc)


class Torus2D:
    def __init__(self, M=64, L=np.pi):
        self.M = M
        self.h = 2 * np.pi / M
        self.x = -np.pi + self.h * np.arange(M)
        m = np.fft.fftfreq(M, d=1.0 / M)
        self.MX, self.MY = np.meshgrid(m, m, indexing="ij")
        self.m2 = self.MX**2 + self.MY**2
        self.L = L
        # Fourier symbol of g = exp(-|x|/L) on R^2 sampled at integer frequencies
        self.G = 2 * np.pi * L**2 / (1 + L**2 * self.m2) ** 1.5
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing="ij")

    # cloud-in-cell deposit -> density (mass per area)
    def deposit(self, P, w):
        M, h = self.M, self.h
        u = (P + np.pi) / h
        i0_ = np.floor(u).astype(int)
        f = u - i0_
        rho = np.zeros((M, M))
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                np.add.at(rho, ((i0_[:, 0] + dx) % M, (i0_[:, 1] + dy) % M), w * wx * wy)
        return rho / h**2

    def gather(self, F, P):
        M, h = self.M, self.h
        u = (P + np.pi) / h
        i0_ = np.floor(u).astype(int)
        f = u - i0_
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                out = out + F[(i0_[:, 0] + dx) % M, (i0_[:, 1] + dy) % M] * wx * wy
        return out

    def grad(self, Fh):
        """gradient of a field given by its FFT."""
        gx = np.real(np.fft.ifft2(1j * self.MX * Fh))
        gy = np.real(np.fft.ifft2(1j * self.MY * Fh))
        return gx, gy


def herder_reference_2d(tor: Torus2D, D, logrhoT, MH):
    Fl = np.fft.fft2(logrhoT)
    Hh = np.zeros_like(Fl)
    nz = tor.m2 > 0
    Hh[nz] = -(D / tor.L) * Fl[nz] / tor.G[nz]
    H = np.real(np.fft.ifft2(Hh))
    area = (2 * np.pi) ** 2
    Mc = (H - H.min()).sum() * tor.h**2
    if MH >= Mc:
        ref = H - H.min() + (MH - Mc) / area
    else:
        fun = lambda A: np.maximum(H + A, 0).sum() * tor.h**2 - MH
        A = brentq(fun, -H.max() - 1, -H.min() + 1)
        ref = np.maximum(H + A, 0)
    return ref, Mc


def lattice(n, rng):
    s = int(np.ceil(np.sqrt(n)))
    g = -np.pi + (np.arange(s) + 0.5) * 2 * np.pi / s
    P = np.array([(a, b) for a in g for b in g])[:n]
    return P + 1e-3 * rng.standard_normal(P.shape)


def run_2d(NT=720, NH=None, D=0.01, L=np.pi, KH=10.0, k=6 / np.pi, rstar=np.pi / 2,
           M=64, dt=0.01, T=200.0, bw=0.4, seed=0, n_frames=0, vmax=3.0):
    rng = np.random.default_rng(seed)
    tor = Torus2D(M, L)
    Mc_cf = mcrit_2d(D, k, L)
    if NH is None:
        NH = int(np.ceil(n_herders(NT, Mc_cf)))
    Ntot = NT + NH
    MH, MT = NH / Ntot, NT / Ntot
    logrhoT = k * np.cos(tor.X) + k * np.cos(tor.Y)
    ref, Mc = herder_reference_2d(tor, D, logrhoT, MH)
    smooth = np.exp(-0.5 * bw**2 * tor.m2)                 # Gaussian KDE in Fourier space
    floor = 0.02 * MH / (2 * np.pi) ** 2

    H = lattice(NH, rng)
    Tg = rng.uniform(-np.pi, np.pi, size=(NT, 2))
    w = np.full(NH, 1.0 / Ntot)
    nsteps = int(round(T / dt))
    rec_every = max(1, nsteps // 400)
    times, chis = [], []
    frames = []
    frame_every = max(1, nsteps // n_frames) if n_frames else None
    snap0 = (H.copy(), Tg.copy())
    for n in range(nsteps + 1):
        rhoH_raw = tor.deposit(H, w)
        Fh = np.fft.fft2(rhoH_raw)
        # targets: v = f * rhoH = -L grad (g * rhoH)
        gx, gy = tor.grad(tor.G * Fh)
        vT = -L * np.stack([tor.gather(gx, Tg), tor.gather(gy, Tg)], 1)
        # herders: KDE, error, Poisson problem  Lap psi = -KH e,  w = grad psi,  u = w / rho_hat
        rho_hat = np.real(np.fft.ifft2(smooth * Fh))
        e = ref - rho_hat
        eh = np.fft.fft2(e)
        psih = np.zeros_like(eh)
        nz = tor.m2 > 0
        psih[nz] = KH * eh[nz] / tor.m2[nz]
        wx, wy = tor.grad(psih)
        rho_c = np.maximum(rho_hat, floor)
        uH = np.stack([tor.gather(wx / rho_c, H), tor.gather(wy / rho_c, H)], 1)
        sp = np.linalg.norm(uH, axis=1, keepdims=True)
        uH = np.where(sp > vmax, uH * vmax / np.maximum(sp, 1e-12), uH)
        if n % rec_every == 0:
            d = np.linalg.norm(Tg, axis=1)
            times.append(n * dt)
            chis.append(np.mean(d < rstar))
        if frame_every and n % frame_every == 0:
            frames.append((n * dt, H.copy(), Tg.copy()))
        if n == nsteps:
            break
        H = H + dt * uH
        Tg = Tg + dt * vT + np.sqrt(2 * D * dt) * rng.standard_normal(Tg.shape)
        H = (H + np.pi) % (2 * np.pi) - np.pi
        Tg = (Tg + np.pi) % (2 * np.pi) - np.pi
    return dict(NH=NH, NT=NT, Mc=Mc, Mc_cf=Mc_cf, MH=MH, times=np.array(times), chi=np.array(chis),
                H=H, T=Tg, snap0=snap0, ref=ref, tor_x=tor.x, frames=frames)
