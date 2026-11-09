"""The Jadhav ↔ Di Lorenzo bridge: one simulator with separable followers, controller and objective.

Three independent choices make a configuration:

  Followers  how the sheep move (rule, social terms, noise, activation),
  Herders    how the dogs move (controller) and how many there are,
  Target     the desired flock density and how the error is measured,

on a Domain that is either the open plane or a periodic square (wrapped displacements and a
circular-mean centroid throughout).

Two follower rules are implemented. They are different update laws, not two settings of one law:

  'heading'      Jadhav et al. (2024) / Strömbom et al. (2014). Each sheep has a unit heading
                 and a fixed speed v. The new heading is the normalized weighted sum
                     w_inertia h + w_rep ρ + w_dog Σ_dogs ê_away + w_att â + w_ali l̂ + e η̂
                 with â the unit sum toward n_att of its k_perceive nearest neighbours (random
                 each step), l̂ the heading of n_ali of those, and η̂ a unit vector of uniform
                 random angle (additive *directional* noise). The dog term is a unit vector,
                 applied only within r_dog. Activation decides which sheep step at all:
                   'published'  (herding_model.m) a sheep moves iff a dog is within r_dog or a
                                neighbour is within r_rep; otherwise it stands with its heading
                                unchanged. There is no grazing or random walk.
                   'contagion'  (herding_model_sim.m, Supp. Sec. 3) every sheep applies the full
                                social rule; the dog term only within r_dog; its speed is the mean
                                speed of its alignment neighbours, reset to v within r_dog, and
                                all speeds start at 0.
                   'always'     (adaptation; model.py social_always) the full rule for every
                                sheep at speed v, the dog term only within r_dog.
                 Extensions, off by default and labelled as such wherever they appear:
                   wander_speed > 0   (extension) a sheep the published rule would leave standing
                                steps wander_speed·dt along unit(w_inertia h + e η̂) instead.
                   D > 0        (hybrid) positional Brownian displacement √(2D dt) ξ added after
                                the heading step. Not in Jadhav.

  'first_order'  Di Lorenzo, Maffettone & di Bernardo (arXiv:2411.04791; interacting followers
                 arXiv:2604.11353). Overdamped positions, no heading, no fixed speed:
                     dx = s [ Σ_dogs f(|x−h|) ê_away + Σ_j g(|x−x_j|) ê_away_from_j ] dt + √(2D) dW
                 f(r) = a_dog e^(−r/L_dog) (their Eq. 2, non-periodic form; on a torus the
                 kernel is summed over periodic images), g the Morse-type follower kernel of the
                 2026 paper, g(r) = g_rep e^(−r/l_rep) − g_att e^(−r/l_att) (positive = repulsion;
                 g = 0 is the 2024 paper). s = 1/(N + N_dogs) when population_scaling is on
                 (their α) and 1 otherwise (kernels in m/s, the convention of feasibility.py).
                 Noise is positional Brownian diffusion with coefficient D. Extension:
                   ali_fo > 0   (extension) a drift ali_fo × the mean displacement rate of
                                neighbours within r_ali over the previous step. Neither Di
                                Lorenzo paper has alignment.

Controllers:
  'strombom'     Jadhav's published dog: collect the sheep farthest from the centroid when it is
                 beyond l_sep, otherwise drive from l_drive behind the centroid on the line from
                 the target; step v_dog with heading noise e; creep by slow_step when within r_rep
                 of a sheep. With n > 1 (adaptation, labelled 'multi-dog Strömbom'): drive points
                 fanned by spread_angle around the rear, the dog nearest a straggler collects.
                 Optional reach-and-release hold (model.py hold_radius).
  'dilorenzo'    Their two-layer control, periodic domain only. Layer 1 (feedforward): the
                 herder density ρ̄_H = H − min H with H the deconvolution by f of the required
                 drift v̄ = D∇ln ρ̄_T − (K_g * ρ̄_T) (Eqs. 9–11 of 2024; Eq. 17 of 2026 adds the
                 follower term). Layer 2 (Theorem 1): herders follow u = w/ρ_H with ∇·w = −K_H
                 (ρ̄_H − ρ_H), ∇×w = 0, solved by Fourier series; ρ_H is a Gaussian kernel density
                 estimate of the herders. Dog speed caps u when finite (they have no cap).
  'static_ring'  n dogs fixed on a ring around the goal (feedforward, no sensing). Heuristic.
  'density_feedback'  HEURISTIC, not from either paper. Each step the dogs read the sheep and
                 the target: with the centroid farther than fb_drive_radius from the goal they
                 drive from behind it (Strömbom's drive geometry, fanned); otherwise each dog
                 takes the sheep with the largest kernel-density surplus ρ̂_b(x_i) − ρ̄_b(x_i)
                 above fb_threshold × the target's peak (greedy, one per dog, at least fb_exclusion
                 apart) and stands fb_standoff beyond it on the ray from the goal, so the surplus is
                 pushed inward; dogs with nothing to act on wait beyond reach of every sheep. When
                 the L² error falls below fb_release every dog withdraws (so published sheep stop)
                 and re-engages above fb_engage. Dogs move at most v per step, with
                 no heading noise, and creep when within r_rep of a sheep. It controls the
                 per-sheep density surplus; it is feedback in the sheep, not a density design.
  'none'         no dogs.

Objective: a smooth localized target density ('gaussian' with sd sigma, or 'vonmises' product
exp{k(cos x + cos y)} on a torus, their Eq. 5 form). Error = ||ρ̂_b − ρ̄_b||² / ||ρ̄_b||² with ρ̂_b the
Gaussian KDE (bandwidth b) of the sheep and ρ̄_b the target smoothed by the same kernel. The
finite-population floor E||ρ̂_b − ρ̄_b||²/||ρ̄_b||² for N independent draws from ρ̄ is returned so
residual error can be read against it. Occupancy = share of sheep within r_goal of the centre;
cohesion = mean distance to the (circular-mean) centroid.

Reference check: Followers(rule='heading') with Jadhav's parameters, Herders('strombom', n=1) and
the open plane consume random numbers in the same order as shepherd.model.simulate and reproduce
its trajectories exactly (tests/test_bridge.py).
"""

from dataclasses import dataclass, field, replace

import numpy as np

EPS = np.finfo(float).eps
INIT, DRIVE, COLLECT, SLOW, HOLD, FLOW, STATIC, SHAPE = 0, 1, 2, 3, 5, 6, 7, 8


# ---------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Domain:
    periodic: bool = False
    size: float = 80.0  # box side when periodic; positions live in [-size/2, size/2)
    images: int = 0  # periodic: sum first-order kernels over (2·images+1)² periodic images (0 = minimum image)

    def wrap(self, d):
        if not self.periodic:
            return d
        return (d + self.size / 2) % self.size - self.size / 2

    def centroid(self, pos):
        """Mean position; on a torus the circular mean per coordinate."""
        if not self.periodic:
            return pos.mean(0)
        ang = pos * (2 * np.pi / self.size)
        m = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
        return m * self.size / (2 * np.pi)


@dataclass(frozen=True)
class Followers:
    rule: str = "heading"  # 'heading' | 'first_order'
    n: int = 14
    dt: float = 1.0
    v: float = 1.0  # heading rule: fixed step speed
    # heading rule (Jadhav): social terms
    w_inertia: float = 0.5
    w_rep: float = 2.0
    w_att: float = 1.5
    w_ali: float = 1.3
    w_dog: float = 1.0
    e_noise: float = 0.3  # additive directional noise weight
    r_rep: float = 2.0
    k_perceive: int = 10
    n_att: int = 4
    n_ali: int = 1
    r_dog: float = 12.0
    activation: str = "published"  # 'published' | 'contagion' | 'always'
    wander_speed: float = 0.0  # extension (heading rule)
    # first-order rule (Di Lorenzo)
    a_dog: float = 1.0
    L_dog: float = np.pi
    g_rep: float = 0.0
    l_rep: float = 1.0
    g_att: float = 0.0
    l_att: float = 1.0
    population_scaling: bool = True
    ali_fo: float = 0.0  # extension (first-order rule)
    r_ali: float = 3.0
    # both rules
    D: float = 0.0  # positional diffusion coefficient

    def resolved(self):
        k = min(self.k_perceive, self.n - 1)
        return replace(self, k_perceive=k, n_att=min(self.n_att, k), n_ali=min(self.n_ali, min(self.n_att, k)))

    def g(self, r):
        return self.g_rep * np.exp(-r / self.l_rep) - self.g_att * np.exp(-r / self.l_att)

    def f(self, r):
        return self.a_dog * np.exp(-r / self.L_dog)

    @property
    def has_pair(self):
        return self.g_rep != 0 or self.g_att != 0


@dataclass(frozen=True)
class Herders:
    controller: str = "strombom"  # 'strombom' | 'dilorenzo' | 'static_ring' | 'none'
    n: int = 1
    v: float = 1.5  # strombom: step speed; dilorenzo: speed cap (inf = none); static_ring: unused
    # strombom
    slow_step: float = 0.05
    l_sep: float | None = None
    l_drive: float | None = None
    l_collect: float | None = None
    noise: float | None = None  # heading noise (None = followers' e_noise)
    spread_angle: float = 0.5  # rad between neighbouring drive points when n > 1
    hold_radius: float | None = None
    hold_release: float | None = None
    hold_standoff: float | None = None
    # dilorenzo
    K_H: float = 10.0
    bandwidth: float | None = None  # KDE bandwidth (None = 0.4 · size/2π)
    grid: int = 64
    reg: float = 1e-3  # Tikhonov weight of the deconvolution (fraction of the kernel's peak power)
    # The follower model the design assumes (None: the simulated followers). Lets Di Lorenzo's herders keep
    # their own model of the sheep (push kernel, diffusion, population scaling) while the simulated sheep
    # follow another rule, e.g. Jadhav's heading rule, which has no drift to deconvolve.
    design_model: "Followers | None" = None
    substeps: int = 1  # herder flow sub-steps per follower step (Di Lorenzo integrate at dt = 0.01)
    # static_ring
    ring_radius: float = 15.0
    # density_feedback (heuristic)
    fb_standoff: float | None = None  # distance beyond the surplus sheep (None: r_dog − 2 for heading, L_dog/2 first-order)
    fb_exclusion: float | None = None  # minimum distance between the sheep different dogs act on (None: r_dog)
    fb_drive_radius: float | None = None  # drive when the centroid is farther than this from the goal (None: r_goal)
    fb_release: float | None = None  # withdraw when the relative L² error is below this (None: never)
    fb_engage: float | None = None  # re-engage above this (None: 1.5 × fb_release)
    fb_threshold: float = 0.25  # act only on sheep whose surplus exceeds this fraction of the smoothed target's peak

    def resolve(self, F: Followers, dom: Domain, T=None):
        r = F.r_rep
        reach = F.r_dog if F.rule == "heading" else F.L_dog
        return replace(
            self,
            l_sep=r * F.n ** (2 / 3) if self.l_sep is None else self.l_sep,
            l_drive=r * np.sqrt(F.n) if self.l_drive is None else self.l_drive,
            l_collect=r if self.l_collect is None else self.l_collect,
            noise=F.e_noise if self.noise is None else self.noise,
            bandwidth=0.4 * dom.size / (2 * np.pi) if self.bandwidth is None else self.bandwidth,
            fb_standoff=(max(F.r_dog - 2.0, 1.0) if F.rule == "heading" else F.L_dog / 2) if self.fb_standoff is None else self.fb_standoff,
            fb_exclusion=reach if self.fb_exclusion is None else self.fb_exclusion,
            fb_drive_radius=(T.resolve(dom).r_goal if T is not None else None) if self.fb_drive_radius is None else self.fb_drive_radius,
            fb_engage=(None if self.fb_release is None else 1.5 * self.fb_release) if self.fb_engage is None else self.fb_engage,
        )


@dataclass(frozen=True)
class Target:
    kind: str = "gaussian"  # 'gaussian' | 'vonmises'
    centre: tuple = (0.0, 0.0)
    sigma: float = 2.5  # gaussian sd
    k: float = 6 / np.pi  # vonmises concentration (torus only)
    r_goal: float | None = None  # occupancy radius (None: 95% radius of the gaussian, or size/4 for vonmises)
    bandwidth: float = 1.0  # KDE bandwidth b of the L² error
    # The required drift D∇ρ̄/ρ̄ is unconstrained where ρ̄ ≈ 0; it is computed as D∇ρ̄/(ρ̄ + floor·max ρ̄),
    # which leaves the von Mises reference unchanged (M̂ 0.2754 → 0.2760) and removes the seam
    # artefact of a Gaussian on a torus (∇ln ρ̄ = −x/σ² is a sawtooth across the seam).
    floor: float = 1e-4

    def resolve(self, dom: Domain):
        if self.r_goal is not None:
            return self
        if self.kind == "gaussian":
            return replace(self, r_goal=self.sigma * np.sqrt(-2 * np.log(0.05)))
        return replace(self, r_goal=dom.size / 4)  # their r* = π/2 on [−π, π]²

    def log_density(self, X, Y, dom: Domain):
        """ln of the (unnormalized) target at grid points, with its gradient."""
        cx, cy = self.centre
        if self.kind == "gaussian":
            dx, dy = dom.wrap(X - cx), dom.wrap(Y - cy)
            return -(dx ** 2 + dy ** 2) / (2 * self.sigma ** 2), (-dx / self.sigma ** 2, -dy / self.sigma ** 2)
        if not dom.periodic:
            raise ValueError("the von Mises target needs a periodic domain")
        w = 2 * np.pi / dom.size
        ln = self.k * (np.cos(w * (X - cx)) + np.cos(w * (Y - cy)))
        return ln, (-self.k * w * np.sin(w * (X - cx)), -self.k * w * np.sin(w * (Y - cy)))


# ---------------------------------------------------------------------------- periodic grid tools

class Grid:
    """n × n periodic grid over [−size/2, size/2)², with FFT convolution helpers."""

    def __init__(self, size, n):
        self.size, self.n, self.h = float(size), int(n), float(size) / n
        a = (np.arange(n) - n // 2) * self.h
        self.X, self.Y = np.meshgrid(a, a, indexing="ij")
        d = np.fft.fftfreq(n, d=1.0 / n) * self.h  # displacements from the origin, wrapped
        self.DX, self.DY = np.meshgrid(d, d, indexing="ij")
        k = 2 * np.pi * np.fft.fftfreq(n, d=self.h)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2

    def gaussian_kernel(self, b):
        """Periodic Gaussian K_b on the wrapped grid, normalized to unit mass."""
        K = np.zeros((self.n, self.n))
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                K += np.exp(-((self.DX + i * self.size) ** 2 + (self.DY + j * self.size) ** 2) / (2 * b ** 2))
        return K / (K.sum() * self.h ** 2)

    def vector_kernel(self, radial, images=0):
        """K(z) = radial(|z|) z/|z| summed over periodic images, as (2, n, n)."""
        kx = np.zeros((self.n, self.n))
        ky = np.zeros_like(kx)
        for i in range(-images, images + 1):
            for j in range(-images, images + 1):
                zx, zy = self.DX + i * self.size, self.DY + j * self.size
                r = np.hypot(zx, zy)
                with np.errstate(invalid="ignore", divide="ignore"):
                    m = np.where(r > 0, radial(r) / r, 0.0)
                kx += m * zx
                ky += m * zy
        return np.stack([kx, ky])

    def convolve(self, kernel, density):
        """(kernel * density)(x) for a scalar or (2, n, n) vector kernel (circular)."""
        Fd = np.fft.fft2(density)
        if kernel.ndim == 2:
            return np.real(np.fft.ifft2(np.fft.fft2(kernel) * Fd)) * self.h ** 2
        return np.stack([np.real(np.fft.ifft2(np.fft.fft2(k) * Fd)) for k in kernel]) * self.h ** 2

    def deposit(self, pos, weights):
        """Bilinear deposit of point masses onto the grid, as a density (mass / h²)."""
        n, h = self.n, self.h
        u = (pos[:, 0] + self.size / 2) / h
        v = (pos[:, 1] + self.size / 2) / h
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        rho = np.zeros((n, n))
        for di, wu in ((0, 1 - fu), (1, fu)):
            for dj, wv in ((0, 1 - fv), (1, fv)):
                np.add.at(rho, ((i0 + di) % n, (j0 + dj) % n), weights * wu * wv)
        return rho / h ** 2

    def sample(self, field, pos, wrapped=False):
        """Bilinear interpolation of a (..., n, n) field at positions (M, 2). `wrapped`: the field
        is on the FFT-ordered displacement grid (DX, DY) rather than the centred one (X, Y)."""
        n, h = self.n, self.h
        off = 0.0 if wrapped else self.size / 2
        u = (pos[:, 0] + off) / h
        v = (pos[:, 1] + off) / h
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        out = 0
        for di, wu in ((0, 1 - fu), (1, fu)):
            for dj, wv in ((0, 1 - fv), (1, fv)):
                out = out + field[..., (i0 + di) % n, (j0 + dj) % n] * (wu * wv)
        return out

    def target(self, T: Target, dom: Domain):
        """Unit-mass target density ρ̂ and ∇ρ̂/(ρ̂ + floor·max ρ̂) (= ∇ln ρ̂ where ρ̂ is not tiny)."""
        ln, (gx, gy) = T.log_density(self.X, self.Y, dom)
        rho = np.exp(ln - ln.max())
        fac = rho / (rho + T.floor)
        return rho / (rho.sum() * self.h ** 2), np.stack([gx, gy]) * fac


class PeriodicKernel:
    """K(z) = radial(|z|) z/|z| summed over the (2·images+1)² nearest periodic images (images = 0:
    minimum image only). The 2024 paper's periodic f is the infinite image sum; with L = π on a 2π
    box the terms fall as e^(−2n), so images = 1 is within 10% and images = 3 within 1%. The design
    grid (Grid.vector_kernel) uses the same sum, so controller and dynamics see one kernel."""

    def __init__(self, radial, dom: Domain, images):
        self.radial, self.dom = radial, dom
        self.images = int(images) if dom.periodic else 0

    def __call__(self, z):
        """z (..., 2) displacements already wrapped. Returns (..., 2)."""
        out = 0.0
        for i in range(-self.images, self.images + 1):
            for j in range(-self.images, self.images + 1):
                zz = z + np.array([i, j]) * self.dom.size if (i or j) else z
                r = np.maximum(np.linalg.norm(zz, axis=-1), 1e-9)
                out = out + (self.radial(r) / r)[..., None] * zz
        return out


# ---------------------------------------------------------------------------- Di Lorenzo design and feasibility

@dataclass
class Design:
    grid: Grid
    rho_hat: np.ndarray  # unit-mass target on the grid
    required_drift: np.ndarray  # v̄ (2, n, n) the dogs must supply
    H: np.ndarray  # deconvolution of v̄ by f (before the shift)
    rho_H: np.ndarray  # designed herder density actually used by the controller
    mass_needed: float  # M̂_H = ∫(H − min H) (or N̂_H in number units)
    mass_available: float
    feasible: bool
    drift_error: float  # relative error of the drift the shifted design produces
    kernel_f: np.ndarray


def design_herders(F: Followers, Hp: Herders, T: Target, dom: Domain, grid=None):
    """Di Lorenzo's feedforward herder density for this follower model, target and dog count.

    With population scaling the densities are mass fractions (M_T + M_H = 1, their convention)
    and mass_needed is M̂_H of their Eq. 11; without it they are number densities and
    mass_needed is a dog count. If the available mass exceeds what is needed the surplus is
    spread uniformly (a uniform herder density exerts no drift on a torus, since K_f is odd);
    if it falls short the design is scaled down and marked infeasible, so the shortfall can be
    observed rather than hidden.
    """
    if not dom.periodic:
        raise ValueError("Di Lorenzo's design is a deconvolution on a periodic domain")
    g = grid or Grid(dom.size, Hp.grid)
    rho_hat, grad_ln = g.target(T, dom)
    N, M = F.n, Hp.n
    F = Hp.design_model or F  # what the herders believe about the followers (counts stay the simulated ones)
    if F.population_scaling:
        m_T, m_H = N / (N + M), M / (N + M)
    else:
        m_T, m_H = float(N), float(M)
    v = F.D * grad_ln
    if F.has_pair:
        Kg = g.vector_kernel(F.g, dom.images)  # drift on x due to a neighbour at x − z: g(|z|) z/|z|
        v = v - m_T * g.convolve(Kg, rho_hat)
    Kf = g.vector_kernel(F.f, dom.images)
    Fx, Fy = np.fft.fft2(Kf[0]) * g.h ** 2, np.fft.fft2(Kf[1]) * g.h ** 2
    Vx, Vy = np.fft.fft2(v[0]), np.fft.fft2(v[1])
    den = np.abs(Fx) ** 2 + np.abs(Fy) ** 2
    Hhat = (np.conj(Fx) * Vx + np.conj(Fy) * Vy) / (den + Hp.reg * den.max())
    Hhat[0, 0] = 0.0
    H = np.real(np.fft.ifft2(Hhat))
    shifted = H - H.min()
    needed = float(shifted.sum() * g.h ** 2)
    if m_H >= needed:
        rho_H = shifted + (m_H - needed) / g.size ** 2
        feasible = True
    else:
        rho_H = shifted * (m_H / needed if needed > 0 else 0.0)
        feasible = False
    achieved = g.convolve(Kf, rho_H)
    w = rho_hat / rho_hat.max()
    err = float(np.sqrt(np.sum(w * ((achieved - v) ** 2).sum(0)) / max(np.sum(w * (v ** 2).sum(0)), 1e-300)))
    return Design(g, rho_hat, v, H, rho_H, needed, m_H, feasible, err, Kf)


def feasibility_scan(F: Followers, T: Target, dom: Domain, masses=None, grid=64, reg=1e-3):
    """Minimum herder mass as a function of the herder mass fraction (population scaling).

    With no follower interaction M̂_H does not depend on M_H and the condition is M_H ≥ M̂_H
    (2024, Eqs. 11–12: N_H ≥ N_T M̂/(1 − M̂)). With interactions the required drift contains
    M_T (K_g * ρ̂) with M_T = 1 − M_H, so M̂ depends on M_H and the feasible set is where
    M_H ≥ M̂(M_H) (2026, Theorem 1). Returns the scan and the feasible masses.
    """
    masses = np.linspace(0.01, 0.99, 99) if masses is None else np.asarray(masses, float)
    g = Grid(dom.size, grid)
    rho_hat, grad_ln = g.target(T, dom)
    Kf = g.vector_kernel(F.f, dom.images)
    Fx, Fy = np.fft.fft2(Kf[0]) * g.h ** 2, np.fft.fft2(Kf[1]) * g.h ** 2
    den = np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + reg * (np.abs(Fx) ** 2 + np.abs(Fy) ** 2).max()

    def deconv(v):
        Vx, Vy = np.fft.fft2(v[0]), np.fft.fft2(v[1])
        Hh = (np.conj(Fx) * Vx + np.conj(Fy) * Vy) / den
        Hh[0, 0] = 0.0
        return np.real(np.fft.ifft2(Hh))

    H_D = deconv(F.D * grad_ln)
    H_g = deconv(g.convolve(g.vector_kernel(F.g, dom.images), rho_hat)) if F.has_pair else 0.0
    needed = []
    for mH in masses:
        H = H_D - (1 - mH) * H_g
        needed.append(float((H - H.min()).sum() * g.h ** 2))
    needed = np.array(needed)
    ok = masses >= needed
    return {"mass": masses, "needed": needed, "feasible": ok,
            "min_feasible_mass": float(masses[ok].min()) if ok.any() else None,
            "max_feasible_mass": float(masses[ok].max()) if ok.any() else None}


def herders_from_mass(n_sheep, mass):
    """Their Eq. 12: N_H = N_T M̂ / (1 − M̂)."""
    return n_sheep * mass / (1 - mass)


# ---------------------------------------------------------------------------- metrics

def finite_population_floor(T: Target, dom: Domain, n, grid=None):
    """E||ρ̂_b − ρ̄_b||² / ||ρ̄_b||² for n independent draws from the target: s²/(n b²) for a
    Gaussian on the plane; on a torus (1/n)(K_{b√2}(0) − ∫ρ̄_b²)/∫ρ̄_b² on the grid."""
    b = T.bandwidth
    if not dom.periodic and T.kind == "gaussian":
        return T.sigma ** 2 / (n * b ** 2)
    g = grid or Grid(dom.size, 64)
    rho_b = g.convolve(g.gaussian_kernel(b), g.target(T, dom)[0])
    i_rb2 = float((rho_b ** 2).sum() * g.h ** 2)
    k0 = float(g.gaussian_kernel(b * np.sqrt(2)).max())
    return (k0 - i_rb2) / (n * i_rb2)


class Metrics:
    """Density error, occupancy, cohesion and centroid error for one configuration."""

    def __init__(self, T: Target, dom: Domain, n, grid=None):
        self.T, self.dom = T.resolve(dom), dom
        self.c = np.asarray(self.T.centre, float)
        self.floor = finite_population_floor(self.T, dom, n, grid)
        self.grid = None
        if dom.periodic or self.T.kind != "gaussian":
            self.grid = grid or Grid(dom.size, 64)
            self.Kb = self.grid.gaussian_kernel(self.T.bandwidth)
            self.rho_b = self.grid.convolve(self.Kb, self.grid.target(self.T, dom)[0])
            self.norm = float((self.rho_b ** 2).sum() * self.grid.h ** 2)
            self.peak = float(self.rho_b.max())
        else:
            self.peak = 1.0 / (2 * np.pi * (self.T.sigma ** 2 + self.T.bandwidth ** 2))  # the smoothed target's maximum

    def density_error(self, pos):
        b, N = self.T.bandwidth, len(pos)
        if self.grid is None:  # closed form for a Gaussian target on the plane
            s = self.T.sigma
            d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
            pair = np.exp(-d2 / (4 * b ** 2)).sum() / (4 * np.pi * b ** 2) / N ** 2
            a2 = s ** 2 + 2 * b ** 2  # ∫ K_b(x − x_i) ρ̄_b(x) dx is a Gaussian of variance s² + 2b²
            cross = np.exp(-((pos - self.c) ** 2).sum(-1) / (2 * a2)).sum() / (2 * np.pi * a2) / N
            norm = 1 / (4 * np.pi * (s ** 2 + b ** 2))  # ∫ ρ̄_b²
            return (pair - 2 * cross + norm) / norm
        rho = self.grid.convolve(self.Kb, self.grid.deposit(pos, np.full(N, 1.0 / N)))
        return float(((rho - self.rho_b) ** 2).sum() * self.grid.h ** 2 / self.norm)

    def surplus(self, pos):
        """ρ̂_b(x_i) − ρ̄_b(x_i) at the sheep: the kernel density estimate of the sheep (bandwidth b)
        minus the smoothed target, both unit mass. Positive where sheep are in excess."""
        b, N = self.T.bandwidth, len(pos)
        d2 = (self.dom.wrap(pos[:, None] - pos[None]) ** 2).sum(-1)
        rho_hat = np.exp(-d2 / (2 * b ** 2)).sum(1) / (2 * np.pi * b ** 2 * N)
        if self.grid is None:
            a2 = self.T.sigma ** 2 + b ** 2
            rho_bar = np.exp(-((pos - self.c) ** 2).sum(-1) / (2 * a2)) / (2 * np.pi * a2)
        else:
            rho_bar = self.grid.sample(self.rho_b, self.dom.wrap(pos))
        return rho_hat - rho_bar

    def occupancy(self, pos):
        return float((np.linalg.norm(self.dom.wrap(pos - self.c), axis=-1) <= self.T.r_goal).mean())

    def cohesion(self, pos):
        G = self.dom.centroid(pos)
        return float(np.linalg.norm(self.dom.wrap(pos - G), axis=-1).mean())

    def centroid_error(self, pos):
        return float(np.linalg.norm(self.dom.wrap(self.dom.centroid(pos) - self.c)))

    def all(self, pos):
        return {"density_error": self.density_error(pos), "occupancy": self.occupancy(pos),
                "cohesion": self.cohesion(pos), "centroid_error": self.centroid_error(pos)}


# ---------------------------------------------------------------------------- simulation

@dataclass
class BridgeResult:
    followers: Followers
    herders: Herders
    target: Target
    domain: Domain
    t: np.ndarray  # stored times (S,)
    pos_s: np.ndarray  # (S, N, 2)
    hd_s: np.ndarray  # (S, N, 2) headings (heading rule) or last displacement direction
    pos_d: np.ndarray  # (S, M, 2)
    mode: np.ndarray  # (S, M)
    metrics: dict  # name -> (S,)
    floor: float
    design: Design | None = None
    goal: np.ndarray | None = None  # (S, 2) goal centre handed to the herders, when a goal schedule is used

    def final(self):
        return {k: float(v[-1]) for k, v in self.metrics.items()}

    def tail_mean(self, frac=0.25):
        k = max(1, int(len(self.t) * frac))
        return {k_: float(np.mean(v[-k:])) for k_, v in self.metrics.items()}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, v / n, 0.0)


def initial_state(F: Followers, Hp: Herders, T: Target, dom: Domain, rng, how="jadhav", box_length=250.0):
    """Starting positions and headings.

    'jadhav'   herding_model.m: sheep within 3 r_rep of a point box_length from the target in a
               random direction, the dog within 3 r_dog of it (same random draws as model.py).
    'uniform'  sheep uniform over the box (Di Lorenzo), dogs on a lattice over the box.
    'spread'   sheep Gaussian-scattered around the goal with sd `box_length`, dogs on a ring
               of radius 2·box_length (density experiments).
    """
    N, M = F.n, Hp.n
    c = np.asarray(T.centre, float)
    if how == "jadhav":
        theta_pos = rng.random()
        side = c + box_length * np.array([np.cos(2 * np.pi * theta_pos), np.sin(2 * np.pi * theta_pos)])
        pos_s = side - 3 * F.r_rep * rng.random((N, 2))
        pos_d = side[None] - 3 * F.r_dog * rng.random((M, 2))
        ts, td = 2 * np.pi * rng.random(N), 2 * np.pi * rng.random(M)
    elif how == "uniform":
        L = dom.size
        pos_s = rng.uniform(-L / 2, L / 2, (N, 2))
        k = max(int(np.ceil(np.sqrt(M))), 1)  # M = 0 (no dogs): empty lattice, same sheep draws
        a = (np.arange(k) + 0.5) * L / k - L / 2
        lat = np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2)[:M]
        pos_d = lat + 0.01 * L / k * rng.standard_normal((M, 2))
        ts, td = 2 * np.pi * rng.random(N), 2 * np.pi * rng.random(M)
    elif how == "spread":
        pos_s = c + box_length * rng.standard_normal((N, 2))
        ang = 2 * np.pi * (np.arange(M) + rng.random()) / max(M, 1)
        pos_d = c + 2 * box_length * np.stack([np.cos(ang), np.sin(ang)], -1)
        ts, td = 2 * np.pi * rng.random(N), 2 * np.pi * rng.random(M)
    else:
        raise ValueError(how)
    hd_s = np.stack([np.cos(ts), np.sin(ts)], -1)
    vel_d = np.stack([np.cos(td), np.sin(td)], -1)
    if dom.periodic:
        pos_s, pos_d = dom.wrap(pos_s), dom.wrap(pos_d)
    return pos_s, hd_s, pos_d, vel_d


def _rotate(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def feedback_aims(pos_s, pos_d, c, surplus, Hp: Herders, dom: Domain, holding, reach, peak, driving=False):
    """Where the density-feedback dogs go, from the sheep positions and their density surplus.

    reach: the distance within which a dog moves the sheep (r_dog for the heading rule, L_dog for
    the first-order rule); peak: the smoothed target's maximum, the scale of the surplus threshold.
    Dogs that have no sheep to act on wait beyond reach of every sheep (the flock's spread plus
    reach, measured from the centroid). Driving starts when the centroid is farther than
    fb_drive_radius from the goal and stops when it is within half of it (`driving` carries the
    state). Returns (aims (M, 2), mode (M,), driving). Deterministic; the JS port gives the same values.
    """
    M = len(pos_d)
    G = dom.centroid(pos_s)
    away_t = dom.wrap(G - c)
    g_norm = np.linalg.norm(away_t)
    e_gt = away_t / g_norm if g_norm > 0 else np.array([1.0, 0.0])
    spread = np.linalg.norm(dom.wrap(pos_s - G), axis=-1).max()
    standoff_hold = Hp.hold_standoff if Hp.hold_standoff is not None else spread + reach + 1.0
    aims = np.zeros((M, 2))
    mode = np.full(M, HOLD, dtype=np.int8)

    def hold_aims():
        od = dom.wrap(pos_d - G)
        on = np.linalg.norm(od, axis=-1, keepdims=True)
        out_dir = np.where(on > 1e-9, od / np.maximum(on, 1e-9), e_gt)
        return G + standoff_hold * out_dir

    driving = g_norm > 0.5 * Hp.fb_drive_radius if driving else g_norm > Hp.fb_drive_radius
    if holding:
        return hold_aims(), mode, driving
    if driving:  # far: Strömbom's drive geometry, fanned
        p_drive = G + Hp.l_drive * e_gt
        offsets = Hp.spread_angle * (np.arange(M) - (M - 1) / 2)
        return G + _rotate(np.broadcast_to(dom.wrap(p_drive - G), (M, 2)), offsets), np.full(M, DRIVE, dtype=np.int8), driving
    # near: one surplus sheep per dog, greedily, at least fb_exclusion apart
    order = np.argsort(-surplus, kind="stable")
    targets = []
    for i in order:
        if surplus[i] <= Hp.fb_threshold * peak or len(targets) == M:
            break
        if all(np.linalg.norm(dom.wrap(pos_s[i] - pos_s[j])) >= Hp.fb_exclusion for j in targets):
            targets.append(int(i))
    aims = hold_aims()
    free = list(range(M))
    for i in targets:  # the nearest free dog takes each target, in surplus order
        d = [np.linalg.norm(dom.wrap(pos_s[i] - pos_d[m])) for m in free]
        m = free.pop(int(np.argmin(d)))
        u = dom.wrap(pos_s[i] - c)
        un = np.linalg.norm(u)
        if un < 1e-9:
            u = dom.wrap(pos_s[i] - G)
            un = np.linalg.norm(u)
        u = u / un if un > 1e-9 else np.array([1.0, 0.0])
        aims[m] = pos_s[i] + Hp.fb_standoff * u
        mode[m] = SHAPE
    return aims, mode, driving


def simulate_bridge(F: Followers, Hp: Herders, T: Target, dom: Domain = Domain(), n_steps=370, seed=None,
                    init="jadhav", box_length=250.0, init_state=None, store_every=1, metric_every=None, goal=None):
    """Run one configuration. Returns a BridgeResult with trajectories and metric series.

    metric_every defaults to store_every. init_state = (pos_s, hd_s, pos_d, vel_d) overrides the
    random start (so controlled comparisons share identical initial configurations).

    goal: optional planner, a function t ↦ goal centre (2,), for the 'dilorenzo' controller only. It is
    read once per follower step at the state's time and only translates the herders' precomputed design
    ρ̄_H by goal(t) − T.centre (exact on the periodic grid: the target, the deconvolution and the flow are
    translation-equivariant). The followers, the herder flow and its integration, the random stream and
    the metrics (still against T) are untouched; goal=None is the original code path.
    """
    F = F.resolved()
    T = T.resolve(dom)
    Hp = Hp.resolve(F, dom, T)
    rng = np.random.default_rng(seed)
    N, M, dt = F.n, Hp.n if Hp.controller != "none" else 0, F.dt
    Hp = replace(Hp, n=M)
    c = np.asarray(T.centre, float)
    metric_every = store_every if metric_every is None else metric_every

    if init_state is None:
        pos_s, hd_s, pos_d, vel_d = initial_state(F, Hp, T, dom, rng, init, box_length)
    else:
        pos_s, hd_s, pos_d, vel_d = (np.array(a, float) for a in init_state)
    pos_d, vel_d = pos_d.reshape(-1, 2)[:M], vel_d.reshape(-1, 2)[:M]  # a "none" controller keeps no dogs
    spd = np.zeros(N) if F.activation == "contagion" else np.full(N, F.v)  # contagion starts at rest
    last_disp = np.zeros((N, 2))
    mode = np.full(M, INIT, dtype=np.int8)
    holding = driving = False

    design = grid = None
    if Hp.controller == "dilorenzo":
        design = design_herders(F, Hp, T, dom)
        grid = design.grid
        Kb_H = grid.gaussian_kernel(Hp.bandwidth)
        w_dog = 1.0 / (N + M) if (Hp.design_model or F).population_scaling else 1.0
        with np.errstate(divide="ignore"):
            inv_k2 = np.where(grid.K2 > 0, 1.0 / grid.K2, 0.0)
    if Hp.controller == "static_ring":
        ang = 2 * np.pi * np.arange(M) / max(M, 1)
        pos_d = c + Hp.ring_radius * np.stack([np.cos(ang), np.sin(ang)], -1)
        mode[:] = STATIC
    met = Metrics(T, dom, N, grid)
    c0 = np.asarray(T.centre, float)
    rho_bar_H = design.rho_H if design is not None else None  # the design the flow tracks this step
    c_now = c0.copy()
    if goal is not None:
        if Hp.controller != "dilorenzo" or design is None:
            raise ValueError("a goal schedule translates Di Lorenzo's design; use controller='dilorenzo'")
        design_hat = np.fft.fft2(design.rho_H)
        shift_now = np.zeros(2)

        def shifted_design(c):
            d = dom.wrap(np.asarray(c, float) - c0)
            if not d.any():
                return design.rho_H
            return np.real(np.fft.ifft2(design_hat * np.exp(-1j * (grid.KX * d[0] + grid.KY * d[1]))))

        c_now = np.asarray(goal(0.0), float)
        rho_bar_H, shift_now = shifted_design(c_now), c_now.copy()
    if F.rule == "first_order":
        Kf_fn = PeriodicKernel(F.f, dom, dom.images)
        Kg_fn = PeriodicKernel(F.g, dom, dom.images)

    S = n_steps // store_every + 1
    t_out = np.zeros(S)
    G_out = np.zeros((S, 2)) if goal is not None else None  # goal centre at each stored step
    P_s, H_s = np.zeros((S, N, 2)), np.zeros((S, N, 2))
    P_d, MODE = np.zeros((S, M, 2)), np.zeros((S, M), dtype=np.int8)
    series = {k: [] for k in ("density_error", "occupancy", "cohesion", "centroid_error", "polarization",
                              "active_fraction", "dog_speed")}
    hold_release = hold_standoff = None
    if Hp.hold_radius is not None:
        hold_release = 2 * Hp.hold_radius if Hp.hold_release is None else Hp.hold_release
        hold_standoff = F.r_dog + 3.0 if Hp.hold_standoff is None else Hp.hold_standoff

    def record(k, t, active, vd):
        t_out[k] = t
        P_s[k], H_s[k], P_d[k], MODE[k] = pos_s, hd_s, pos_d, mode
        if G_out is not None:
            G_out[k] = c_now
        m = met.all(pos_s)
        for kk, vv in m.items():
            series[kk].append(vv)
        series["polarization"].append(float(np.linalg.norm(hd_s.mean(0))) if F.rule == "heading" else np.nan)
        series["active_fraction"].append(float(active.mean()))
        series["dog_speed"].append(float(np.linalg.norm(vd, axis=-1).mean()) if M else np.nan)

    record(0, 0.0, np.zeros(N, bool), np.zeros((M, 2)))
    k_out = 1
    for step in range(1, n_steps + 1):
        if goal is not None:  # planner: the goal for this step, read at the state's time
            c_now = np.asarray(goal((step - 1) * dt), float)
            if (c_now != shift_now).any():
                rho_bar_H, shift_now = shifted_design(c_now), c_now.copy()
        # ------------------------------------------------------------- followers (read state at t−1)
        if F.rule == "heading":
            d_att = rng.random((N, F.k_perceive))
            d_ali = rng.random((N, F.n_att))
            d_noise = rng.random(N)
            d_noise_d = rng.random(M) if Hp.controller == "strombom" else None

            r_ij = dom.wrap(pos_s[None, :, :] - pos_s[:, None, :])  # [i, j] = x_j − x_i
            dist = np.linalg.norm(r_ij, axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                u_ij = r_ij / dist[..., None]
            near = (dist < F.r_rep) & ~np.eye(N, dtype=bool)
            has_rep = near.any(-1)
            rep = -_unit(np.where(near[..., None], u_ij, 0).sum(1))

            if M:
                to_dog = dom.wrap(pos_d[None, :, :] - pos_s[:, None, :])  # [i, m]
                dd = np.linalg.norm(to_dog, axis=-1)
                sees_m = dd <= F.r_dog
                with np.errstate(invalid="ignore", divide="ignore"):
                    away = np.where(sees_m[..., None], -to_dog / dd[..., None], 0.0).sum(1)
                sees = sees_m.any(1)
            else:
                away, sees = np.zeros((N, 2)), np.zeros(N, bool)

            knn = np.argsort(dist, axis=-1, kind="stable")[:, 1 : F.k_perceive + 1]
            att = np.take_along_axis(knn, np.argsort(d_att, -1)[:, : F.n_att], -1)
            ali = np.take_along_axis(att, np.argsort(d_ali, -1)[:, : F.n_ali], -1)
            r_att = r_ij[np.arange(N)[:, None], att]
            s_att = _unit((r_att / (np.linalg.norm(r_att, axis=-1, keepdims=True) + EPS)).sum(1))
            s_ali = _unit(hd_s[ali].sum(1))
            th = 2 * np.pi * d_noise
            eta = np.stack([np.cos(th), np.sin(th)], -1)

            active_dir = (F.w_inertia * hd_s + F.w_rep * rep + F.w_dog * away + F.w_att * s_att
                          + F.e_noise * eta + F.w_ali * s_ali)
            grazing = F.w_inertia * hd_s + F.w_rep * rep
            if F.activation == "published":
                new_hd = np.where(sees[:, None], _unit(active_dir), _unit(grazing))
                moves = sees | has_rep
                step_len = np.where(moves, F.v, 0.0)
                if F.wander_speed > 0:  # extension: standing sheep wander instead
                    idle = ~moves
                    new_hd = np.where(idle[:, None], _unit(F.w_inertia * hd_s + F.e_noise * eta), new_hd)
                    step_len = np.where(idle, F.wander_speed, step_len)
                    moves = np.ones(N, bool)
                new_hd = np.where(moves[:, None], new_hd, hd_s)
            elif F.activation == "always":
                new_hd = _unit(active_dir)
                moves = np.ones(N, bool)
                step_len = np.full(N, F.v)
            elif F.activation == "contagion":
                new_hd = _unit(active_dir)
                s_nb = spd[ali].mean(1)
                step_len = np.where(sees, F.v, s_nb)
                moves = step_len > 0
                spd = step_len.copy()
            else:
                raise ValueError(F.activation)
            new_pos = pos_s + (step_len * dt)[:, None] * new_hd
            if F.D > 0:  # hybrid: positional Brownian noise on top of the heading step
                new_pos = new_pos + np.sqrt(2 * F.D * dt) * rng.standard_normal((N, 2))
            active = sees
        elif F.rule == "first_order":
            drift = np.zeros((N, 2))
            if M:
                rd0 = dom.wrap(pos_s[:, None] - pos_d[None])  # sheep − dog
                drift += Kf_fn(rd0).sum(1)
                active = (np.linalg.norm(rd0, axis=-1) <= F.L_dog).any(1)
            else:
                active = np.zeros(N, bool)
            if F.has_pair:
                rij0 = dom.wrap(pos_s[:, None] - pos_s[None])  # x_i − x_j
                push = Kg_fn(rij0)
                push[np.arange(N), np.arange(N)] = 0.0  # no self-interaction
                drift += push.sum(1)
            if F.population_scaling:
                drift *= 1.0 / (N + M)
            if F.ali_fo > 0:  # extension: align with neighbours' recent displacement rate
                rij0 = dom.wrap(pos_s[:, None] - pos_s[None])
                nb = (np.linalg.norm(rij0, axis=-1) <= F.r_ali) & ~np.eye(N, dtype=bool)
                cnt = np.maximum(nb.sum(1), 1)[:, None]
                drift += F.ali_fo * (nb[..., None] * last_disp[None] / dt).sum(1) / cnt
            new_pos = pos_s + drift * dt + np.sqrt(2 * F.D * dt) * rng.standard_normal((N, 2))
            new_hd = _unit(new_pos - pos_s)
            new_hd = np.where((new_hd == 0).all(-1, keepdims=True), hd_s, new_hd)
            last_disp = new_pos - pos_s
        else:
            raise ValueError(F.rule)

        # ------------------------------------------------------------- herders (read sheep at t−1)
        new_pos_d, new_vel_d = pos_d.copy(), vel_d.copy()
        if M and Hp.controller == "strombom":
            G = dom.centroid(pos_s)
            from_g = dom.wrap(pos_s - G)
            spread = np.linalg.norm(from_g, axis=-1)
            far = int(spread.argmax())
            d_sd = np.linalg.norm(dom.wrap(pos_s[None] - pos_d[:, None]), axis=-1)  # [m, i]
            min_d = d_sd.min(1)
            if Hp.hold_radius is not None:
                dist_t = np.linalg.norm(dom.wrap(G - c))
                holding = dist_t <= hold_release if holding else dist_t <= Hp.hold_radius
            slow = (min_d <= F.r_rep) & (not holding)
            need_collect = spread[far] >= Hp.l_sep and not holding
            away_t = dom.wrap(G - c)
            g_norm = np.linalg.norm(away_t)
            e_gt = away_t / g_norm if g_norm > 0 else np.array([1.0, 0.0])
            p_drive = c + (g_norm + Hp.l_drive) * away_t / g_norm if g_norm > 0 else G + Hp.l_drive * e_gt  # as herding_model.m
            offsets = Hp.spread_angle * (np.arange(M) - (M - 1) / 2)
            aims = G + _rotate(np.broadcast_to(dom.wrap(p_drive - G), (M, 2)), offsets)
            mode_now = np.full(M, DRIVE, dtype=np.int8)
            if need_collect:
                collector = int(d_sd[:, far].argmin())
                aims[collector] = G + (spread[far] + Hp.l_collect) * from_g[far] / spread[far]
                mode_now[collector] = COLLECT
            if holding:
                od = dom.wrap(pos_d - G)
                on = np.linalg.norm(od, axis=-1, keepdims=True)
                out_dir = np.where(on > 1e-9, od / np.maximum(on, 1e-9), e_gt)
                aims = G + hold_standoff * out_dir
                mode_now[:] = HOLD
            to_aim = dom.wrap(aims - pos_d)
            th_d = 2 * np.pi * d_noise_d if F.rule == "heading" else 2 * np.pi * rng.random(M)
            eta_d = np.stack([np.cos(th_d), np.sin(th_d)], -1)
            dog_dir = _unit(_unit(to_aim) + Hp.noise * eta_d)
            v_dog = np.full(M, Hp.v)
            if holding:
                dist_aim = np.linalg.norm(to_aim, axis=-1)
                dog_dir = to_aim / np.maximum(dist_aim, 1e-9)[:, None]
                v_dog = np.minimum(Hp.v, dist_aim / dt)
            step_d = np.where(slow[:, None], Hp.slow_step * dt * vel_d, (v_dog * dt)[:, None] * dog_dir)
            new_pos_d = pos_d + step_d
            new_vel_d = np.where(slow[:, None], vel_d, dog_dir)
            mode_now[slow] = SLOW
            mode = mode_now
        elif M and Hp.controller == "dilorenzo":
            # The herder flow does not read the sheep (feedforward), so sub-stepping it inside one follower
            # step is exact for Di Lorenzo's dynamics; substeps = 1 is the original single Euler step.
            n_sub = max(int(Hp.substeps), 1)
            h_sub = dt / n_sub
            y = pos_d
            for j in range(n_sub):
                if j:
                    y = dom.wrap(y)
                rho_H = grid.convolve(Kb_H, grid.deposit(y, np.full(M, w_dog)))
                e_H = rho_bar_H - rho_H
                Eh = np.fft.fft2(e_H)
                # ∇·w = −K_H e_H, ∇×w = 0  ⇒  ŵ = i k K_H ê / |k|²
                Wx = np.real(np.fft.ifft2(1j * grid.KX * Hp.K_H * Eh * inv_k2))
                Wy = np.real(np.fft.ifft2(1j * grid.KY * Hp.K_H * Eh * inv_k2))
                u = np.stack([Wx, Wy]) / np.maximum(rho_H, 1e-3 * design.rho_H.max())
                u_d = grid.sample(u, y).T  # (M, 2)
                if np.isfinite(Hp.v):
                    sp = np.linalg.norm(u_d, axis=-1, keepdims=True)
                    u_d = np.where(sp > Hp.v, u_d * Hp.v / np.maximum(sp, 1e-12), u_d)
                y = y + u_d * h_sub
            new_pos_d = y
            new_vel_d = u_d if n_sub == 1 else dom.wrap(y - pos_d) / dt  # mean velocity over the step
            mode = np.full(M, FLOW, dtype=np.int8)
        elif M and Hp.controller == "static_ring":
            new_vel_d = np.zeros((M, 2))
        elif M and Hp.controller == "density_feedback":
            if Hp.fb_release is not None:
                err_now = met.density_error(pos_s)
                holding = err_now < Hp.fb_engage if holding else err_now < Hp.fb_release
            aims, mode_now, driving = feedback_aims(pos_s, pos_d, c, met.surplus(pos_s), Hp, dom, holding,
                                                    F.r_dog if F.rule == "heading" else F.L_dog, met.peak, driving)
            d_sd = np.linalg.norm(dom.wrap(pos_s[None] - pos_d[:, None]), axis=-1)
            slow = (d_sd.min(1) <= F.r_rep) & (mode_now != HOLD)
            to_aim = dom.wrap(aims - pos_d)
            dist_aim = np.linalg.norm(to_aim, axis=-1)
            dog_dir = to_aim / np.maximum(dist_aim, 1e-9)[:, None]
            v_dog = np.minimum(Hp.v, dist_aim / dt)  # stop at the aim
            step_d = np.where(slow[:, None], Hp.slow_step * dt * _unit(vel_d), (v_dog * dt)[:, None] * dog_dir)
            new_pos_d = pos_d + step_d
            new_vel_d = step_d / dt
            mode_now[slow] = SLOW
            mode = mode_now

        pos_s, hd_s = dom.wrap(new_pos), new_hd
        pos_d, vel_d = dom.wrap(new_pos_d), new_vel_d
        if step % store_every == 0 and k_out < S:
            record(k_out, step * dt, active, vel_d if Hp.controller != "strombom" else vel_d * Hp.v)
            k_out += 1

    series = {k: np.array(v[:k_out]) for k, v in series.items()}
    return BridgeResult(F, Hp, T, dom, t_out[:k_out], P_s[:k_out], H_s[:k_out], P_d[:k_out], MODE[:k_out],
                        series, met.floor, design, None if G_out is None else G_out[:k_out])


# ---------------------------------------------------------------------------- presets

def preset(name):
    """Named configurations. Each is (Followers, Herders, Target, Domain, run kwargs).

    Reference configurations reproduce a paper's own setting; the others are labelled
    adaptations or hybrids that change several components at once on purpose.
    """
    if name == "jadhav_reference":
        # herding_model.m / simulation_hm.m: the published model, start 250 m from the target.
        return (Followers(), Herders("strombom", n=1), Target("gaussian", sigma=2.5),
                Domain(False), dict(init="jadhav", box_length=250.0, n_steps=370))
    if name == "jadhav_modified":
        # herding_model_sim.m / simulation_hm_sim.m (Supp. Sec. 3): contagion speeds, r_dog 3, 8 of 10
        # neighbours attract, 2 align, drive point 2 m behind.
        return (Followers(activation="contagion", r_dog=3.0, n_att=8, n_ali=2),
                Herders("strombom", n=1, l_drive=2.0), Target("gaussian", sigma=2.5),
                Domain(False), dict(init="jadhav", box_length=250.0, n_steps=500))
    if name == "dilorenzo_reference":
        # arXiv:2411.04791 Sec. IV-A: [−π, π]², 720 targets + 280 herders, D = 0.01, L = π, K_H = 10,
        # von Mises goal k = 6/π with r* = π/2, dt = 0.01, T = 200, bandwidth 0.4.
        return (Followers("first_order", n=720, dt=0.01, D=0.01, a_dog=1.0, L_dog=np.pi, population_scaling=True),
                Herders("dilorenzo", n=280, v=np.inf, K_H=10.0, bandwidth=0.4, grid=64),
                Target("vonmises", k=6 / np.pi, r_goal=np.pi / 2, bandwidth=0.3),
                Domain(True, 2 * np.pi, images=1), dict(init="uniform", n_steps=20000))
    if name == "hybrid_jadhav_sheep_dilorenzo_dogs":
        # LABELLED HYBRID: Jadhav heading sheep (always active, plus Brownian D = 0.5 m²/s so the design has
        # a diffusion to confine) on an 80 m torus with their von Mises goal (r* = 10 m) and Di Lorenzo
        # herders designed for a weak push 0.8 e^(−r/12 m) m/s, which these sheep do not have.
        return (Followers("heading", n=60, activation="always", a_dog=0.8, L_dog=12.0, population_scaling=False, D=0.5),
                Herders("dilorenzo", n=16, v=1.5, K_H=1.0, grid=64),
                Target("vonmises", k=3 / (2 * np.pi * 10 / 80), r_goal=10.0, bandwidth=1.5), Domain(True, 80.0),
                dict(init="uniform", n_steps=600))
    if name == "hybrid_first_order_sheep_strombom_dogs":
        # LABELLED HYBRID: Di Lorenzo sheep in metres (push a e^(−r/L), Morse cohesion, D) driven by
        # Jadhav's closed-loop dog(s) on the open plane.
        return (Followers("first_order", n=50, dt=0.1, D=0.12, a_dog=2.3, L_dog=5.0, g_rep=1.0, l_rep=1.0,
                          g_att=0.2, l_att=4.0, population_scaling=False),
                Herders("strombom", n=1, v=1.5), Target("gaussian", sigma=3.0), Domain(False),
                dict(init="spread", box_length=8.0, n_steps=2000))
    if name == "dilorenzo_jadhav_sheep":
        return dilorenzo_scenario("jadhav")
    if name == "dilorenzo_metres":
        return dilorenzo_scenario("first_order")
    raise KeyError(name)


def dilorenzo_scenario(sheep="jadhav", n_sheep=36, n_dogs=None, scale=None, time_unit=1.0, images=1,
                       v_dogs=np.inf, **sheep_kw):
    """Di Lorenzo et al. 2024 (2411.04791, Sec. IV-A) in metres, herders unchanged, followers chosen.

    Everything that belongs to the scenario and the herders is theirs: the periodic box of side 2π, the
    von Mises goal k = 6/π with r* = π/2, the herder mass fraction 280/1000, uniform starts, K_H = 10,
    KDE bandwidth 0.4, uncapped herder speed, their dt = 0.01 for the herder flow, T = 200, and the
    feedforward design computed from THEIR follower model (push f = e^(−r/L), L = π, D = 0.01,
    α = 1/(N_T + N_H)). Only the followers change:

      sheep = 'jadhav'       Jadhav's heading rule at the published baseline (Followers() defaults: n_att 4,
                             e 0.3, r_dog 12 m, published activation, no positional noise). The design's
                             follower model is then what the herders believe, not what the sheep do.
      sheep = 'first_order'  their own follower, in metres (the reference: design and sheep agree).

    Units: `scale` metres per their length unit (default r_dog/π, so their push range L = π is the
    sheep's notice radius, 12 m, and the box is 24 m); `time_unit` seconds per their time unit (default
    1 s, so the published sheep's 1 m/s step is one follower step and the herder flow runs 100 sub-steps
    of 0.01 s). The design's shape and mass fraction do not depend on either: v̄ ∝ D and the push both
    scale as length/time, so M̂ is the same 0.241 (images = 1, grid 64) as in their units.

    n_sheep defaults to 36, not 720: 720 sheep with a 2 m repulsion radius do not fit a 24 m box. The
    herder count keeps their fraction, n_dogs = round(n_sheep · 280/720) (14 for 36).
    """
    s = (Followers(**sheep_kw).r_dog / np.pi) if scale is None else scale
    tau = time_unit
    n_dogs = int(round(n_sheep * 280 / 720)) if n_dogs is None else n_dogs
    model = Followers("first_order", n=n_sheep, dt=0.1 * tau, D=0.01 * s ** 2 / tau, a_dog=s / tau,
                      L_dog=np.pi * s, population_scaling=True)
    if sheep == "jadhav":
        F = Followers(n=n_sheep, **sheep_kw)
    elif sheep == "first_order":
        F = replace(model, **sheep_kw)
    else:
        raise ValueError(sheep)
    H = Herders("dilorenzo", n=n_dogs, v=v_dogs, K_H=10.0 / tau, bandwidth=0.4 * s, grid=64,
                design_model=model, substeps=max(int(round(F.dt / (0.01 * tau))), 1))
    T = Target("vonmises", k=6 / np.pi, r_goal=np.pi / 2 * s, bandwidth=0.3 * s)
    dom = Domain(True, 2 * np.pi * s, images=images)
    return F, H, T, dom, dict(init="uniform", n_steps=int(round(200 * tau / F.dt)))


PRESETS = ("jadhav_reference", "jadhav_modified", "dilorenzo_reference", "hybrid_jadhav_sheep_dilorenzo_dogs",
           "hybrid_first_order_sheep_strombom_dogs", "dilorenzo_jadhav_sheep", "dilorenzo_metres")
