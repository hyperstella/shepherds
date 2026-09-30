"""Nonparametric interaction laws learned from trajectories.

A second-order interacting-particle model for the sheep, with the dog as an
external agent:

    dv_i/dt = a1 v_i + a2 |v_i|^2 v_i                     self-propulsion / friction
            + sum_j phi_E(r_ij) (x_j - x_i) / r_ij         attraction (>0) / repulsion (<0)
            + sum_j phi_A(r_ij) (v_j - v_i)                alignment
            + phi_D(r_iD) (x_i - x_D) / r_iD               flight from the dog (>0)
            [+ psi_V(r_iD) v_D                             "pressure": moving with the dog, optional]
            [+ psi_L(r_iD) max(0, looming) (x_i - x_D)/r_iD  flight scaled by approach speed, optional]
            [- gamma v_i + chi(r_iD) v_i/|v_i|             speed relaxing to a dog-dependent preferred speed, optional
                                                           (terms "fric" and "S"; use instead of "self")]
            + sigma * dW/dt

Each kernel phi is a sum of piecewise-linear "hat" functions on fixed knots
and vanishes beyond its last knot, so the model is linear in its
coefficients (in the spirit of Lu, Maggioni & Tang, PNAS 2019).

Fitting uses an integral (weak) form that avoids differentiating noisy
positions twice: over each window of H seconds,
    v_i(t+H) - v_i(t) = integral_t^{t+H} F_i ds,
where velocities come from a Savitzky-Golay filter and the integral of the
features uses the trapezoid rule. Normal equations are accumulated per drive,
so leave-one-session-out cross-validation needs no refitting of features.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from . import data

N = data.N_SHEEP


@dataclass(frozen=True)
class HatBasis:
    """Hat functions peaking at knots[:-1]; the kernel is 0 from knots[-1] on."""

    knots: tuple

    @property
    def size(self):
        return len(self.knots) - 1

    @property
    def peaks(self):
        return np.asarray(self.knots[:-1], float)

    def __call__(self, r):
        k = np.asarray(self.knots, float)
        r = np.asarray(r, float)[..., None]
        peak = k[:-1]
        left = np.concatenate([[k[0] - (k[1] - k[0])], k[:-2]])
        right = k[1:]
        up = (r - left) / (peak - left)
        down = (right - r) / (right - peak)
        return np.clip(np.minimum(up, down), 0.0, None)

    def curve(self, coef, r):
        return self(r) @ coef


ORDER = ("self", "fric", "S", "E", "A", "D", "DV", "DL")
PAIR_KNOTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
DOG_KNOTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0)


@dataclass(frozen=True)
class KernelModel:
    """Which terms are present and how kernels are represented.

    terms: subset of ("self", "fric", "S", "E", "A", "D", "DV", "DL"); topo_k: if set, pair
    sums run only over each sheep's topo_k nearest neighbours (topological
    interactions). DV and DL need the dog's velocity.
    """

    terms: tuple = ("self", "E", "A", "D")
    pair: HatBasis = field(default_factory=lambda: HatBasis(PAIR_KNOTS))
    dog: HatBasis = field(default_factory=lambda: HatBasis(DOG_KNOTS))
    topo_k: int | None = None

    def slices(self):
        sizes = {"self": 2, "fric": 1, "S": self.dog.size, "E": self.pair.size, "A": self.pair.size,
                 "D": self.dog.size, "DV": self.dog.size, "DL": self.dog.size}
        out, i = {}, 0
        for t in ORDER:
            if t in self.terms:
                out[t] = slice(i, i + sizes[t])
                i += sizes[t]
        return out

    @property
    def n_coef(self):
        return max(s.stop for s in self.slices().values())

    def features(self, pos, vel, dog, dog_vel=None):
        """pos, vel: (..., N, 2); dog, dog_vel: (..., 2). Returns (..., N, 2, P)."""
        n = pos.shape[-2]
        parts = []
        if "self" in self.terms:
            sp2 = (vel ** 2).sum(-1, keepdims=True)
            parts += [vel[..., None], (sp2 * vel)[..., None]]
        if "fric" in self.terms:
            parts.append(vel[..., None])
        if "S" in self.terms:
            dd = np.linalg.norm(pos - dog[..., None, :], axis=-1)
            sp = np.linalg.norm(vel, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                head = np.where(sp > 1e-6, vel / sp, 0.0)
            parts.append(self.dog(dd)[..., None, :] * head[..., :, None])
        if "E" in self.terms or "A" in self.terms:
            rij = pos[..., None, :, :] - pos[..., :, None, :]  # [i, j] = x_j - x_i
            dist = np.linalg.norm(rij, axis=-1)
            eye = np.eye(n, dtype=bool)
            mask = ~eye
            if self.topo_k is not None:
                order = np.argsort(np.where(eye, np.inf, dist), axis=-1)
                ranks = np.argsort(order, axis=-1)
                mask = mask & (ranks < self.topo_k)
            B = self.pair(dist) * mask[..., None]  # (..., N, N, K)
            if "E" in self.terms:
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.where(mask[..., None], rij / dist[..., None], 0.0)
                parts.append(np.einsum("...ijk,...ijc->...ick", B, e))
            if "A" in self.terms:
                dv = vel[..., None, :, :] - vel[..., :, None, :]  # [i, j] = v_j - v_i
                parts.append(np.einsum("...ijk,...ijc->...ick", B, dv))
        if {"D", "DV", "DL"} & set(self.terms):
            rd = pos - dog[..., None, :]
            dd = np.linalg.norm(rd, axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                e = rd / dd[..., None]  # unit vector from the dog to the sheep
            Bd = self.dog(dd)  # (..., N, K)
            if "D" in self.terms:
                parts.append(Bd[..., None, :] * e[..., :, None])
            if "DV" in self.terms:
                dv = np.broadcast_to(dog_vel[..., None, :], pos.shape)
                parts.append(Bd[..., None, :] * dv[..., :, None])
            if "DL" in self.terms:
                closing = np.maximum(0.0, ((dog_vel[..., None, :] - vel) * e).sum(-1))  # dog closing in, m/s
                parts.append((Bd * closing[..., None])[..., None, :] * e[..., :, None])
        return np.concatenate(parts, axis=-1)

    def penalty(self):
        """Second-difference smoothness penalty on each kernel's coefficients."""
        P = np.zeros((self.n_coef, self.n_coef))
        for t, sl in self.slices().items():
            if t in ("self", "fric"):
                continue
            m = sl.stop - sl.start  # smooth in distance for every kernel
            D = np.diff(np.eye(m), n=2, axis=0)
            P[sl, sl] = D.T @ D
        return P


# ------------------------------------------------------------------ data prep

def smooth_drive(pos, dt=data.DT, window_s=1.1, poly=2):
    """Savitzky-Golay positions and velocities for (n, 2, T) tracks."""
    w = max(poly + 2, int(round(window_s / dt)) | 1)
    w = min(w, pos.shape[-1] - (1 - pos.shape[-1] % 2))
    if w <= poly:
        return pos.copy(), np.gradient(pos, dt, axis=-1)
    return savgol_filter(pos, w, poly, axis=-1), savgol_filter(pos, w, poly, deriv=1, delta=dt, axis=-1)


@dataclass
class Normal:
    """Accumulated normal equations of one drive."""

    G: np.ndarray
    b: np.ndarray
    yy: float
    n: int
    session: int
    label: str


def weak_form(model, pos, vel, dt=data.DT, H=0.5, stride=0.2):
    """Rows of the integral-form regression for one drive.

    pos, vel: (N+1, 2, T) smoothed tracks (dog last). Returns X (rows, P), y (rows,).
    """
    h, s = int(round(H / dt)), max(1, int(round(stride / dt)))
    x = np.moveaxis(pos, -1, 0)  # (T, N+1, 2)
    v = np.moveaxis(vel, -1, 0)
    F = model.features(x[:, :N], v[:, :N], x[:, N], v[:, N])  # (T, N, 2, P)
    C = cumulative_trapezoid(F, dx=dt, axis=0, initial=0)
    t0 = np.arange(0, x.shape[0] - h, s)
    X = C[t0 + h] - C[t0]  # (n_t, N, 2, P)
    y = v[t0 + h, :N] - v[t0, :N]  # (n_t, N, 2)
    ok = np.isfinite(X).all(-1) & np.isfinite(y)
    return X[ok], y[ok]


def normal_equations(model, drives, H=0.5, stride=0.2, window_s=1.1, tracks=None):
    """Per-drive normal equations. `tracks` may supply (pos, vel) per drive
    (e.g. synthetic data); otherwise the drives' positions are smoothed."""
    out = []
    for i, d in enumerate(drives):
        pos, vel = tracks[i] if tracks is not None else smooth_drive(d.pos[: N + 1], window_s=window_s)
        X, y = weak_form(model, pos, vel, H=H, stride=stride)
        out.append(Normal(X.T @ X, X.T @ y, float(y @ y), len(y), d.session, f"s{d.session}d{d.index + 1}"))
    return out


def solve(model, normals, lam, ridge=1e-6):
    """Penalized least squares; lam is relative to the mean diagonal of G."""
    G = sum(n.G for n in normals)
    b = sum(n.b for n in normals)
    scale = np.trace(G) / G.shape[0]
    A = G + lam * scale * model.penalty() + ridge * scale * np.eye(G.shape[0])
    return np.linalg.solve(A, b)


def sse(normals, c):
    return sum(n.yy - 2 * c @ n.b + c @ n.G @ c for n in normals)


def cross_validate(model, normals, lams=np.logspace(-4, 2, 13)):
    """Leave-one-session-out CV of the penalty weight. Returns (best lam, table)."""
    sessions = sorted({n.session for n in normals})
    rows = []
    for lam in lams:
        err, tot = 0.0, 0.0
        for s in sessions:
            train = [n for n in normals if n.session != s]
            test = [n for n in normals if n.session == s]
            c = solve(model, train, lam)
            err += sse(test, c)
            tot += sum(n.yy for n in test)
        rows.append((float(lam), float(1 - err / tot)))
    best = max(rows, key=lambda r: r[1])[0]
    return best, rows


def noise_level(normals, c, H=0.5):
    """sigma such that the residual of v(t+H) - v(t) has variance sigma^2 H per component."""
    n = sum(nn.n for nn in normals)
    return float(np.sqrt(max(sse(normals, c), 0) / n / H))


def coercivity(model, normals):
    """Smallest eigenvalue of each kernel's normalized Gram block (identifiability)."""
    G = sum(n.G for n in normals)
    out = {}
    for t, sl in model.slices().items():
        g = G[sl, sl]
        d = np.sqrt(np.diag(g))
        with np.errstate(invalid="ignore", divide="ignore"):
            gn = g / np.outer(d, d)
        ev = np.linalg.eigvalsh(np.nan_to_num(gn))
        out[t] = float(ev.min())
    return out


def kernel_curves(model, c, r_pair=np.linspace(0, 8, 161), r_dog=np.linspace(0, 20, 201)):
    sl = model.slices()
    out = {}
    for t in ("self", "fric"):
        if t in sl:
            out[t] = c[sl[t]].tolist()
    for t in ("E", "A"):
        if t in sl:
            out[t] = (r_pair, model.pair.curve(c[sl[t]], r_pair))
    for t in ("S", "D", "DV", "DL"):
        if t in sl:
            out[t] = (r_dog, model.dog.curve(c[sl[t]], r_dog))
    return out


# ------------------------------------------------------------------ simulation

def simulate(model, c, sigma, pos0, vel0, dog_path, dt=data.DT, seed=None, v_max=4.0, speed_path=None):
    """Euler-Maruyama integration with the dog path replayed.

    pos0, vel0: (R, N, 2); dog_path: (2, T); c: (P,) or (R, P); sigma: scalar or
    (R,). If speed_path (T,) is given, all
    sheep velocities are rescaled each step so the flock's mean speed follows
    it (the flock's overall speed is then an external input, like the dog).
    Returns pos, vel of shape (R, N+1, 2, T) with the dog last.
    """
    rng = np.random.default_rng(seed)
    R, T = pos0.shape[0], dog_path.shape[-1]
    c = np.asarray(c, float)  # (P,) shared or (R, P) per run
    sig = np.broadcast_to(np.asarray(sigma, float), (R,))[:, None, None]
    P = np.empty((R, N + 1, 2, T))
    V = np.empty_like(P)
    x, v = pos0.astype(float).copy(), vel0.astype(float).copy()
    dog_v = np.gradient(dog_path, dt, axis=-1)
    for t in range(T):
        P[:, :N, :, t], V[:, :N, :, t] = x, v
        P[:, N, :, t], V[:, N, :, t] = dog_path[:, t], dog_v[:, t]
        if t == T - 1:
            break
        feats = model.features(x, v, np.broadcast_to(dog_path[:, t], (R, 2)), np.broadcast_to(dog_v[:, t], (R, 2)))
        F = feats @ c if c.ndim == 1 else np.einsum("rnkp,rp->rnk", feats, c)
        v = v + dt * F + sig * np.sqrt(dt) * rng.standard_normal(v.shape)
        sp = np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.where(sp > v_max, v * v_max / sp, v)
        if speed_path is not None:
            mean_sp = np.linalg.norm(v, axis=-1).mean(-1)[:, None, None]
            v = v * (speed_path[t + 1] / np.maximum(mean_sp, 1e-9))
        x = x + dt * v
    return P, V


# ------------------------------------------------------------------ synthetic data

class Piece:
    """A stretch of track used like a drive (session + label) in fitting."""

    def __init__(self, session, index, pos):
        self.session, self.index, self.pos = session, index, pos


def uwb_noise(pos, sigma, rng, dt=data.DT, rate_hz=2.0):
    """Tag-like measurement noise: independent errors at `rate_hz`, linearly
    interpolated to the 10 Hz sampling (as UWB fixes at 2 Hz are)."""
    T = pos.shape[-1]
    step = int(round(1 / (rate_hz * dt)))
    knots = np.arange(0, T + step, step)
    err = rng.normal(0, sigma, (*pos.shape[:-1], len(knots)))
    t = np.arange(T)
    flat = err.reshape(-1, len(knots))
    interp = np.stack([np.interp(t, knots, row) for row in flat]).reshape(*pos.shape[:-1], T)
    return pos + interp


def synthetic_pieces(model, c, sigma, drives, horizon=15.0, seed=0, meas_sigma=0.1, window_s=2.1, speed_replay=False):
    """Synthetic flocks with known kernels, driven by the real dog paths.

    Each drive is cut into `horizon`-second pieces; each piece starts from the
    real (smoothed) sheep state, runs the model with the real dog replayed,
    then gets tag-like noise and the same smoothing as real data. With
    speed_replay the flock's mean speed follows the real one (as in the
    forward simulations), which the fitted model does not describe.
    Returns (pieces, tracks) for `normal_equations(..., tracks=...)`.
    """
    rng = np.random.default_rng(seed)
    L = int(round(horizon / data.DT))
    pieces, tracks = [], []
    for d in drives:
        pos, vel = smooth_drive(d.pos[: N + 1], window_s=window_s)
        T = pos.shape[-1]
        for s in range(0, max(1, T - L // 2), L):
            e = min(T, s + L)
            if e - s < 20:
                continue
            speed = np.linalg.norm(vel[:N, :, s:e], axis=1).mean(0) if speed_replay else None
            P, _ = simulate(model, c, sigma, pos[None, :N, :, s], vel[None, :N, :, s], pos[N, :, s:e],
                            seed=int(rng.integers(1 << 31)), speed_path=speed)
            noisy = uwb_noise(P[0], meas_sigma, rng)
            pieces.append(Piece(d.session, len(pieces), noisy))
            tracks.append(smooth_drive(noisy, window_s=window_s))
    return pieces, tracks


def curve_error(model, c_true, c_hat, weights):
    """Relative L2 error of each kernel, weighted by where data actually lie.

    weights: {"pair": (r, density), "dog": (r, density)}.
    """
    out = {}
    sl = model.slices()
    for t, s in sl.items():
        if t in ("self", "fric"):
            out[t] = float(np.linalg.norm(c_hat[s] - c_true[s]) / (np.linalg.norm(c_true[s]) + 1e-12))
            continue
        basis, (r, w) = (model.pair, weights["pair"]) if t in ("E", "A") else (model.dog, weights["dog"])
        ft, fh = basis.curve(c_true[s], r), basis.curve(c_hat[s], r)
        out[t] = float(np.sqrt(np.sum(w * (fh - ft) ** 2) / np.sum(w * ft ** 2)))
    return out


def distance_weights(drives, r_pair=np.linspace(0, 8, 161), r_dog=np.linspace(0, 20, 201)):
    """Empirical densities of sheep-sheep and sheep-dog distances in the drives."""
    pair, dog = [], []
    for d in drives:
        x = d.pos[:N]
        dist = np.linalg.norm(x[:, None] - x[None], axis=2)
        pair.append(dist[~np.eye(N, dtype=bool)].ravel())
        dog.append(np.linalg.norm(x - d.pos[N][None], axis=1).ravel())
    hp, _ = np.histogram(np.concatenate(pair), bins=np.append(r_pair, r_pair[-1] + (r_pair[1] - r_pair[0])), density=True)
    hd, _ = np.histogram(np.concatenate(dog), bins=np.append(r_dog, r_dog[-1] + (r_dog[1] - r_dog[0])), density=True)
    return {"pair": (r_pair, hp), "dog": (r_dog, hd)}


def apply_gains(model, c, gains):
    """Per-run coefficients: each kernel's learned shape times a gain.

    gains: {term: (R,) array}, e.g. {"E": ..., "D": ...}; missing terms keep gain 1.
    Returns (R, P).
    """
    R = len(next(iter(gains.values())))
    out = np.tile(np.asarray(c, float), (R, 1))
    for t, sl in model.slices().items():
        if t in gains:
            out[:, sl] *= np.asarray(gains[t], float)[:, None]
    return out
