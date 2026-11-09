"""Goal schedules: a planner above Di Lorenzo's herders that only decides where the goal centre is.

The sheep (Jadhav's published rules) and the herders (Di Lorenzo's design and flow) are untouched;
`bridge.simulate_bridge(..., goal=schedule)` translates the precomputed herder design to the
schedule's centre once per follower step inside one continuous simulation.

The requested centre distribution p_c comes from the fixed-shape approximation: if the flock keeps a
shape S about its centroid, its time-averaged density is S ∗ (centroid occupation), so a target ρ̄ asks
for p_c = ρ̄ ⊘ S. A non-negative, normalized p_c is a distributional feasibility condition under that
approximation only; realizing it also needs the flock to track the goal safely.
"""

import numpy as np


def flock_shape(positions, dom, grid):
    """Mean density of the sheep about their own (circular-mean) centroid, unit mass on the grid."""
    acc = np.zeros((grid.n, grid.n))
    for x in positions:
        acc += grid.deposit(dom.wrap(x - dom.centroid(x)), np.full(len(x), 1.0 / len(x)))
    S = acc / len(positions)
    return S / (S.sum() * grid.h ** 2)


def centre_distribution(S, T, dom, grid, reg=1e-2):
    """p_c = ρ̄ ⊘ S by Tikhonov-regularized deconvolution on the grid.

    Returns (cell probabilities clipped at 0 and renormalized, negative mass share before clipping).
    The negative share is the feasibility check: near 0 means a non-negative p_c exists for this S.
    """
    h2 = grid.h ** 2
    rho, _ = grid.target(T, dom)
    rho = rho / (rho.sum() * h2)
    Sh = np.fft.fft2(np.fft.ifftshift(S)) * h2
    P = np.real(np.fft.ifft2(np.fft.fft2(rho) * np.conj(Sh) / (np.abs(Sh) ** 2 + reg * np.abs(Sh).max() ** 2)))
    neg = float(-P[P < 0].sum() / P[P > 0].sum())
    P = np.clip(P, 0, None)
    return P / P.sum(), neg


def draw_waypoints(P, grid, n, rng):
    """n independent centres from the cell probabilities P, uniform within the drawn cell."""
    cells = np.stack([grid.X.ravel(), grid.Y.ravel()], -1)
    idx = rng.choice(len(cells), size=n, p=P.ravel())
    return cells[idx] + (rng.random((n, 2)) - 0.5) * grid.h


class FixedGoal:
    def __init__(self, centre=(0.0, 0.0)):
        self.c = np.asarray(centre, float)

    def __call__(self, t):
        return self.c


class HopSchedule:
    """The goal sits at waypoint i for t ∈ [i·dwell, (i+1)·dwell) and jumps to the next."""

    def __init__(self, waypoints, dwell):
        self.w, self.dwell = np.asarray(waypoints, float), float(dwell)

    def __call__(self, t):
        return self.w[min(int(t // self.dwell), len(self.w) - 1)]


class GlideSchedule:
    """Same waypoints and dwell, but the goal moves at constant speed from waypoint i to i+1 over
    [i·dwell, (i+1)·dwell) along the shortest path on the torus: continuous, never jumps."""

    def __init__(self, waypoints, dwell, dom):
        self.w, self.dwell, self.dom = np.asarray(waypoints, float), float(dwell), dom

    def __call__(self, t):
        i = min(int(t // self.dwell), len(self.w) - 2)
        f = min(max((t - i * self.dwell) / self.dwell, 0.0), 1.0)
        return self.dom.wrap(self.w[i] + f * self.dom.wrap(self.w[i + 1] - self.w[i]))
