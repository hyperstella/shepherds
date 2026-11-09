# Josef — Part III: from agents to densities

This folder reproduces **the two figures in Josef's Applied Mathematics Senior Seminar presentation** without relying on any other folders, external data files, Jupyter, internet APIs, videos, LaTeX, or the rest of the group's code. The source numerical kernel in `model.py` is **extracted unchanged** from the group's `part3_continuification.py` (Torus2D, target inversion, herder feedback, and stochastic time stepping), while the two figure scripts supply the plotting and analysis.

## Requirements

- **Python 3.12 or 3.13** (64-bit). Tested on CPython 3.13.5.
- Windows, macOS, or Linux; no GPU required.
- Package versions pinned in [`requirements.txt`](requirements.txt): NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8.

Clone this repository and from its root run:

```bash
git clone https://github.com/hyperstella/shepherds.git
cd shepherds
python -m venv .venv
```

Activate the environment (choose one):

```bash
# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r josef/requirements.txt
```

If PowerShell refuses activation, you can instead invoke `.venv\Scripts\python.exe` directly in the commands below (or use `py -3.12 -m venv .venv` to select Python 3.12).

## Reproduce both slides (one command)

```bash
python josef/reproduce_all.py
```

This generates **both** figures plus the reproducibility metrics in `josef/output/`.
A successful full run reports `138` sheep initially and `577` sheep finally inside the goal, with a theoretical target of `79.6905%`. You can also run each figure separately, as below.

## Reproduce the two slides

### Figure 1 — Visualizing our density design

```bash
python josef/analytic_density.py
```

Creates:
- `josef/output/josef_analytic_illustration_figure.pdf`
- `josef/output/josef_analytic_illustration_figure.png`

Three panels: (a) product von Mises sheep target, (b) analytically inverted herder density, and (c) required inward drift. **This is an analytic formula visualization, not an agent simulation.** Parameters: periodic box `[-π,π]²`, sheep `N_T = 720`, herders `N_H = 300`, `D = 0.01`, `L = π`, `k = 6/π`, goal disk radius `π/2`. The panel colors are normalized to the maximum for readability. Since `300` is above the feasibility threshold `277`, the analytical herder density remains positive.

### Figure 2 — 2D agents: reproducible shepherding simulation

```bash
python josef/simulate_agents.py
```

Creates:
- `josef/output/reproduced_2d_agents.pdf`
- `josef/output/reproduced_2d_agents.png`
- `josef/output/reproduce_2d_agent_metrics.csv` (time series)
- `josef/output/agent_metrics.json` (parameters, results, software versions)

Simulates **720 sheep and 277 herders** in the 2D periodic box with `D = 0.01`, `L = π`, `k = 6/π`, `K_H = 10`, Gaussian-KDE bandwidth `0.4`, 64×64 Fourier grid, Euler time step `0.01`, final time `200`, herder speed cap `3`, and NumPy random seed `0`. It takes longer than the analytic visualization, since it makes **20,000 time steps**. Runtime varies with CPU.

Expected with the pinned software versions:

| Quantity | Expected result |
|---|---:|
| Sheep in goal at `t=0` | `138 / 720 = 19.17%` |
| Sheep in goal at `t=200` | `577 / 720 = 80.14%` |
| Exact von Mises goal-disk fraction | `79.6904786%` |
| Minimum continuum herder mass | approximately `0.27738` |
| Actual herder mass for 277/997 | approximately `0.27783` |

The dotted line in the right-hand plot is **not** fitted to the agent simulation: it is computed by integrating the *prescribed target density* over the goal disk. Near agreement of occupancy does **not** establish convergence of the full sheep density. This is one seeded stochastic run, not a guarantee for every realization.

The numerical controller estimates the herder density from discrete agents (cloud-in-cell deposition + Gaussian Fourier smoothing), solves a periodic Poisson problem by FFT, and evaluates the control field at agent positions. The continuum proof applies to ideal densities; this code adds finite-agent, smoothing, discretization, and capped-speed approximations.

## Reproducibility checks

Fast test (before the long simulation):

```bash
python josef/test_repro.py
python josef/simulate_agents.py --quick  # smoke test only, t=2
```

The `--quick` option **cannot** reproduce the `t=200` slide; remove it for the final result. To use another destination:

```bash
python josef/simulate_agents.py --output-dir ~/sheep-reproduction
```

No file elsewhere in the repository is overwritten. Output files are generated under `josef/output/` by default and do not need to be checked into Git. The source scripts use paths relative to their own location and Matplotlib's noninteractive `Agg` renderer, so they work when run from another working directory or a remote/headless computer.

## Files

| File | Purpose |
|---|---|
| [`reproduce_all.py`](reproduce_all.py) | One-command reproduction for both plots |
| [`analytic_density.py`](analytic_density.py) | Produces the three-panel analytic density/drift figure |
| [`model.py`](model.py) | Unmodified numerical solver extracted from the seminar's original `part3_continuification.py` |
| [`simulate_agents.py`](simulate_agents.py) | Runs the full 2D agent experiment and produces plots + data |
| [`test_repro.py`](test_repro.py) | Checks threshold, exact target fraction, and short numerical run |
| [`requirements.txt`](requirements.txt) | Pinned dependencies |

### Mathematical references

- Di Lorenzo, Maffettone & di Bernardo, *A Continuification-Based Control Solution for Large-Scale Shepherding*, European Journal of Control 86 (2025), 101324 (arXiv:2411.04791).
- Maffettone, Boldini, Porfiri & di Bernardo, *Leader–Follower Density Control of Spatial Dynamics in Large-Scale Multiagent Systems*, IEEE TAC 70(10) (2025), 6783–6798.
