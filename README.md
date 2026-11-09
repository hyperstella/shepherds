# Shepherds: sheep-herding simulation research

This is a Python reproduction and extension of Jadhav et al. (2024), *Collective responses of
flocking sheep to a herding dog* (Communications Biology 7:1543). It includes their UWB tracking
data (14 sheep + dog + shepherd, 10 Hz, 29 drives) and their agent-based shepherding model.

See **[REPORT.md](REPORT.md)** for what was reproduced, where the model and data disagree, and a
prioritized list of improvements.

## Setup

Requires Python 3.12 or newer. The pinned NumPy and SciPy do not install on older versions.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python reproduce.py          # writes results/figures/*.png and results/summary.json
```

```bash
.venv/bin/python -m pytest tests
```

```python
from shepherd import data, metrics
from shepherd.model import Params, simulate

sessions = data.load_sessions()              # 9 trials, pos/vel (16, 2, T)
drives = data.load_drives(sessions)          # 29 hand-labelled drives
res = simulate(Params(n_sheep=50), n_runs=100, seed=0)   # vectorized over runs
pos, vel = res.run(0)                        # (N+1, 2, T), dog last
metrics.cohesion(pos, 50)
```

## Interactive viewer

`viz/sheep-and-collie.html` is a self-contained page with two views. The first replays the 29 real
drives, with leader links drawn on top. The second runs the model live with sliders and compares
its distributions against the real drives. `viz/sim.js` is a JavaScript port of the model and
matches the Python version's statistics.

```bash
.venv/bin/python viz/build.py      # rebuild after changing viz/app.html or viz/sim.js
```

## Jadhav ↔ Di Lorenzo bridge (interactive density experiment)

`viz/bridge.html` runs one simulator in which the followers (Jadhav's heading rule or Di
Lorenzo's first-order rule), the controller (Strömbom dog(s), Di Lorenzo's feedforward herders, a
density-feedback dog that reads the flock's density surplus, a static ring) and the density
objective are chosen separately, on the open plane or a periodic square. Every configuration that is not a paper's own is labelled (adaptation, extension, hybrid).
The Python reference is `shepherd/bridge.py`; the page is its port (`viz/bridge_sim.js`). Equations,
parameter mappings, the discrepancies between papers, code and this repository, and the
comparison results are in **[docs/bridge_simulation.md](docs/bridge_simulation.md)**.

```bash
.venv/bin/python -m http.server 8765 --directory viz    # then open http://localhost:8765/bridge.html
```

```bash
.venv/bin/python experiments/bridge_compare.py           # results/bridge_compare.json + figures (about 30 min; --quick for less)
```

```python
from shepherd import bridge as B
F, H, T, D, kw = B.preset("dilorenzo_reference")
B.design_herders(F, H, T, D).mass_needed                  # their minimum herder mass fraction
r = B.simulate_bridge(*B.preset("jadhav_reference")[:4], n_steps=369, seed=0)   # = model.simulate(Params(), seed=0)
```

## Layout

| Path | Contents |
|---|---|
| `shepherd/data.py` | loads `sheepR.dat`; ports `pre_ana.m`, `pre_ini_ana.m`, `drivs_dat.m` |
| `shepherd/metrics.py` | cohesion, polarization, elongation, flock frame, viewing angles, turning rates |
| `shepherd/network.py` | directional cross-correlation, leader–follower networks, d_i vs indegree |
| `shepherd/model.py` | the herding model: vectorized `simulate` (per-run parameters, dog/speed replay, dt/τ) plus a literal reference port |
| `shepherd/expected_heading.py` | each sheep's exact one-step expected heading (normalization, random neighbour choice and noise kept; no Monte Carlo), and the exact one-step change of M₂ |
| `shepherd/collecting.py` | the published dog's decision rule, and an exact bound on the expected change of a two-phase collecting potential (farthest sheep's distance plus the dog's distance to it) |
| `shepherd/protocol.py` | one analysis protocol for data and models: segments, summaries, sheep-level hierarchy, bootstrap, surrogates |
| `shepherd/scenario.py` | simulated drives matched to real ones: the real dog replayed in 15 s windows that restart from the real flock |
| `shepherd/abc.py` | NumPy ABC: Sobol reference tables, regression adjustment, coverage checks |
| `shepherd/kernels.py` | learned interaction kernels: integral-form least squares, CV, synthetic data, forward simulation |
| `shepherd/feasibility.py`, `shepherd/dog_design.py` | continuification with measured kernels: dog counts, cohesion stiffness, the deconvolution-designed single dog |
| `shepherd/density_control.py` | a Di Lorenzo-inspired density controller for one Jadhav dog: placed each step where the exact expected descent of a density-error functional (second moment about the goal, or the L² density error) is largest, then moved there at v_dog |
| `shepherd/phases.py` | rule-based draft phase labels (drive, turn, flank, idle, away) for any `Track`: positions, sampling interval, which row is the dog |
| `experiments/` | `protocol_check`, `replay_check`, `build_table`, `fit_kernels`, `compare_models`, `label_phases`, `label_simulations`, `pair_attraction` (see REPORT.md); `closed_loop_convergence`, `target_density`, `hold_strategies`, `designed_dog`, `cohesion_stiffness`, `residual_persistence` (Di Lorenzo's convergence and the Jadhav sheep, see docs/continuification_driving.md); `steer_and_shape`, `steering_gain`, `path_tracking`, `shape_drift`, `shape_stress`, `shape_trajectories`, `size_sweep`, `fission_threshold`, `fission_provenance` (can one dog steer and hold a shape, see docs/steer_and_shape.md); `tracking_bound` (mean-square bound on the centroid's path error, see docs/jadhav_dilorenzo_bridge.md §9); `restoring_condition` (deriving the restoring condition C4 from exact expected headings, see docs/jadhav_dilorenzo_bridge.md §10); `memory_inequality` (a drift function for M₂ that charges current headings, see docs/jadhav_dilorenzo_bridge.md §11); `certify_mi`, `mi_counterexamples`, `certify_d3` (a certificate for the memory inequality, its counterexamples forward, and a statistical certificate for a 3-step drift, see docs/jadhav_dilorenzo_bridge.md §12); `collecting_chain` (the published dog's collecting, from drift to cycles, see docs/collecting_chain.md); `density_controller` (one dog under the density controller: reaching, holding, and converging a scattered flock at the goal, against the published dog, the reach-and-release rule and the static ring, see §13), `hold_condition` (the controller's hold radius against dog speed, predicted from the steady-turn feasibility of the measured heading law, see §13.6) |
| `viz/` | interactive viewer: `app.html` (page), `sim.js` (model in JS), `build.py` (embeds the data); `phases_app.html`, `build_phases.py` (phase replay) |
| `docs/continuification_driving.md` | whether Di Lorenzo et al.'s convergence theorem says anything about Jadhav's sheep and dog, and what survives |
| `docs/jadhav_dilorenzo_bridge.md` | the synthesis: macroscopic dynamics of Jadhav's flock, minimal closures, and what carries over from Di Lorenzo's feasibility and stability analysis |
| `docs/steer_and_shape.md` | whether one dog can steer a flock and hold a prescribed distribution, and under what conditions |
| `reproduce.py` | regenerates the paper's Figs 1b–6 and the summary numbers |
| `data/upstream/` | the authors' data, `.mat` files (used to verify the port) and MATLAB model, from commit `2c02a88` |

This project is licensed under GPL-3.0 (`LICENSE`), because it is a port of the upstream
GPL-3.0 code and data (`data/upstream/LICENSE`).
