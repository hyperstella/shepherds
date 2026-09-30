# Shepherds: sheep-herding simulation research

This is a Python reproduction and extension of Jadhav et al. (2024), *Collective responses of
flocking sheep to a herding dog* (Communications Biology 7:1543). It includes their UWB tracking
data (14 sheep + dog + shepherd, 10 Hz, 29 drives) and their agent-based shepherding model.

See **[REPORT.md](REPORT.md)** for what was reproduced, where the model and data disagree, and a
prioritized list of improvements.

## Setup

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

## Layout

| Path | Contents |
|---|---|
| `shepherd/data.py` | loads `sheepR.dat`; ports `pre_ana.m`, `pre_ini_ana.m`, `drivs_dat.m` |
| `shepherd/metrics.py` | cohesion, polarization, elongation, flock frame, viewing angles, turning rates |
| `shepherd/network.py` | directional cross-correlation, leader–follower networks, d_i vs indegree |
| `shepherd/model.py` | the herding model: vectorized `simulate` plus a literal reference port |
| `viz/` | interactive viewer: `app.html` (page), `sim.js` (model in JS), `build.py` (embeds the data) |
| `reproduce.py` | regenerates the paper's Figs 1b–6 and the summary numbers |
| `data/upstream/` | the authors' data, `.mat` files (used to verify the port) and MATLAB model, from commit `2c02a88` |

This project is licensed under GPL-3.0 (`LICENSE`), because it is a port of the upstream
GPL-3.0 code and data (`data/upstream/LICENSE`).
