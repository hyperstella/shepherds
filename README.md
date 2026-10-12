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

## Labelling video with a local model

`experiments/label_video.py` turns a herding video into clean data with no manual labelling. It
runs on your machine:

1. **Detect and track.** A COCO-pretrained YOLO model (`yolo11n.pt`, downloaded once) finds
   `sheep`, `dog` and `person` (taken as the shepherd), and ByteTrack links the boxes into tracks.
2. **Clean.** Tracks shorter than 1 s are dropped, and each track takes its majority class.
   Positions are converted to metres and resampled to 10 Hz like the UWB data. Gaps up to 1 s are
   interpolated and the tracks are smoothed over 0.5 s. Dog and shepherd fragments are stitched
   into one track each.
3. **Simplify.** The flock centre, velocity (the mean of each sheep's own velocity), cohesion and
   sheep count are computed alongside the dog and shepherd positions.
4. **Label.** The phase rules in `shepherd/phases.py` (drive, turn, flank, idle, away) run on those
   series. A dog out of shot counts as `away`, and the `dog_seen` column says how often it was
   actually in view.

```bash
.venv/bin/pip install -r requirements-video.txt         # once; installs PyTorch + ultralytics
.venv/bin/python experiments/label_video.py drive.mp4   # writes results/video/drive_*.csv
```

| Output | Contents |
|---|---|
| `_detections.csv` | raw tracked boxes; re-clean without the model via `--from-detections` |
| `_tracks.csv` | cleaned tracks: `t_s, track, kind, x_m, y_m, x_norm, y_norm` every 0.1 s |
| `_flock.csv` | `flock_x/y, flock_vx/vy, cohesion, n_sheep, dog_x/y, shepherd_x/y` every 0.1 s |
| `_phases.csv` | phase segments in the `draft_phases.csv` columns, plus `n_sheep`, `dog_seen` and video times |

Units matter: the phase thresholds are in metres and m/s. For oblique footage, pass
`--homography h.json`, a 3×3 pixel-to-ground matrix. For top-down (drone) footage, pass
`--px-per-m`. With neither, the scale is guessed from the median sheep box, assuming a 1.2 m
sheep, which is only reasonable for top-down video. `--session` and `--offset` put the times on
the UWB clock (tracking time = video time + offset). `--stride 2` halves the detector's work.
`--model` takes other YOLO weights. Classes are matched by name (`sheep`, `dog`, `person`/`human`), so a sheep-only drone model works too.

`tests/test_video.py` checks the cleaning and labelling without the model. It renders a real
session as noisy top-down detections (missed frames, ID switches, false positives) and gets the
same phase as the tracking-based labels on more than 99% of samples. How well the detector does
depends on the footage. COCO models find sheep well from the ground and from low drones, but
small, densely packed sheep in high aerial shots may need `--imgsz 1920` or a fine-tuned model.

### Reviewing in the browser

`viz/labeler.html` is the review tool. It needs no build step: open it in a browser and drop in
the video together with its `_phases.csv` and `_tracks.csv`. The video stays on your machine.

- Model tracks are drawn on the frame with 2 s trails, and model phases fill the timeline.
- Press **1–5** to relabel from the playhead onward (drive, turn, flank, idle, away); **0**
  clears the label. Drag boundaries on the timeline, scroll over it to zoom, and click a segment
  to edit its times, label or notes.
- **Q / W / E** turn on point marks for the dog, shepherd or sheep: click the frame to place one,
  right-click to remove it, and press **R** to stop marking.
- **Export segments** writes the reviewed phases (same columns, tracking time). Work autosaves in
  the browser for each video, and Ctrl+Z undoes.

Read reviewed labels back as per-sample arrays:

```python
from shepherd import data, phases
lab = phases.read_reviewed("drive_phases.csv", data.load_sessions()[0])  # -1 = unlabelled
```

## Layout

| Path | Contents |
|---|---|
| `shepherd/data.py` | loads `sheepR.dat`; ports `pre_ana.m`, `pre_ini_ana.m`, `drivs_dat.m` |
| `shepherd/metrics.py` | cohesion, polarization, elongation, flock frame, viewing angles, turning rates |
| `shepherd/network.py` | directional cross-correlation, leader–follower networks, d_i vs indegree |
| `shepherd/model.py` | the herding model: vectorized `simulate` (per-run parameters, dog/speed replay, dt/τ) plus a literal reference port |
| `shepherd/protocol.py` | one analysis protocol for data and models: segments, summaries, sheep-level hierarchy, bootstrap, surrogates |
| `shepherd/scenario.py` | simulated drives matched to real ones: the real dog replayed in 15 s windows that restart from the real flock |
| `shepherd/abc.py` | NumPy ABC: Sobol reference tables, regression adjustment, coverage checks |
| `shepherd/kernels.py` | learned interaction kernels: integral-form least squares, CV, synthetic data, forward simulation |
| `shepherd/phases.py` | rule-based draft phase labels (drive, turn, flank, idle, away) for all tracked time |
| `shepherd/video.py` | video → clean tracks: YOLO + ByteTrack detection, cleaning to metres at 10 Hz, flock series, phase labels |
| `experiments/` | `protocol_check`, `replay_check`, `build_table`, `fit_kernels`, `compare_models`, `label_phases` (see REPORT.md), `label_video` (video → clean data) |
| `viz/` | interactive viewer: `app.html` (page), `sim.js` (model in JS), `build.py` (embeds the data); `labeler.html` (video labeler) |
| `reproduce.py` | regenerates the paper's Figs 1b–6 and the summary numbers |
| `data/upstream/` | the authors' data, `.mat` files (used to verify the port) and MATLAB model, from commit `2c02a88` |

This project is licensed under GPL-3.0 (`LICENSE`), because it is a port of the upstream
GPL-3.0 code and data (`data/upstream/LICENSE`).
