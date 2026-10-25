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
`--model` takes other YOLO weights or a shortcut (`icaerus`, `icaerus-n`; see below). Classes are
matched by name (`sheep`, `dog`, `person`/`human`), so a sheep-only drone model works too.

`tests/test_video.py` checks the cleaning and labelling without the model. It renders a real
session as noisy top-down detections (missed frames, ID switches, false positives) and gets the
same phase as the tracking-based labels on more than 99% of samples, so in practice the detector
is what limits the results.

### Detector choice

We measured detectors on `sheep_crossing.MP4` from
[ICAERUS UC3](https://github.com/ICAERUS-EU/UC3_Livestock_Monitoring) (GPL-3.0). It is a 14 s,
5 fps nadir drone clip of about 50 sheep, a white dog and a shepherd. Boxes were checked by eye on
frames 0, 35 and 70:

| Model | Sheep boxes on frames 0 / 35 / 70 (≈50 / 40 / 9 visible) | Dog |
|---|---|---|
| `yolo11n.pt` (COCO, the default) | 0 / 0 / 0 | never |
| `yolo11x.pt` (COCO), `--conf 0.05` | 29 / 26 / 0 | never |
| AUTH-Sheep `hbb_model.pt` | 0–11 / 0–3 / 0 | never |
| `--model icaerus-n` (YOLO11n) | 41 / 21 / 3 | never |
| **`--model icaerus`** (YOLOv8m v4) | **51 / 30 / 8**: about 94% recall on frame 0 and 80% on frame 35, with misses in the shadowed back of the flock | boxed as a sheep |

COCO models were trained on side views and miss sheep seen from above. Use `--model icaerus` for
drone footage. It downloads once into `models/`, pinned to a commit and a SHA-256, and runs at its
training settings (640 px, conf 0.5). This clip probably overlaps ICAERUS's training data, so
expect lower recall on your own footage.

No available model finds the dog from above, so phases on drone video read `away` until you
fine-tune.

### Fine-tuning for the dog

`experiments/finetune_detector.py` trains a sheep/dog/person model from your own footage in four
steps:

```bash
.venv/bin/python experiments/finetune_detector.py prelabel drive*.mp4 --out data/finetune   # ICAERUS pre-draws sheep
.venv/bin/python experiments/box_labeler.py data/finetune                                   # fix boxes, add dogs and people
.venv/bin/python experiments/finetune_detector.py train data/finetune --epochs 60           # starts from ICAERUS
.venv/bin/python experiments/finetune_detector.py eval data/finetune --model runs/finetune/sheepdog/weights/best.pt
.venv/bin/python experiments/label_video.py drive9.mp4 --model runs/finetune/sheepdog/weights/best.pt
```

- **Box labeler** (`experiments/box_labeler.py`, which opens `viz/box_labeler.html`): the model's
  boxes come pre-drawn, and the ones it was unsure of are dashed. Most frames need only a few
  keys:
  - Click the dog the model called a sheep and press **2**; **X** deletes a wrong box.
  - Double-click a missed sheep to drop a sheep-sized box, or drag to draw any box (this works on
    top of other boxes too).
  - **Tab** jumps to the next unsure box, **P** pastes the previous frame's dog and person boxes
    to drag into place, and **Enter** accepts the frame and opens the next one to review.
  - The wheel zooms, and right-drag pans.

  Every change is saved straight to `labels/`, and accepted frames are recorded in
  `review.json`. The page shows progress toward about 200 reviewed frames with a dog. It runs on
  localhost only. CVAT ("Ultralytics YOLO Detection") and Label Studio (YOLO) read and write the
  same folder if you prefer them.
- **Reviewed frames only:** once you've used the box labeler, `train` and `eval` use only
  accepted frames, because unchecked pre-labels teach the model its own mistakes.
  `--include-unreviewed` overrides this.
- **Validation:** whole videos are held out, because frames from one drive are near-duplicates
  and would overstate accuracy.
- **Augmentation:** rotations and flips are on, since nadir footage has no "up".
- **Comparing models:** `eval` scores any model on the held-out frames. Run it on `icaerus` and on
  your model to see the gain. Scores mean nothing until a person has corrected the held-out
  labels, because uncorrected pre-labels only measure the base model against itself.
- **How much to label:** a few hundred frames with the dog in shot, spread across videos, light
  and altitude.
- **Checked pipeline:** on the ICAERUS clip we ran pre-labelling, five hand-drawn dog boxes, a
  3-epoch CPU fine-tune, `eval`, and `label_video` with the result, which then produced dog and
  shepherd tracks. That shows the mechanics work, not that the model is accurate.

YOLO weights are pickles, and unpickling can run code. Every external `.pt` is therefore scanned
before use, without being loaded, and refused if it imports anything beyond torch/ultralytics
classes (`shepherd/weights.py`).

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
| `shepherd/phases.py` | rule-based draft phase labels (drive, turn, flank, idle, away) for any `Track`: positions, sampling interval, which row is the dog |
| `shepherd/video.py` | video → clean tracks: YOLO + ByteTrack detection, cleaning to metres at 10 Hz, flock series, phase labels |
| `shepherd/weights.py`, `shepherd/finetune.py` | pinned detector downloads with checksum and pickle checks; YOLO dataset helpers for fine-tuning |
| `shepherd/box_labeler.py` | local server for `viz/box_labeler.html`: reads and saves YOLO labels and the review state |
| `experiments/` | `protocol_check`, `replay_check`, `build_table`, `fit_kernels`, `compare_models`, `label_phases`, `label_simulations`, `pair_attraction` (see REPORT.md), `label_video` (video → clean data), `finetune_detector` (sheep/dog/person detector), `box_labeler` (fix detector boxes) |
| `viz/` | interactive viewer: `app.html` (page), `sim.js` (model in JS), `build.py` (embeds the data); `labeler.html` (video labeler); `phases_app.html`, `build_phases.py` (phase replay) |
| `reproduce.py` | regenerates the paper's Figs 1b–6 and the summary numbers |
| `data/upstream/` | the authors' data, `.mat` files (used to verify the port) and MATLAB model, from commit `2c02a88` |

This project is licensed under GPL-3.0 (`LICENSE`), because it is a port of the upstream
GPL-3.0 code and data (`data/upstream/LICENSE`).
