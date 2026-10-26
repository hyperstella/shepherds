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

1. **Detect and track.** A YOLO model finds sheep, and ByteTrack links the boxes into tracks. The
   default, `icaerus-n`, is a drone-sheep detector downloaded once (see "Detector choice"). It
   detects sheep only, so dogs and people need a fine-tuned model (see "Fine-tuning for the dog")
   or a COCO model such as `--model yolo11x.pt` for ground-level video, which also finds `dog` and
   `person` (taken as the shepherd).
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
`--model` takes other YOLO weights or a shortcut (`icaerus-n`, the default, or `icaerus`; see below). Classes are
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

COCO models were trained on side views and miss sheep seen from above. On this clip the larger
`icaerus` looked best, but the clip probably overlaps ICAERUS's training data. The independent
benchmark below reverses the order, so the default is `icaerus-n` at 1280 px. Shortcut weights
download once into `models/`, pinned to a commit and a SHA-256.

**Benchmark against hand-drawn boxes.** `experiments/benchmark_auth_sheep.py` scores detectors on
[AUTH-Sheep](https://zenodo.org/records/13843376) (CC BY 4.0), which is 4K drone video from a
different farm with every sheep, goat and person boxed by hand. On videos 2–4 (138 frames, one per
second, 3,613 sheep and goats), counting a box as a hit at IoU 0.5:

| Model, input size | AP50 | Precision | Recall | Sheep recall | Goat recall | Boxes on people | s/frame (CPU) |
|---|---|---|---|---|---|---|---|
| `icaerus-n`, 1280 | **0.51** | 0.69 | **0.54** | 0.66 | **0.28** | 2 | 0.2 |
| `icaerus`, 1920 | 0.38 | 0.63 | 0.49 | **0.70** | 0.02 | 0 | 1.7 |
| `icaerus`, 1280 | 0.36 | 0.62 | 0.48 | 0.67 | 0.04 | 1 | 1.8 |
| `icaerus`, 640 | 0.35 | 0.61 | 0.40 | 0.53 | 0.11 | 8 | 0.5 |
| `yolo11x.pt` (COCO), 1280 | 0.15 | 0.68 | 0.08 | 0.11 | 0.00 | 0 | 2.7 |

On footage it was not trained on, ICAERUS finds about half the flock, far below what it achieves on
its own clips. Goats are mostly missed, so mixed flocks will be undercounted. The small
`icaerus-n` at 1280 px scores best overall and runs fastest, so it is the default for
`label_video.py`, `prelabel` and `train`. `--model icaerus` (at 640 px unless `--imgsz` is
given) remains available.

No available model finds the dog from above, so phases on drone video read `away` until you
fine-tune.

### Fine-tuning for the dog

`experiments/finetune_detector.py` trains a sheep/dog/person model from your own footage in four
steps:

```bash
.venv/bin/python experiments/finetune_detector.py prelabel drive*.mp4 --out data/finetune   # icaerus-n pre-draws sheep
.venv/bin/python experiments/box_labeler.py data/finetune                                   # fix boxes, add dogs and people
.venv/bin/python experiments/finetune_detector.py train data/finetune --epochs 60           # starts from icaerus-n
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
- **Frozen backbone (`--freeze 10`, the default):** only the detection head learns, so a small
  dataset adds the dog class without eroding the base model's sheep detection. On a 14 s test clip
  starting from `icaerus`, full fine-tuning on 36 frames cut held-out sheep recall from 97% to 78%;
  with the backbone frozen it stayed at 94%. `--freeze 0` trains every layer, which is worth it once you have
  thousands of labelled frames.
- **No early stopping (`--patience 0`, the default):** the stopping score sums all classes, so
  the many sheep boxes dominate it. With early stopping on, training stopped on epoch-1 weights
  that had learned nothing about the dog. For the same reason, compare `last.pt` and `best.pt`
  with `eval`.
- **Dog data:** these settings protect sheep accuracy but do not make the dog appear. On that
  test clip, 36 boxes of one dog over 7 s gave 0% held-out dog recall under every setting. The
  dog needs varied examples: different dogs, light, poses and videos.
- **Comparing models:** `eval` scores any model on the held-out frames. Run it on `icaerus-n` and on
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

### Video against the Jadhav UWB data

`experiments/video_vs_jadhav.py` measures the paper's flock and dog observables from drone video and
from the UWB drives with the same code. The dog is located by a white-blob tracker, checked by eye,
in 6 of the 16 ICAERUS flights (`results/icaerus_dog_tracks.csv`). Sheep are tracked with
`icaerus-n` at 10 Hz. 159 frames (16 s) have the dog and at least 8 moving sheep in view; the UWB
drives give 848 s.

| Observable (median, IQR) | Video | UWB (Jadhav) | Model parameter |
|---|---|---|---|
| Polarization P | 0.98 (0.95–0.99) | 0.92 (0.83–0.96) | |
| Elongation E | 2.31 (1.54–3.38) | 1.02 (0.77–1.39) | |
| Nearest-neighbour distance (m) | 0.60 (0.49–0.82) | 0.50 (0.43–0.57) | `r_rep` scale |
| Sheep speed (m/s) | 1.10 (0.81–1.55) | 1.09 (0.83–1.60) | `v_sheep` = 1.0 |
| Dog speed (m/s) | 1.24 (0.64–1.93) | 1.26 (0.77–2.02) | `v_dog` = 1.5 |
| Dog–barycentre distance (m) | 5.8 (3.6–6.9) | 5.0 (4.0–6.1) | |
| Dog–barycentre / NND | 7.6 (6.3–11.3) | 10.3 (7.9–13.2) | |
| \|ψ_BD\| (deg; 180 = directly behind) | 162 (81–173) | 156 (140–168) | |
| Dog behind the rearmost sheep | 36% | 94% | |

Speeds and spacing agree to within the ±20% scale uncertainty. The metres come from sheep box size,
because the flight log's altitudes give scales that cannot fit the visible flock. The flocks differ
where the setting does:
- **Elongation and polarization are higher:** these clips are sheep walking in single file down a
  lane behind a shepherd.
- **The dog is behind the flock far less often:** it accompanies the flock rather than driving it.

The pooled medians also hide large differences between clips. In one flight the dog runs ahead of
the flock at 3 m/s; in another it stays behind it throughout. Sheep speed shows no clear dependence on
distance to the dog, so `r_dog` cannot be estimated from this footage. Treat these numbers as a
consistency check on scale and speed, not as a test of the herding model, which needs footage of a
dog actually driving a flock.

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
