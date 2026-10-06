# Reproducing Jadhav et al. (2024): sheep flock + herding dog

Source: Jadhav, Pasqua, Zanon, Roy, Tredan, Bon, Guttal & Theraulaz, *Collective responses of flocking
sheep (Ovis aries) to a herding dog (border collie)*, Communications Biology 7:1543 (2024).
Code and data: [tee-lab/collective-responses-of-flocking-sheep-to-herding-dog](https://github.com/tee-lab/collective-responses-of-flocking-sheep-to-herding-dog)
at commit `2c02a88`.

Everything here was re-implemented in Python (`shepherd/`) from the authors' MATLAB. Run
`.venv/bin/python reproduce.py` to regenerate `results/` (~15 s).

## How the port was verified

| Check | Result |
|---|---|
| `data.load_sessions()` vs authors' `sheep_all_dat.mat` | positions and velocities bit-identical; headings within 1e-15 |
| `data.concat_drives_by_session()` vs authors' `drives_data.mat` | bit-identical, 8 sessions, 29 drives |
| Vectorized model (`simulate`) vs line-by-line port of `herding_model.m` (`simulate_reference`) | identical trajectories and dog modes to 1e-8 over 30 steps, 6 seeds, N = 5 and 14 (after that, rounding differences are amplified by chaos) |
| `xcorr_unbiased` vs `np.correlate` | exact |

`pytest tests/` runs all of these.

## Part 1: data analyses (Figs 1–5d), reproduced

| Quantity | Paper | This reproduction |
|---|---|---|
| Mean cohesion ⟨C⟩ ± SD | 1.21 ± 0.34 m | **1.21 ± 0.34 m** |
| Mean polarization ⟨P⟩ ± SD | 0.85 ± 0.17 | **0.85 ± 0.17** |
| Mode of elongation E | 0.8 | **0.8** |
| Mean speed, sheep / dog | 1.3 / 1.5 m/s | 1.29 / 1.47 m/s (barycenter: 1.17) |
| Speed-PDF peak, barycenter / dog | 0.9 / 0.8 m/s | 0.9 / 0.7 m/s (0.2 m/s bins centred on odd tenths, so 0.8 is not a possible bin) |
| Dog–barycenter distance d_BD | "≈ 3 to 7 m" | 10–90%: 3.4–7.4 m |
| Dog–rear-sheep distance ȳ_RD | "≈ 1 to 4 m", rarely < 0 | 10–90%: 0.6–5.1 m; dog behind rear sheep 94% of the time |
| Dog lateral excursion x̄_D | "−6 to 6 m" | 5–95%: −4.7 to 3.5 m |
| Flock turns away from a dog on its flank (Fig 4a) | yes | yes (r = −0.28 between ψ_BD and Δφ_B on the flanks) |
| Dog turning ∝ its viewing angle ψ_DB (Fig 4b) | yes | yes (r = 0.57; slope ≈ 1) |
| ⟨d_i⟩ vs indegree, Pearson r (Fig 5d) | 0.85, p < 0.0001 | **0.87**, p = 4e-5 |
| Dog's place in the leadership hierarchy (Fig 5c) | lowest | **lowest** (15th of 15; mean indegree 1.1 vs 4.6–10.2 for sheep) |

Figures: `results/figures/fig1b_fig2ae_timeseries.png`, `fig2fj_pdfs.png`,
`fig3_fig4_angles_turning.png`, `fig5_leader_follower.png`.

## Part 2: model (Figs 5e and 6), qualitatively reproduced

The model was run with the parameters in `model/simulation_hm.m`: 300 runs × 370 steps, N = 14,
Δt = 1 s. See `results/figures/fig6_model_vs_data.png`.

| Claim | Paper | Our port |
|---|---|---|
| Front sheep have more followers (Fig 5e) | yes (ρ = 0.99 in authors' CSV) | yes, ρ = 0.96; but the ⟨d_i⟩ range is about half the authors' saved output |
| Dog is lowest in the hierarchy | "consistently" | dog lowest in 86% of runs (mean indegree 1.1 vs 7.0 for sheep) |
| Null model without a dog has no leader–follower pairs | 0 to 3 pairs | 0 pairs per run |
| E mostly < 1, high P, dog behind the flock | "qualitative agreement" | yes: E mode 0.6, ⟨P⟩ 0.82, dog behind rear sheep 95% |
| x̄_D weakly bimodal | yes | **not reproduced**: unimodal and narrower than the data |

The model **does not match quantitatively**:

- **Cohesion is about 2.3× too loose:** model ⟨C⟩ is 2.8 m vs 1.21 m in the data. The sheep repulsion
  radius is 2 m, larger than a sheep's 1.2 m body length, so the flock cannot pack as tightly as real
  sheep do.
- **The dog stays too far back and too directly behind:** model d_BD is 6.3–8.9 m vs 3.4–7.4 m, and
  ȳ_RD is 3.3–6.7 m vs 0.6–5.1 m. In the model, ψ_BD is within 30° of straight behind 97% of the
  time; in the data it is 61%. The real dog zigzags widely on the shepherd's commands, while the
  model dog aims at a single drive point, with only noise added.
- **The "collect" behaviour never runs:** the dog is in drive mode 95% of steps and slowed 5%. The
  threshold l_sep = 2·N^(2/3) ≈ 11.6 m is never exceeded, so the model is effectively drive-only.

## Differences between the paper, the code and the analysis (worth knowing before building on it)

1. **The paper's Methods and the code disagree on parameters.** The paper gives n_Att = 5 and
   e = 0.5, and says the dog slows to 0.05·v_D. The code uses n_Att = 4 and e = 0.3, and a slow
   step of 0.05 m. With the paper-text values, the hierarchy result still holds (ρ = 0.95).
2. **Model and data are analysed differently:**
   - The model uses |CCF| to pick the lag; the data uses the signed CCF.
   - The model's lag window is 20 steps = 20 s (Δt = 1 s); the data's is 20 samples = 2 s (10 Hz).
     The model's leader–follower lags therefore sit on a time scale 10× longer.
   - Model statistics pool the whole run, including the start; the data uses only hand-labelled drives.
3. **ȳ_RD has the opposite sign for the model:** `model_results.m` computes ȳ_D − ȳ_R, while the
   data scripts use ȳ_R − ȳ_D. We use the data convention for both.
4. **Drive count:** the code defines **29** drives; the paper states 26.
5. **Turning rates straddle drive boundaries:** the Fig 2f–j, 3 and 4 statistics are computed on
   drives concatenated per session, so 3-second turning windows cross the gaps between drives. We
   kept this to reproduce faithfully; the effect is small.
6. **The r = 0.85 has an inflated p-value:** it correlates 14 per-indegree bin means as if they were
   independent points, so the p-value overstates the evidence. A drive-level permutation test would
   be more appropriate.
7. **The null model is confounded:** without a dog, the flock is disordered (⟨P⟩ = 0.32). The lack
   of leader–follower pairs may therefore come from having no coherent motion, not from having no
   dog.

---

## Possible improvements, in priority order

### A. Make the model–data comparison fair (small, do first)
- Analyse model and data the same way: signed CCF, the same **lag window in seconds**, and only
  drive-phase steps after the dog first reaches the flock.
- Run the model at Δt = 0.1 s with the measured speeds (sheep 1.3 m/s, dog 1.5 m/s), so the time
  scales of Figs 4 and 5 are comparable.
- Put error bars on the hierarchy result:
  - bootstrap over drives;
  - a permutation null that shuffles headings in time within a drive, to calibrate the
    c_min = 0.5 threshold;
  - a sensitivity sweep over c_min and the maximum lag.

### B. Calibrate the model to the data (turns "qualitative" into "quantitative")
- Set the length scales from sheep biology: r_rep ≈ 1 m (body length), and choose l_drive to match
  the measured d_BD.
- Fit the remaining weights (w_att, w_ali, w_dog, noise) with **simulation-based inference**,
  either ABC or neural posterior estimation (`sbi`). Use the Fig 2, 3 and 6 PDFs plus the
  d_i–indegree slope as summary statistics, with a Wasserstein distance.
- Add **speed modulation**. Real sheep speed varies from 0 to 3 m/s and is correlated with dog
  speed and cohesion (Supp. Fig), while model speed is constant. For example, make speed depend on
  the distance to the dog.

### C. A realistic dog
- Replace the fixed-point pursuit with a dog that zigzags (flanks), as the real one does
  (x̄_D ±4 m, a broad ψ_BD). Three options:
  - a stochastic left/right flanking offset around P_drive;
  - a controller fitted to the data. Fig 4b shows the dog's turning is almost linear in ψ_DB, so
    a simple rule fits well;
  - behavioural cloning on the 29 drives.
- Check whether that reproduces the bimodal x̄_D the paper reports and the wide ψ_BD seen in the
  data.

### D. Infer the interaction rules from the data (the paper's own suggested next step)
- Use the tracks to estimate how each sheep's turn and speed change depend on its neighbours and
  on the dog. Options: force maps (Katz et al. 2011), the Calovi/Escobedo burst-and-coast
  decomposition, or a small graph neural network.
- Test directly **whether alignment is needed**. The paper assumes it, but notes that some
  studies find no velocity matching.
- Add **individual heterogeneity**, such as per-sheep sensitivity to the dog. The data show some
  sheep consistently at the front, but in the model all sheep are interchangeable.

### E. Better causal tests of "the dog creates the hierarchy"
- A **polarized null model**: directed motion with no dog, for example a moving goal or a virtual
  leader, with P matched to the data. This separates "caused by the dog" from "caused by
  coherent motion".
- Replace or complement CCF with **transfer entropy**, and use the surrogate data from A to
  set significance.

### F. Extensions linking to *The Flock* (emergentbehavior.com)
- **Scale:** swap the O(N²) neighbour search for `scipy.spatial.cKDTree` and study
  N = 14 → 1000. Questions:
  - Do the heuristics l_sep ∝ N^(2/3) and l_drive ∝ √N still work?
  - How do time to target, the rate of flock splitting and the depth of the hierarchy scale with N?
- **Dog policies:** compare the Strömbom heuristic, a controller fitted to the data, and a policy
  trained with reinforcement learning (PPO). Objectives: time to target, flock cohesion, and a
  "stress" proxy (sheep speed, closeness to the dog).
- Multiple dogs, obstacles and terrain, plus a browser-based viewer (Canvas or WebGL) with
  parameter sliders and real drives replayed on top.

---

## Phase 1: a fair protocol, and three findings confirmed

`shepherd/protocol.py` analyses real and simulated drives the same way:
- positions sampled every dt;
- headings from displacement, with samples slower than 0.1 m/s excluded;
- signed cross-correlation, with the lag window given in seconds;
- statistics at the level of individual sheep, with session-level bootstrap intervals and
  time-shift surrogates.

Run `experiments/protocol_check.py` to reproduce this section; it writes
`results/protocol_check.json` and `results/figures/protocol_check.png`.

| Finding | Result |
|---|---|
| **F2: the paper's r = 0.87 comes from bin means** | The legacy pipeline reproduces the bin-mean r of 0.874 exactly. Per sheep, the same data give Spearman ρ = 0.36. Under the fair protocol ρ = **0.40 (95% CI 0.30–0.48, session bootstrap)**. All 500 time-shift surrogates fall below it (p ≈ 0.002; null mean ρ ≈ −0.04). **The front-to-back hierarchy in the real flock is real, but moderate.** |
| **F3: follow delays are sub-second** | Median leader→follower lag is **0.5 s** (IQR 0.2–0.8 s). ρ stays at 0.36–0.42 for maximum lags from 1 to 4 s, and for sampling steps from 0.1 to 0.5 s. At a 1 s step it falls to **0.25**, so a model that steps once per second cannot show the effect at its natural time scale. |
| **F1: the published model's hierarchy is a start-up transient** | These results are from 300 runs of M0. With the authors' own model analysis (\|CCF\|, 20 s lag window), per-sheep ρ is 0.34 in steps 1–100 but only **0.10 in steady driving** (steps 100–370). Its bin-mean r stays at 0.81–0.94, which is why the paper's Fig 5e looked like a match. **Under the fair protocol the published model shows no hierarchy during steady driving: ρ = 0.00 (95% CI −0.03 to 0.03).** |

**Consequence for the paper.** The real flock has a statistically robust front-to-back hierarchy
at sub-second lags, and the published model does not reproduce it once it is analysed like the
data. That makes the question of whether a *learned* interaction model can reproduce it (Q3) a
sharp test.

## Phase 2: the published model driven by the real dog

`shepherd/scenario.py` replays the collie's recorded path, and starts each simulated flock from
the real flock's state, so model and data face exactly the same dog.

**Full-drive replay does not work.** Over a 30–100 s drive the model flock drifts from the real
flock's route. The recorded dog then drifts away from the model flock (median dog–flock distance
12.8 m vs 5.2 m) and leaves the sheep's 12 m notice radius, so the model sheep stop.

**The harness therefore uses multiple shooting.** Each drive is cut into 15 s windows. At the
start of each window the model restarts from the real sheep positions and headings, and the first
3 s are dropped as burn-in. The data statistics use exactly the same windows. Run
`experiments/replay_check.py` to reproduce this.

| Statistic (15 s windows, dt = 1 s) | Real flock | Published model |
|---|---|---|
| Nearest-neighbour distance (median) | **0.51 m** | 1.23 m |
| Cohesion C (median) | **1.19 m** | 2.63 m |
| Elongation E (median) | **1.12** | 0.68 |
| Polarization P (median) | **0.94** | 0.78 |
| Dog–barycenter distance d_BD (median) | 5.2 m | 7.4 m |
| Share of time dog within 30° of straight behind | 0.58 | 0.74 |

These gaps persist at dt = 0.2 s and with the sheep speed set to the measured 1.29 m/s. They are
the targets for Phase 3 (calibration) and Phase 4 (learned interaction kernels).

## Phase 4: learning the interaction laws from the tracks

`shepherd/kernels.py` fits a second-order interacting-particle model for the sheep, with the dog
as an external agent. Each interaction is a nonparametric kernel of distance, built from
piecewise-linear "hat" functions:
- φ_E: attraction/repulsion;
- φ_A: alignment;
- φ_D: flight from the dog;
- ψ_V: response to the dog's velocity;
- ψ_L: flight scaled by how fast the dog is closing in.

Fitting method:
- **Integral form:** the fit matches the change in each sheep's velocity over 1 s against the
  time-integrated forces, rather than differentiating noisy tag positions twice.
- **Smoothing:** a 2.1 s Savitzky–Golay filter.
- **Penalty:** second-difference smoothness, with its weight chosen by leave-one-session-out
  cross-validation.

Run `experiments/fit_kernels.py` to reproduce this section.

### What explains the sheep's accelerations (Q3)

Scores are leave-one-session-out R² of the velocity change over 1 s. The whole-flock part is 55% of
the variance.

| Model | Total | Whole-flock part | Within-flock part |
|---|---|---|---|
| All terms plus self-propulsion | 0.19 | 0.27 | 0.10 |
| **Main model** (E, A, D, DV, DL; flock speed external) | 0.12 | 0.14 | 0.09 |
| No alignment | 0.12 | 0.14 | 0.09 |
| No attraction/repulsion | 0.07 | 0.14 | **−0.02** |
| Dog position only (no dog-velocity terms) | 0.06 | **0.02** | 0.09 |
| No dog terms | 0.05 | 0.00 | 0.10 |
| Topological (4 or 7 nearest neighbours) | 0.10–0.11 | 0.13–0.14 | 0.05–0.07 |

- **Attraction/repulsion carries the structure within the flock.** Removing it takes the
  within-flock R² from 0.09 to below zero. The kernel changes sign from repulsion to attraction at
  **r = 1.25 m, the sheep's body length (1.2 m)**.
- **Alignment adds nothing measurable** (0.117 with or without it), and it is not identifiable
  (see Q1). This agrees with the paper's caveat that velocity matching has little direct support.
- **The flock as a whole responds to the dog's motion, not just its position.** The dog-velocity
  and closing-speed kernels lift the whole-flock R² from 0.02 to 0.14. ψ_V is *negative*: sheep
  accelerate against the dog's lateral motion, so when the collie swings right the flock veers
  left. This is anticipatory flight, consistent with the turn-away in Fig 4a.
- **Distance-based (metric) interactions fit better than nearest-k (topological) ones.**
- **The flock's own speed regulation is not identifiable.** Every fit gives *negative* friction,
  and in the data the flock's speed barely depends on dog distance (1.2–1.3 m/s from 0 to 8 m).
  In a synthetic check the method did recover a restoring speed term when one existed (shrunk
  30–50% toward zero), so its absence in the real fit is a property of the data. The flock's
  overall speed is set from outside, by the shepherd and dog. All forward simulations therefore
  **replay the real flock's mean speed**, just as they replay the dog, for every model family.

### Identifiability (Q1)

These results use synthetic flocks from known kernels (the real-data fit), driven by the real dog
paths. Tag-like noise was added at 2 Hz and interpolated to 10 Hz, then the fitting pipeline was
run unchanged. Errors are relative L2, weighted by where data actually lie. See
`results/figures/identifiability.png`.

- **φ_E** (attraction/repulsion) is recovered well: error 0.06–0.15, falling with more drives.
- **The dog kernels** (φ_D, ψ_V, ψ_L) are recovered with errors of about 0.25–0.35. These level
  off rather than falling with more data, which points to a bias from the flock-speed input the
  model does not describe.
- **φ_A (alignment) is not identifiable.** Its error is 0.85–1.4 even with all 29 drives, and it
  stays above 1 even without tag noise. In a cohesive, polarized flock, velocity differences
  between neighbours are small and collinear with the attraction term.

## Phase 3: calibrating the published model (NumPy ABC)

`shepherd/abc.py` together with `experiments/build_table.py` fits the Strömbom/Jadhav sheep rules
to the real drives. Every simulation uses the same inputs as the data: the real dog's path and the
real flock's mean speed, in 15 s windows that restart from the real flock.

**Method:**
- Sobol reference tables: 65,536 + 32,768 parameter sets at dt = 1 s (M1), and
  32,768 + 16,384 at dt = 0.2 s with heading relaxation (M1τ). Each table takes about 5–8 minutes
  on 10 cores.
- Rejection ABC keeping the closest 1%, with a local-linear regression adjustment.
- 15 summary statistics; the hierarchy is deliberately excluded.
- A coverage check with 50 pseudo-observed rows: 90% intervals contain the truth 86–96% of the
  time, so the posteriors are well calibrated.

**Posterior (median and 90% CI):**

| Parameter | M1 (dt = 1 s) | M1τ (dt = 0.2 s) |
|---|---|---|
| Personal space r_rep | 0.87 m (0.31–1.58) | **0.34 m (0.22–0.47)** |
| Notice radius for the dog r_dog | **8.6 m (7.4–10.1)** | **9.8 m (7.6–12.0)** |
| Heading relaxation τ_s | — | 0.21 s (0.20–0.22), at the prior floor (= dt) |
| Alignment weight w_ali (w_rep = 2 fixed) | ≈ 900, at the upper prior edge | ≈ 900, at the upper prior edge |

The spacing and dog-notice radii are well identified (contraction 0.73–0.97). The individual
weights are only weakly identified, because only their ratios matter, and the fit pushes the
alignment weight to the prior's edge. In effect, the calibrated model is "copy a random
neighbour's heading almost instantly, with very little repulsion". A reparametrization that fixes
the alignment weight instead would be cleaner, and is on the to-do list.

## Phase 5: held-out prediction (Q2) and the hierarchy test (Q3)

Run `experiments/compare_models.py` to reproduce this section; it writes
`results/compare_models.json` and `results/figures/heldout_hierarchy.png`.

**Q2, predicting unseen sessions.** Each model is scored with leave-one-session-out on the 15
summary statistics, as an RMS of errors in units of the data's window-to-window SD. The
training-data average (0.59–0.63) is the floor.

| Model | dt = 1 s | dt = 0.2 s |
|---|---|---|
| Published (M0), constant speed | 3.13 | — |
| Published, real flock speed | 3.18 | 3.69 |
| Published, body-length spacing | 2.10 | — |
| **Calibrated M1 / M1τ** | **1.33** | **1.03** |
| Learned kernels + 3 calibrated gains | — | 1.42 |

- **Calibration cuts the published model's error by 60–70%.**
- **On summary statistics, the calibrated parametric model beats the learned-kernel model.** The
  learned flock is too loosely aligned: its polarization error is 2.5 SD.
- **The learned kernels are too weak:** the calibrated gain on attraction/repulsion is 3.9 (90% CI
  1.9–7.8). This matches the shrinkage toward zero seen in the synthetic study (noisy velocities
  on both sides of the regression).

**Q3, front-to-back leadership.** All rows use the same 15 s windows at dt = 0.2 s and the fair
protocol. ρ is the sheep-level Spearman correlation between the number of followers and front
position; the CIs come from a session bootstrap.

| Real flock and models | ρ (95% CI) |
|---|---|
| **Real flock** (time-shift surrogate p = 0.005) | **0.30 (0.23–0.34)** |
| Published model | 0.09 (−0.03–0.15) |
| Calibrated M1τ (posterior draws) | 0.08 (0.05–0.11) |
| **Learned kernels (posterior draws)** | **0.27 (0.22–0.32)** |
| Learned, dog removed (flock speed replayed; P drops to 0.54) | 0.21 (0.17–0.27) |
| Learned, flock *velocity* replayed | 0.45 (0.33–0.58) |
| Learned, flock *velocity* replayed, dog removed | 0.45 (0.36–0.57) |

- **Matching the summary statistics is not enough.** Even calibrated, the Strömbom/Jadhav model
  shows no front-to-back hierarchy. Its sheep copy randomly chosen neighbours, which has no
  front/back asymmetry.
- **The learned kernels reproduce the real flock's hierarchy** (0.27 vs 0.30), although nothing in
  the fit used hierarchy statistics.
- **With the flock's collective motion held fixed, the dog adds nothing to the hierarchy** (0.45
  with vs 0.45 without). In the learned model, leadership comes from sheep–sheep
  attraction/repulsion in a flock moving in one direction; the dog's role is to steer the flock as
  a whole. This refines the paper's conclusion that the dog is needed: their null model lost
  coherent motion entirely.
- **Caveats:**
  - The velocity-replay runs are less polarized than the real flock (P ≈ 0.65–0.68 vs 0.89) and
    overshoot ρ.
  - Each window starts from the real configuration, so front/back positions are partly inherited.
  - One flock and one dog.

### Robustness of the leadership result (`experiments/hierarchy_robustness.py`)

All rows use the learned-kernel model with posterior gains and the same fair protocol. The real
flock's ρ is 0.30 in 15 s windows and 0.40 in 30 s windows.

| Check | ρ (95% CI) | Polarization P |
|---|---|---|
| Learned model, as in Phase 5 | 0.27 (0.23–0.33) | 0.61 |
| **Front/back mirrored at the start of each window** | **0.27** (0.23–0.33) | 0.61 |
| **All sheep start with the flock's mean heading** | **0.27** (0.23–0.31) | 0.66 |
| 30 s windows | 0.24 (0.18–0.27) | 0.52 |
| Flock velocity replayed, noise 0.03 → 0.2: dog vs no dog | 0.49 / 0.50 → 0.46 / 0.47 | 0.79 → 0.66 |
| **+ weak uniform alignment 0.005/s: dog vs no dog** | **0.39 / 0.42** | **0.90 / 0.88** (real flock 0.89) |
| + alignment 0.01/s: dog vs no dog | 0.28 / 0.32 | 0.93 / 0.92 |
| + alignment 0.02 / 0.05 / 0.10 per s | 0.15 → 0.04 → 0.00 | 0.97 → 1.00 |

1. **The hierarchy is produced by the dynamics, not inherited.** Mirroring the starting layout
   front-to-back, or erasing individual starting headings, leaves ρ unchanged.
2. **The dog does not create it.** With the flock's collective motion held fixed, removing the dog
   leaves ρ the same, or slightly higher, at every noise level. This includes the alignment strength
   at which polarization matches the real flock (0.89).
3. **Strong alignment destroys front-to-back leadership.** As alignment grows, polarization goes to
   1 and ρ to 0: when every sheep copies every other, heading information no longer flows from
   front to back. Weak alignment (about 0.005–0.01 per neighbour per second) reproduces both the
   real flock's polarization and its hierarchy. This is also why the alignment-dominated calibrated
   parametric model shows no hierarchy.

## Phase 7 (feasibility): a second dataset

A desk check of public sources; nothing was downloaded.

| Source | Contents | Dog? | Licence or terms | Verdict |
|---|---|---|---|---|
| **King et al. 2012 / Strömbom et al. 2014 GPS data** ([shoalgroup.org/data](https://www.shoalgroup.org/data)) | 46 merino sheep and 1 kelpie, 3 herding events, 1 Hz GPS, 10–20 cm accuracy, already tracked | Yes | "Cite our papers, email us"; no formal licence | **Best candidate**: a different flock, dog breed and country. Needs permission to publish derived trajectories. |
| Gómez-Nava et al. 2022 ([Zenodo 6905807](https://zenodo.org/records/6905807)) | Videos of 2–4 sheep from a tower, 2.1 GB, no tracks | No | CC-BY 4.0 | Not a herding dataset |
| AUTH-Sheep / ICAERUS drone sets (Zenodo) | Drone video, some boxes and tracks | No | CC-BY 4.0 | No dog |
| de Marcken & Sarfati 2020 (YouTube drone clip) | About 1 min of a dense flock and 1 dog | Yes | Standard YouTube licence (assume no redistribution) | Qualitative check only |
| Pexels aerial herding clip | 32 s, 4K, 2 dogs | Yes | Pexels licence (free use) | Too short; sanity check only |

**Recommendation:** go, conditionally. Ask the King lab for terms and metadata first; a draft
request is in `docs/data_request_king_lab.md`. If they agree, extract the driving phases (dog
behind the flock) so they match how the Jadhav data defines a drive, resample to their 1 Hz, and
use them as an out-of-distribution test of the learned kernels and the calibrated model.

## Phase 3b: recalibration with weight-share priors (M2, M2τ)

The heading rule depends only on the *ratios* of its six weights. M2 and M2τ therefore give each
weight an independent Exponential(1) prior, which is a uniform prior on their shares, and report
shares instead of fixing w_rep.

| | M2 (dt = 1 s) | M2τ (dt = 0.2 s) |
|---|---|---|
| Personal space r_rep | 0.46 m (0.14–0.86) | 0.11 m (at the prior floor) |
| Notice radius for the dog r_dog | 5.8 m (5.4–6.2) | 7.0 m (6.6–7.4) |
| Heading relaxation τ_s | — | 2.0 s (0.3–2.6) |
| **Alignment share** | **0.81** | **0.99** |
| Repulsion share | 0.17 | < 0.01 |
| Attraction, dog, inertia and noise shares | < 0.01 each | < 0.01 each |
| Held-out error (Q2) | **0.86** (best at dt = 1 s; M1: 1.33) | 1.15 (M1τ: 1.03) |
| Hierarchy ρ / dog's rank | — | 0.23 (0.15–0.32) / **1 of 15** |

**Short windows choose an alignment-only model.** Each 12 s window starts from a real, already
cohesive flock, and copying a random neighbour's heading preserves that formation well enough to
match the summaries. Attraction is therefore not constrained. M2τ even comes close to the data's ρ,
but for the wrong reason: it makes **the dog the most-followed animal** (rank 1, against rank 15 in
the real flock), because the sheep simply lag the replayed dog's turns. Matching summary statistics
does not guarantee the right causal structure, and the hierarchy test catches that.

## Phase 6 (Q4): closed-loop herding

`experiments/dog_control.py`. The task is to drive the flock 60 m to a target within 300 s, at
N = 14, 50 and 100, with 64 runs per condition. The tuned heuristic is chosen on separate runs.

**The dogs:**
- **Published:** the Strömbom heuristic (drive point 2√N m behind the flock, 1.5 m/s, instant
  turning).
- **Data-derived** (`shepherd/dog_policy.py`): every quantity measured from the real collie. It
  works 4.2 m behind the flock centre, flanks sideways as an Ornstein–Uhlenbeck process (SD 2.5 m,
  correlation time 2.7 s), turns toward its aim with a 2.2 s time constant, and has speed
  0.39 m/s per metre of distance to its aim, capped at 3.0 m/s.
- **Tuned heuristic:** the best of a small grid over drive distance, speed and flanking.

| Flock model | N | Published dog | Data-derived dog | Tuned heuristic |
|---|---|---|---|---|
| **Published sheep (M0)** | 14 | 100% success, 68 s | 100%, 76.5 s, dog < 3 m from a sheep 25% of the time | 100%, 69 s |
| Published sheep (M0) | 50, 100 | 0% (flock larger than the 12 m notice radius) | 0% | 0% |
| Calibrated M2τ, grazing | 14–100 | 0% | 0–3% | 0% |
| Calibrated M2τ, social always on | 14–100 | 0–2% | 2–9% | 2–3% |

- **The calibrated flocks fall apart in closed loop:** cohesion 13–180 m. They were fitted only on
  12 s windows that start from real flocks, and they carry almost no attraction. So **calibrating
  on short horizons does not give a stable closed-loop flock simulator.** The fix is to calibrate on
  several horizons, adding long windows or a closed-loop cohesion statistic; that is the next
  modelling step.
- **On the published flock** at N = 14, all three dogs succeed. The collie-style dog is about 10%
  slower and works much closer to the sheep. With fixed sheep perception, no single dog can drive
  50 or 100 sheep.
- **This answers Q4 only partly.** A data-derived dog shows no performance advantage here, and a
  real test needs a flock model that is cohesive over long horizons.

## Phase 3c: multi-horizon calibration (M3, M3τ)

The short-window fits (M2, M2τ) fall apart over whole drives, because nothing in 12 s windows that
start from real, cohesive flocks constrains cohesion. M3 and M3τ add a second horizon:
- **The long runs:** every real drive of at least 20 s is re-run from its first frame for its full
  length, in **closed loop** with the dog policy measured from the collie
  (`scenario.closed_loop_drives`).
- **The extra statistics:** flock-only, so the fixed dog model cannot bias the sheep parameters.
  They are cohesion, nearest-neighbour distance, polarization, elongation, **cohesion drift** (end
  minus start) and the **share of sheep more than 5 m from the centre**.

Priors are on weight shares, as in M2. Run `experiments/build_table.py --model M3 | M3tau`, then
`experiments/multi_horizon.py`.

| | Held-out error (short windows) | Long drives: C | Drift | Far (> 5 m) | P | E | Hierarchy ρ, dog's rank |
|---|---|---|---|---|---|---|---|
| **Real flock** | — | **1.13** | **0.00** | **0.00** | **0.95** | **1.02** | **0.30, 15th** |
| Published M0 | 3.1 | 2.90 | 0.35 | 0.08 | 0.83 | 0.70 | 0.09 |
| M2 (short only, 1 s) | 0.86 | 3.18 | 1.81 | 0.16 | 0.81 | 0.88 | — |
| **M3 (two horizons, 1 s)** | 0.87 | **2.42** | **1.35** | **0.09** | **0.95** | 0.75 | — |
| M2τ (short only, 0.2 s) | 1.15 | 1.96 | 0.58 | 0.08 | 1.00 | 0.81 | 0.23, **1st** |
| **M3τ (two horizons, 0.2 s)** | 1.15 | **1.71** | 0.70 | **0.05** | **0.98** | **0.94** | **0.20, 11th** |

- **Adding the long horizon makes whole drives clearly more realistic at no cost on short windows.**
  M3τ is the closest parametric model to the real flock over whole drives.
- **The family still cannot hold the flock as tightly as real sheep do** (1.7 vs 1.1 m), and it
  drifts slowly. Its attraction share stays under 1%: random-neighbour attraction looks
  structurally too weak. The learned metric attraction kernel (Phase 4) is the better candidate
  mechanism.
- **The response to the dog is not identifiable from driving data.** The posterior share of
  "flee the dog" is 0.7% (90% CI 0.2–2.4%) for M3, and below 0.1% for M3τ. Real drives start with
  the flock already moving in the drive direction, so copying neighbours and inertia carry it, and
  the data never require the dog's push. As a result, the calibrated flock **cannot be turned
  around**: in the viewer's back-and-forth task it completes 0.2–0.3 trips per 10 minutes, against
  5 for the published model. Identifying the dog response needs data where the dog changes the
  flock's direction (collecting, turns at the posts) or a direct estimate from the tracks. The
  learned dog kernels (Phase 4) are such an estimate.

**The viewer** (`viz/`, republished) now opens on the **Calibrated** preset (M3τ posterior medians
and the measured collie) in a new **"Re-run the real drives"** scenario, where the model is
validated. The **back-and-forth** scenario and the **Short-horizon fit** preset are there to show
the limits above. `viz/make_presets.py` regenerates the presets from the tables, and the JS
simulator matches the Python model with all extensions on (cohesion 0.850 vs 0.849, P 0.935 vs
0.934).

## Phase 3d: fixing the dog response with the learned dog kernels (M4τ)

Driving data cannot identify how strongly sheep react to the dog (Phase 3c). The learned kernels
(Phase 4) estimate that reaction directly from the tracks:

a_dog = φ_D(r) ê_away + ψ_V(r) v_dog + ψ_L(r) · max(0, closing speed) · ê_away

`model.Params.dog_kernel` adds it to the heading rule as a turn of dt · a_dog / v_sheep, weighted by
the sum of the other weights times a gain g. The second-order model would rotate the heading by
that much per step, so g = 1 is the direct translation. Run `experiments/dog_response.py`, which
uses M3τ posterior draws, the measured collie, and dt = 0.2 s.

| Gain g | Short-window error | Long drives: C | Drift | P | 90° turn test success |
|---|---|---|---|---|---|
| 0 (M3τ) | 1.04 | 1.71 m | 0.70 | 0.98 | 11% |
| 1 | 0.86 | 1.96 m | 1.04 | 0.96 | 69% |
| **2** | **0.78** | **1.44 m** | **0.40** | **0.94** | **75%** (about 70 s) |
| 4 | 0.83 | 1.32 m | 0.21 | 0.94 | 34% |
| Real flock | — | 1.13 m | 0.00 | 0.95 | — |

- **g = 2 gives the best fit and makes the flock steerable.** It lowers the short-window error by
  25%, brings long-drive cohesion and drift closer to the real flock, and raises the share of
  successful 90° turns from 11% to 75%. A gain above 1 matches the synthetic finding that noisy
  tracks shrink the learned kernels toward zero.
- **Still missing:**
  - the flock's turn away from a flanking dog (Fig 4a slope −0.27 vs −0.77);
  - elongation (the flock is too wide);
  - sustained cohesion over many 180° reversals. In the viewer's back-and-forth task: about 1 trip
    per 10 min, against about 0.1 without the learned response and 5 for the published model.

  All three point to the same gap: random-neighbour attraction is too weak. The natural next step
  is to put the learned metric attraction kernel into the heading rule as well.
- **Viewer:** the Calibrated preset is now M3τ plus the learned dog response (g = 2). The JS port
  matches Python with the term on (C 0.581 vs 0.584, P 0.742 vs 0.739). M3τ without it is kept for
  contrast.
