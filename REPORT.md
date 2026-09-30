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
