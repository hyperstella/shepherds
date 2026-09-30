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
