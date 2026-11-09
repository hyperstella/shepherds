# The Jadhav ↔ Di Lorenzo bridge simulation: assumptions, equations, mappings and a first comparison

**Question.** How do sheep–sheep interactions, movement rules and the available herder resources
affect the ability to reach a desired flock density?

**What this adds.** One simulator, `shepherd/bridge.py` (Python reference) and `viz/bridge_sim.js`
+ `viz/bridge.html` (the interactive page, a port of it), in which the three parts of a shepherding
problem are chosen independently:

| Part | Choices | Provenance labels |
|---|---|---|
| Followers | the Jadhav/Strömbom **heading rule** or Di Lorenzo's **first-order rule**; activation; social terms; noise mechanism | reference / adaptation / extension / hybrid |
| Controller | Jadhav's **Strömbom dog** (1 dog published; n dogs adaptation); Di Lorenzo's **feedforward design + Theorem-1 flow**; a static ring; none | reference / adaptation / heuristic |
| Objective | a smooth localized target density (Gaussian, or von Mises on a torus); the L² density error with its finite-population floor; occupancy; cohesion | measurement only |
| Domain | open plane or periodic square (wrapped displacements, circular-mean centroid everywhere) | |

The published Jadhav model is untouched (`shepherd/model.py`, its results and tests). The bridge
reproduces it bit for bit in its reference configuration (`tests/test_bridge.py`), and the
Di Lorenzo design reproduces the herder count of their own example (§3.3).

Evidence labels, as in the other documents: **[E]** exact identity, **[P]** proved under stated
assumptions, **[S]** simulation result, **[H]** hypothesis.

Run: `.venv/bin/python viz/build.py` is not needed; open `viz/bridge.html` through the `viz` launch
configuration (or any static server in `viz/`). Experiments: `.venv/bin/python
experiments/bridge_compare.py` writes `results/bridge_compare.json` and two figures. Tests:
`tests/test_bridge.py` (Python) and `tests/test_bridge_js.py` (the JS port against Python, needs
node).

---

## 1. What each model actually assumes

### 1.1 Side by side

| | Jadhav et al. 2024 (Strömbom-type) | Di Lorenzo et al. 2024 (2411.04791) | Di Lorenzo et al. 2026 (2604.11353) |
|---|---|---|---|
| State per follower | position, unit heading; fixed speed v_s = 1 m/step | position only (overdamped) | position only |
| Movement equation | new heading = unit( h v + ρ_a r̂ + ρ_d d̂ + c â + e η̂ + a l̂ ); x += v_s·heading | dT = α Σ_j f(T − H_j) dt + √(2D) dB, α = 1/(N_T+N_H) | dx_k = (1/(N_L+N_F)) [Σ_j f^FL + Σ_m f^FF] dt + √(2D) dB |
| Sheep–sheep | repulsion (unit sum, r < r_a = 2 m); attraction to 4 random of the 10 nearest (code; paper says 5); alignment: copy 1 of those | none ("non-cohesive") | Morse kernel f^FF = (1/ℓ_r) e^(−r/ℓ_r) − (ζ/ℓ_a) e^(−r/ℓ_a) (repulsion − attraction); no alignment |
| Dog → sheep | unit vector away from the dog, weight ρ_d = 1, only within r_s = 12 m | f(x) = sign(x) e^(−‖x‖/L), additive, all distances, periodized | f^FL as in 2024 |
| Noise | additive **directional**: a unit vector of uniform random angle, weight e = 0.3 (code; paper 0.5), inside the normalized heading sum; speed unaffected | **positional Brownian**, diffusion D = 0.01 | positional Brownian, D |
| Activation / grazing | a sheep with no dog within r_s and no neighbour within r_a **does not move**; its heading is unchanged. No grazing walk, no noise while standing. Supp. Sec. 3 "modified" model: every sheep applies the social rule; the dog term only within r_s = 3 m; speed = mean speed of its alignment neighbours, reset to v_s within r_s, all speeds start at 0 | always moving (diffusing) | always moving |
| Boundary | open plane; `box_length` = 250 m only sets the start point | periodic square [−π, π]² | periodic (1D ring; 2D/3D square) |
| Controller | one dog, closed loop: if a sheep is more than f_n = r_a N^(2/3) from the centroid, go to l_c = r_a beyond it (collect), else to p_d = r_a √N behind the centroid on the line from the origin (drive); speed 1.5 with heading noise e; creep 0.05 m/step when within r_a of a sheep | many herders, single integrators, **feedforward in the targets**: herder density ρ̄_H computed once from the goal by deconvolution; herders converge to it by a Poisson-solved velocity field with gain K_H = 10 (Theorem 1); density estimated by a Gaussian KDE (bandwidth 0.4) | same two layers; leaders' control law with gain K |
| Objective | reach the origin; no density objective, no hold (the dog keeps driving) | the target density ρ̄_T (von Mises, k = 3/r*) in L²; metric: share of targets inside r* | ρ̄_F in L² and KL divergence |
| Herder resource | one dog | minimum herder mass M̂_H = ∫(H − min H) → N_H = N_T M̂/(1 − M̂) (Eqs. 11–12) | feasible interval M̂_1 ≤ M_L ≤ M̂_2 (Theorem 1), depends on f^FF |
| Guarantee | none (simulation) | Theorem 2: ‖e_T‖₂ decays as exp(−D(2 − ‖G‖∞) t) if ‖G‖∞ < 2, G = ∇·(∇ρ̄/ρ̄) | local stability if D(2 − ‖G‖∞) > F, with F built from norms of f^FF |

### 1.2 Discrepancies between paper, published code and this repository

Jadhav:
1. **Parameters.** The Methods text gives n_att = 5 and e = 0.5; `simulation_hm.m` uses 4 and 0.3. The
   paper says the dog slows to 0.05·v_D; the code steps 0.05 m. We use the code (REPORT.md, item 1).
2. **Standing sheep.** `herding_model.m` moves a sheep outside r_s only if it is repelled, and then
   without noise; otherwise `pos_s(i,:) = pos_s(i,:)` and the heading is kept. A prototype that
   let distant sheep turn randomly while standing still was *not* faithful: the published rule
   neither turns nor moves them. `activation='published'` reproduces this. Background wandering
   (`wander_speed`) and always-on social motion (`activation='always'`, the repository's old
   `social_always`) are labelled extensions/adaptations. The paper's own alternative is the
   Supplementary modified model, reproduced as `activation='contagion'` from
   `herding_model_sim.m`: it also has no random walk; motion spreads through the speed average.
3. **Modified model details.** `simulation_hm_sim.m`: r_s = 3, 8 of 10 neighbours attract, 2
   align, drive point 2 m behind, 500 steps. Its dog uses `<` where the base model uses `<=`
   (contact) and `>` for `>=` (collect); these differ only at exact equality and are ignored.
4. **Objective.** The code drives toward the origin and never stops. The Gaussian target used here
   is a *measurement*, not something the Strömbom dog knows about. The reach-and-release hold
   (`hold_radius`, from `model.py`) is the repository's extension.
5. **One dog.** Jadhav has one dog. With n > 1 the bridge fans the drive points by `spread_angle`
   and sends the dog nearest a straggler to collect: an adaptation, labelled "multi-dog Strömbom".
6. **Periodic domain.** Not in Jadhav. The bridge wraps every displacement and uses the circular
   mean for the centroid, so the published dog's rules read the same on a torus.

Di Lorenzo 2024:
7. **Kernel.** The periodic kernel has no closed form in 2D; the paper periodizes
   sign(x) e^(−‖x‖/L). We sum the (2·images+1)² nearest images (`Domain.images`); the design grid
   uses the same sum, so controller and dynamics see one kernel. With L = π on a 2π box the image
   terms fall as e^(−2n): images = 0, 1, 2, 3 give M̂ = 0.177, 0.231, 0.267, 0.275. Their N_H = 280
   for N_T = 720 is M̂ = 0.28 [S, reproduced at images ≥ 3].
8. **Target density.** Their Eq. 5 as printed, Z exp{kᵀc₁ + c₂ I₂ c₂ᵀ}, carries a term cos²x₁ +
   sin²x₂ beyond the product von Mises exp{k(cos x₁ + cos x₂)}. With that term M̂ = 0.70, which
   contradicts their 280 herders; the product form gives 0.275. We use the product form, as
   `continuification_driving.md` did, and read the extra term as a typesetting slip.
9. **Deconvolution.** Their Appendix A pseudo-inverts the discretized convolution matrix. We divide
   in Fourier space with Tikhonov weight `reg` = 10⁻³ of the kernel's peak power; M̂ changes by
   < 0.001 between reg = 10⁻² and 10⁻⁵. They write the shift as A = min H; we use ρ̄_H = H − min H,
   the only reading that is non-negative with minimal mass.
10. **Herder flow sign.** Their Eqs. 20–21 and 26 are not mutually consistent in sign; we take
    ∇·w = −K_H (ρ̄_H − ρ_H), which makes the herder error decay (checked:
    `test_dilorenzo_herders_converge_onto_the_designed_density`).
11. **Density estimate.** Theirs: Gaussian KDE, bandwidth 0.4. Ours: bilinear deposit on the grid,
    then FFT convolution with the periodic Gaussian: a KDE up to one cell of smoothing.
12. **Surplus and shortfall.** They pick N_H from M̂. When the user picks a different N_H, the
    surplus mass is spread uniformly (a uniform herder density exerts no drift on a torus, since the
    kernel is odd) and a shortfall scales the design down and is marked **infeasible** in the page
    and in `Design.feasible`, so the shortfall can be observed.
13. **Time step and size.** Their Euler–Maruyama dt = 0.01, T = 200, 720 + 280 agents. The Python
    experiments use dt = 0.05; the page's preset uses 144 + 56 agents so it stays interactive (the
    feasibility panel is independent of N_T).
14. **Where the target vanishes.** D∇ρ̄/ρ̄ is unconstrained where ρ̄ ≈ 0 (there is no flux to
    balance). We compute it as D∇ρ̄/(ρ̄ + 10⁻⁴ max ρ̄) (`Target.floor`). For their von Mises goal
    this changes M̂ from 0.2754 to 0.2760; for a Gaussian on a torus it removes a seam artefact
    (∇ln ρ̄ = −x/σ² is a sawtooth across the seam, and its Gibbs ringing dominated the
    deconvolution: 15.7 dogs with 77 % drift error became 1.7 dogs with 5 % in the E3 setting).
15. **Speed.** Their herders have no speed limit. `Herders.v` caps the flow speed when finite
    (labelled adaptation); the reference presets use ∞.

Di Lorenzo 2026:
16. The theory is one-dimensional with a 2D/3D extension. We use the 2D Morse kernel with the 1D
    amplitudes (1/ℓ_r, ζ/ℓ_a), periodized like f. Their follower alignment does not exist; the
    bridge's `ali_fo` (a drift toward the neighbours' mean displacement rate over the last step)
    is an **extension** so that the question "does alignment change the requirement" can be asked.

### 1.3 Reference configurations vs adaptations (presets)

| Preset | What it is | Label |
|---|---|---|
| `jadhav_reference` | `herding_model.m` with `simulation_hm.m` parameters, open plane, start 250 m out | reference (equal to `model.simulate` up to rounding) |
| `jadhav_modified` | `herding_model_sim.m` (Supp. Sec. 3) | reference |
| `dilorenzo_reference` | 2411.04791 §IV-A: torus 2π, 720 + 280, D = 0.01, L = π, k = 6/π, K_H = 10 | reference up to items 7, 9–11, 13 |
| `dilorenzo_interacting` (page) | the 2024 setting plus the 2026 weak Morse kernel | 2D analogue of their 1D example |
| `hybrid_jadhav_sheep_dilorenzo_dogs` | heading sheep, always active, 80 m torus, Di Lorenzo herders | **hybrid**: rule, activation, domain and controller change together, on purpose |
| `hybrid_first_order_sheep_strombom_dogs` | first-order sheep in metres (push 2.3 e^(−r/5 m), Morse cohesion, D = 0.12) under one Strömbom dog | **hybrid** |
| `dilorenzo_metres` | `dilorenzo_reference` rescaled to metres (12/π m and 1 s per unit: 24 m box), 36 + 14 | reference, rescaled (§4.5) |
| `dilorenzo_jadhav_sheep` | the same scenario and herders, designed from *their* follower model; the sheep are Jadhav's published baseline | **one component swapped**: followers only (§4.5) |

---

## 2. The equations, as implemented

Notation: x_i sheep positions, h_i unit headings, y_m dog positions, Δ(a, b) = b − a wrapped to the
box when periodic, ê(z) = z/|z|, dt the step.

### 2.1 Heading rule (`rule='heading'`) [E, reproduces `herding_model.m`]

For each sheep i, reading the state at t − 1:

    r_i  = −ê( Σ_{j ≠ i, |Δ(x_i,x_j)| < r_rep} ê(Δ(x_i, x_j)) )             (0 if no such j)
    d_i  = Σ_{m : |Δ(x_i,y_m)| ≤ r_dog} −ê(Δ(x_i, y_m))                      (sees_i = that set is non-empty)
    A_i  = n_att of the k_perceive nearest j, chosen uniformly at random each step
    a_i  = ê( Σ_{j ∈ A_i} Δ(x_i,x_j) / (|Δ(x_i,x_j)| + ε) )
    l_i  = ê( Σ_{j ∈ L_i} h_j ),   L_i = n_ali of A_i at random
    η_i  = (cos θ_i, sin θ_i),  θ_i ~ U(0, 2π)
    u_i  = ê( w_inertia h_i + w_rep r_i + w_dog d_i + w_att a_i + w_ali l_i + e_noise η_i )
    g_i  = ê( w_inertia h_i + w_rep r_i )                                     (the "grazing" direction)

Activation decides the new heading h_i⁺ and the step length s_i:

| activation | h_i⁺ | s_i |
|---|---|---|
| `published` | u_i if sees_i, else g_i if repelled, else h_i | v if sees_i or repelled, else 0 |
| `contagion` | u_i (d_i only within r_dog, as above) | v if sees_i, else mean_{j ∈ L_i} s_j (speeds start at 0) |
| `always` | u_i | v |
| `published` + `wander_speed` > 0 | as published, but a standing sheep takes ê(w_inertia h_i + e_noise η_i) | wander_speed for a standing sheep |

    x_i⁺ = x_i + s_i dt h_i⁺  [ + √(2 D dt) ξ_i,  ξ_i ~ N(0, I), only if D > 0: hybrid ]

With one dog, `published`, D = 0 and `wander_speed` = 0 this consumes the same random numbers in
the same order as `shepherd.model.simulate` and gives the same trajectories up to rounding.

### 2.2 First-order rule (`rule='first_order'`) [E, Di Lorenzo Eq. 1b with the 2026 pair term]

    f(r) = a_dog e^(−r/L_dog)                        push from a dog, outward
    g(r) = g_rep e^(−r/l_rep) − g_att e^(−r/l_att)   pair kernel, positive = repulsion
    K_f(z) = Σ_images f(|z|) ê(z),  K_g likewise     (images = (2·Domain.images + 1)² periodic copies)
    v_i = s [ Σ_m K_f(Δ(y_m, x_i)) + Σ_{j ≠ i} K_g(Δ(x_j, x_i)) ]     s = 1/(N + N_dogs) if population_scaling else 1
        + ali_fo · mean_{j ≠ i, |Δ| ≤ r_ali} (x_j − x_j⁻)/dt           (extension; 0 by default)
    x_i⁺ = x_i + v_i dt + √(2 D dt) ξ_i

Their reference: a_dog = 1, L_dog = π, population scaling on, g = 0 (2024) or Morse (2026).
In metres the repository's convention (`feasibility.py`) is population scaling off and a_dog in
m/s (fitted 2.3 m/s at L = 5 m).

### 2.3 Controllers

**Strömbom / Jadhav** (`'strombom'`) [E for n = 1]. With centroid G (circular mean on a torus),
straggler index k = argmax |Δ(G, x_i)|, goal c:

    collect if |Δ(G, x_k)| ≥ l_sep:    aim = G + (|Δ(G,x_k)| + l_collect) ê(Δ(G, x_k))
    drive otherwise:                   aim = c + (|Δ(c,G)| + l_drive) ê(Δ(c, G))
    direction = ê( ê(Δ(y, aim)) + noise η ),   y⁺ = y + v_dog dt · direction
    contact (a sheep within r_rep):    y⁺ = y + slow_step dt · previous direction

with l_sep = r_rep N^(2/3), l_drive = r_rep √N, l_collect = r_rep, noise = e_noise. For n > 1 the
drive aims are rotated about G by spread_angle·(m − (n−1)/2) and the dog nearest the straggler
collects (adaptation). Optional hold: when |Δ(c, G)| ≤ hold_radius every dog withdraws to
hold_standoff = r_dog + 3 m from G and waits (extension from `model.py`).

**Di Lorenzo** (`'dilorenzo'`, periodic only) [E for the design; the flow is their Theorem 1].
On an n × n grid with the unit-mass target ρ̂ and mass fractions m_T, m_H (or counts when
population scaling is off):

    v̄     = D ∇ρ̂/(ρ̂ + floor·max ρ̂) − m_T (K_g ∗ ρ̂)       required drift (2024 Eq. 9; 2026 Eq. 17)
    Ĥ(k)  = conj(F̂(k))·V̂(k) / (|F̂(k)|² + reg·max|F̂|²),  Ĥ(0) = 0      deconvolution by K_f
    ρ̄_H   = H − min H  (+ (m_H − M̂)/|Ω| if m_H ≥ M̂ := ∫(H − min H), else scaled by m_H/M̂: infeasible)
    ρ_H   = K_b ∗ deposit(y)                               Gaussian KDE of the herders, bandwidth b
    ŵ(k)  = i k K_H ê_H(k) / |k|²,   e_H = ρ̄_H − ρ_H       so ∇·w = −K_H e_H, ∇×w = 0
    u     = w / ρ_H sampled at y_m,  y_m⁺ = y_m + u dt   (capped at v_dog if finite)

**Static ring** (`'static_ring'`): n dogs fixed at ring_radius around c (heuristic, no sensing).

**Density feedback** (`'density_feedback'`) [heuristic, from neither paper]. Each step, from the
sheep positions and the target:

    s_i   = ρ̂_b(x_i) − ρ̄_b(x_i)                              per-sheep surplus (KDE minus smoothed target)
    drive if |Δ(c, G)| > fb_drive_radius (hysteresis: until < ½ of it): Strömbom drive aims, fanned
    else:  targets = greedy max-s_i with s_i > fb_threshold·max ρ̄_b, at least fb_exclusion apart, one per dog
           aim_m = x_i + fb_standoff ê(Δ(c, x_i))              the nearest free dog, beyond the sheep on the ray from the goal
           dogs without a target wait at G + (max_i|Δ(G,x_i)| + reach + 1) ê_out   (beyond reach of every sheep)
    release (optional): hold all dogs while error < fb_release, re-engage above fb_engage = 1.5 fb_release
    y_m⁺ = y_m + min(v_dog, |Δ(y_m, aim_m)|/dt) dt ê(Δ(y_m, aim_m)); creep slow_step when within r_rep of a sheep

reach = r_dog (heading rule) or L_dog (first-order). Defaults: fb_standoff = r_dog − 2 (heading) or
L_dog/2, fb_exclusion = reach, fb_drive_radius = r_goal, fb_threshold = 0.25. It controls the
per-sheep density surplus by pushing surplus sheep toward the goal; it is feedback in the sheep,
not a density design, and the Jadhav sheep's unit response within r_dog means that any dog
within reach moves every nearby sheep, so for them it acts on the flock, not on its shape.
Design notes from building it: without the threshold the dogs chase sampling noise in a matched
flock and scatter it; without the spread-dependent wait distance a waiting dog sits inside the
notice radius of the nearest sheep; without the drive hysteresis the dogs chatter at r_goal.

### 2.4 Objective and measurements

    ρ̄(x) ∝ exp(−|Δ(c,x)|²/2σ²)  (Gaussian)   or   exp{k (cos(2πx/L) + cos(2πy/L))}  (von Mises, torus)
    error(t)   = ‖ρ̂_b − ρ̄_b‖² / ‖ρ̄_b‖²,   ρ̂_b = (1/N) Σ_i K_b(· − x_i),  ρ̄_b = K_b ∗ ρ̄
    floor      = E error for N independent draws from ρ̄ = σ²/(N b²) on the plane  [E]
                 (on a torus: (K_{b√2}(0) − ∫ρ̄_b²)/(N ∫ρ̄_b²), computed on the grid)
    occupancy  = #{i : |Δ(c, x_i)| ≤ r_goal}/N,  r_goal = σ √(−2 ln 0.05) (95 % of ρ̄) or r* = L/4
    cohesion   = mean_i |Δ(G, x_i)|,   centroid error = |Δ(c, G)|

On the plane the error is a closed-form sum of Gaussians of pair and goal distances; on a torus it
is computed on the grid. The two agree to 0.02 on a 60 m box (`test_density_error_closed_form…`).
The floor is what to read residual error against: error at or below it is sampling, not control.
Ongoing noise adds a second, dynamic floor, read as the tail mean of each run.

### 2.5 Parameter mapping

| Python / JS field | Jadhav (MATLAB) | Di Lorenzo | Note |
|---|---|---|---|
| `n` | `no_shp` = 14 | N_T (720), N_F | |
| `v` | `vs` = 1 | — | first-order followers have no speed |
| `w_inertia` | `h` = 0.5 | — | |
| `w_rep`, `r_rep` | `rho_a` = 2, `rad_rep_s` = 2 | ≈ g_rep, l_rep (2026) | different functional forms |
| `w_att`, `k_perceive`, `n_att` | `c` = 1.5, `K_atr` = 10, `k_atr` = 4 | ≈ g_att, l_att (2026) | topological vs metric |
| `w_ali`, `n_ali` | `alg_str` = 1.3, `k_alg` = 1 | none | `ali_fo` is an extension |
| `w_dog`, `r_dog` | `rho_d` = 1, `rad_rep_dog` = 12 | `a_dog`, `L_dog` ↔ f = e^(−r/L), L = π | unit push vs decaying push |
| `e_noise` | `e` = 0.3 | — | directional |
| `D` | — | D = 0.01 | positional Brownian |
| `population_scaling` | — | α = 1/(N_T + N_H) | |
| `Herders.n`, `v` | 1 dog, `v_dog` = 1.5 | N_H = 280, no cap | |
| `l_sep`, `l_drive`, `l_collect`, `slow_step` | `f_n`, `pd`, `pc`, 0.05 | — | |
| `K_H`, `bandwidth`, `grid`, `reg` | — | K_H = 10, 0.4, (their mesh), Appendix A | |
| `Target.sigma` / `k`, `r_goal` | — (origin) | k = 3/r* = 6/π, r* = π/2 | |
| `Domain.periodic`, `size`, `images` | open plane | [−π, π]², periodic kernel | |

### 2.6 The convenience slider, defined

λ ∈ [0, 1] on the page changes **only** these fields of the heading rule, linearly:
w_att 1.5 → 0, w_ali 1.3 → 0, w_inertia 0.5 → 0, e_noise 0.3 → 0, D 0 → 0.12 m²/s. It never changes
the update rule, the activation, the dog response, the speed or the domain. So λ = 1 is a
memoryless unit-speed walker that flees dogs within r_dog with Brownian jitter: "Di Lorenzo-like"
in its social structure (no interaction, positional noise) but **not** Di Lorenzo's follower,
which has no fixed speed and an additive, decaying push. Reproducing either paper requires the
rule switch and the presets, which is why the page keeps those separate.

---

## 3. Herder resources: the population constraint

### 3.1 Their calculation, reproduced [E, S]

Di Lorenzo's bound deconvolves the required drift v̄ = D∇ln ρ̄_T by the push kernel and shifts
the result up to be non-negative; the shifted mass M̂_H is the least herder *mass fraction*
N_H/(N_T + N_H) that can hold ρ̄_T, and N_H = N_T M̂/(1 − M̂) (their Eqs. 9–12).
`bridge.design_herders` does this on the grid; `bridge.feasibility_scan` does it as a function of
the herder mass, which is needed once followers interact (2026: the required drift carries
M_T = 1 − M_H, so the feasible set is where M_H ≥ M̂(M_H), an interval [M̂_1, M̂_2]).

For their example (k = 6/π, D = 0.01, L = π): M̂ = 0.275 at images = 3, i.e. 274 herders for 720
targets, against their 280 [S]. M̂ is proportional to D and independent of N_T
(`test_herder_mass_scales_with_D_and_is_independent_of_N`): it is a fraction. Their simulation
reported 92 % of targets inside r*; ours, 720 + 280, dt = 0.05, images = 1, reaches 83 % at
T = 200 with the density error at the finite-N floor from t ≈ 160 [S].

### 3.2 Its assumptions, and why its threshold does not transfer

The bound assumes (i) followers whose *only* drift is the herders' additive push (plus a pair
kernel), (ii) positional Brownian noise, (iii) a periodic domain, (iv) a herder *density*, i.e.
the many-herder limit, and (v) a static goal. Under (i)–(ii) the target equation with feedforward
herders is a linear Fokker–Planck equation and converges unconditionally
(`continuification_driving.md` §5).

The Jadhav sheep break (i) and (ii): the dog term is a unit vector inside r_dog, normalized
together with the social terms, the speed is fixed, and the noise is directional. There is no
drift to deconvolve, so M̂ is not defined for them, and transferring 0.28 (or any M̂) to the
heading sheep would be unjustified. The page says so when the heading rule is selected. For the
hybrid with heading sheep under Di Lorenzo herders, the design is still computed (it needs a
push kernel: `a_dog`, `L_dog` are then *fictitious* parameters of the controller, not of the
sheep) and the panel labels the configuration a hybrid.

The measurable statement for Jadhav and the hybrids is the **observed resource requirement**:
the dog count (and social weights) at which occupancy and density error stop improving. §4
reports it.

---

## 4. The comparison experiment (`experiments/bridge_compare.py`) [S]

All runs: identical sheep starts across the variants of one comparison (the seed draws the sheep
before the dogs), the last quarter of each run averaged, and the density error read against the
finite-N floor. Numbers are in `results/bridge_compare.json`; figures
`results/figures/bridge_compare.png` and `bridge_compare_jadhav.png`.

### 4.1 E1: first-order followers on the torus, herder count × follower interactions [S]

Their units: torus [−π, π]², 200 followers, D = 0.01, L = π, von Mises goal k = 6/π, K_H = 10,
bandwidth 0.4, dt = 0.05, T = 100, 2 seeds, dogs 5–160, kernel images = 1 (so M̂ = 0.231 and the
feasibility count is N̂_H = 200·0.231/0.769 = 60; the scan's 0.02 grid reports 63). Tail-mean
occupancy within r* = π/2 (uniform would be 0.20) and density error (floor 0.040):

| followers \ herders N_H | 5 | 10 | 20 | 30 | 40 | 50 | 65 | 80 | 100 | 120 | 160 | N̂_H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-interacting (2024): occupancy | 0.25 | 0.25 | 0.42 | 0.46 | 0.53 | 0.57 | 0.62 | 0.62 | 0.63 | 0.66 | 0.66 | 63 |
| non-interacting: density error | 0.70 | 0.66 | 0.39 | 0.34 | 0.24 | 0.17 | 0.12 | 0.14 | 0.14 | 0.09 | 0.08 | |
| attraction only (1/π e^(−r/π)): occupancy | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 56 |
| attraction only: cohesion | 0.17 | 0.11 | 0.11 | 0.11 | 0.12 | 0.12 | 0.13 | 0.14 | 0.17 | 0.21 | 0.33 | |
| weak Morse (2026: (1/ℓ_r)e^(−r/ℓ_r) − (ζ/ℓ_a)e^(−r/ℓ_a), ℓ_r = π/2, ℓ_a = π, ζ = 1): occupancy | 0.20 | 0.19 | 0.25 | 0.24 | 0.24 | 0.25 | 0.27 | 0.29 | 0.30 | 0.35 | 0.41 | 569 |
| weak Morse: density error | 0.77 | 0.79 | 0.67 | 0.71 | 0.66 | 0.68 | 0.62 | 0.59 | 0.56 | 0.48 | 0.38 | |
| weak Morse + alignment (extension, ali_fo 0.5, r_ali 1): occupancy | 0.20 | 0.20 | 0.25 | 0.24 | 0.23 | 0.24 | 0.27 | 0.29 | 0.29 | 0.34 | 0.41 | 569 |

Reading:
- **Non-interacting followers: the feasibility count is where the curve bends.** Occupancy and
  density error improve steadily up to N̂_H ≈ 60 and flatten beyond it (0.62 → 0.66 from 65 to
  160 herders; error 0.12 → 0.08). The runs are 100 time units; the 720 + 280 reference reaches
  the floor at t ≈ 160, so these tails are still converging and the plateau is below the
  reference's 83 %. Below the count the design is scaled down and the confinement is partial,
  not absent.
- **Attraction alone lowers the continuum count and destroys the outcome.** The feasibility scan
  says attraction supplies part of the goal's curvature (M̂ falls from 0.231 to 0.22 at this
  mass, 56 herders), as `continuification_driving.md` §14 derived. In the simulation 200
  attracting followers collapse into one clump of radius 0.1–0.3 that sits off the goal at every
  herder count (occupancy 0). This is §5d of that document seen directly: cohesion is
  self-*cohering*, not self-*confining*; pair forces cancel in the centroid, so the clump is a
  single particle whose position only the herders hold, with a stiffness the feedforward design
  made smaller on purpose, and with 200 followers in one place the kernel density estimate behind
  the design (a smooth ρ̄_T) is no longer what the herders face.
- **The 2026 paper's weak Morse kernel is net repulsive at this flock's pair distances, and
  makes the problem infeasible for any herder mass below 0.74 (569 herders for 200).** Its zero
  crossing is at r = π ln 2 = 2.18, while 95 % of the goal lies within r* = 1.57, so the pairs that
  matter repel. The feasibility scan decomposes it: attraction alone lowers the needed mass
  (0.231 → 0.19 at M_H = 0.3), repulsion alone raises it to 3.5, and the sum sits at 1.6. In the
  simulation occupancy creeps from 0.20 to 0.41 at 160 herders with the flock's cohesion falling
  only from 2.3 to 1.9: the (scaled-down) design confines partially and the pair repulsion undoes
  the rest. The 1D paper reports a milder shift (M̂₁ 0.14 → 0.24) for the same kernel, since in one
  dimension the kernel's integral is what enters; the 2D figures here are ours, not theirs.
- **Alignment changes nothing at the density level.** The alignment extension (a drift toward the
  neighbours' mean displacement rate within one length unit) leaves every row within 0.01 of the
  Morse row. A first-order flock in diffusive equilibrium has no persistent velocity to align
  with; alignment matters where there is self-propulsion, which is the heading rule (E2, E3).


### 4.2 E2: Jadhav's sheep on the plane, Strömbom dog(s), dog count × attraction × alignment [S]

14 sheep start 60 m from the goal (published activation, published parameters except where
varied), 400 steps, 20 seeds, 1–4 dogs. Tail means (steps 300–400) of occupancy within
r_goal = 6.1 m; the finite-N floor of the density error is 0.45.

| | dogs | w_att 0.5 | **baseline** (1.5, 1.3) | w_att 2.5 | w_ali 0 | w_ali 2.6 |
|---|---|---|---|---|---|---|
| published loop (no hold) | 1 | 0.02 | 0.15 | 0.17 | 0.33 | 0.08 |
| | 2 | 0.11 | 0.12 | 0.13 | 0.24 | 0.08 |
| | 4 | 0.02 | 0.10 | 0.14 | 0.12 | 0.09 |
| with reach-and-release hold (6 m) | 1 | 0.15 | 0.64 | 0.75 | 0.59 | 0.65 |
| | 2 | 0.55 | 0.59 | 0.79 | 0.57 | 0.68 |
| | 3 | 0.34 | 0.60 | 0.81 | 0.61 | 0.63 |
| | 4 | 0.28 | 0.61 | 0.73 | 0.63 | 0.67 |

Centroid error in the tail: 8–38 m without the hold, 1–4 m with it (except w_att 0.5). Density
error with the hold: 0.4–0.9 at w_att ≥ 1.5 (floor 0.45; a cohesive frozen flock can sit below
the floor of random draws), 1.0–1.5 at w_att 0.5.

Reading:
- **The controller feature, not the dog count, is the resource that matters.** Without the hold
  the published loop drives through the goal and keeps going (the overshoot of
  `continuification_driving.md` §9), so the tail occupancy is 0.1–0.3 whatever the dog count. With
  the hold one dog reaches 0.6–0.8, and dogs 2–4 add nothing at the baseline.
- **Attraction sets the occupancy reached.** With the hold, w_att 0.5 → 2.5 raises occupancy from
  0.15 to 0.75 (one dog). At w_att 0.5 the flock splits: cohesion 6–11 m, and *more dogs make it
  worse* (0.55 → 0.28 from 2 to 4 dogs; without the hold cohesion reaches 18 m with 4 dogs), because
  the fanned drive points push on a flock that does not pull itself together. This is the lateral
  inequality of `steer_and_shape.md` seen from the density side.
- **Alignment cuts both ways.** Without the hold, removing alignment is the best of the no-hold
  rows (0.33: the flock does not run through the goal as a body); with the hold, alignment above
  the baseline helps slightly (0.64 → 0.65–0.68), since the released flock freezes together.
- **Finite-population floor.** The best held flocks reach the floor (0.39–0.56 against 0.45), so
  their residual error is sampling; the no-hold rows sit at 1.5–3.3 because the flock is elsewhere.

### 4.3 E3: hybrid on the torus, Di Lorenzo herders over Jadhav sheep and over first-order sheep [S]

60 sheep on an 80 m torus, Di Lorenzo's von Mises goal with r* = 10 m (k = 3/r* = 3.82 in box
angle), D = 0.5 m²/s for every follower, herders designed for a weak push 0.8 e^(−r/12 m) m/s
(feasibility count 6.9 dogs), K_H = 1, speed cap 1.5 m/s, 600 s, 3 seeds, identical sheep starts.
The heading sheep rows are a **hybrid**: always active, with Brownian jitter D so the design has
a diffusion to confine (with D = 0 the design is empty), and the push the design assumes is not
theirs. Tail-mean occupancy within r* (floor of the density error 0.36):

| followers \ dogs N_H | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| first-order (the design's assumption) | 0.08 | 0.22 | 0.45 | 0.53 | 0.12 | 0.65 | 0.37 | 0.56 | 0.54 |
| first-order + Morse (net repulsive at 2 m) | 0.02 | 0.36 | 0.00 | 0.02 | 0.01 | 0.00 | 0.02 | 0.04 | 0.05 |
| heading, baseline (w_att 1.5, w_ali 1.3) | 0.15 | 0.11 | 0.07 | 0.09 | 0.00 | 0.11 | 0.06 | 0.13 | 0.14 |
| heading, w_att 0.5 | 0.09 | 0.06 | 0.16 | 0.12 | 0.01 | 0.32 | 0.10 | 0.31 | 0.32 |
| heading, w_att 2.5 | 0.08 | 0.18 | 0.03 | 0.25 | 0.00 | 0.16 | 0.04 | 0.02 | 0.14 |
| heading, w_ali 0 | 0.08 | 0.10 | 0.08 | 0.03 | 0.01 | 0.11 | 0.06 | 0.03 | 0.09 |
| heading, published activation | 0.06 | 0.06 | 0.06 | 0.07 | 0.01 | 0.02 | 0.11 | 0.09 | 0.14 |

(Uniform occupancy would be π r*²/L² = 0.05.) Density error of the first-order rows at N_H ≥ 6:
0.44–0.76 against the floor 0.36; of every heading row: 1.1–3.5.

Reading:
- **The design works for the followers it was derived for, and the feasibility count is where it
  starts to.** Occupancy rises from 0.1–0.2 below 6.9 dogs to 0.45–0.65 above it (the N_H = 12 dip
  is seed variance: single seeds range 0.1–0.7 at the same count), and the density error comes
  within a factor two of the floor. The remaining gap to the designed 95 % is the discrete-herder
  gap of H1: 16 dogs with a 12 m push leave holes a continuum does not.
- **The push scale is the real resource.** With a real dog's amplitude, 2.3 m/s, the design asks
  for 1.7 dogs but every count from 1 to 64 scatters the flock (occupancy ≤ 0.2; an earlier run
  of this experiment). The design needs a drift of order 0.05 m/s; a dog that pushes at 2.3 m/s
  within a few metres is a disturbance 50 times larger than the signal. Di Lorenzo's α = 1/1000
  is what makes each herder weak enough for the continuum to hold; in metres the equivalent is a
  weak push, which is not what a sheep's response to a dog looks like.
- **The heading sheep never gather, at any dog count or social setting.** Occupancy stays at
  0–0.3 and cohesion at 15–25 m. On the page the mechanism is visible: with attraction the sheep
  form a tight flock (cohesion 4 m after 800 steps in the preset) that walks at 1 m/s as a body and
  is deflected by the designed ring of herders without being held by it; the static design has no
  term for a self-propelled centroid (`continuification_driving.md` §7, §13). Attraction, alignment
  and the published standing rule move the numbers within noise. Switching the rule to first-order
  under the same 16 herders gives 87 % in the preset.
- **Net-repulsive pair forces defeat the design in metres as in their units.** The Morse rows fail
  at every count although the feasibility count is lower (5.6): the kernel is repulsive below 2 m,
  where neighbours in a 60-sheep goal of radius 10 m sit, and the feedforward design does not
  react to the spreading flock.



### 4.4 E4: a density-feedback dog against the other controllers, matched resources [S]

Same dog count and speed (1.5 m/s), same seeds and starts for every controller. Two protocols:
**hold** (the flock starts gathered at the goal: can the controller keep it matched under the
sheep's own motion?) and **reach** (the flock starts cohesive 40 m away on the plane, 30 m on
the torus). A "no dogs" row is the null. Each cell: tail-mean occupancy, and in brackets the
share of the second half of the run with occupancy ≥ 0.6 (the *sustained* measure). Controllers:
Strömbom with the 6 m / 5 m hold, the density-feedback dog with and without release at twice the
floor, and on the torus Di Lorenzo's herders. Plane: 14 published sheep, 20 seeds, 600 steps;
torus: the E3 setting, 60 sheep, 3 seeds, 600 s. `results/figures/bridge_controllers.png`.

**Plane, Jadhav's published sheep** (floor 0.45):

| protocol | dogs | no dogs | Strömbom + hold | density feedback | feedback + release |
|---|---|---|---|---|---|
| hold | 1 | 0.90 (1.00) | 0.82 (1.00) | 0.10 (0.11) | 0.90 (1.00) |
| hold | 2–3 | | 0.45–0.47 (0.3) | ≤ 0.03 | 0.90 (1.00) |
| reach | 1 | 0.00 | 0.65 (0.80) | 0.15 (0.12) | **0.86 (1.00)**, error 0.38 |
| reach | 2 | | 0.65 (0.75) | 0.05 | 0.72 (0.90) |
| reach | 3 | | 0.72 (0.95) | 0.02 | 0.60 (0.75) |

**Plane, always-active sheep** (adaptation; the flock walks off at 1 m/s on its own: no dogs 0.09):
the best of any controller is one feedback dog with release, 0.38 (hold) and 0.47 (reach), against
0.27–0.33 for Strömbom with the hold at every count; nothing sustains more than 0.44 of the time.

**Torus, heading sheep (always + D) and first-order followers** (floor 0.36):

| followers / protocol | dogs | no dogs | Di Lorenzo | Strömbom + hold | density feedback | feedback + release |
|---|---|---|---|---|---|---|
| heading / hold | 1 | 0.55 (0.43) | 0.54 | 0.26 | 0.33 | 0.18 |
| heading / hold | 8 | | 0.34 | 0.13 | 0.24 | 0.61 (0.58) |
| heading / hold | 16 | | **0.88 (1.00)** | 0.03 | 0.01 | 0.01 |
| heading / reach | 1 | 0.00 | 0.00 | **0.75 (0.52)** | 0.28 | 0.28 |
| heading / reach | 2 | | 0.00 | 0.11 | 0.23 | 0.54 (0.39) |
| heading / reach | 8 | | 0.00 | 0.22 | 0.43 | 0.43 |
| first-order / hold | 8 | 0.09 | **0.70 (0.99)**, error 0.33 | 0.09 | 0.31 | 0.49 |
| first-order / hold | 16 | | 0.87 (1.00) | 0.02 | 0.41 | 0.04 |
| first-order / reach | 8 | 0.05 | **0.74 (1.00)** | 0.05 | 0.18 | 0.30 |
| first-order / reach | 16 | | 0.66 | 0.01 | 0.16 | 0.01 |

Reading:
- **For the published sheep the feedback dog with release is the best controller at one dog, and
  the answer to the question is "yes, through the release".** It reaches 0.86 with the error below
  the floor and holds it the whole second half, against 0.65 for Strömbom with the hold; on the
  hold protocol it equals the null (0.90) because it never engages a matched flock. Without
  release it is the worst controller: it keeps pushing, and the published sheep never settle under
  a dog within r_dog. The sustained matching comes from a controller that *stops*, triggered by
  the density error rather than by the centroid distance; the surplus-pushing itself buys little
  for these sheep (any dog within 12 m moves every nearby sheep).
- **More dogs do not help the heading sheep with any closed-loop controller.** Two or three dogs
  lower every closed-loop row; on the torus Strömbom with one dog reaches 0.75 and with two
  0.11. The flock is moved as a body and more movers mean more overshoot and more splitting.
- **Holding a gathered heading flock is done best by a fence, not by a dog that reads it.**
  Di Lorenzo's 16 static herders keep the always-moving heading flock in the goal for the whole
  run (0.88, held 1.00), where every feedback controller at 16 dogs wrecks it (≤ 0.03) and at
  fewer dogs the null (0.55) is as good as anything. The static ring confines through the notice
  radius, as `continuification_driving.md` §15 says, and does nothing else; a dog that approaches
  a self-propelled flock sets it off. But the same herders cannot *gather* it (reach: 0.00 at every
  count): reaching needs one closed-loop dog, holding needs many static ones.
- **For the first-order followers Di Lorenzo's herders are best at both tasks** (0.70–0.87 held
  all the time at 8–16 dogs; 0.74 reached), the feedback dog is second (0.3–0.5) and Strömbom's
  dog fails (0.02–0.15): a diffusing flock with no fixed speed is held by a designed density, not
  pushed by one dog. Release hurts first-order followers at 16 dogs (0.04): they diffuse away
  while the dogs wait and are gone by the re-engage threshold.
- **On sustained matching beyond reaching.** With the published sheep, reaching and sustaining
  coincide once the controller withdraws (held 1.00). With sheep that keep moving, no controller
  sustains matching more than about half the time on the plane; on the torus only the static
  fence does, and only for a flock that is already there. The density-feedback heuristic adds a
  better-timed release and a modest gain at low dog counts; it does not change the picture that
  the Jadhav sheep are steered as a body and confined by the edge of the notice disc.

### 4.5 E5: Di Lorenzo's scenario with Jadhav's sheep, herders unchanged [S]

Script: `experiments/dilorenzo_jadhav_sheep.py` → `results/dilorenzo_jadhav_sheep.json`,
`results/figures/dilorenzo_jadhav_sheep.png`. Replay page: `experiments/dilorenzo_jadhav_replay.py` →
`viz/dilorenzo_jadhav_sheep.html` (one seed, three panels, identical starts). Scenario:
`bridge.dilorenzo_scenario`, presets `dilorenzo_metres` and `dilorenzo_jadhav_sheep`.

Interactive: the same presets are in `viz/bridge.html`, plus `dilorenzo_jadhav_flat` (the
flat-design control). The JS port has the same `design_model` and `substeps` fields.
`tests/test_bridge_js.py` checks that the JS port matches Python: the design to 1e-9, and the
herder trajectories through 100 sub-steps to 1e-6. Each 1 s step costs about 50 ms in the page,
so it runs at roughly 18 steps/s; a hidden browser pane throttles this to about one frame per
second.

Page controls added with this scenario:
- **"Set N_H to Di Lorenzo's count"** (Herders panel, Di Lorenzo controller). Sets the smallest
  N_H whose herder mass covers M̂, checked with the design the simulation runs; a search is
  needed when a pair kernel makes M̂ depend on N_H. Tested against Python: feasible at the
  count, infeasible one below. It reports why when no count is defined (open plane, empty design).
- **Fast-forward** ("+60 s", "To end"). Steps in about 40 ms chunks and redraws between them.
- **"herder flow sub-steps"** slider. On the Jadhav-sheep preset, 200 s takes 12.7 s at the
  faithful 100 sub-steps and 2.7 s at 20, with the same kind of end state.
- **"lock zoom"** (open plane). Stops the view refitting, which otherwise changes the on-screen
  size of the r_goal circle as the flock spreads and gathers.

**What changes and what does not.** E3 changed four things at once (rule, activation, domain,
added diffusion). Here only the followers change. The scenario is 2411.04791 §IV-A rescaled
to metres. Lengths: 12/π m per unit, so their push range L = π is the sheep's 12 m notice
radius; the box is 24 m and r* = 6 m. Time: 1 s per unit, so one published sheep step is one
unit of 0.01-step herder flow (100 sub-steps, `Herders.substeps`). Agents: 36 sheep and 14
herders, their 28 % herder fraction; 720 sheep with 2 m repulsion do not fit the box. Starts:
uniform. T = 200 s, 5 seeds, tail mean of the last quarter.

The herders keep everything of theirs:
- K_H = 10, uncapped speed, KDE bandwidth 0.4 units, 64 grid, images = 1.
- The design is computed from **their** follower model (`Herders.design_model`: push e^(−r/L),
  D = 0.01, α = 1/(N_T + N_H)), not from the sheep that are simulated.

The rescaling is exact: M̂ = 0.2406 in their units and in metres
(`test_dilorenzo_scenario_in_metres_keeps_their_design`). The sheep are `Followers()`, the
published baseline: n_att 4, e 0.3, r_dog 12 m, published activation, no positional noise.
Without `design_model` a heading sheep's design would be empty: it has D = 0 and no additive
push, so there is nothing to deconvolve (§3.2).

| Followers / herders | within r* | density error (floor 0.22) | cohesion (target 4.2 m) | centroid error | sheep seeing a herder |
|---|---|---|---|---|---|
| their followers, designed (reference) | 0.86 ± 0.01 | 0.24 | 3.6 m | 0.7 m | 1.00 |
| **Jadhav sheep, designed** | **1.00 ± 0.00** | **1.29 ± 0.17** | **1.9 m** | **0.4 m** | 1.00 |
| Jadhav sheep, designed, herders capped at 1.5 m/s | 1.00 | 1.13 | 2.0 m | 0.4 m | 1.00 |
| Jadhav sheep, flat design (same herders, D = 0 in their model) | 0.17 ± 0.16 | 1.75 | 5.3 m | 10.0 m | 1.00 |
| Jadhav sheep, no herders | 0.17 | 0.92 | 8.5 m | 8.4 m | 0 |

Herder count (feasibility count 11.4, from M̂ = 0.241):

| herders | 4 | 8 | 12 | 14 | 20 | 28 |
|---|---|---|---|---|---|---|
| their followers: within r* | 0.60 | 0.77 | 0.87 | 0.86 | 0.84 | 0.81 |
| Jadhav sheep: within r* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Jadhav sheep: density error | 0.73 | 0.75 | 0.55 | 1.29 | 0.85 | 1.46 |
| Jadhav sheep: cohesion (m) | 2.3 | 2.3 | 2.5 | 1.9 | 2.1 | 1.8 |

Box size at fixed sheep density and herder fraction. L = k·12 m, box 2k·12 m, N = 36k²:

| box (N + M) | 24 m (36 + 14) | 36 m (81 + 32) | 48 m (144 + 56) |
|---|---|---|---|
| their followers: within r* / error / floor | 0.86 / 0.24 / 0.22 | 0.80 / 0.11 / 0.10 | 0.80 / 0.05 / 0.06 |
| Jadhav sheep: within r* / error / cohesion | 1.00 / 1.29 / 1.9 m | 1.00 / 1.13 / 2.8 m | 1.00 / 1.51 / 3.4 m |
| Jadhav sheep: r* / target cohesion | 6 m / 4.2 m | 9 m / 6.3 m | 12 m / 8.4 m |

Reading:
- **The reference reproduces their result in metres.** 86 % inside r* against their 92 %
  and our 83 % in §3.1. The density error reaches the finite-N floor at every box size, and
  occupancy rises up to the feasibility count, then plateaus.
- **Jadhav's sheep reach the goal disc, in every run, and stay.** All sheep are inside r*
  within about 20 s, with the centroid within 0.4 m. The design is what does it. The same
  herders with a flat design leave the flock clumped 10 m off the goal (occupancy 0.17). With
  no herders the published sheep stand where they started.

  The mechanism: the designed herder density is a ring outside r*, heaviest in the box corners
  (the grey shading). Every sheep is within 12 m of some herder at all times (active fraction
  1.00), so each takes the unit flight step away from the nearby herders, and that points into
  the goal. This is position control by a static fence. It is not the drift balance the
  design was computed for.
- **The density is not reached, and the design does not set it.** Cohesion is 1.9–3.4 m
  against the target's 4.2–8.4 m: the flock is about twice as compact as the von Mises goal.
  Its density error, 1.1–1.5, is 5–25 times the floor. It is even above the no-herder value,
  since L² penalizes a concentrated clump more than a spread one. The shape comes from the
  sheep's attraction and repulsion: the flock packs at about the repulsion radius.
  `continuification_driving.md` §14 found the same split with Strömbom-type dogs: cohesion
  holds the shape, the dogs hold the position.
- **The feasibility count does not transfer, as §3.2 predicted.** With 4 herders, a third of
  the count, the sheep still reach r* in every run. The count is a statement about an additive
  drift these sheep do not have.

  The density error does not improve with more herders either. It is non-monotone and within
  about 2 SD across counts. If anything, more herders compress the flock further: cohesion is
  2.3 m at 4–8 herders and 1.8–1.9 m at 14 and 28. A cap at the Jadhav dog's 1.5 m/s changes
  nothing, because the herders sit almost still once their design is reached (mean speed
  0.003 m/s).
- **Limitation, untested here: notice-disc coverage.** At all three box sizes the designed ring
  keeps every sheep within 12 m of a herder. The sweep therefore never reaches the regime where
  the goal region lies outside every notice disc. There the published sheep would stand still,
  and this result should fail. That condition is r_dog against the ring's distance from the
  flock, not the herder count.

**Release: where the flock comes to rest** (`--release`, 5 seeds). A published sheep stands only
when two conditions both hold: no herder within r_dog = 12 m and no neighbour within r_rep = 2 m.

Under the design neither ever holds. At t = 200 s, all 14 herders are 4–12 m from the nearest
sheep and every sheep has a neighbour within 2 m (seeds 0–2). Spots more than 12 m from every
sheep exist: they make up 3–5 % of the box, in the corner of the torus opposite the flock. But
a herder there is seen by no sheep and does nothing. For these sheep "a place to stand where
the sheep stay still" therefore means "leave".

The test: the herders run until the flock is gathered, or until t = 200 s, then disappear and
the flock runs alone for 120 s.

| release | at release: within r* / error / cohesion | at rest: within r* / error / cohesion / centroid error | every sheep still after |
|---|---|---|---|
| first time every sheep is inside r* (t = 14 s on average) | 1.00 / 1.00 / 2.1 m | 0.42 (0.33–0.47) / 0.47 (0.43–0.57) / 6.3 m / 0.7 m | 22 s (16–27) |
| t = 200 s | 1.00 / 1.23 / 1.9 m | 0.41 (0.39–0.42) / 0.48 (0.42–0.51) / 6.4 m / 0.6 m | 25 s (16–31) |

- **The released flock always comes to rest, centred on the goal.** Repulsion spreads it until
  no two sheep are within 2 m (minimum spacing at rest 2.03–2.04 m); then nothing moves. With no
  herder in sight, the published rule has no attraction, so the flock stays as it is.
- **The resting shape is set by the sheep, and it is looser than the target.** Cohesion is
  6.1–6.6 m against the target's 4.2 m. Even perfect hexagonal packing of 36 sheep at 2 m spacing
  gives 4.2 m. That this equals the target's value is a coincidence of this N, not a design.

  Random spreading stops well short of that packing. So any standing flock has cohesion of at
  least about 4.2 m, and in practice about 6.3 m. Herders can choose where it rests and when to
  leave, but not its shape.
- **Resting beats holding on the density error.** 0.47 at rest against 1.0–1.3 while held; the
  floor is 0.22. Releasing at the first full gathering (14 s) gives the same rest state as
  releasing at 200 s. The 186 s of holding add nothing.

  This is the repository's reach-and-release result (`continuification_driving.md` §11,
  `jadhav_dilorenzo_bridge.md` §6), here with Di Lorenzo's herders doing the reaching. It also
  explains why their controller has no way to make the flock stand. It is designed for followers
  that diffuse all the time, so a herder density that stops pushing does not occur in it.

**Time average: does the moving flock fill the target on average?** (`--timeavg`, 5 seeds,
600 s runs, averaged over t ∈ [100, 600] s; figure `results/figures/dilorenzo_jadhav_timeavg.png`.)
The KDE density is averaged over time, and its relative L² error is taken against the smoothed
target, the same functional as the instantaneous error.

| | their followers | Jadhav sheep | Jadhav sheep, flat design | target |
|---|---|---|---|---|
| error of the time-averaged density | **0.03** | **1.02** | 1.07 | — |
| mean instantaneous error (floor 0.22) | 0.24 | 1.23 | 1.83 | — |
| mean density inside r* (sheep/m²) | 0.27 | 0.32 | 0.04 | 0.245 |
| mean density inside r*/2 (sheep/m²) | 0.50 | **0.90** | 0.04 | 0.42 |
| mass inside r* | 0.83 | 0.995 | 0.14 | 0.76 |
| mean distance of a sheep from the goal | 3.7 m | 2.0 m | 10.1 m | 4.4 m |
| centroid distance from goal: mean / median / 95 % / max | 0.66 / 0.62 / 1.25 / 1.9 m | **0.43 / 0.41 / 0.84 / 1.4 m** | 10.7 / 11.0 / 15.6 / 16.7 m | — |

- **Their followers fill the target over time.** Each snapshot sits at the finite-N floor, but the
  time average matches the target to 0.03. Diffusing followers sample the target as time passes.
- **The Jadhav flock does not.** Its time average is nearly as wrong as a snapshot (1.02 against
  1.23). The centroid stays within 0.84 m of the goal 95 % of the time, which is too little drift
  to smear a clump about 2 m in cohesion over a goal of radius 6 m. The core holds 2.1 times the
  target density, and the mass inside r* is 0.995 against 0.76. The herders pin the clump rather
  than letting it roam, so time-averaging does not recover the density.

Conditional statement [S]: under Di Lorenzo's feedforward herders, designed from their own
follower model, the published Jadhav flock reaches the goal disc and stays there. This
requires that the designed ring keeps the flock inside some herder's notice radius. It holds
at every herder count tried, including counts below their feasibility bound. The flock's
density is set by its cohesion, not by the target, so their density guarantee (Theorem 2)
does not carry over. While the herders act, the flock is held tight and keeps moving. Released,
it comes to rest centred on the goal at the spacing set by r_rep (cohesion about 1.5× the
target's), with less than half the held density error.

### 4.6 Four things the page shows, explained [S, E where marked]

Script: `experiments/bridge_diagnostics.py` → `results/bridge_diagnostics.json` (A–C below). The
fourth item was measured with the page's own code on seeds 1–10. These are limitations of the
configurations, recorded briefly, not new theories.

**A. Di Lorenzo's first-order followers look like a buzzing gas** (`dilorenzo_metres`, seed 1).
- Their per-step motion is pure Brownian jitter. The mean step speed, 2.14 m/s at dt = 0.1 s,
  equals the Brownian expectation √(πD/dt) = 2.14 m/s [E].
- The drift the design supplies is at most D k (2π/L) = 0.073 m/s [E], about 3 % of that jitter.
  Drift displacement overtakes diffusion only after 4D/v² ≈ 110 s [E]. So progress shows only as
  a slow statistical shift: occupancy 0.08 → 0.47 → 0.75 → 0.86 at 0, 50, 100, 150 s.
- The herders' flow has no speed cap. They leap from the starting lattice at up to 9 m/s within
  0.2 s, then barely move: a mean of about 0.003 m/s, at most 0.23 m/s after 10 s. They are
  feedforward and never react to the followers.
- Each herder pushes every follower in the box (range 12 m, box 24 m), at most α·a = 0.076 m/s at
  contact [E]. Followers therefore drift through herders.
- Their 2024 followers do not interact, so no flock forms.

**B. Hybrid A: a group that splits off does not come back** (4 seeds, 600 s).
- The flock is split into 3.1–4.1 groups (at least 3 sheep, linked within 5 m) in every sample.
- Split-off groups of 12–16 sheep sit 23–39 m from the goal and are inside r* 0–1 % of the
  time. Final occupancy is 0–0.17.

Two causes:
- *The herders' push does not point home for these sheep.* The design's push sums weak long-range
  contributions from all 16 herders. Its direction is goalward at 89 % / 82 % of points 10–20 m /
  20–30 m from the goal (mean cosine 0.59 / 0.58). The heading sheep feel only the 1–3 herders
  within r_dog = 12 m, each as a unit vector. On the same herder positions that term is goalward
  at 49–59 % of points at every distance (mean cosine −0.01 to +0.13): it is essentially sideways.
  Herder notice discs cover 92–94 % of the box, so the issue is direction, not coverage.
- *Nothing pulls the groups back together* [E]. Attraction acts on 4 of a sheep's 10 nearest
  neighbours. In a group of 11 or more, all 10 nearest are in the same group, so groups exert no
  attraction on each other. A split lasts until chance motion brings the groups within reach.

**C. Hybrid B: the Strömbom dog loses stragglers** (4 seeds, 200 s).
- Some sheep is more than 12 m from the flock centroid 43–100 % of the time, up to 18 at once.
- *The collect threshold sits beyond the reach of cohesion.* The dog collects only when the
  farthest sheep is beyond l_sep = r_rep N^(2/3) = 27.1 m, Jadhav's rule. The Morse pull of the
  whole 50-sheep flock on one sheep is 0.82, 0.24, 0.067 and 0.012 m/s at 10, 15, 20 and 27 m [E].
  A sheep 15–27 m out is effectively detached, yet the dog keeps driving. While driving, the
  farthest sheep sits at a median of 25–27 m in three of four seeds, just under the threshold.
- *The approach pushes the straggler away.* The dog runs straight at a point 2 m beyond the
  straggler, so it comes from the flock's side. It was between the flock and the straggler 34–68 %
  of the collecting time, and its push (all distances) shoves the sheep outward before it gets
  around.
- *It cannot close the gap.* Within 5 ln(2.3/1.5) = 2.1 m the push on the sheep exceeds the dog's
  1.5 m/s [E]. Within r_rep = 2 m of any sheep the dog creeps at 0.05 m/s.
- *It loses track.* The target is always the currently farthest sheep. With several out, it
  switched up to 15 times during collecting in one run. The dog was more than 15 m from the
  straggler 14–34 % of the time it was out.

**D. Designed versus flat herders over 10 page seeds** (`dilorenzo_jadhav_sheep`,
`dilorenzo_jadhav_flat`, 200 s).
- **Designed herders:** every seed ends with all sheep inside r* and the centroid within 0.9 m of
  the goal (density error 0.87–1.65). The herders leave the start lattice within about a second
  (each moves about 6.4 m) and settle on the designed ring. Its one large herder-free hole is
  centred on the goal, so the lattice start is irrelevant.
- **Flat design:** the herders move only 3–5 m, spreading the lattice into an even layout. Every
  gap is then like any other: the flock clumps (cohesion 4–7 m) in whichever gap it reaches, with
  its centroid 3–16 m from the goal and occupancy 0–0.61 (six of ten seeds below 0.1). The
  position control in §4.5 is the design's doing.

**E. Several Strömbom dogs act as copies, not as a team** (`jadhav_reference` sheep, start 60 m
out, 600 s, 5 seeds; `--only multi_dog`).
- Jadhav has one dog. The n-dog version here is our adaptation (§1.2 item 5). Each dog runs the
  single-dog rule; their drive aims are rotated about the centroid by a fixed 0.5 rad, and only
  the dog nearest the straggler collects. No dog knows where the others are, there is no density
  objective, and each dog has its own heading noise. That is why they look independent.

| dogs | arc of the flock they cover (median / 90 %) | goal crossings per run | inside r*, last half | cohesion |
|---|---|---|---|---|
| 1 | — | 7.8 | 0.14 | 3.1 m |
| 2 | 30° / 103° | 9.8 | 0.13 | 4.8 m |
| 4 | 84° / 145° | 10.4 | 0.10 | 5.7 m |
| 6 | 184° / 256° | 3.2 | 0.03 | 13.4 m |

- *They never surround the flock.* The arc is the fixed fan, 29° × (dogs − 1), behind it.
- *There is no hold.* The rule drives toward a point. When the centroid passes the goal, "behind"
  flips to the other side and the dogs run around; the flock crosses the goal 8–10 times per run.
- *More dogs hurt.* Six dogs fanned over 184° push the outer sheep sideways and split the flock
  (cohesion 13 m), as E2 found.
- *Spreading them evenly would not hold the target density either* [E, `continuification_driving.md`
  §15]. With all dogs in view, the sum of unit flight vectors is the gradient of the summed distance
  to the dogs, which points outward everywhere inside a surrounding ring. A ring confines only where
  sheep stop seeing a dog, at the edge of the 12 m notice disc. Measured there, the flock is held on
  a thin circle about 1.2 m in radius, hollow at the centre. Because the response is all-or-nothing
  (full speed within r_dog, standing beyond), dogs can place and wall these sheep but not shape them.

### 4.7 A different objective: keep the flock tight, control where it spends time [S, E where marked]

Script: `experiments/goal_schedule_validation.py` → `results/goal_schedule_validation.json`. Planner:
`shepherd/goal_schedule.py`.

**Why a different objective.** §4.5–4.6 show that Di Lorenzo's instantaneous density objective
transfers poorly to the cohesive Jadhav flock. The flock's shape is set by the sheep (held RMS
radius about 2.1 m), and herders can place the clump but not reshape it. The objective
investigated here: keep the flock tight, and control where its centroid spends time.

Under the fixed-shape approximation, the time-averaged density is S ∗ μ, where S is the held shape
about the centroid and μ is the centroid's occupation (the share of time it spends at each place).
A target ρ̄ then asks for μ = p_c := ρ̄ ⊘ S.

**What is designed, and what is not.** Only the goal centre's schedule c(t) is designed: a planner
above the controller. `simulate_bridge(..., goal=c)` reads c once per step at the state's time and
translates the precomputed herder design ρ̄_H by c(t) − c₀. This is exact on the periodic grid
[E, `test_shifted_design_equals_a_design_at_the_new_centre`, to 1e-8].
- Unchanged: Jadhav's sheep, Di Lorenzo's herder flow and its integration (100 sub-steps per 1 s),
  the random stream, the metrics (still against the fixed target), and the existing presets.
- With `goal=None` the original code path runs; a goal held at the target centre is bitwise the
  original run.

**What the earlier pilot's restarts were.** The pilot and the replay page `viz/hopping_goal.html`
(still built that way) restarted `simulate_bridge` for each 20 s segment.
- Carried over: sheep positions and headings, herder positions.
- Reset: herder velocities (the flow never reads them), the random stream (reseeded), simulation time
  (unused), and the design (recomputed at the new centre).
- No physical state was lost, but the random stream was not continuous. The comparison below runs in
  one continuous simulation.

**Comparison.** Settings:
- `dilorenzo_jadhav_sheep` unchanged (36 sheep, 14 herders), 1200 s, seeds 0–3. Initial conditions
  and the random stream are matched across schedules.
- Schedules:
  - *fixed*: the goal stays at the target centre;
  - *hop*: every 20 s the goal jumps to the next waypoint;
  - *glide*: the same waypoints, with the goal moving at constant speed between them, so it never
    jumps.
- Waypoints are drawn independently from p_c. S and p_c come from separate fixed-goal runs
  (seeds 100–102): S has RMS 2.15 m, p_c has RMS 4.43 m, and 0.05 % negative mass was clipped.
- One estimator for every quantity: unit-mass deposit on the 64-grid, Gaussian smoothing with the
  target's b = 1.15 m, samples every 1 s, relative L² against the smoothed target.
- The time-averaged error is the error *of the averaged field*, never a mean of instantaneous errors.
- Window t ∈ [100, 1200] s excludes only the initial gathering; every goal change, travel and
  settling is included. The window from t = 0 gives the same picture (time-averaged error
  0.985 / 0.052 / 0.091).

Mean over seeds (range):

| | fixed | hop | glide |
|---|---|---|---|
| instantaneous density error (floor 0.22) | 1.22 | 2.10 | 2.06 |
| **error of the time-averaged density** | 1.01 | **0.058** (0.026–0.117) | **0.096** (0.053–0.172) |
| centroid occupation vs requested p_c | 3.62 | 0.100 | 0.157 |
| the goal path's own occupation vs p_c | 4.18 | 0.122 | 0.155 |
| S ∗ (measured occupation) vs measured time average | 0.008 | 0.004 | 0.005 |
| centroid to current goal | 0.44 m | 1.58 m | 0.68 m |
| cohesion: mean / s.d. / 99th percentile | 1.93 / 0.18 / 2.33 m | 2.09 / 0.43 / 4.07 m | 1.97 / 0.20 / 2.43 m |
| RMS-radius coefficient of variation / elongation | 0.09 / 1.33 | 0.23 / 1.55 | 0.10 / 1.42 |
| bursts (cohesion > 3 m) per 1100 s; recovery mean / max | 0 | 4.75 (2–8); 8.3 / 14 s | 0 |
| splits (largest 5 m-linked group < 90 %); unrecovered | 0 | 0.5, each ≤ 1 s; 0 | 0 |

Reading:
- **The improvement survives continuous dynamics.** The error of the time-averaged density falls from
  1.01 (fixed) to 0.058 (hop) or 0.096 (glide). The restarted pilot (20 sub-steps) reported 0.036;
  continuous hops are the same order. The result did not depend on the restarts.
- **Every snapshot gets worse.** The instantaneous error rises from 1.2 to about 2.1, because the clump
  spends its time away from the target's centre. Only the time average converges.
- **The fixed-shape approximation holds.** S ∗ (measured occupation) reproduces the measured
  time-averaged density to 0.4–0.8 % (relative L²), even with the hops' bursts.
  - The remaining gap to the planned S ∗ p_c (0.05 hop, 0.07 glide) is occupation error.
  - That error comes mostly from the goal path itself: 60 waypoints are a finite sample of p_c, and
    glide's straight paths between waypoints cut across the middle.
  - Tracking adds little: the flock's occupation matches p_c about as well as the goal path does.
- **Hop versus glide is a trade-off in the schedule alone.** Hops burst the flock about five times per
  1100 s (the new herder ring lands on it; §4.6 has the mechanism), and all bursts recover within 14 s.
  Glide keeps the flock as tight and steady as a fixed goal, at about 1.7× the hops' time-averaged
  error over this duration.

Qualifications:
- **Feasibility.** A non-negative, normalized ρ̄ ⊘ S is a distributional feasibility condition *under
  the fixed-shape approximation*. Comparing RMS widths is not a general criterion. Realizing p_c also
  needs tracking and safety, which the hops show can conflict.
- **Convergence.** For n independent hops, the mean squared occupation error may scale as 1/n, so the
  typical error scales as 1/√n [H]. Requested goal samples are not independent samples of the actual
  occupation: lag, travel, and glide's correlated path intervene. The measured occupation error is the
  quantity to quote.
- **Bursts.** Ring geometry (hop length against the ring radius) gives a candidate predictor, or a
  sufficient safety condition. It is not a proved universal threshold.
- **Deferred:** a feasibility sweep over target widths, convergence proofs, and the interface. Note
  that `viz/hopping_goal.html` still uses the restarted segments.

### 4.8 Hybrid B's sheep: Strömbom's dog reaches but cannot stop; Di Lorenzo's herders cannot hold them [S]

Script: `experiments/hybrid_b_herders.py` → `results/hybrid_b_herders.json`. The overshoot trace was
measured inline on the same preset.

**The published dog overshoots.** In `hybrid_first_order_sheep_strombom_dogs` (50 first-order sheep
with Morse cohesion, D = 0.12, one Strömbom dog on the open plane), the dog gathers the flock and
then pushes it out of the goal circle again. Over 600 s (4 seeds) the flock centre is inside r_goal
28 % of the time and crosses the goal 5–10 times; occupancy is 0.26. Seed 0 shows the sequence:
- the share inside r_goal rises 0.38 → 0.60 by t = 9 s;
- at t = 15 s the dog is 0.9 m from the flock centre and the share is 0.06;
- by t = 24 s it is 0, with the flock 8 m away and the dog leaving to collect stragglers.

The mechanism:
1. *The drive point never stops at the goal.* It is 14 m behind the flock on the goal→flock line.
   Once the centre reaches the goal, a small offset sets that line, so the aim point swings to the far
   side and the dog runs straight through the flock.
2. *These sheep feel the dog at every distance, most strongly at contact* (2.3 m/s), so a dog inside the
   flock pushes everyone outward.
3. *After an overshoot "behind" flips.* The dog runs back around, often on the goal side of the flock
   (21–82 % of the time, depending on the seed).
4. *It leaves to collect* stragglers beyond l_sep = 27 m, while the flock's pull fades from about 15 m
   (§4.6 C).

The repository's reach-and-release option (`hold_radius = r_goal`, an existing extension) makes it
worse: occupancy 0.09, cohesion 3.7 m. These sheep keep diffusing when the dog withdraws, and the push
has no cut-off, so a dog waiting 15 m away still pushes at about 0.11 m/s.

**Swapping the controller as a component.** The sheep stay exactly as in the preset, with the
preset's starts (scattered around the goal, sd 8 m), seeds 0–3, 600 s, window t ∈ [100, 600] s.
- Di Lorenzo's design is a deconvolution on a periodic domain, so it runs in a 40 m periodic box. The
  same Strömbom dog in that box isolates the domain change.
- The herders run unchanged: K_H = 10, no speed cap, flow at their dt = 0.01, and a design from these
  sheep's own model (feasible, drift error 0.004).
- The minimum count is M̂ = 6.14 → 7 herders (2.24 without the Morse term; 11.1 in a 60 m box).

| controller | inside r_goal | centre → goal | time-averaged density error (floor 0.18) | cohesion (99th percentile) |
|---|---|---|---|---|
| Strömbom, open plane (the preset) | 0.20 (0.00–0.28) | 12.9 m | 1.17 | 2.2 m (3.8) |
| Strömbom, 40 m periodic box | **0.98** (0.92–1.00) | 0.7 m | 1.15 | 1.7 m (2.2) |
| Di Lorenzo, 7 herders | 0.00 | 17.1 m | 3.12 | 7.9 m (9.5) |
| Di Lorenzo, 14 herders | 0.00 | 20.5 m | 3.87 | 3.6 m (6.4) |

(The plane run is evaluated on an 80 m periodic grid at the same resolution; sheep beyond ±40 m wrap.)

Reading:
- **Di Lorenzo's herders cannot hold these sheep, because the design must spread a cohesive flock.**
  - Cohesion pulls 50 sheep far tighter than σ = 3 m, so the required drift points *outward*: +1.2 m/s
    at 5 m from the goal, almost all of it cancelling the pair pull. Holding against diffusion needs only
    0.07 m/s inward.
  - 40 % of the herder mass therefore sits inside r_goal, pushing out. The pair forces cancel at the
    centroid (§4.1, E1), so the clump moves as one body: any offset is pushed further out, and the clump
    is expelled whole, 17–20 m off the goal. More herders push it farther.
  - This is E1's attraction failure seen for Hybrid B's sheep. It is consistent with Di Lorenzo's 2026
    local stability condition, which needs diffusion to dominate the pair term; here the pair drift
    outweighs the diffusive drift by about tenfold. Their bound was not evaluated exactly.
- **The Strömbom dog holds position in the box only because the box disables collecting.**
  - In the 40 m box no sheep can be more than 28.3 m from the centroid, barely past l_sep = 27.1 m, so
    the dog drives 100 % of the time (on the plane: drive 69 %, collect 28 %).
  - Driving only, it stays 15–17 m from the flock, where its push is about 0.1 m/s, and the clump stays
    at the goal.
  - This is a domain artefact, not a fix. It does locate the cause on the plane: the collect mode and
    the dog's straight run through the flock.
- **Neither reaches the target density.** Held in place, the flock is still a tight clump (time-averaged
  error 1.15). As for Jadhav's sheep, a cohesive flock can be placed but not spread. This supports the
  objective of §4.7 for these sheep too: keep the flock tight and control where it spends time.

### 4.9 How solid is "100 % in the goal circle" for Jadhav's sheep under Di Lorenzo's herders? [S]

`experiments/follower_swap_validation.py` → `results/follower_swap_validation.json`. The configuration of
§4.5 with a fixed goal, unchanged: 36 Jadhav sheep, 14 herders, uniform start, 24 m torus, r_goal = 6 m,
dt = 1 s. Occupancy is an integer count of sheep inside the circle, checked at every simulation step.

**Design.** 50 fresh seeds (5000–5049) for 600 s. The first six of them, chosen before any result, continue
to 6000 s in one run (their first 600 s match the short runs bitwise). The late window is the last 25 % of a
run, endpoints included (450–600 s; 4500–6000 s for the long runs), so the start-up is excluded. Density error
is §2.4's squared relative L² against the smoothed target. "Instantaneous" is its mean over the window;
"time-averaged" is the error of the window-averaged density.

**Findings**

- **The reported 1.00 is exact in the four scorecard seeds** (36/36 at all 151 late steps), but it does not
  hold in every run.
- **Full late-time occupancy: 47 of 50 runs** (94 %; Wilson 95 % interval 84–98 %).
  - Seeds 5036, 5048 and 5049 each lost one sheep for one step (35/36, at t = 516, 561, 467 s). The farthest
    any sheep reached was 6.16 m.
  - Mean late occupancy over the 50 runs is 0.99999.
- **Reaching and retention.** All 50 runs reached success (≥ 0.8 inside for 30 s; median 9 s, range 7–12 s),
  with no escapes (< 0.5 inside for ≥ 5 s).
- **Splits.** 14 runs had a split (largest 5 m-linked group < 90 %, as low as 61 %). Every one ended by t = 6 s,
  while the flock was gathering; none persisted.
- **Persistence.** 5 of the 6 long runs were full over 4500–6000 s. The sixth (5003) lost one sheep for one
  step at t = 5033 s. After t = 600 s, 53 of 54 blocks of 600 s are full, spread stays at 2.07–2.12 m and the
  block time-averaged error at 1.0–1.09, with no trend.
- **Rate of exits.** Four one-step exits in about 59,000 steps after settling. The three in the short runs
  all fell in their last 150 s, but the long runs show no rise with time, so this is read as chance.

**Density error over the late window (50 seeds; finite-N floor 0.22)**

| | min | 5 % | 25 % | median | 75 % | 95 % | max |
|---|---|---|---|---|---|---|---|
| Instantaneous | 1.09 | 1.11 | 1.15 | 1.17 | 1.20 | 1.23 | 1.26 |
| Error of the time-averaged density | 0.92 | 0.95 | 0.98 | 0.99 | 1.02 | 1.04 | 1.07 |

The error is about 4.5 times the floor and narrowly distributed: the mismatch with the target is as
reproducible as the occupancy. Time averaging barely helps because the centroid hardly moves (mean offset
0.44 m, largest 1.35 m).

**Share of mass inside radius r**

| r | 2 m | 3 m | 4 m | 5 m | 6 m (r_goal) |
|---|---|---|---|---|---|
| Target (smoothed) | 0.17 | 0.33 | 0.50 | 0.65 | **0.76** |
| Flock (late window, mean of 50) | 0.52 | 0.86 | 0.98 | 0.999 | **1.00** |

**What 100 % means here**

- **It is over-concentration, not a match.** The target itself puts only 76 % of its mass inside r_goal
  (79 % before smoothing). A flock with all its sheep inside exceeds the target's own occupancy, so occupancy
  against this circle cannot certify the target and is best read as a placement measure.
- **The margin is thin at the edge.** RMS spread is 2.1 m, but the farthest sheep reaches a median of 5.45 m
  per window (range 4.93–6.16 m). The one-step exits are this tail crossing the edge.
- **It is an active hold.** Over the last 150 s every sheep is in flight at 1 m/s at every step, with the
  nearest herder 4.3 m from the nearest sheep on average (minimum 2.25 m). The result is for this box:
  r_dog = 12 m is half its width, and other box sizes were not part of this check.

---

## 5. Checks

(`tests/test_bridge.py`, `tests/test_bridge_js.py`)

| Check | What it verifies |
|---|---|
| reference | the heading rule with Jadhav's parameters and one Strömbom dog reproduces `model.simulate` trajectories and modes (3 seeds, first 50 steps, atol 1e-6; later, a last-bit rounding difference that appears on some CPUs is amplified by the flock) |
| activation | published: a sheep with no dog within r_dog and no neighbour within r_rep neither moves nor turns; wander: off by default, and when on every standing sheep walks at exactly `wander_speed`; contagion: only sheep within 3 m of the dog move on step 1 and motion spreads to sheep that never saw the dog |
| noise | directional noise keeps every step at exactly v; Brownian noise changes step lengths; the first-order rule's mean square displacement is 4Dt |
| periodic geometry | wrapped differences, circular-mean centroid across the seam, translation invariance of the dynamics and metrics under a shift across the seam, image sums converge to the periodic kernel and agree with the design grid |
| metrics | closed-form density error = grid error on a large torus; the finite-N floor σ²/(N b²) matches sampling |
| Di Lorenzo | M̂ = 0.28 ± 0.01 for their example; M̂ ∝ D and independent of N_T; surplus spread uniformly, shortfall marked infeasible; the feasibility scan is flat without pair interactions and mass-dependent with them; the herder density error decays under the Theorem-1 flow |
| multi-dog | the n-dog Strömbom reduces to the published dog at n = 1 |
| density feedback | the largest surplus is targeted inward by the nearest dog, drive geometry when far, hold aims beyond reach; release withdraws and the published sheep stop; a matched flock is left alone (dogs beyond r_dog of every sheep); JS placement and surplus equal Python to 1e-9 |
| Di Lorenzo scenario in metres | the rescaled scenario needs the same M̂ as their units (to 1e-9); with Jadhav sheep the herders design from their own follower model (an empty design without it); herder sub-steps converge to the fine-step flow; the page's presets equal Python's (design to 1e-9, herder trajectories to 1e-6) |
| goal schedule (§4.7) | a goal held at the target centre is bitwise the original run; translating the design by whole grid cells equals recomputing it at the new centre (sheep and herder trajectories to 1e-8); hop and glide share waypoints, glide never jumps; a schedule with a non-Di Lorenzo controller is refused |
| JS port | design, feasibility scan, density error and floor agree with Python to 1e-9; the published model's 60-step statistics agree within sampling error |

---

## 6. What is mathematics, what is observation, what is hypothesis

**Mathematical results [E, P].** The equations of §2 are the published rules (Jadhav: bit-for-bit;
Di Lorenzo: up to the kernel periodization and discretization stated in §1.2). The finite-N floor
is exact. Di Lorenzo's feasibility and Theorem 1 apply to the first-order followers on the torus
exactly as implemented; Theorem 2's convergence holds for them unconditionally
(`continuification_driving.md` §5b). None of these applies to the heading sheep (§3.2); for them
the repository's centroid result holds (`jadhav_dilorenzo_bridge.md` §6, §9): under notice-radius
coverage and sufficient attraction, one Strömbom dog brings the centroid to the goal in finite
time, and holding requires withdrawing the dog.

**Experimental observations [S].** §3.1 and §4.

**Hypotheses [H], and what the first comparison did to them.**
- H1. For first-order followers the feasibility count N̂_H is a lower bound that discrete herders
  only approach from above, with the gap closing as N_T grows. *Consistent with E1 and E3*: the
  curves bend at N̂_H and plateau below the designed 95 %; the N_T dependence is untested here
  (the earlier `feasibility.py` result for 14 vs 50 sheep points the same way).
- H2. Cohesion lowers the continuum requirement only when it supplies part of the goal's
  curvature, and raises it when net repulsive at the flock's pair distances; alignment has no
  density-level counterpart. *The feasibility scan confirms the first two clauses exactly [E];
  E1 adds that a lowered continuum count can coincide with a collapsed clump off the goal, so
  "fewer herders needed" and "goal reached" come apart; the alignment clause held in E1.*
- H3. Heading sheep under static feedforward herders do not reach a held density at any dog
  count; closed-loop dogs reach the goal, and holding needs the release. *E3 and E2 agree.* What
  E3 adds is that the push *scale* is part of the assumption: a dog-strength push breaks the
  design even for first-order followers.
- H4. With closed-loop dogs, attraction lowers the dog count needed for a given occupancy, and
  alignment past the baseline raises it. *E2 revises this*: with the hold, attraction sets the
  occupancy reached (0.15 → 0.75) but the dog count barely matters (one dog suffices, more dogs
  split a weakly cohesive flock); alignment hurts without the hold (the flock runs through the
  goal as a body) and helps slightly with it.
- H5. The design's herder count in metres is set by the push amplitude, a knob the Jadhav sheep
  do not have. *E4 tested the density-feedback alternative*: for the published sheep one such dog
  with a density-triggered release beats Strömbom with the hold (0.86 vs 0.65, sustained), but
  the gain is the release, not the surplus-pushing, and more dogs hurt; for an already-gathered
  self-propelled flock the static Di Lorenzo ring is the only controller that sustains the match,
  and it cannot gather. The next step this suggests is a two-stage controller: one closed-loop dog
  to reach, then a static fence at the notice radius to hold.
