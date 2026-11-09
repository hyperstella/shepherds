# Jadhav's flock in Di Lorenzo's framework: macroscopic dynamics, closures, and conditional conclusions

**Question.** Using the Jadhav baseline parameters, what macroscopic dynamics can be derived, and
what conditional feasibility or stability conclusions can be established with Di Lorenzo's
approach?

This document is the synthesis. The detailed derivations and experiments are in
[steer_and_shape.md](steer_and_shape.md) and [continuification_driving.md](continuification_driving.md);
a section number without a file name refers to this document. Sections 1–8 required no new
simulations; their numbers come from result files already on disk. §9 (the tracking bound),
§10 (deriving C4) and §11 (the multi-step restoring inequality) each add one experiment. §12
(certifying MI⁺, its counterexamples, and a multi-step drift) adds three.

## 0. Baseline, and the kinds of evidence

**Baseline, fixed throughout.** These are the parameters of the code that produced Jadhav et al.'s
figures (`shepherd/model.py` defaults).

| Item | Value |
|---|---|
| Flock and perception | N = 14; each sheep perceives its K = 10 nearest, attracts to 4 of them at random, copies 1 of those |
| Heading weights | memory 0.5, repulsion 2, dog 1, attraction 1.5, alignment 1.3, noise 0.3 |
| Distances and speeds | r_rep = 2 m, r_dog = 12 m; sheep 1 m/step, dog 1.5 m/step |
| Rule | published: a sheep with no dog within r_dog stands still unless repelled |
| Dog | the published Strömbom dog: drives from l_drive = 2√N ≈ 7.5 m behind, and collects when a sheep is more than l_sep = 2N^(2/3) ≈ 11.6 m from the centroid |

The paper's Methods text differs (5 attraction neighbours, noise 0.5); those values are not used.
Where a result below comes from another setting (r_dog 20, a custom controller, a drive-only dog),
it is labelled.

**Evidence labels.**
- **[E]** exact identity for the full rule.
- **[P]** proved, given stated assumptions.
- **[D]** derived under a stated approximation.
- **[S]** measured in simulation of the baseline model.
- **[R]** measured on the real tracks.

## 1. Exact one-step macroscopic dynamics [E]

The state x is all positions, headings and the dog's position. A sheep moves by a_i = v m_i h_i′,
where:
- m_i ∈ {0, 1} is set by x (it sees the dog or is repelled);
- h_i′ = unit(S_i) is the new heading.

Given x, each sheep's random draws (neighbour subsets, noise) are independent of the others',
because alignment copies *current* headings. Let μ_i = E[h_i′ | x], f_m = (1/N)Σ m_i and
M = (1/N)Σ m_i μ_i. With c the centroid, ξ_i = x_i − c and M₂ = (1/N)Σ|ξ_i|²:

| Quantity | One-step identity | Where |
|---|---|---|
| Centroid, mean | E[Δc \| x] = v M, so the speed is v f_m P_μ with P_μ = \|M\|/f_m, along the mean expected heading | new |
| Centroid, noise | Var(Δc \| x) = (v²/N²) Σ m_i (1 − \|μ_i\|²), roughly v²(1 − P²)/N per step | new |
| Second moment | E[ΔM₂ \| x] = C + G | steer_and_shape §10 |
| | C = (2v/N) Σ m_i ξ_i·(μ_i − μ̄) − (2v/N) (Σ_frozen ξ_i)·μ̄, with μ̄ = M/f_m | |
| | G = v²(f_m − \|M\|²) − Var(Δc \| x) ≤ v²(f_m − f_m² P_μ²) | |
| With every sheep moving | C = −2v κ γ √M₂, with κ the weighted inward relative heading and γ = mean\|ξ\|/√M₂ ≈ 0.9 | steer_and_shape §10 |

**Reading.**
- *Each sheep's heading spread is spent twice.* It splits exactly into shape spreading G and
  centroid wandering Var(Δc). G plays the role diffusion plays in Di Lorenzo's model, but it is set
  by polarization, not by a diffusion constant.
- *Only relative headings change the shape.* The common heading cancels in C.
- *The stop rule enters exactly.* Frozen sheep add the term −(2v/N)(Σ_frozen ξ_i)·μ̄. It contracts
  the flock when the frozen sheep are ahead of the movers (the dog behind), and stalls it otherwise.

Higher moments and the relative density have identities of the same kind. They need the whole
field μ_i(ξ), not just its aggregate κ, so they do not close.

## 2. The minimal closures

The exact identities involve μ_i, which depends on the full state. These four assumptions are
enough to close the centroid and second-moment levels. Nothing closes the density level.

| Closure | Statement | Needed for | Baseline evidence | Status |
|---|---|---|---|---|
| **C1 coverage, or frozen sheep ahead** | f_m = 1, or (Σ_frozen ξ_i)·μ̄ ≥ 0, so C_frz ≤ 0 | centroid speed; M₂ drift | Observed states: every sheep moves in 97%. Dilated states: coverage drops, but C_frz < 0 (−0.52, −1.33, −2.26 per step at 1.5×, 2×, 3×) [S]. Violated by sheep left out to the side; the collecting dog restores coverage in every tested case [S] | holds by assumption, or via the dog's feedback |
| **C2 heading law** | E[θ_{t+1} − θ_t \| x] = δ(φ_t) + β, with δ(φ) = atan2(w_dog sin φ, w_inertia + w_ali + w_dog cos φ), for P ≳ 0.7 and \|φ\| ≤ 60° | steering, centroid | Gain 0.372 measured vs 0.357 derived. Turn rates at 10°, 30°, 60°: 2.7, 11.4, 24.2 °/step against 3.6, 10.6, 20.6. Closed-loop one-step RMS 7.4 °/step; accumulated turn 0.72–0.96 of predicted [S, steer_and_shape §5, §7] | [D], with measured error |
| **C3 polarization** | P_μ(x) ≈ P_eq(φ), a fast variable | centroid speed; G | P_eq = 0.85, 0.80, 0.71 at 0°, 30°, 60°. Reaches 63% of a change in a median of 3 steps (at most 7). The arc-averaged P is within 0.01 of P_eq [S, steer_and_shape §2 R3, §7] | [S] |
| **C4 restoring** | κγ ≥ k* > 0 on the states of interest outside a ball | M₂ drift | Derived in §10. Exact κ (no Monte Carlo): 0.047 at the observed size (23% of states negative, inside the ball), 0.145 at 1.5×, positive on all 255 driving states outside the ball; γ ≥ 0.86. A sufficient condition (attraction − dog − repulsion + heading − normalization bound) certifies 62–98% of them. It fails uniformly once headings fan outward; a multi-step memory inequality replaces it | [P] sufficient condition; [S] where it holds |

**Why these four are minimal.**
- C1 and C3 turn the exact centroid identity into motion along θ at speed vP_eq(φ).
- C2 closes θ.
- C1, C3 and C4 turn the exact M₂ identity into a scalar drift inequality.
- Dropping any one leaves the corresponding level open.

None of them requires the flock to stay connected at every instant. C1 concerns who moves; C4
concerns the aggregate inward relative heading. A temporary split is allowed as long as these
hold on average, or are restored.

## 3. The reduced dynamics (exact identities plus C1–C4)

    θ_{t+1} = θ_t + δ(φ_t) + ε_t                       heading, steered by the dog's bearing φ
    P_t ≈ P_eq(φ_t)                                     polarization, fast
    c_{t+1} = c_t + v P_{t+1} e(θ_{t+1})                exact when every sheep moves; the noise
                                                        of §1, ≈ v²(1 − P²)/N, rides on ε_t and P
    E[ΔM₂ | x] ≤ −2v k* √M₂ + v²(1 − P_t²)              shape, on the region where C1, C4 hold
    ‖h_{t+1} − h_t‖ ≤ v_dog,   φ = arg(h − c) − θ − π  actuator: the dog's own motion (exact)

**The levels are coupled through φ.** Steering lowers P, which slows the centroid and raises the
dispersion G. At baseline, D⊥ rises from 0.110 to 0.160 m²/step between φ = 0 and 60°, but M₂ stays
at 8.8–9.3 m²; only the aspect changes [S, steer_and_shape §5].

## 4. What can be established, level by level

These levels are not equivalent. A flock whose centroid tracks a path can have any shape; a flock
whose M₂ is held can have many densities with that M₂.

### 4.1 Centroid motion: the strongest level

- **Reaching** [S + D]. In the published closed loop the centroid closes on the target linearly,
  not exponentially: slope −0.837 m/step against −0.833 predicted, and t_reach 134 steps against
  132.6 predicted (continuification_driving §9).
- **No hold.** The published loop has none: after reaching, the flock overshoots 19.6 m on
  average. Releasing the dog freezes it within a median of 6 steps.
- **Path feasibility** [D, from C2, C3 and the exact actuator geometry]. A steady turn at standoff
  d and bearing φ needs dog speed √(V² + 2Vdω sin φ + d²ω²), with ω = δ(φ) and V = vP_eq(φ). At
  baseline (v_dog 1.5, d 7.5) the smallest steady radius is R_min ≈ 6.0 m (5.8 m in the one-step
  chord version), and the bearing can change by at most about 0.16 rad/step. This is a fixed-standoff,
  fixed-bearing *estimate*.
- **Tracking** [S; custom cascade controller, drive only, baseline sheep]. On a straight-then-arc
  path, the maximum cross-track error is 1.4–1.7 m (median over runs) for R ≥ 8 m, and 2.1 and
  3.3 m for R = 5 and 3 m. No flock split.
- **Tracking bound** [P, given measured disturbance sizes; §9]. The path error obeys an exact
  two-state recursion. A lemma bounds its mean square at every step:
  - about 1.4 m on the first straight;
  - 3.0, 2.0 and 1.8 m at arcs of R = 8, 12 and 20 m;
  - arc transients decay at 0.959 per step toward a stationary 1.6 m.

  The published runs stay below it at every step: their RMS error is 0.5–1.1 m after step 15.
  The bound is at least 1.4× the measured value, with a median ratio of 2.3–2.9. Separately, the
  frozen recursion predicts the measured RMS error within 0.07 m.
- **Centroid noise** [E]: about v²(1 − P²)/N per step.

### 4.2 Shape moments: a conditional drift result

- **Lemma** [P, from §1 and C1, C4]. On the region 𝒢 where C1 and C4 hold and P_μ ≥ P*,

      E[ΔM₂ | x] ≤ −a √M₂ + b,   a = 2v k*,  b = v²(1 − P*²).

  At baseline (κ* = 0.05, γ* = 0.88, P* = 0.79): a = 0.088, b = 0.37 per step. The drift is
  therefore negative outside √M₂ ≈ (b + ε)/a ≈ 4.5 m, against an observed √M₂ of 3.1 m. The form
  −a√M₂, not −aM₂, is forced by unit-vector attraction, which restores at constant speed.
- **Finite-horizon corollary** [P, given the lemma]. Let σ be entry into the ball and τ exit from
  𝒢. Stopped at σ∧τ, M₂ is a supermartingale, so:
  - E[σ∧τ] ≤ M₂(0)/ε;
  - P(M₂ reaches L before σ∧τ) ≤ M₂(0)/L.
- **Long-run boundedness: not proved.** It would need the drift on every reachable state, or
  invariance of 𝒢.
- **C4 derived** (§10) [E + P + S].
  - In the driving region with full coverage, κ splits exactly into attraction − dog divergence −
    repulsion + heading terms + a normalization correction + a front–back speed term.
  - The resulting sufficient condition certifies 62–98% of the tested states outside the ball.
  - A uniform one-step C4 is false: outward heading fans break it.
  - A drift function W = M₂ + βQ, given a memory inequality that is tested but not proved,
    allows such temporary outward motion.
- **A 3-step drift** (§12.6) [S, probability 0.999 per claim].
  - D₃ = E[M₂(t+3)] − M₂(t) ≤ −0.17 m² is certified at all 285 tested states, adversarial headings
    included.
  - On states that recovery actually visits, it is not uniform. It fails with probability at least
    0.022: at lateral splits that stay split, and where the three steps cross into collecting.
- **Observed recovery** [S, baseline sheep and dog with collecting]:
  - From all ten tested deformations (stretches, 8–12 m splits, outliers, heading noise), every
    trajectory re-entered the ball within 150 steps: median 2–64 steps, 90th percentile ≤ 110.
    None ended split.
  - Most of these trajectories start outside 𝒢 (some sheep frozen in the stressed state) or leave
    it on the way, so the recovery is not explained by the lemma alone. Collecting restores
    coverage, after which the drift applies.
- **Attainable second moments** [S]. Stationary M₂ is about 9–13 m² over usable standoffs and
  flanking at baseline (steer_and_shape §4). The push term itself cannot compress pairs that see the same dogs
  [P, additive model only, Prop. 1]. Tightening happens only indirectly, and at baseline weights
  the dog does not get below the dog-free value of about 8 m².

### 4.3 Convergence toward a prescribed density: not established

**Supported:**
- Each sheep's motion relative to the flock is diffusive at the flock's scale:
  - Knudsen number ≤ 0.5 in the model [S];
  - 0.2 on the real tracks, with individual D ≈ 0.12 m²/s [R, continuification_driving §13].
- So a co-moving Fokker–Planck description of the *shape* is plausible.

**Missing:**
- **No drift field.** The field μ_i(ξ) has not been identified beyond its aggregate κ.
- **The dog's effect is not linear in a herder density.** Normalization makes it additive only
  across the heading, near polarization.
- **At N = 14 a density is barely resolved.** The L² error of a 14-sample density estimate has a
  floor near 0.5; moments are the meaningful observables.
- **The realizable set is one-sided.** In the additive approximation the push cannot compress
  (Prop. 1).

## 5. Di Lorenzo's analysis, piece by piece

| His component | Adapts directly | Needs a new argument | Unsupported |
|---|---|---|---|
| **Macroscopic model**: first-order mean-field PDE, drift linear in the herder density (2024), plus a follower–follower kernel (2026) | The *form* of a conservation law for the shape in the co-moving frame (diffusive relative motion) | A moment hierarchy (c, θ, P, M₂) replaces the PDE, because Jadhav sheep move at fixed speed with normalized, memory-carrying headings. The exact identities of §1 are the new base | A closed PDE for the relative density; any convolution (linear) structure in the dog's effect |
| **Required field** (zero flux): v̄ = D∇ρ̄/ρ̄ − f^FF*ρ̄ | Its stationary moment version is exact: E[C] = −E[G] under the stationary law | The translation part (ċ) is not a required drift but a *heading control problem*: sheep propel themselves, and the dog supplies the bearing that turns them (C2) | The full matching condition for a prescribed density |
| **Feasibility**: a non-negative herder density with admissible mass, via deconvolution | The *logic*: required action → can the herders produce it? | Centroid: an admissible dog path (speed ≤ v_dog, \|φ\| ≤ 60°, coverage) giving the required turn rate, hence R_min. Shape: the attainable interval of stationary M₂ under the dog's knobs, and the sign constraint from Prop. 1, the analogue of his non-negativity | Feasibility of a prescribed density |
| **Herder-layer convergence** (his Theorem 1) | – | One rate-limited dog; its bearing dynamics are exact geometry plus a speed bound | – |
| **Target-layer stability**: L² Lyapunov estimate (his Theorem 2; locally in 2026) | The Lyapunov idea, with V = M₂ in place of ‖e‖²_{L²} | The drift lemma (√M₂ rate, local, until exit) and its supermartingale corollary. For the centroid, a mean-square ISS bound on the path error (§9). For the shape, a drift function W = M₂ + βQ that allows temporary outward motion (§10) | Long-run (recurrence) guarantees; density convergence; rates beyond the moment level |

Two side results matter for the comparison:
- Di Lorenzo's own condition ‖G‖∞ < 2 is not needed in his setting: his own example violates it
  and still converges at a positive weighted rate (continuification_driving §5).
- His 2026 follow-up's interaction term is the closest analogue of Jadhav cohesion, but it is
  positional and additive. Here cohesion acts through normalized unit vectors and enters only via κ.

## 6. Conditional result, as it stands (baseline)

Under C1–C4 and the dog's speed limit:

1. **Centroid** [D + S, and P for the bound]. The heading follows the bearing law δ(φ) with
   measured error. A path is steady-turn feasible when its radii are at least R_min ≈ 6 m. Given
   the measured disturbance sizes, the RMS cross-track error is then bounded at every step, with
   transients decaying at 0.959 per step (§9). Below R_min the bound still holds, but the arc bias
   persists and the bound grows to 6–10 m.
2. **Shape** [P + S]. While the state stays in 𝒢, M₂ drifts down outside √M₂ ≈ 4.5 m, and returns
   to that ball or leaves 𝒢 within expected time M₂(0)/ε. On every tested deformation the baseline
   flock and dog actually returned, including after temporary splits.
   - C4 has a sufficient condition in the driving region (§10.3).
   - Without a heading condition C4 fails. The drift function W of §10.5 replaces it, given a
     memory inequality that is tested but not proved.
   - A 3-step drift is certified on the tested states (§12.6) but is not uniform over visited
     driving states. Lateral splits and the handover to collecting break it.
   - Collecting is what brings disturbed flocks back into that region (§10.6).
3. **Density** [–]. No statement.

**What this does not claim:**
- convergence to a prescribed density;
- an exponential rate;
- that the flock never leaves 𝒢;
- anything for flocks much larger than 14 (see §7).

## 7. Limitations recorded, not pursued

- **Fission at larger N** (steer_and_shape §11.4–§11.5; `experiments/size_sweep.py`,
  `fission_threshold.py`, `fission_provenance.py`, `fission_bias.py`).
  - At baseline weights, a *drive-only* dog splits compact flocks laterally from about 17–18
    sheep; collecting does not prevent it. The 50% point is 17.4 sheep within 300 steps, about
    17.0 within 1000.
  - A one-constant threshold model predicts eight other settings within about 1.4 sheep. It runs
    low by about 0.6 sheep at 300 steps and about 0.3 at 1000, the gap shrinking as runs lengthen.
    That work is paused.
  - For this bridge it bounds the scope: a single-centroid, single-M₂ description is appropriate
    only below that size. The baseline flock of 14 is below it.
- **Coverage loss under a drive-only dog.** Sheep pushed out to the side, beyond r_dog, freeze
  and stay apart (steer_and_shape §11.2). That is a failure of the drive-only strategy; the
  baseline dog's collecting rescues every tested case.
- **Alignment's stopping cost** (modified rule only). Released flocks run on 15× further at
  w_ali 3 than at 0. Under the baseline (published) rule they freeze.

## 8. What would most strengthen the bridge

1. *Done:* the mean-square tracking bound for the centroid (§9). Two things would sharpen it:
   - derive the noise size S from the §1 identity instead of measuring it;
   - use a non-quadratic Lyapunov function to recover the 0.83 spectral radius.
2. *Done in part (§10, §11):* C4 has an exact decomposition and a sufficient condition in the
   driving region, and the one-step version is false without a heading condition. The
   multi-step version reduces to one hypothesis, MI⁺: V ≤ λQ⁺ − k·mean|ξ|.
   - §12 certifies MI⁺ at 178 of 255 states and refutes it at 6.
   - At those 6, no drift function built from M₂ and Q alone can work.
   - Under the baseline dog the counterexamples recover within 3–5 steps.
   - §12.6 certifies the 3-step drift D₃ ≤ −0.17 m² at all tested states (probability 0.999).
     On visited driving states it fails with probability at least 0.022, at lateral splits and at
     the handover to collecting.

   - The collecting chain of §10.6 is followed up in [collecting_chain.md](collecting_chain.md).
     It gives an exact collecting drift bound, a gap lemma, a handover with an analysis band, and
     a cycle bound E[τ_ball] ≤ 214 steps from a handover. These hold on sampled recovery states,
     not uniformly.

   Target switching inside a collecting cycle is bounded in cost but not in number: two stragglers
   on opposite sides take turns with no ratchet (collecting_chain §3.5). That is a limitation of the
   rule. Next: a drift function common to both modes, with a lateral term for split flocks.
3. Identify the relative-velocity field μ_i(ξ) − μ̄ as a function of position in the flock. That is
   the step from moments toward a density-level (Fokker–Planck) description, and the only route
   to a Di Lorenzo-style statement about prescribed densities.

## 9. The centroid tracking bound

This writes out the claim of §4.1 and steer_and_shape §9. It needed one new experiment,
`experiments/tracking_bound.py` (results/tracking_bound.json, results/figures/tracking_bound.png).

**Setting.** Baseline sheep (§0, published stop rule) and baseline dog speed v_dog = 1.5; the
controller is custom and drive-only (steer_and_shape §7, `path_tracking.py`).
- *Path:* 25 m straight, a quarter circle of radius R, then straight.
- *Controller:* guidance θ_d = θ_path − atan(e/Δ) with Δ = 6 m; turn command
  ω_t = V_t κ_t + k(θ_d − θ_t) with k = 0.3; bearing φ_d = δ̂⁻¹(ω_t) from the measured steering
  table, at most 60°. The dog heads for the point at standoff 7.5 m with that bearing, moving at
  most v_dog per step.
- *State:* x_t = (ψ_t, e_t), where ψ is the mean heading minus the path tangent (rad) and e the
  signed cross-track error (m, left > 0).
- *Other notation:* V_t = |c_t − c_{t−1}|; κ_t is the path curvature at c_t (0 or 1/R); F_t is
  everything known when the dog moves at step t.

### 9.1 One centroid step, exactly [E]

- **No separate centroid noise.** When every sheep moves (C1), the centroid step is exactly
  V_{t+1} e(θ_{t+1}), where θ_{t+1} is the direction and V_{t+1} = vP_{t+1} the length of the mean
  new heading. Checked in every run: the angle error is at most 3·10⁻¹⁴ rad. So the Var(Δc | x) of
  §1 appears as noise in the heading and in the speed.
- **Arc geometry.** On an arc of radius R (centre on the left), let ψ̃ = θ_{t+1} − θ_path(s_t)
  and ρ = R − e. Then

      e_{t+1} = e_t + V sin ψ̃ − V² cos²ψ̃ / (ρ + ρ′ − V sin ψ̃),     ψ_{t+1} = ψ̃ − Δα,

  where Δα is the angle the step sweeps about the centre. The last term of the first equation is
  the chord: the step is straight while the path curves. On a straight both corrections vanish.
- **The recursion.** Linearize, and write the realized turn as g times the command. This gives

      x_{t+1} = A(V_t, κ_t) x_t + c(V_t, κ_t) + w_t

      A(V, κ) = [ 1 − gk       −(gk/Δ + Vκ²) ]      c(V, κ) = ( (g − 1) Vκ ,  V²κ (g − ½) )
                [ V (1 − gk)    1 − gkV/Δ    ]

  w_t is *defined* as the remainder, so this is an identity, not an approximation. The closures
  only decide how small w is.
- **Reading A and c.**
  - A has complex eigenvalues of modulus √((1 − gk)(1 + V²κ²)) ≈ 0.83: a lightly damped
    oscillation between heading error and offset.
  - c's second entry combines the chord with the controller turning *before* the step. With
    g = 1 it pulls the flock inward by about V²κ/2 per step. On geometry alone, the frozen
    recursion would hold a steady inward offset of 0.30, 0.22 and 0.14 m at R = 8, 12 and 20 m.

### 9.2 Lemma: mean-square ISS for a contracting random recursion [P]

Let x_{t+1} = A_t x_t + d_t + z_t, with:
- A_t and d_t F_t-measurable;
- E[z_t | F_t] = 0;
- τ a stopping time and P ≻ 0.

Suppose that for t < τ:
- ‖A_t‖_P ≤ λ < 1;
- (E[‖d_t‖²_P ; τ > t])^{1/2} ≤ B_t;
- E[‖z_t‖²_P ; τ > t] ≤ S_t².

Then, with c_e² = (P⁻¹)_ee,

    (E[e_t² ; τ > t])^{1/2} ≤ c_e (a_t + s_t),
    a_0 = (E‖x_0‖²_P)^{1/2},  a_{t+1} = λ a_t + B_t,      s_0 = 0,  s_{t+1}² = λ² s_t² + S_t².

For constant B and S this is c_e [λ^t ‖x_0‖ + B/(1 − λ) + S/√(1 − λ²)]: a decaying transient, a
bias floor and a noise floor.

*Proof.* Let X_t² = E[‖x_t‖²_P ; τ > t].
1. {τ > t + 1} ⊂ {τ > t}, and {τ > t} ∈ F_t. Hence
   X²_{t+1} ≤ E[‖A_t x_t + d_t‖²_P ; τ > t] + E[‖z_t‖²_P ; τ > t]. The cross term vanishes because
   A_t x_t + d_t is F_t-measurable and E[z_t | F_t] = 0.
2. By Minkowski in L², the square root of the first term is at most λX_t + B_t. So
   X²_{t+1} ≤ (λX_t + B_t)² + S_t².
3. Induction: if X_t ≤ a_t + s_t, then
   X²_{t+1} ≤ (a_{t+1} + λs_t)² + S_t² = (a_{t+1} + s_{t+1})² − 2a_{t+1}(s_{t+1} − λs_t),
   using s²_{t+1} − λ²s_t² = S_t². The last term is non-negative because s_{t+1} ≥ λs_t.
4. |e| ≤ c_e‖x‖_P by Cauchy–Schwarz in the P inner product. ∎

**Remarks.**
- No distribution is assumed for z beyond its conditional mean and second moment.
- Curvature jumps and saturation episodes are not special cases. They are steps where B_t is
  large, and their effect decays like λ^{t−t_j}.
- **Checking contraction is a finite computation.** A(V, κ²) is affine in V and in κ²
  separately, so on a box it is a convex combination of its four corner values. The operator
  norm is convex, so its maximum over the box is at a corner.

### 9.3 Proposition: cascade tracking at baseline [P, given the measured H2–H4]

Fix g = 1.026 (the closed-loop gain, below). Take the box V ∈ [0.25, 1], κ ∈ [0, 1/3], and τ the
first step with V_t < 0.25. The lemma then applies with d_t = c_t + b_t and z_t = w_t − b_t,
where b_t = E[w_t | F_t], under these hypotheses:

| | Hypothesis | Status at baseline |
|---|---|---|
| H1 | ‖A(V, κ)‖_P ≤ λ on the box | [P], checked at the corners: λ = 0.959 with P = [[4.33, 0.91], [0.91, 1]] (ψ in rad, e in m), so c_e = 1.11. P is chosen to minimize the bound on the straight |
| H2 | V_t ≥ 0.25 before τ (V ≤ v = 1 always) | V ≤ 1 is exact. Reached in 1 of the 79 published runs: R = 8, one step on the arc at 0.19, where that run's bound stops [S] |
| H3 | RMS of ‖c_t + b_t‖_P ≤ B_t at each step | B_t is estimated on 320 fresh calibration runs (seed 131, 64 per radius), with b_t from a regression of w_t on 14 quantities known at t (state, curvature, saturation, bearing, coverage, last step's saturation deficit) [S] |
| H4 | E‖z_t‖²_P ≤ S_t², and z unpredictable | S_t is estimated the same way. Unpredictability is checked out of sample on the 79 published runs: autocorrelation within ±0.015 at lags 1–8 (2/√n = 0.018), and correlation with every regressor at most 0.02 [S] |

**Where the closures enter.**
- *C2, the heading law*, makes A the right matrix:
  - On unsaturated steps with full coverage, the realized turn is 1.026 × the command
    (0.99–1.06 by radius), with intercept −0.002 rad and residual sd 0.127 rad (7.3°/step;
    n = 49 334 steps).
  - Earlier, the accumulated turn was 0.72–0.96 of the *table* value at the achieved bearing
    (§2). That ratio uses a different reference, the measured δ̂ at the bearing the dog actually
    reached. Relative to its own command, the loop turns almost exactly as asked.
- *C3, polarization*, keeps V in a narrow box.
- *R ≥ R_min* keeps the arc bias transient:
  - On a saturated arc step the heading falls short by 0.86κ rad (regression coefficient), close
    to the whole feedforward Vκ, and the shortfall carries into the next step while the dog
    stays saturated.
  - The dog is at its speed limit on 52%, 10% and 1.4% of arc steps at R = 8, 12 and 20 m, and on
    every arc step at R = 3 and 5 m.
- *C1, coverage*, matters only through the size of w.

### 9.4 Baseline values, against the published runs

Constants:
- Straight: B = 0.012 and S = 0.32 in the P-norm. The stationary bound is 0.32 m (bias)
  + 1.26 m (noise) = 1.58 m.
- Arc bias B: 0.125, 0.035 and 0.018 at R = 8, 12 and 20 m, against 0.41 and 0.88 at R = 5 and 3 m.
  S stays about 0.33 everywhere.

The bound starts from x_0 = 0, which holds exactly because the flock starts on the path heading
along it. It uses only calibration constants. "Measured" is the RMS over the published runs at
each step (`path_tracking.py`, seed 31; 16 runs per radius, 15 at R = 20).

| R (m) | Before the arc (t ≥ 15): measured / bound | Arc peak: measured / bound | ≥ 20 steps after the arc: measured / bound | Bound ÷ measured, min / median over steps |
|---|---|---|---|---|
| 8 | 0.56 / 1.37 | 0.77 / 2.99 | 0.59 / 2.37 | 1.7 / 2.9 |
| 12 | 0.48 / 1.43 | 1.12 / 2.03 | 0.67 / 1.79 | 1.7 / 2.4 |
| 20 | 0.70 / 1.39 | 0.83 / 1.78 | 0.66 / 1.69 | 1.4 / 2.3 |
| 5 (below R_min) | 0.63 / 1.35 | 2.19 / 5.96 | 0.63 / 4.22 | 1.8 / 2.9 |
| 3 (below R_min) | 0.66 / 1.40 | 3.51 / 9.77 | 0.64 / 6.65 | 1.6 / 3.2 |

In each window, "measured" is the RMS over the window's steps and "bound" its largest value.
The arc window runs from the arc entry to 10 steps after the exit.

- **The bound holds at every step for every radius.** It is at least 1.4× the measured value.
  The median ratio over steps is 2.3–2.9 above R_min and 2.9–3.2 below.
- **Where the slack comes from.** The robust norm contracts at 0.959 per step, against a
  spectral radius of 0.83. The cost is the non-normal A and the low-speed corner of the box.
  Adding the bias and noise terms by the triangle inequality costs more.
- **A narrower box.** V ∈ [0.5, 1] gives λ = 0.932, and arc peaks of 2.65, 1.70 and 1.48 m at
  R = 8, 12 and 20. But 2–4 of 16 published runs per radius (none at R = 20) have a step below
  0.5, and the bound stops there.
- **Below R_min the bound stays valid but grows**, because the saturation bias persists for the
  whole arc. R_min is where the arc bias stops being transient.

**The frozen recursion is accurate** [S]. Freeze V at its mean and use the calibration noise
covariance and mean bias. The exact stationary moments (discrete Lyapunov equation) then
predict the following:

| | Predicted RMS e (mean) | Measured, 64 calibration runs per radius | Measured, published runs |
|---|---|---|---|
| Straight | 0.65 (sd ψ 0.177) | 0.66 (sd ψ 0.172) | 0.61 (sd ψ 0.166) |
| Arc R = 8, second half | 0.68 (−0.23) | 0.74 (−0.21) | 0.61 (+0.04) |
| Arc R = 12 | 0.67 (+0.10) | 0.67 (−0.11) | 0.95 (−0.05) |
| Arc R = 20 | 0.67 (+0.07) | 0.68 (+0.04) | 0.72 (−0.05) |

- With 64 runs per radius, the one-step reduction predicts the multi-step tracking error within
  0.07 m. The published R = 12 value (16 runs) is the outlier.
- **Where the noise comes from.** The cross-track noise of 0.10 m per step is within the exact
  ceiling of §1, √((1 − P²)/N) v ≈ 0.14 m per step. That is consistent with the tracking-noise
  floor being finite-N centroid noise, which would predict a smaller floor for larger flocks.
  That prediction is untested.

### 9.5 What this establishes, and what it does not

**Establishes.**
- For the baseline flock and dog speed, with this controller, the mean-square cross-track error
  is bounded at every step by an explicit recursion:
  - about 1.4 m on the first straight;
  - 3.0, 2.0 and 1.8 m at arcs of radius 8, 12 and 20 m;
  - arc transients decay at 0.959 per step, toward the stationary 1.6 m.
- The proof is complete given H2–H4. H1 is a finite check.
- This is the centroid-level counterpart of Di Lorenzo's Lyapunov estimate: a quadratic
  function decays geometrically. Two differences:
  - the state is the centroid's path error, not a density error;
  - there is a noise floor, because 14 sheep are not a continuum.

**Does not establish.**
- *A bound on the largest error along a run.* The bound is per step. It gives a per-step tail via
  Chebyshev, P(|e_t| ≥ a) ≤ bound_t²/a², but nothing uniform over the path.
- *H3 and H4 from the model.* They are measured, and b_t = E[w_t | F_t] is only estimated. The
  regression captures what its 14 regressors can see; the whiteness checks are the evidence for
  the rest.
- *Anything at other parameters, flock sizes or controllers*, or for the shape (§4.2) or the
  density (§4.3).
- *Behaviour after τ*, and the probability of reaching τ, beyond the observed one run in 79.

*Provenance note.* Writing this up exposed a bug in `path_frame`, which `path_tracking.py` also
uses in its controller. A centroid more than R to the left of the final straight was read as
back on the first straight. Segments now only advance. The baseline published runs never hit
it. One non-baseline setting did: w_ali 0, v_dog 3, R 3. Its only changes are uncited fields:
RMS error on the straight 5.18 → 0.95 m, plus small shifts in a few other fields.

## 10. Deriving C4 (the restoring condition)

C4 asks for κγ ≥ k* > 0 outside the ball of the shape lemma (√M₂ ≥ 4.46 m at baseline, §4.2).
This section derives it rather than measuring it, with three constraints:
- normalization and random neighbour choice are kept exactly;
- the driving region is treated first;
- if a uniform one-step version is false, the next target is a multi-step inequality.

It uses `shepherd/expected_heading.py` (tested in `tests/test_expected_heading.py`) and
`experiments/restoring_condition.py` (results/restoring_condition.json,
results/figures/restoring_condition.png).

**States.** 96 runs of the baseline closed loop (published Strömbom dog, collecting on), sampled
every 10 steps from step 100 to 290: 1,920 states, all with the dog in drive mode. Larger spreads
are made by dilating these states about the centroid, keeping headings and the dog fixed. Only
states in which every sheep is within r_dog of the dog are used:

| Dilation | States kept | √M₂ (median) | Outside the ball |
|---|---|---|---|
| ×1 | 1,787 | 3.0 m | 0 |
| ×1.25 | 1,293 | 3.7 m | 13 |
| ×1.5 | 450 | 4.4 m | 181 |
| ×1.75 | 63 | 5.0 m | 61 |
| ×2 | 0 | – | – |

So the driving region with full coverage is thin outside the ball, and it ends before 2× the
observed spread.

### 10.1 The expected heading, exactly [E]

A sheep that sees the dog takes h′ = unit(c + 0.3 e(ψ)), where:
- c = B + 1.5 a(J) + 1.3 h_j and B = 0.5 h + 2 rep + away;
- J is a uniformly random set of 4 of the 10 nearest neighbours;
- a(J) = unit(Σ_{k∈J} unit(x_k − x));
- j is uniform in J, so it is uniform over the 10 nearest;
- ψ is uniform on the circle.

Averaging over ψ only shrinks the unit vector radially:

    E_ψ[unit(c + r e(ψ))] = m(r/|c|) unit(c),   m(q) = [(1 + q) E(k) + (1 − q) K(k)] / π,  k = 2√q/(1 + q),

with E and K the complete elliptic integrals. So m(0) = 1, m(1) = 2/π, and m(q) ≈ 1/(2q) for
large q. The conditional mean μ_i = E[h_i′ | x] is therefore an exact average over the 210 × 4
equally likely (J, j) choices. There is no Monte Carlo: it agrees with 40,000 simulated
replicates within 5 standard errors for every sheep. κ is now an exact, computable function of
the state.

### 10.2 Exact decomposition [E]

Work in the frame (e, n) of the flock's mean expected heading, and write ξ_i = x_i e + y_i n and
S = Σ|ξ_i|. With every sheep seeing the dog, κ = κ∥ + κ⊥, where:
- κ∥ = (1/S) Σ x_i Δ_i with Δ_i = 1 − μ_i·e ≥ 0. This is the *speed-deficit asymmetry*: it is
  restoring when the front sheep are slower.
- κ⊥ = −(1/S) Σ y_i μ_i·n: restoring when the sheep turn inward.

**The normalization identity.** For each realization c, let ω = m(0.3/|c|)/|c| > 0. Then
m·unit(c) = ω c, and taking expectations gives

    μ_i = ω̄_i F_i + Cov(ω_i, c_i),    F_i = E c_i = 0.5 h_i + 2 rep_i + away_i + 1.5 ā_i + 1.3 h̄_N(i),

where ā_i averages a(J) over the 210 subsets and h̄_N(i) is the mean heading of the 10 nearest.
So κ⊥ splits exactly into one ω̄-weighted term per force, plus the normalization covariance ℰ:

    κ⊥ = 𝒜 − 𝒟 − ℛ + ℋ + ℰ,     each force term = −(1/S) Σ y_i ω̄_i (weight × force_i)·n.

**What normalization does to inward attraction.**
- *Across the heading* the pull survives normalization. It enters with a positive per-sheep
  weight ω̄_i ≈ 1/|F_i|, and is corrected by ℰ, which is small: medians −0.002 to −0.019.
  Cauchy–Schwarz bounds it: |ℰ| ≤ ℰ_CS = (1/S) Σ |y_i| sd(ω_i) sd(c_i·n) ≈ 0.06.
- *Along the heading* it does not survive. Every moving sheep moves at speed v, so the front–back
  pull changes only |F_i|, and through it the angular spread and the speed deficit Δ_i. The
  front–back restoring is a second-order normalization effect.
- *The naive reading fails by 8–40×.* Divide the attraction's front–back pull by a common |F̄|
  and you predict κ∥ ≈ 0.16. The exact κ∥ is 0.004–0.020.

### 10.3 Driving-region result

**Proposition 4 (one step, driving region)** [P]. If the dog drives and every sheep sees it, then

    κ ≥ 𝒜 − 𝒟 − ℛ + ℋ − ℰ_CS + min(κ∥, 0).

C4 holds wherever the right-hand side is at least k*/γ. Each term depends on one factor:

| Term | Factor | Formula (all exact) |
|---|---|---|
| 𝒜 | neighbour geometry | −(1/S) Σ y_i ω̄_i 1.5 ā_i·n, with ā_i from the directions to the 10 nearest, averaged over the random subsets |
| 𝒟 | dog position | (1/S) Σ ω̄_i y_i (y_i − y_d)/d_i, since away_i·n = (y_i − y_d)/d_i exactly (d_i the sheep's distance to the dog, y_d the dog's lateral offset) |
| ℛ | crowding | (1/S) Σ y_i ω̄_i 2 rep_i·n; zero unless a neighbour is within r_rep |
| ℋ | heading dispersion | −(1/S) Σ y_i ω̄_i (0.5 h_i + 1.3 h̄_N(i))·n: the lateral correlation of own and neighbour-mean headings with position |
| κ∥ | speed-deficit asymmetry | (1/S) Σ x_i Δ_i |
| ℰ_CS | normalization error bound | (1/S) Σ \|y_i\| sd(ω_i) sd(c_i·n) |

Coverage enters as the hypothesis. A sheep out of range either stands still (C_frz, restoring
when it is ahead) or follows a different rule.

**Baseline values** [S], medians with [10th, 90th percentiles]; positive means restoring:

| | ×1 (√M₂ 3.0) | ×1.5 (4.4) | ×1.75 (5.0) |
|---|---|---|---|
| Attraction 𝒜 | +0.359 [+0.310, +0.426] | +0.371 [+0.334, +0.422] | +0.378 [+0.346, +0.418] |
| Dog −𝒟 | −0.123 [−0.155, −0.099] | −0.174 [−0.204, −0.147] | −0.196 [−0.217, −0.166] |
| Repulsion −ℛ | −0.183 [−0.282, −0.096] | −0.043 [−0.108, −0.007] | −0.028 [−0.068, −0.001] |
| Own heading | −0.008 [−0.039, +0.020] | −0.007 [−0.037, +0.022] | −0.001 [−0.029, +0.026] |
| Neighbours' headings | −0.002 [−0.023, +0.017] | −0.002 [−0.022, +0.017] | +0.001 [−0.015, +0.019] |
| Normalization ℰ | −0.002 [−0.020, +0.012] | −0.016 [−0.035, −0.003] | −0.019 [−0.041, −0.007] |
| κ⊥ | +0.038 [−0.040, +0.116] | +0.126 [+0.067, +0.176] | +0.127 [+0.087, +0.175] |
| κ∥ (naive reading) | +0.004 (0.158) | +0.017 (0.161) | +0.020 (0.162) |
| Share with κ < 0 | 23% | 0% | 0% |
| Sheep with a neighbour within 2 m | 86% | 43% | 36% |

**Reading the table.**
- *Attraction does not grow with size.* It is a unit vector, which is why the lemma's drift has
  the −a√M₂ form.
- *The dog's divergence grows with size*, as y²/d. Extrapolated, it would reach attraction near
  3× the observed spread. But coverage ends before 2×, so inside the driving region the dog never
  overpowers attraction at baseline.
- *Repulsion is what holds the flock at its observed size*: 86% of sheep have a neighbour within
  2 m. That is why 23% of observed states have κ < 0. They are inside the ball, so this does not
  violate C4.

**What the proposition certifies.** Outside the ball, the right-hand side is positive on 62%
(×1.25, 13 states), 87% (×1.5, 181) and 98% (×1.75, 61) of states. The exact κ is positive on all
255; the smallest is +0.004.
- *The slack* is mostly ℰ_CS (0.06, against an actual |ℰ| of about 0.02).
- *A coarser version fails.* Replace each ω̄_i by its range over the flock, so the conditions
  involve only force geometry, and it certifies none of them. The per-sheep normalization weight
  cannot be dropped.

### 10.4 A uniform one-step C4 is false [S, exact on constructed states]

Take the 255 outside-ball driving states and turn each sheep's heading outward (away from the
flock's axis) by φ:

| Fan φ | 0° | 5° | 10° | 20° | 30° | 40° | 60° | 90° |
|---|---|---|---|---|---|---|---|---|
| κ, median | +0.143 | +0.126 | +0.109 | +0.077 | +0.051 | +0.033 | +0.007 | +0.012 |
| κ, smallest | +0.004 | −0.003 | −0.013 | −0.043 | −0.067 | −0.085 | −0.099 | −0.103 |
| Share with κ < 0 | 0% | 0.4% | 0.8% | 1.6% | 7.8% | 16% | 38% | 35% |

- **Critical fans.** The fan at which κ turns negative has a 10th percentile of 33° and a minimum
  of 2.9°; 54% of states never turn negative up to 90°.
- **Real flocks fan too.** Undisturbed driving has fans of median +2.9°, 90th percentile +11.8°
  and maximum +26°.
- **Consequence.** No one-step C4 holds uniformly on the driving region outside the ball without a
  heading condition. ℋ ≥ −η is that condition. It is a real restriction: an outward fan is a state
  where the sheep have just moved apart and still head apart.

### 10.5 Multi-step restoring: a drift function that allows temporary outward motion

**Exact identities** [E]. Let Q_t = (1/N) Σ ξ_i(t)·h_i(t) be the position–heading covariance of
the current headings. With every sheep moving, ξ_i(t+1) = ξ_i + v(h_i′ − h̄′). This gives

    M₂(t+1) − M₂(t) = 2v Q_{t+1} − v²(1 − P²_{t+1}),
    E[Q_{t+1} | x_t] = V(x_t) + v (1 − E[P²_{t+1} | x_t]),   V = (1/N) Σ ξ_i·μ_i = −κ·mean|ξ|,
    E[P²_{t+1} | x_t] = |μ̄|² + (1/N²) Σ (1 − |μ_i|²).

**Proposition 5** [P]. Suppose that on a region the *memory inequality*

    V(x) ≤ λ Q(x) − k·mean|ξ|,   with λ ∈ [0, 1) and k > 0,

holds, and 1 − E[P²_{t+1} | x] ≤ D. Then W = M₂ + βQ with β = 2vλ/(1 − λ) satisfies

    E[ΔW | x] ≤ −(2v/(1 − λ)) k·mean|ξ| + v² D (1 + λ)/(1 − λ).

*Proof.*
1. By the identities, E[ΔW] = (2v + β)(V + vD_x) − βQ − v²D_x.
2. The memory inequality bounds V. Because (2v + β)λ = β, the Q terms cancel.
3. What remains is −(2v + β)k·mean|ξ| + v²D_x(2/(1 − λ) − 1). ∎

Two remarks:
- Since |Q| ≤ √M₂ √(1 − P²), W and M₂ differ by at most β√M₂. So this gives the shape lemma's
  −a√M₂ + b form for W.
- The memory inequality is a relaxation of one-step C4. Whenever V ≤ −κ*·mean|ξ| and Q ≥ 0, it
  holds with k = κ* for every λ. An outward fan raises V, but also raises Q, which W charges and
  repays over the following steps.

**Evidence** [S, exact on tested states]. The pairs tested are the 255 outside-ball states with
three kinds of fan: all sheep (0–90°), only the inner two-thirds (10–60°, acting through
alignment), and every other sheep (10–60°). Over these 4,335 pairs:
- **The memory inequality holds for every λ ≥ 0.15.** The best k is 0.069, at λ = 0.35–0.40. On
  the same pairs the one-step version (λ = 0) fails, with k = −0.103.
- **The ball.** At λ = 0.35, β = 1.08. The ball for W has √M₂ ≈ 3.7 m at typical polarization
  (1 − E P′² = 0.32), comparable to the one-step ball. It grows to 11 m if polarization collapses,
  as in 90° fans. The W drift bound is negative on 76% of the tested pairs.

**Monte Carlo under the published dog** (dog rules on, 64 replicates per state, 255 states).
Number of states with positive expected change of M₂ after s steps, and how many of those are
more than 2 SE above zero:

| s steps | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Fan 0° | 2 (2) | 0 | 0 | 0 | 0 |
| Fan 20° | 36 (19) | 3 (0) | 0 | 0 | 0 |
| Fan 40° | 168 (77) | 13 (4) | 3 (1) | 1 (0) | 0 |

Outward fans cause a one-step increase of M₂ in two-thirds of the states. That outward motion is
temporary: by 5 steps every state's M₂ has fallen.

**Status.**
- *Proved:* the identities and Proposition 5.
- *Tested, not proved:* the memory inequality, on the families above only. §11 takes this
  further: a drift identity for any memory matrix, a positive-part variant, and an adversarial
  test.
- *Path to a proof:* bound the two heading terms of §10.3 by λQ. The own-heading part of V is
  0.5 ω̄ Q when ω̄_i is uniform. The neighbour part involves a different moment,
  Q_N = (1/N) Σ ξ_i·h̄_N(i). So the response of V to a fan, dV/dQ at 20°, depends on the fan's
  pattern:

  | Fan pattern | Median dV/dQ | Largest |
  |---|---|---|
  | All sheep | 0.26 | 0.49 |
  | Inner two-thirds | 0.16 | 0.62 |
  | Every other sheep | 0.26 | 1.04 |

  No scalar memory factor below 1 covers every pattern pointwise. The inequality holds on the
  tested pairs only because of its restoring margin k. The natural next step is a two-component
  memory inequality in (Q, Q_N).

### 10.6 From collecting to driving: what is still needed

At baseline the dog never collects during undisturbed driving: all 1,920 snapshots are in drive
mode. Collecting requires a sheep at least l_sep = 11.6 m from the centroid. It matters after a
disturbance, and there it is decisive (steer_and_shape §11.3, r_dog 12):
- In the 2× dilation and in the split, stretch and side-outlier families, 93–100% of starts lie
  outside the shape lemma's region 𝒢. The ahead-outlier and heading families start inside it.
- With a drive-only dog, the share that recovers is:
  - 0% for a 12 m lateral split and for a 3× lateral stretch;
  - 13% for an outlier 14 m to the side;
  - 53% for an 8 m lateral split;
  - 77% for the 2× dilation.
- With collecting, every family recovers in all runs (median 2–64 steps, 90th percentile ≤ 110).

**A gap in the switching rule** [D]. A sheep is within r_dog of the drive point (7.5 m behind the
centroid) only if it is at most 4.5 m ahead or 9.4 m to the side. So a sheep 9.4–11.6 m to the
side is neither covered nor collected. It freezes, and is collected only once the flock has moved
far enough that it exceeds 11.6 m. That is consistent with the slow recoveries from 12 m lateral
splits (median 64 steps).
*Corrected in [collecting_chain.md](collecting_chain.md) §4:* the dog drives from behind, so a
frozen sheep lies ahead of its notice disc. The advancing dog passes it and wakes it within about
7 m of flock advance, and on recovery paths frozen sheep are woken in a median of 1 step. The slow
lateral-split recoveries come from the drive's lateral push (§12.6), not from the gap.

**What would connect collecting to Propositions 4 and 5:**
1. *A collecting drift.* While max|ξ_i| ≥ l_sep, the dog behind the farthest sheep reduces its
   distance: E[Δ max|ξ| | collect] ≤ −c₁. This is the reaching argument of
   continuification_driving §9 applied to one sheep, and gives a finite expected collecting time.
2. *Escape from the gap.* A frozen lateral sheep's distance to the moving flock grows at about the
   flock's speed, so it reaches l_sep in finite time. This needs the flock to keep moving, which
   driving supplies.
3. *A handover condition.* When the dog returns to driving, the state must reach full coverage
   within bounded time, with a bounded fan. Proposition 5 tolerates fans; Proposition 4 does not.
4. *No endless switching.* The rule has no hysteresis, so the dog can alternate between modes.
   This needs either a bound on the number of switches or a drift function common to both modes,
   for example W plus a term in max|ξ|.

None of these is proved here. The 100% recovery with collecting is the evidence that they hold
at baseline.

**Followed up in [collecting_chain.md](collecting_chain.md).**
1. A one-step drift for max|ξ| is false: it is positive exactly at 29% of collecting states. A
   two-phase potential (the farthest sheep's distance plus the dog's distance to it) has an exact
   drift bound, ≤ −0.1 at 98.9% of collecting states.
2. The gap is short-lived (Lemma G there).
3. A handover with a 3 m analysis band reaches the ball before collecting again with probability
   at least 0.51 at every sampled handover.
4. Cycles are geometric, so from a handover E[τ_ball] ≤ 214 steps on the sampled states.

Target switches inside a collecting cycle are bounded in cost (each at most 4v + 2R′) but not in
number. Two stragglers on opposite sides take turns with no ratchet. With no rival beyond 9 m there
are no switches across the flock (collecting_chain §3.5).

## 11. The multi-step restoring inequality: what can be proved

§10.5 left a memory inequality that held on every tested fan but was not proved, and proposed a
two-component version in (Q, Q_N). This section settles what can be proved. It adds
`experiments/memory_inequality.py` (results/memory_inequality.json,
results/figures/memory_inequality.png) and uses the same 255 driving states outside the ball as
§10. Every sheep sees the dog in all of them.

### 11.1 Why (Q, Q_N) does not close

The heading part of the virial V involves own headings (through Q) and neighbour-mean headings
(through Q_N). To control Q_N one step later, one needs the next neighbour-mean headings. Those
copy the current headings of the neighbours' neighbours, which brings in Q_NN, and so on.
Alignment carries headings along the neighbour graph, so no finite set of such moments closes.
The closure sums the chain: charge each current heading for its whole discounted effect on
future virials. That is a resolvent.

### 11.2 The rule is linear in current headings, with state-dependent weights [E]

With the normalization weights ω of §10.2, a sheep that sees the dog has exactly

    μ_i = Σ_j L_ij h_j + g_i,   L_ii = 0.5 ω̄_i,   L_ij = 1.3 × (mean of ω over the realizations that copy j), j among the 10 nearest,
                                g_i = E[ω (2 rep_i + away_i + 1.5 a_i(J))].

L ≥ 0 and its row sums are 1.8 ω̄_i. The identity holds to 3·10⁻¹⁵. But the exact L is no use as
a memory:
- Its spectral radius reaches 1.02 on the 255 states (median 0.81). Sheep whose forces nearly
  cancel have very large normalization weights, and the resolvent blows up.
- It depends on the current headings, through ω.

So the construction below uses a surrogate L̂.

### 11.3 A drift identity for any memory matrix [P]

**Proposition 6.** Let L̂(x) be any matrix known at time t with spectral radius below 1, and set:
- R̂ = L̂(I − L̂)⁻¹;
- u_i = h_i − h̄;
- Φ = (1/N) Σ_i ξ_i·(R̂u)_i;
- W = M₂ + 2vΦ.

If every sheep moves, then exactly

    E[ΔW | x] = 2v Π + v² D + 2v (Err_a + Err_b),
    Π     = (1/N) ξᵀ (I − L̂)⁻¹ (μ − μ̄ − L̂u)
    Err_a = (v/N) E[u′ᵀ R̂(x′) u′],             |Err_a| ≤ v λ_max(sym R̂(x′)) D
    Err_b = (1/N) E[ξᵀ (R̂(x′) − R̂(x)) u′],     zero when R̂ is constant.

*Proof.*
1. Moving every sheep gives ξ′ = ξ + v u′. So, sample by sample,
   Φ(x′) = (1/N)ξᵀR̂u′ + (1/N)ξᵀ(R̂′ − R̂)u′ + (v/N)u′ᵀR̂′u′.
2. R̂ is known at time t and E[u′] = μ − μ̄, so E[(1/N)ξᵀR̂u′] = (1/N)ξᵀR̂(μ − μ̄).
3. With E[ΔM₂] = 2vV + v²D (§10.5) and V = (1/N)ξᵀ(μ − μ̄), this gives
   E[ΔW] = 2v(1/N)ξᵀ[(I + R̂)(μ − μ̄) − R̂u] + v²D + 2v(Err_a + Err_b).
4. Finally I + R̂ = (I − L̂)⁻¹ and R̂ = (I − L̂)⁻¹L̂. ∎

The identity holds for every choice of L̂; the choice decides how much Π still depends on
headings. Suppose μ − μ̄ responded to current headings exactly as L̂u. Then Π would be the virial of
the relative headings the flock would settle into with its positions frozen,
u* = (I − L̂)⁻¹(μ − μ̄ − L̂u), whatever the current headings. Special cases:
- *L̂ = 0* gives back the one-step lemma: Φ = 0 and Π = V.
- *L̂ = λI* gives Proposition 5: Φ = λQ/(1 − λ), Π = (V − λQ)/(1 − λ), Err_a = λvD/(1 − λ)
  exactly, and Err_b = 0. Then E[ΔW] = 2vΠ + v²D(1 + λ)/(1 − λ) exactly.

**Corollary 7 (drift)** [P]. Suppose that on a region:
- Π ≤ −k·mean|ξ|;
- D ≤ D̄;
- Err_b ≤ e·mean|ξ| with e < k;
- λ_max(sym R̂) ≤ c_R at the next states.

Then

    E[ΔW | x] ≤ −2v(k − e)·mean|ξ| + v² D̄ (1 + 2c_R),

and |W − M₂| ≤ 2v‖R̂‖₂ √M₂ √(1 − P²). This is the shape lemma's −a√M₂ + b form, for W.

**Constants that need no measurement.**
- *Scalar λ:* c_R = λ/(1 − λ) and Err_b = 0.
- *Graph-aware L̂ = ω₀(0.5 I + 1.3 A)*, with A the average over each sheep's 10 nearest:
  - for any neighbour graph of 14 sheep, ‖L̂‖_∞ = 1.8 ω₀ and ‖L̂‖₁ ≤ 2.19 ω₀;
  - at ω₀ = 0.40, ‖R̂‖₂ ≤ √(‖R̂‖₁‖R̂‖_∞) ≤ √(7.06 × 2.57) = 4.26, so c_R ≤ 4.26;
  - Err_b comes from changes in the neighbour graph and has no analytic bound here.

### 11.4 Charging only outward headings [P]

**Proposition 8.** Let W⁺ = M₂ + βQ⁺ with β = 2vλ/(1 − λ). If

    V ≤ λ Q⁺ − k·mean|ξ|            (MI⁺)

then

    E[ΔW⁺ | x] ≤ −2v k·mean|ξ| + v² D (1 + λ)/(1 − λ) + β σ/2,
    σ = (1/N) √(Σ |ξ_i|² (1 − |μ_i|²)) + v/2 ≥ sd(Q′ | x).

*Proof.*
1. E[Q′⁺] ≤ (E Q′)⁺ + E|Q′ − E Q′|/2 ≤ (E Q′)⁺ + σ/2, with E Q′ = V + vD (§10.5). The bound on σ
   holds because the sheep draw independently given x, so Var((1/N) Σ ξ_i·h_i′) ≤
   (1/N²) Σ |ξ_i|² (1 − |μ_i|²), and v(1 − |h̄′|²) lies in [0, v].
2. *If V + vD ≥ 0:* the choice of β cancels the Q⁺ terms, as in Proposition 5.
3. *If V + vD < 0:* E[ΔW⁺] ≤ 2vV + v²D + β(σ/2 − Q⁺), and 2vλ ≤ β.
4. In both cases the bound above follows. ∎

MI⁺ is weaker than the linear memory inequality wherever Q < 0, so inward headings cannot break
it. The price is the fluctuation term βσ/2.

### 11.5 Results on the 255 states [S]

The identity is confirmed sample by sample (to 2.5·10⁻¹³) and in mean: Monte Carlo z-scores
under the published dog have RMS 1.04–1.09.

The table gives the restoring margin −Π/mean|ξ| (the smallest over states) for each memory
matrix. The adversary searches all 14 headings of each state, keeping polarization ≥ 0.5, to make
Π ≥ 0.

| Memory matrix | ρ(L̂) | Natural headings | Worst tested family (fans, random) | Adversary: states where Π ≥ 0 | Adversary's largest Π/mean\|ξ\| |
|---|---|---|---|---|---|
| One-step, L̂ = 0 | 0 | +0.004 | −0.103 | 253 of 255 | +0.188 |
| Scalar, λ = 0.35 | 0.35 | +0.106 | +0.079 | 8 | +0.049 |
| Scalar, λ = 0.50 | 0.50 | +0.137 | +0.038 | 159 | +0.207 |
| Graph-aware, ω₀ = 0.35 | 0.63 | +0.083 | +0.074 | 7 | +0.072 |
| Graph-aware, ω₀ = 0.40 | 0.72 | +0.096 | +0.080 | 3 | +0.081 |
| Positive part (MI⁺), λ = 0.35 | – | +0.107 | +0.101 | **0** | −0.007 |

For MI⁺ the margin is shown as −(V − λQ⁺)/((1 − λ)·mean|ξ|). The constant k of Proposition 8 is
(1 − λ) times it: 0.070 on natural headings, 0.066 on the worst family, 0.005 under the
adversary.

**The drift of W, term by term** (medians over states, m²; for the graph-aware matrix, Err_a and
Err_b are from Monte Carlo):

| | 2vΠ | v²D | 2v Err_a | 2v Err_b | States with E[ΔW] < 0 |
|---|---|---|---|---|---|
| Scalar λ 0.35, natural | −2.15 | +0.24 | +0.26 | 0 | 255 of 255 (exact) |
| Scalar λ 0.35, 40° fan | −2.81 | +0.40 | +0.43 | 0 | 255 of 255 (exact) |
| Scalar λ 0.35, alternate 60° fan | −2.70 | +0.41 | +0.44 | 0 | 255 of 255 (exact) |
| Graph ω₀ 0.40, natural | −1.95 | +0.24 | +0.09 | +0.07 | 255 of 255 |
| Graph ω₀ 0.40, 40° fan | −2.18 | +0.40 | +0.15 | +0.12 | 254 of 255 |
| Graph ω₀ 0.40, alternate 60° fan | −2.15 | +0.41 | +0.15 | +0.12 | 254 of 255 |

**Reading the tables.**
- *For the graph-aware matrix, the rigorous bound on Err_a is loose:* its median is 0.64–1.08,
  against an actual 0.05–0.08.
- *Err_b is small:* at most 0.042·mean|ξ|.
- *M₂ versus W:* under these fans, M₂'s own one-step drift is positive in up to two-thirds of
  states (§10.5). W's drift is negative in all of them, so the temporary outward motion is
  absorbed.

**The positive-part bound, rigorous apart from MI⁺:**

| Heading family | Natural | 20° fan | 40° fan | 60° fan | 90° fan | Alternate 60° | Random ±60° |
|---|---|---|---|---|---|---|---|
| Share of states with the bound below 0 | 90% | 92% | 89% | 69% | 26% | 74% | 90% |

The fluctuation term costs about a third of the restoring term (median βσ/2 = 0.51 against
2vk·mean|ξ| = 1.49 on natural headings). It grows as polarization collapses under wide fans.

**What the adversary finds.**
- *Against the one-step version:* the outer sheep are turned outward by about 82°, giving
  Q = 3.1 against a natural 0.23.
- *Against the scalar versions:* the headings are turned inward (outer sheep −31°, Q = −1.27 at
  λ = 0.35; Q = −3.2 at λ = 0.5). A linear W counts inward headings as stored restoring and
  demands it back later, so the larger λ, the easier it is to break. This is a defect of the
  Lyapunov function, not an outward push, and the positive part removes it.
- *Against the graph-aware versions:* outer sheep are turned outward and inner sheep inward.
  This exploits the mismatch between ω₀ and the true per-sheep weights. Following the graph
  helps a little (3 failing states instead of 8) but does not remove the failures.

### 11.6 Status

- **Proved.**
  - No finite moment hierarchy in (Q, Q_N, …) closes (§11.1); the resolvent does.
  - The exact linear structure of the rule (§11.2).
  - Proposition 6 and Corollary 7, with constants that need no measurement for the scalar and
    graph-aware memories.
  - Proposition 8 for the positive-part W⁺.
- **Shown false.** For every linear memory matrix tried, Π ≤ 0 is not uniform over heading
  fields with polarization ≥ 0.5. Adversarial headings break it in 1–3% of the states outside the
  ball, against 99% for the one-step version.
- **Not proved here: MI⁺, the one remaining hypothesis.** §12 certifies it at 178 states,
  refutes it at 6, and follows the counterexamples forward.
  - It holds on every tested family and survives the adversary in all 255 states, but in the
    worst state the margin is only 0.005.
  - The adversary is a local search, not a certificate, so this is evidence, not proof.
  - Given MI⁺, Proposition 8 proves negative drift of W⁺ in 90% of the undisturbed states
    outside the ball.
- **Not addressed.** Staying in the driving region (coverage) is still the collecting chain of
  §10.6. Fans wide enough to collapse polarization (90°) defeat the rigorous W⁺ bound in most
  states.

**What would close it.**
- *A certificate for MI⁺ over heading fields.* This is a 14-dimensional global maximization per
  position configuration. The heading-dependent part of V is a sum of single-sheep terms coupled
  only through the copied neighbour, which suggests a branch-and-bound over each sheep's
  neighbourhood.
- *Or a restriction to reachable heading fields.* Every heading at t ≥ 1 is an output of the rule
  and lies within the cone of its sheep's realizable drives. The adversary's fields, turned by 50°
  on average, may not be reachable at all.

## 12. Certifying MI⁺, its counterexamples, and a multi-step drift

§11 reduced the multi-step restoring inequality to one hypothesis, MI⁺: V ≤ λQ⁺ − k·mean|ξ| with
λ = 0.35. This section does three things:
- certifies MI⁺ where it holds;
- shows it fails at a few states;
- follows those states forward under the baseline controller.

It adds `experiments/certify_mi.py` (results/certify_mi.json, results/figures/certify_mi.png) and
`experiments/mi_counterexamples.py` (results/mi_counterexamples.json,
results/figures/mi_counterexamples.png). As before, the states are the 255 driving states outside
the ball.

### 12.1 A certificate [P]

**Reduction.** For any α ∈ [0, λ], −λQ⁺ ≤ −αQ. So MI⁺ holds with constant k at a state if

    sup over all heading fields θ of (V − αQ) ≤ −k·mean|ξ|.

No polarization condition is needed, so a certificate covers every heading field.

**The structure that makes it computable.** The noise average has a closed form, and the subset
and copy average is finite. So V is a sum of terms that each depend on two headings:

    V = Σ_i Σ_{k∈N(i)} t_ik(θ_i, θ_k),
    t_ik = (1/(10N)) × mean over the 84 subsets J that contain k of ξ_i·g(G_iJ + 0.5 h(θ_i) + 1.3 h(θ_k)),

where g(c) = m(r/|c|)·unit(c) and G_iJ = 2 rep_i + away_i + 1.5 a_i(J) depends on positions only.

**The bound.**
1. *Cells.* Cut each heading angle into M cells. Over a pair of cells, c lies within
   ε = 1.8·2 sin(δ/4) of its value at the cell centres. So |c| and the angle between c and ξ_i lie
   in known intervals. Because m decreases in q = r/|c|, interval arithmetic gives an upper bound
   on ξ_i·g(c) with no Lipschitz constant. That matters: m has a logarithmically singular derivative
   at |c| = r, where the noise circle passes through the origin, so a Lipschitz bound would fail
   there.
2. *Unary terms.* The terms −αξ_j·h_j have exact cell maxima.
3. *Discrete problem.* Every heading field lies in some tuple of cells, so the maximum of the
   discrete problem built from these cell bounds is at least sup(V − αQ).
4. *Relaxation.* Bound that discrete maximum by the dual of its linear-programming relaxation,
   reduced by message passing (MPLP). For any messages λ,
   Σ_j max[u_j + Σ_e λ_{e→j}] + Σ_e max[T_e − λ_{e→j} − λ_{e→l}] is an upper bound, because it
   bounds the objective at every assignment term by term.
5. Take the smallest bound over α ∈ {0.20, 0.25, 0.30, 0.35}.

**Floating point.** Writing this up exposed an inaccuracy in `noise_shrink`: within 10⁻⁵ of q = 1
it dropped the (1 − q)K term. It now uses the complementary elliptic parameter (`ellipkm1`) and
matches 2²⁴-node quadrature to 1.5·10⁻⁹ everywhere. The certificate adds a margin of 2·10⁻⁹ to
every bound on m. Re-running `restoring_condition.py` changed nothing beyond 10⁻⁹. Re-running
`memory_inequality.py` changed nothing beyond 10⁻⁶, apart from one descriptive angle that moved
by 3·10⁻⁶ degrees.

### 12.2 What it certifies [P on the certified states, S elsewhere]

Without any polarization condition:

| Status | States (of 255) |
|---|---|
| **Certified** (MI⁺ holds for every heading field) | **178**: 134 at M = 72, 44 more at M = 144 |
| Not certifiable this way: at the best α, an attack finds V − αQ > 0 | 15 |
| Undecided | 62 |

A positive V − αQ at one α does not refute MI⁺, which is weaker. It only shows that this route,
at that α, cannot certify the state. MI⁺ itself is refuted only by the counterexamples of §12.3.

- **The certified constants are small** (median k = 0.018, 10th percentile 0.005, largest 0.068),
  because they hold against the worst heading field, not the natural one.
- **The gap is too wide to decide the rest.** The gap between the certificate and the best attack
  has median 0.055·mean|ξ| and largest 0.13. It comes from the cells and from the relaxation. So the
  undecided states could go either way. Cells finer than M = 144, or the polarization condition as
  a further multiplier, might decide some of them; neither was tried.
- **Consistency.** The attack never exceeds the certificate, and every state where a
  counterexample was found is uncertified.

### 12.3 Counterexamples, and an obstruction for any function of Q [P]

With polarization at least 0.5, a multi-start adversary (12 starts on the 30 tightest states)
finds heading fields that violate MI⁺ at 6 states. The violation is 0.001 to 0.041·mean|ξ|.
- Every sheep sees the dog, and P is 0.53–0.81.
- Each was checked exactly.
- In five of the six, Q is essentially 0 (|Q| ≤ 0.005) while V > 0. Alignment carries outward
  motion that the current position–heading covariance does not show.

**Proposition 9 (Jensen obstruction).** Let W = M₂ + f(Q), with f convex and nondecreasing. At a
state where every sheep moves, V > 0 and Q ≤ V + vD,

    E[ΔW | x] ≥ 2vV + v²D > 0.

*Proof.*
1. E[ΔM₂] = 2vV + v²D (§10.5).
2. By Jensen, E[f(Q′)] ≥ f(E Q′) = f(V + vD) ≥ f(Q), since f is nondecreasing and V + vD ≥ Q. ∎

The condition holds at all six counterexamples. So at these states MI⁺ is false, and so is every
drift function built from M₂ and Q alone, linear or positive-part. Ruling them out would need a
heading functional that sees the alignment graph, or a statement over several steps.

### 12.4 The counterexamples, forward under the baseline controller [S]

Each of the six was run for 80 steps with 256 replicates, under the published Strömbom dog
(collecting on) and the published sheep rule. The control is the same positions with their
natural headings.

| State | Violation | √M₂ (m) | P | Back in the ball | Median steps (natural) | M₂ growth before falling | Out-of-range sheep ahead, steps 1–3 (natural) |
|---|---|---|---|---|---|---|---|
| 161 | 0.041 | 4.58 | 0.76 | 100% | 5 (2) | 1 step, +0.81 m² | 43% (98%) |
| 45 | 0.013 | 4.55 | 0.62 | 100% | 3 (2) | 1 step, +0.83 m² | 74% (85%) |
| 16 | 0.004 | 4.70 | 0.53 | 100% | 4 (3) | 1 step, +0.42 m² | 88% (96%) |
| 35 | 0.005 | 4.48 | 0.63 | 100% | 3 (2) | 1 step, +0.51 m² | 52% (89%) |
| 136 | 0.006 | 4.61 | 0.81 | 100% | 3 (3) | 1 step, +0.46 m² | 97% (97%) |
| 81 | 0.001 | 4.72 | 0.65 | 100% | 4 (3) | 1 step, +0.23 m² | 36% (68%) |

**They recover, and the expansion is a single-step kick.**
- **One step of expansion.** M₂ rises for exactly one step, then falls toward the observed
  equilibrium (√M₂ ≈ 3.0 m by step 40). The 90th percentile of the return time is 5–8 steps.
- **The kick is stored in Q and repaid.** V is positive only at step 0. From step 1 it is −0.07
  to −0.19 per mean|ξ|. Q jumps to +0.07 to +0.19 at step 1, which is the outward step stored,
  then falls below zero within a few steps.
- **Recovery is driving only.** In at least 98.8% of runs the dog never collects before the flock
  is back in the ball. No run ends split; at most 4% split at any point.
- **Coverage is lost almost at once, and not because of the counterexample.** Some sheep is out of
  range within three steps in 94–100% of runs. With natural headings it is 92–100%, because these
  dilated states sit at the edge of the coverage region. With natural headings most of those
  sheep are ahead (68–98%), so they freeze ahead, which restores the shape. With adversarial
  headings more are to the side (only 36–52% ahead in three states), and the flock still recovers.

### 12.5 A multi-step drift condition [S]

Monte Carlo under the baseline dog, 128 replicates per state. The table counts states with
E[M₂(t+m)] − M₂(t) > 0; in brackets, those more than 2 SE above 0.

| States | m = 1 | m = 2 | m = 3 | m = 4…10 | Largest value at m = 3 |
|---|---|---|---|---|---|
| The 6 counterexamples | 6 (6) | 2 (1) | 0 | 0 | −0.15 m² |
| All 30 adversarial fields | 26 (17) | 5 (2) | 0 | 0 | −0.13 m² |
| Natural headings, all 255 | 2 (2) | 0 | 0 | 0 | −1.35 m² |

**D₃: E[M₂(t+3) | x_t] − M₂(t) ≤ −0.12 m² on every tested state outside the ball, adversarial
headings included.**
- No replicate collects in those three steps.
- A sheep is out of range at steps 1–3 in 19–85% of replicates. So D₃ is a property of the
  driving controller with the published stop rule (frozen sheep included), not of strict coverage.
- If D₃ held on every state that the 3-step skeleton visits outside the ball, the stopped
  supermartingale argument of §4.2 would apply to the skeleton. The expected number of 3-step
  blocks before entering the ball would then be at most M₂(0)/0.12. §12.6 certifies D₃ on these
  states and finds that it does *not* hold on every visited state.

### 12.6 A high-confidence certificate for D₃ [S, with stated confidence; E for D₁]

`experiments/certify_d3.py` (results/certify_d3.json, results/figures/certify_d3.png). Write
D_m(x) = E[M₂(t+m) | x_t = x] − M₂(t) under the baseline closed loop: the published sheep rule with
its stop rule, and the published dog with collecting and heading noise.

**The exact first step** [E]. Let d_i = v h_i′ for a sheep that moves and d_i = 0 for one that
stands still. Then M₂′ − M₂ = (2/N) Σ ξ_i·d_i + (1/N) Σ |d_i|² − |d̄|². Given the state the d_i
are independent, with E d_i = v μ_i and E|d_i|² = v² [moves_i]. So

    D₁ = 2vV + v² (n_moving/N − |μ̄|² − (1/N²) Σ ([moves_i] − |μ_i|²)).

This extends §10.5 to frozen and repel-only sheep (`expected_spread_change` in
`shepherd/expected_heading.py`). It matches 200,000 one-step draws at a state with both kinds.

**Estimator.**
- For m = 2, 3, Y_m = D₁(x) + M₂(x_m) − M₂(x₁) is unbiased for D_m. It replaces the realized
  first step by its exact mean. At two probe states this lowered the standard deviation from
  1.25–1.55 m² (raw M₂(x₃) − M₂(x₀)) to 0.93–1.19 m², at no cost. Exact means at the inner steps as
  well would halve it again, to 0.47–0.56 m², but cost about 60 times more per sample, so they
  were not used.
- **A range fixed before sampling.** No sheep moves more than v, so by the triangle inequality in
  L², |√M₂(x_{k+1}) − √M₂(x_k)| ≤ v surely. With s = √M₂(x):

      Y₃ − D₁ ∈ [−4vs, 4vs + 8v²],   Y₂ − D₁ ∈ [−2vs − v², 2vs + 3v²].

  Every sample was checked to lie in its range.

**Bound.** Empirical Bernstein (Maurer & Pontil 2009, Theorem 4). For n i.i.d. samples in an
interval of width R, with probability at least 1 − δ,

    E Y ≤ Ȳ + √(2 V ln(2/δ) / n) + 7 R ln(2/δ) / (3(n − 1)),

with V the unbiased sample variance; the lower bound is symmetric.
- **Stages.** Replicates come in cumulative stages of 2¹³, 2¹⁵, 2¹⁷ and 2¹⁹. At every stage each
  state gets four one-sided bounds (upper and lower, for D₂ and D₃), each with its own share of δ.
  By the union bound they all hold together, so stopping early is legitimate.
- **Stopping.** A state stops once D₃ is decided against both 0 and −0.1, or at 2¹⁹ replicates.
  The 30 adversarial fields run every stage, to decide the sign of D₂.
- **Floating point.** A margin of 10⁻⁸ is added for the exact D₁.

**Tested states: certified.** δ = 10⁻³ covers all 285 states together (2.2·10⁻⁷ per bound).

| States | D₁ > 0 (exact) | D₂ > 0 / < 0 (certified) | Largest certified bound on D₃ | Median | Replicates |
|---|---|---|---|---|---|
| 255, natural headings | 2 | – | −1.14 m² | −3.40 m² | 2¹³ each |
| 30 adversarial fields | 27 | 5 / 25 | −0.171 m² | −1.79 m² | 2¹⁹ each |

- **With probability at least 0.999, D₃ ≤ −0.17 m² at all 285 states at once,** adversarial
  headings included. No replicate collects within the three steps.
- **Three steps is the shortest horizon that works at the adversarial fields.** One step fails at
  27 of them (exactly) and two steps at 5 (certified). Every one of the 30 is decided at m = 2,
  with intervals of ±0.010 m².
- **One state sets the constant.** At counterexample 161, D₃ = −0.184 m²; the next tightest
  state's certified bound is −0.53 m². The §12.5 values from 128 replicates were within their
  noise.

| Counterexample | D₁ (exact) | D₂ | D₃ |
|---|---|---|---|
| 161 | +0.789 | +0.661 | −0.184 |
| 45 | +0.819 | −0.395 | −1.258 |
| 16 | +0.444 | −0.798 | −1.825 |
| 35 | +0.491 | +0.120 | −0.790 |
| 136 | +0.440 | −0.687 | −1.789 |
| 81 | +0.401 | −0.387 | −1.520 |

The certified intervals are ±0.010 m² for D₂ and ±0.016 m² for D₃.

**States the closed loop visits: D₃ is not uniform** [S, PAC]. The tested states are dilated
snapshots, so they say nothing about the states that recovery actually passes through. A sample
of those:
- **Paths.**
  1. Take an independent baseline run and a snapshot at a uniformly random step 100–299.
  2. Deform it by one of 11 deformations, chosen uniformly: a dilation by 1.25–2, or one of the ten
     of `experiments/shape_stress.py` (stretches along and across by 2–3, shear, splits along and
     across by 4–12 m, an outlier 6–14 m to the side, ahead or behind, heading noise, half the
     headings reversed).
  3. Run the closed loop 150 steps.
- **States.** Per path, draw one state uniformly from those before the flock first enters the
  ball at which the dog's next decision is to drive (or to collect, for the collecting family).
  The states of a family are i.i.d.
- **Bounds.** Let B = {D₃ > −c} and C = {certified at −c}. Then P(B) ≤ P(not C) + P(B and C),
  where P(B and C) is at most the state's share of δ (below 10⁻⁶), and Clopper–Pearson bounds
  P(not C). The states certified positive give a lower bound on P(D₃ > 0) the same way. Each
  family has δ = 10⁻³, half for its per-state bounds and half for three Clopper–Pearson statements.
- **Confirmation.** Families 3 and 4 use fresh paths. They test, on new data, what family 1
  suggested: that every failure follows a lateral deformation (split across, stretch across,
  outlier to the side).

| Family | States | D₃ ≤ −0.1 certified | D₃ > 0 certified | P(D₃ > −0.1) ≤ | P(D₃ > 0) ≥ |
|---|---|---|---|---|---|
| 1. Driving, all deformations | 800 | 759 | 35 | 0.085 | 0.022 |
| 2. Collecting, all deformations | 300 | 222 | 74 | 0.359 | 0.164 |
| 3. Driving, after a non-lateral deformation (fresh paths) | 600 | 597 | 3 | 0.025 | 0.0002 |
| 4. Driving, after a lateral deformation (fresh paths) | 600 | 553 | 45 | 0.125 | 0.042 |

The bounds hold with probability 0.999 per family, and with 0.995 for all five claims of this
section together.

**Where D₃ fails** (descriptive). There are two mechanisms, and both are already limitations of
§10.6 and §7.
- **Lateral splits that stay split.** In family 1, every one of the 41 uncertified states follows
  a lateral deformation: 33 lateral splits, 5 lateral stretches and 3 side outliers at 14 m. None of
  the 331 states that follow other deformations fails. In family 4, 41 of the 47 failures follow
  a lateral split. About half the failures in families 1 and 4 (23 and 20) involve neither
  imminent collecting nor a farthest sheep frozen beyond 9.4 m, and some have full coverage. The
  drive pushes the two halves apart: their dog-repulsion vectors diverge (the dog-divergence term
  of §10.3), and a sheep's 10 nearest neighbours are its 6 half-mates and only 4 of the other half.
- **The handover to collecting.** All three failures in family 3 have max|ξ| = 11.5–11.6 m, just
  under the collecting threshold l_sep = 11.62 m. In each, the dog collects within the three steps
  in 86–100% of replicates. Across families 1 and 4, where the dog collects within three steps in
  at least half the replicates, 58–65% of states fail, against 4–6% elsewhere. So D₃ measures M₂
  across the switch into collecting, and that switch can raise M₂.
- **Failures sit on paths that go on to collect.** In families 1, 3 and 4 the dog later collects
  on the path of 25 of 41, 3 of 3 and 36 of 47 uncertified states, against 7%, 2% and 13% of the
  certified ones.
- **Collecting states do not satisfy D₃** (family 2: positive at a quarter). That is expected:
  collecting needs its own drift (§10.6, item 1).

**What this gives, and what it does not.**
- A 3-step skeleton argument holds on the tested states, but those are not invariant. Over the
  driving region outside the ball, D₃ is false with probability at least 0.022 (at least 0.042
  after lateral deformations). So the stopped supermartingale argument of §4.2 does not apply to
  that whole region, at 3-step resolution, with M₂ alone.
- After non-lateral deformations, D₃ ≤ −0.1 m² except on a set of probability at most 0.025. Every
  exception found sits at the threshold for collecting.
- The certificate is statistical and holds pointwise or in distribution. D₃ jumps where neighbour
  sets change and where the dog switches modes, so a certificate at a state does not extend to a
  neighbourhood of it. The PAC statements are relative to the stated mix of deformations.

### 12.7 Status

- **Proved.**
  - The certificate method, and MI⁺ for every heading field at 178 of the 255 states (10th
    percentile of the certified k: 0.0049).
  - The Jensen obstruction: no drift function of the form M₂ + f(Q) exists at the counterexamples.
  - The exact one-step drift D₁, including frozen and repel-only sheep (§12.6).
- **Certified, at probability 0.999 per claim.**
  - D₃ ≤ −0.17 m² at all 285 tested states at once, adversarial headings included.
  - At the adversarial fields, three steps is the shortest uniform horizon: D₁ > 0 at 27 (exact)
    and D₂ > 0 at 5.
  - On states visited during recovery, D₃ ≤ −0.1 m² except with probability at most 0.025 after
    non-lateral deformations.
- **Refuted.**
  - MI⁺ at 6 states with polarization ≥ 0.5. At 15 states the certificate route fails at its best
    α, which does not by itself refute MI⁺.
  - D₃ as a uniform condition on the driving region outside the ball. It fails with probability at
    least 0.022 on visited driving states.
- **Observed.**
  - The counterexamples recover in 3–5 steps, with one step of expansion and no collecting.
  - D₃ fails at lateral splits that stay split, and where the 3-step window crosses into
    collecting.
- **Next.** The two failure mechanisms are §10.6's open items, not new obstacles:
  - the handover (item 3);
  - the lateral gap and the drive's lateral push (item 2, §7).

  A drift statement for the whole recovery needs a function common to driving and collecting
  (item 4). It would need a lateral term, because M₂ alone cannot tell a split flock from a
  dilated one. [collecting_chain.md](collecting_chain.md) takes up items 1–4. The gap turns out to
  be short-lived, and the handover is controlled by a cycle argument with an analysis band; the
  lateral push remains.

## 13. A Di Lorenzo-inspired density controller for one dog

Controller: `shepherd/density_control.py` (tests in `tests/test_density_control.py`). Script:
`experiments/density_controller.py`; numbers in `results/density_controller.json`; figure
`results/figures/density_controller.png`. The sheep are the published model throughout (the
paper's always-moving variant where labelled); the dog is the controller's.

The combination tested here is the one the rest of this document points to: Jadhav's model as
the sheep simulator, and for the dog a controller built on Di Lorenzo's structure rather than
on Strömbom's heuristic. His control has two layers, a herder density computed from the
target density (the deconvolution of the required drift) and herder dynamics that converge
onto it. The Jadhav sheep's response to a dog is not additive (§4.3), so there is no drift to
deconvolve. The controller keeps his two layers and replaces the deconvolution by an exact
computation on the sheep's own rule.

### 13.1 The controller [E for the placement scores]

- **Placement.** At each step, over candidate positions h on rings around the centroid (radii
  4, 6, 8, 10, 15, 18 m, 12 bearings, plus the dog's current position),

      h*(x) = argmin_h E[ΔV | x, h],

  the expected one-step change of a density-error functional V with the dog at h. The
  expectation over each sheep's neighbour choices and noise is the exact one of §10.1
  (`shepherd/expected_heading.py`), batched so that the dog-free part of every realization is
  computed once per step and each candidate only adds w_dog × the unit vector away from it
  (`ExpectedResponse`). Candidates within 2% of the best tie; the nearest to the dog wins.
- **Motion.** The dog moves toward h* by at most v_dog per step (1.5 m, baseline).
- **Functionals**, for the Gaussian target n̄ of continuification_driving §10 (95% within
  5 m, s = 2.04 m; the baseline flock's own √M₂ of 3.0 m is close to its √2·s = 2.89 m):
  - *second moment* V = (1/N) Σ |x_i − g|², the potential-energy part of the relative entropy
    ∫ ρ ln(ρ/n̄), the Fokker–Planck Lyapunov functional whose basin is global. For a mover,
    |x_i + v h_i′ − g|² − |x_i − g|² = 2v (x_i − g)·h_i′ + v², so
    E[ΔV | x, h] = (1/N) Σ m_i [2v (x_i − g)·μ_i + v²] exactly, with μ_i the expected heading
    and m_i whether the sheep moves. The minimizer does not depend on s. The v² term is the
    price of a step: a sheep is worth moving only where its expected displacement toward the
    goal exceeds half a step.
  - *L² error* V = ‖ρ̂_b − n̄_b‖², the functional of his Theorem 2, with ρ̂_b the Gaussian
    kernel density estimate (b = 1 m, as in continuification_driving §10) and n̄_b the target
    smoothed by the same kernel. It is a closed-form sum of Gaussians of pair and goal
    distances; E[ΔV] is taken to first order in the step plus an isotropic second-order term.
  - *lookahead* V_H = (1/N) Σ |x_i − g + H v h_i|², the second moment of the positions
    extrapolated H steps along the current headings (H = 8), which charges the flock's
    momentum, the one thing his overdamped targets lack. E[ΔV_H] is still exact, since a
    mover's new heading is a unit vector and a sheep that stands still keeps its heading.
  - *hybrid*: the second moment while the centroid is more than 5 m from the goal, the L²
    error inside (drive, then regulate).
- **Exactness.** The second-moment and lookahead scores match Monte Carlo on driven states
  (−339.1 against −338.9 ± 0.3; −1137 against −1140 ± 1.3 pooled over 4 × 40,000 draws); the
  batched response equals `expected_headings` to 10⁻¹⁵; the L² gradient matches finite
  differences. In the closed loop, the mean expected and realized ΔV agree to three figures
  in every variant: driving, −21.9 against −21.9 per step overall and −55.0 against −55.0 on
  the steps where the dog stands at h*. The placement layer is exact; what it cannot see is
  the next step.

### 13.2 What the placement recovers [S]

- **The drive point.** On a flock 33 m from the goal (figure, panel C), h* is 8.4 m behind
  the centroid on the goal line (cosine 0.97 from straight behind). By radius, the best
  score is −55 to −56 m²/step at 4–8 m, −41 at 10 m, −39 at 12 m, −14 at 15 m. The published
  dog's own position, l_drive = 2√N = 7.5 m behind, scores within 2% of the best, so the
  controller leaves it there: the Strömbom drive point is the one-step optimum of the second
  moment. In the dog-control comparison (continuification_driving §12) the controller reaches
  the goal at the published dog's time (median 131.5 steps against 135).
- **Nothing reverses a flock in one step.** Every candidate scores negative in that state,
  even 4 m straight ahead (−24). The heading rule's memory and alignment (0.5 + 1.3) outweigh
  the dog (1.0), so δ(180°) = 0 (§4.1, C2): a dog in front slows the flock's polarization but
  does not turn it back. The flock's momentum lives in its headings, and the dog's one-step
  gain on them is at most the 0.37 of C2.
- **At the goal, a packed flock keeps the dog in sight.** Sheep sampled from n̄ sit within
  each other's repulsion radius. With no dog in range the published sheep graze apart,
  repulsion only, which raises the second moment by 2–3 m² per step; with the dog in sight
  attraction is on, and the placement finds in-range positions that cost less (test
  `test_hold_at_the_goal_keeps_a_packed_flock_in_sight`). Release is free exactly when no pair
  is within r_rep: the out-of-range change is then zero (`test_release_is_free_for_a_spread_flock`).
  So the second-moment controller never releases a flock it has just driven, and the sheep
  keep walking.

### 13.3 Reaching and holding (the task of continuification_driving §11) [S]

Start 100 m from the goal, 450 steps, 48 runs (the baselines of §11 used 200). Post-reach:
from 20 steps after the centroid first comes within 5 m.

| Sheep | Dog | reached, median time | post-reach time within 5 m | ρ mean / 90th pct (m) | sheep moving | dog out of range |
|---|---|---|---|---|---|---|
| published | published dog | 1.00, 135 | 0.11 | 21.1 / 37.9 | 0.99 | 0 |
| published | reach-and-release rule | 1.00, 135 | **1.00** | **2.2 / 2.8** | 0.00 | 1.00 |
| published | second moment | 1.00, 131.5 | 0.36 | 6.9 / 12.3 | 0.99 | 0.00 |
| published | lookahead H = 8 | 1.00, 131.5 | 0.34 | 7.3 / 12.9 | 0.97 | 0.00 |
| published | second moment, v_dog = 3 | 1.00, 124 | **0.99** | **2.1 / 3.7** | 0.97 | 0.00 |
| published | hybrid (L² inside 5 m) | 1.00, 131.5 | **0.99** | **3.9 / 4.8** | 0.01 | 0.99 |
| always moving | published dog | 1.00, 135 | 0.11 | 20.5 / 36.8 | 1.00 | 0 |
| always moving | reach-and-release rule | 1.00, 135 | 0.34 | 11.1 / 27.5 | 1.00 | 0.65 |
| always moving | second moment | 1.00, 133 | 0.35 | 7.2 / 12.6 | 1.00 | 0.00 |
| always moving | lookahead H = 8 | 1.00, 132.5 | 0.35 | 7.0 / 12.3 | 1.00 | 0.00 |
| always moving | second moment, v_dog = 3 | 1.00, 123 | **1.00** | **1.9 / 3.4** | 1.00 | 0.00 |
| always moving | hybrid | 1.00, 133 | **0.46** | **5.6 / 9.9** | 1.00 | 0.00 |

The L² functional alone cannot drive: its goal term at 100 m is a Gaussian of e^(−800), a
descent direction exists at 9% of steps, and in 100 steps the flock does not move (16 runs).
Its basin is the target's own width; so is Theorem 2's.

**Layer diagnostics, second moment, published sheep.** A descent direction exists at 77% of
steps (69% after reaching). The dog stands at h* on 23% of steps (15% after reaching); its gap
to h* averages 6.5 m while driving and 7.9 m while holding, 90th percentile 15 m. The expected
descent is −55 m²/step at h* and −22 on average while driving, which still reaches; after
reaching it is −9.9 at h* and −0.15 on average. The motion layer gives back nearly all of
the placement layer's descent, and the flock oscillates about the goal with a median
excursion of 7 m (panel A). The always-moving flock does the same at 7.2 m. Under the
hybrid the published flock is released at 4–5 m, spreads by grazing and freezes: 99% of
post-reach time within 5 m, with no re-engagement (descent available at 2% of steps). It is
not as close as the rule's 2.2 m, because the L² regulator releases where the flock's width
is right, not where its centre is.

**At v_dog = 3 the oscillation becomes a hold.** The dog is at h* on 34% of steps while
driving (gap 4.4 m against 6.5 m), reaches 11 steps sooner, and after reaching keeps the
centroid within 5 m on 99.3% of steps (published sheep) and 99.8% (always moving), at 2.1 and
1.9 m on average and 3.7 and 3.4 m at the 90th percentile, with the sheep walking (97% and
100% of sheep-steps) and the dog never out of range. The post-reach descent averages zero,
−2.8 to −3.1 at h*: a stationary feedback hold, not a freeze. Over the last 20 s, 76–77% of
the sheep are within 5 m of the goal, RMS 4.1 m, L² error 0.68–0.69 (the static ring: 99%,
2.3 m, 0.61). This is the first hold of a walking flock in this project; the rule of §11
holds the always-moving flock at 11 m.

### 13.4 At the goal, against the static ring (the task of continuification_driving §10) [S]

Scattered start inside the 10.9 m ring and a tight cluster 7 m off the goal, 200 steps, 50
runs, one dog starting on the ring. Final values over the last 20 s.

| Start | Dog | inside 5 m | RMS to goal (m) | L² error | centroid off goal (m) | out of range |
|---|---|---|---|---|---|---|
| scattered | static ring of 10 dogs (L5) | **0.99** | **2.3** | **0.61** | 1 | – |
| scattered | published dog, started at the goal (L6) | 0.15 | 21.8 | 1.13 | | 0 |
| scattered | second moment | 0.34 | 8.0 | 1.10 | 6.2 | 0.00 |
| scattered | lookahead H = 8 | 0.26 | 8.8 | 1.16 | 7.1 | 0.00 |
| scattered | second moment, v_dog = 3 | **0.75** | **4.2** | **0.71** | 2.2 | 0.00 |
| scattered | L² error | 0.28 | 6.8 | 1.03 | 4.7 | 1.00 |
| cluster | static ring (L5) | 0.99 | 2.3 | 0.61 | | – |
| cluster | second moment | 0.30 | 8.3 | 1.13 | 6.6 | 0.00 |
| cluster | lookahead H = 8 | 0.32 | 8.9 | 1.13 | 6.7 | 0.00 |
| cluster | second moment, v_dog = 3 | **0.76** | **4.2** | **0.71** | 2.2 | 0.00 |
| cluster | L² error | 0.21 | 8.2 | 1.11 | 7.0 | 1.00 |

- Under the second moment at v_dog = 1.5 the dog is at h* on 14–16% of steps with a mean gap
  of 8.3 m; the expected descent is −9.6 to −12.5 m²/step at h* and +0.1 on average (realized
  +0.1): no net descent, the oscillation again, from either start.
- At v_dog = 3 the gap falls to 5.8 m, the average descent becomes negative (−0.1 to −0.2),
  and the cluster converges to its floor at 0.09/s (the RMS excess reaches 10% of its initial
  value at 23 s). The floor is an oscillation with the centroid 2.2 m off and RMS 4.2 m
  against the ring's 2.3 m.
- The lookahead changes nothing, for the reason of §13.2: the momentum it charges cannot be
  removed by any one-step position.
- The L² controller withdraws at once (out of range on every step) and the flock freezes
  where it stands: descent is available at 4–8% of steps, because sheep 6–9 m from the goal
  are beyond the reach of the target's Gaussian. From the cluster 7 m off the goal it never
  moves.

### 13.5 Reading

1. **What carries over.** His two-layer structure survives, with the deconvolution replaced by
   an exact expected descent; the placement layer is exact, and in the closed loop its
   predicted descent is realized to three figures whenever the dog is where it was placed.
   The functional that drives is the energy part of the relative entropy, not his L² error:
   the L² basin is the target's width, and the second moment's is global. This is the
   moment-level Lyapunov function of §5 made into a controller.
2. **What the second moment recovers without being told.** The Strömbom drive point (the
   published dog stands within 2% of the optimum), the published dog's reaching time, and the
   stop-rule equilibrium (release is free exactly when no pair is within r_rep).
3. **The binding condition is his Theorem 1's.** The herders must converge onto their
   placement faster than the targets move. One dog at 1.5 m/step against a flock at 0.8 is at
   its placement 15–23% of the time, and the flock's momentum (δ(180°) = 0) means that the
   placement cannot be anticipated by a one-step score; the lookahead shows that. The result is
   a bounded oscillation, 7 m median for both sheep rules, against the published dog's 20 m
   and the rule's 11 m on the always-moving flock. Doubling the dog's speed turns the
   oscillation into a hold at 2 m for both sheep rules, with the flock walking and the dog in
   range, and takes a displaced cluster at the goal to the same floor at 0.09/s. This is
   condition (iv) of continuification_driving's surviving statement, measured on the dog's
   placement rather than on the drive point, and it is sharper than stated there: at the
   flock's own speed the dog reaches; at twice it, it also holds.
4. **At baseline dog speed, the published sheep's hold is the freeze, and the L² regulator
   decides when.** The second-moment controller never pays the one-time cost of release and
   keeps the flock walking; the hybrid releases where the flock's width matches the target and
   holds 99% of the time within 5 m. The rule of §11 (release at 5 m) holds closer, at 2.2 m,
   because it releases on position. The faster dog makes the question moot: it holds at 2 m
   without releasing.
5. **The density level stays where §4.3 left it.** One closed-loop dog converges neither a
   scattered flock nor a displaced cluster to the goal density: the second moment oscillates,
   the L² error has no basin there, and the faster dog, which holds the position, still leaves
   a quarter of the sheep outside 5 m, with an RMS of 4.1 m against the ring's 2.3 m. The
   conclusion of continuification_driving §10 stands, now against a controller designed for
   the density rather than against the Strömbom heuristic: in Jadhav's world the density is
   cohesion's and the position is the dog's. What changes is the position's hold: a dog fast
   enough to stay at its placement holds a walking flock, which a single Strömbom dog never did.

**Limitations, recorded.** The placement is one-step and over a ring of 73 candidates (2–3 m
of arc at the drive standoff); the L² score is first order in the step; the dog has no slow
mode and turns instantly; the candidate rings are centred on the centroid, so a split flock is
served by neither half; 48 runs against the baselines' 200.

### 13.6 The hold radius: the tightest steady turn the dog can keep up with [D from C2–C3; S]

Script: `experiments/hold_condition.py`; numbers in `results/hold_condition.json`; figure
`results/figures/hold_condition.png`.

**Mechanism [S].** Under the second-moment controller a flock at the goal does not stop: it is
turned, and it walks a circle about the goal. In a 300-step hold at v_dog = 1.5 the published
flock turns at a median 8.0°/step with the dog at a 12° bearing and 7.2 m standoff, consecutive
turns have the same sign on 61% of steps, and it completes 1.6 laps in the last 240 steps; at
v_dog = 3 it turns at 16°/step with the dog at 25° and 6.7 m, same sign on 78% of steps, 2.2
laps. The dog stays behind the flock (|φ| ≤ 60°) on 80–98% of steps at every speed.

**Derivation [D].** A steady turn with the dog at standoff d and bearing φ turns the flock at
ω = δ(φ) and moves it at V = v P_eq(φ) (C2, C3), so the flock circles at R = V/ω, and the dog,
co-rotating at the same standoff, must move at

    s(φ, d) = √(V² + 2 V d ω sin φ + d² ω²)

(steer_and_shape §8; the kinematics is exact, δ and P_eq are the measured law with its stated
error). Over the bearings where the measured law is monotone, |φ| ≤ 60°, the tightest circle a
dog of speed v_dog can hold is

    R_min(v_dog, d) = min { V(φ)/δ(φ) : s(φ, d) ≤ v_dog }.

A flock that circles the goal has ρ ≈ its turning radius, so the prediction is
ρ ≈ V/|ω| ≈ R_min(v_dog, d) at the standoff the controller keeps, and not below it. From the
steering-gain table (baseline weights; standoffs 5, 7.5 and 10 m, interpolated in d), at
d = 7.5 m: R_min = 6.0, 4.0, 3.0, 2.4 and 1.7 m at v_dog = 1.5, 2, 2.5, 3 and 4, reached at
bearings 20°, 30°, 39°, 47° and 60°. A hold within the 5 m goal needs R_min ≤ 5, that is
**v_dog ≥ 1.7 m/step** at d = 7.5 (1.35 at d = 5, 1.8 at d = 10).

**Test [S].** Holds of 300 steps from a driven flock translated to the goal, 24 runs, both
sheep rules, measured over steps 60–300.

| Sheep | v_dog | ρ median (10–90%) | within 5 m | V/\|ω\| median | turn (°/step) | standoff, bearing | R_min at that standoff |
|---|---|---|---|---|---|---|---|
| published | 1.5 | 6.4 (1.6–12.0) | 0.38 | 5.6 | 8.0 | 7.2 m, 12° | 5.8 |
| published | 2.0 | 3.7 (1.0–6.7) | 0.69 | 4.0 | 10.9 | 6.8 m, 17° | 3.6 |
| published | 2.5 | 2.7 (0.8–4.8) | 0.93 | 3.1 | 13.0 | 6.8 m, 21° | 2.7 |
| published | 3.0 | 2.1 (0.7–3.7) | 0.99 | 2.5 | 16.1 | 6.7 m, 25° | 2.2 |
| published | 4.0 | 1.5 (0.5–2.7) | 1.00 | 1.9 | 19.8 | 7.0 m, 28° | 1.7 |
| always moving | 1.5 | 7.0 (1.6–12.5) | 0.36 | 5.9 | 7.5 | 8.3 m, 13° | 6.2 |
| always moving | 2.0 | 3.7 (1.0–6.6) | 0.68 | 4.0 | 10.5 | 7.1 m, 18° | 3.8 |
| always moving | 2.5 | 2.4 (0.8–4.4) | 0.96 | 2.7 | 14.7 | 6.1 m, 26° | 2.4 |
| always moving | 3.0 | 1.8 (0.6–3.3) | 1.00 | 2.0 | 18.9 | 5.9 m, 30° | 2.1 |
| always moving | 4.0 | 1.2 (0.4–2.3) | 1.00 | 1.3 | 25.7 | 6.1 m, 36° | 1.8 |

**Reading.**

1. **The prediction holds at every speed.** Up to v_dog = 3 the median ρ is within 0.6 m of
   R_min for both rules and the turning radius within 0.4 m; the measured bearings sit near
   the predicted ones (12–30° against 20–47°, the controller choosing slightly smaller
   bearings at a slightly smaller standoff). At v_dog = 4 both measures fall 0.2–0.5 m below
   R_min, because the feasibility caps the bearing at 60°, the law's monotone range, while the
   controller uses bearings beyond it where the measured turn rate keeps rising (43°/step at
   90°); the always-moving flock turns at 26°/step there, above the capped 24°.
2. **The hold condition is a number.** The published dog's 1.5 m/step holds at 6–7 m, and a
   dog 13% faster (1.7) would hold within the 5 m goal: at 1.5 the centroid is inside 36–38%
   of the time, at 2.0 68–69%, at 2.5 93–96%. Together with the reach condition (iv) of
   continuification_driving, that the dog be about as fast as the flock, the single-dog
   closed loop now has two speed thresholds: about 1 m/step to reach, 1.7 to hold within 5 m.
3. **What is derived and what is observed.** The speed identity and the minimization are
   exact given the law; the law is measured (C2, C3). That the flock circles the goal rather
   than some other point, and that ρ tracks the turning radius, are observed: the second
   moment's placement keeps the dog on the far side of the goal, so the centre of the circle
   is the goal, but that is not proved. The 10–90% bands (1.6–12 m at v_dog 1.5) are the
   transients between circles, which the steady-state bound does not cover.
4. **Both sheep rules give the same hold**, within 0.6 m at every speed, because the hold never
   releases: the stop rule plays no part while the dog stays in range.
