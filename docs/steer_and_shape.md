# Steering a flock and holding its shape with one dog

**Question.** For sheep with alignment, cohesion and a finite notice radius, can a dog both
steer the flock's overall motion and keep a prescribed spatial distribution, and under what
conditions?

**The synthesis is in [jadhav_dilorenzo_bridge.md](jadhav_dilorenzo_bridge.md).** It states the
macroscopic dynamics, the minimal closures, and what does and does not carry over from Di Lorenzo
et al. at the baseline parameters. This file holds the detailed derivations and experiments. The
fission-threshold work (§11.4–§11.5) is paused, and is kept as a record of the limits of a
drive-only dog, not as a goal.

Companion to [continuification_driving.md](continuification_driving.md), which asked whether
Di Lorenzo et al.'s convergence theorem says anything about Jadhav's sheep (answer: only a
centroid statement survives). This document takes the constructive route. It builds an
effective model from Jadhav's rules, asks which targets a dog can realize in it, and tests
the answers on the Jadhav model. Scripts: `experiments/steer_and_shape.py`,
`experiments/steering_gain.py`, `experiments/path_tracking.py` and `experiments/shape_drift.py`;
numbers and figures under the same names in `results/`.

**Rule versions.** Jadhav's *published* rule makes a sheep with no dog within r_dog stand
still unless repelled. The paper's *modified* rule (its Supp. Sec. 3; `social_always=True`,
or the dog-free null model `dog_present=False`) keeps such sheep moving under their social
terms.

| Experiment | Rule |
|---|---|
| driven flocks of the grid (§4), steering gain (§5), real-dog path (§7), shape drift (§10) | published |
| dog-free reference flocks of the grid (§3.4, §4) and of the shape drift (§10) | modified (null model) |
| turning and stopping test (§4) | modified |

Under the published rule a released flock freezes after a median of 6 steps
(continuification_driving.md §9.2). The stopping numbers in this document therefore do not
apply to it. At r_dog = 20 every sheep of a driven flock sees the dog on every step, so the
two rules coincide there; that is where driven and dog-free flocks are compared.

## Corrected conclusions, briefly

1. **Shape: an exact identity and a conditional drift bound** (§10).
   - With ξ_i the offsets from the centroid and a_i the actual one-step displacements,
     ΔM₂ = (2/N)Σ ξ_i·a_i + (1/N)Σ|a_i − ā|² holds exactly.
   - Under the full Jadhav rule (normalization, random neighbours, dog, stop rule) its
     conditional mean splits exactly into a radial drift C and a heading dispersion
     G ≤ v²(f_m − f_m²P²), which is v²(1 − P²) when every sheep moves. When every sheep
     moves, C = −2vκγ√M₂, with κ the weighted inward heading of each sheep relative to the
     flock and γ ≈ 0.9 a shape factor.
   - *Proved, local:* on the set 𝒢 where every sheep moves, κ ≥ κ* and P ≥ P*,
     E[ΔM₂ | state] ≤ −a√M₂ + b.
   - *Proved, finite horizon:* until the flock re-enters the ball or leaves 𝒢, M₂ is a
     supermartingale, so one of the two happens within expected time M₂(0)/ε.
   - *Not proved, long term:* recurrence or bounded moments would need the drift on every
     reachable state, and §11 finds reachable states where it fails.
   - The restoring is constant-speed (κ saturates), so the form is −a√M₂, not −aM₂.
2. **Proposition 1 is narrower than previously stated** (§3). In a given state, the
   *additive* push's contribution to d/dt|x − y|² is non-negative for every pair that sees the
   same dogs. That says nothing by itself about the steady size of the nonlinear closed loop.
   Under the dog, both the dispersion G and the restoring κγ fall. Which one decides the
   stationary size cannot be read off their ratio, because κ depends on the state. A ratio
   formula used earlier predicted the published flock would shrink 1.45×; it grew 6%. The
   attribution is therefore only qualitative.
3. **Stress tests and trajectories** (§11). On ten deformations of observed 14-sheep states:
   - At full coverage every trajectory re-cohered within 150 steps (median 3–63 steps) and none
     split, although most left 𝒢 on the way.
   - With a short notice radius and a drive-only dog, lateral displacements froze apart in
     87–100% of runs. Strömbom collecting rescued all of them.
   - *Strongest failure:* larger flocks fission laterally into two halves under the drive,
     with either dog.
   - *The first boundary guess was wrong.* N ≤ 2K + 1 = 21 was predicted to mark the onset of
     fission; it starts at 17–18 sheep at the published weights (§11.4).
   - *A modelled threshold replaces it* (§11.5). Fission is predicted when the dog's lateral
     push on each half beats the attraction across the gap, with the cross-perception share
     max(0, K + 1 − N/2)/K. With one constant fitted to one setting, it predicts the 50% split
     size of eight others (K, d, w_dog, w_att, w_ali) within 1.4 sheep, but systematically
     low by about 0.6 sheep, for reasons not identified.
   - 2K + 2 is the first size at which two separated groups can lose sight of each other, not a
     proven ceiling. Being below N_c means lower risk, not guaranteed recovery.
   - *Candidate sufficient condition, a conjecture supported only on tested states:* coverage
     kept, N below N_c, and average attraction beating the dog's spreading at the edge.
4. **Steering** (§5–§9) is unchanged in substance. R_min is the steady-turn estimate at
   fixed standoff and fixed bearing (the one-step chord version is 0–4% smaller), and
   R_min − R is an empirical relation. The heading and centroid claim is restated as a
   mean-square bound with decaying transients.
5. **Parameters.** All results use the code's values: 4 attraction neighbours and noise
   e = 0.3. The paper's Methods text gives n_att = 5 and e = 0.5. A paper-text variant of the
   shape analysis gives the same picture (§10).

## 1. What each source provides

- **Jadhav et al. (2024)** give the micro mechanism. Each sheep's heading is the normalized sum
  of unit vectors: heading memory (0.5), repulsion (2), flight from the dog within r_dog (1),
  attraction to 4 random members of its 10 nearest (1.5), copying 1 of those (1.3), and noise
  (0.3). It moves at a fixed speed. These are the values in the code that produced the paper's
  figures, and they are used throughout. The paper's Methods text gives 5 attraction neighbours
  and noise 0.5 (REPORT.md, "Differences").
- **Di Lorenzo, Maffettone & di Bernardo (2024, arXiv:2411.04791)** give the macro control
  logic: prescribe the target density, solve for the velocity field the herders must
  produce, deconvolve it into a herder density, then check that density is non-negative.
- **Their follow-up (arXiv:2604.11353, April 2026)** adds follower–follower interactions. Its
  required leader-induced velocity (their Eq. 17, Eq. 60 in d dimensions) is
  v̄^FL = D∇ρ̄/ρ̄ − f^FF*ρ̄, the static case of the matching condition below. Feasibility
  becomes a window on the leader mass, [M̂1, M̂2], proved if-and-only-if in 1D and checked
  numerically in 2D. Stability is local, in L², under D(2 − ‖G1‖∞) > F. It does not treat
  moving targets, alignment, sensing ranges, heading or fixed-speed agents, or a single
  leader.

  One possible gap: as summarized to us, their comparison Lemma 1 (Eq. 40) contains a term
  γe^(−kt)√η. The equation η′ = −η + e^(−t)√η has the nonzero solution
  η = (e^(−t/2) − e^(−t))² from η(0) = 0. We checked this against the summarized equation,
  not against the paper's exact Eq. 40. If the summary is right, local stability also needs
  a small *leader* transient.

## 2. The effective model, and three repairs it needs

Proposed model, with target ρ*(x, t) = q*(x − c(t)), a fixed shape whose centre follows c(t):

    ∂_t ρ + ∇·[ρ (b_collective + u_dog)] = D_ind Δρ,
    u_dog* = ċ + D_ind ∇log ρ* − b_collective[ρ*]          (zero relative flux)

**R1. The response is additive only when noise dominates.** A sheep's heading is unit(S),
with S = Σ_k w_k e_k + ηζ. Its mean is m(|S|/η) Ŝ, which is linear (gain 1/2η) while
|S| ≪ η and becomes a pure direction once |S| ≳ 2η. Around a polarized flock moving along
ê, a perturbation δS changes the velocity by δu = v (I − êê) δS/|S| + O(δS²). That is
additive, but across the heading only. Along it the speed is fixed, and the one first-order
actuator is the freeze outside r_dog.

**R2. D_ind is not a constant, and the dog changes it.**
- **Where the noise comes from.** Each sheep's motion relative to the flock is diffusive at
  the flock's scale in the Jadhav model (Knudsen number ≤ 0.53; the real flock's is 0.2).
  The noise comes mostly from re-drawing neighbours every step.
- **How it is measured.** D⊥ = ⟨v⊥²⟩ τ_int, with τ_int the integrated autocorrelation time
  of the lateral relative velocity. τ_int stays at 0.50–0.67 steps in every condition.
- **What the dog does to it.** Changes in velocity variance are therefore changes in D. For
  the published flock, D⊥ is 0.246 m²/step dog-free and 0.103 under a dog at 7.5–10 m. With
  w_ali = 3 it is 0.182 → 0.060. This comparison is within one rule (r_dog 20). D_ind is
  thus a function of the polarization P the dog imposes.

**R3. The heading and the polarization are state variables.**
- **Heading.** It is steered by the dog (§5).
- **Polarization.** P relaxes to a quasi-static value P_eq(φ) set by the dog's bearing,
  covering 63% of a change in a median of 1–3 steps (at most 7). On the real-dog arcs of §7,
  the arc-averaged P matches P_eq(φ) within 0.01–0.05.
- **Persistence after release** (modified rule only). P halves in 3–5 steps without
  alignment, 5–9 at w_ali = 1.3, and 19–43 at w_ali = 3.

The effective state is therefore (c, θ, P, q), with P closed by P ≈ P_eq(φ) while the
bearing changes slowly against those few steps.

## 3. What a dog's push can and cannot do to the shape

### 3.1 A provable statement for the additive push

**Proposition 1 (additive unit push).** Let dogs at positions h_1(t), …, h_M(t), arbitrary
and possibly chosen by feedback from the sheep's state, push each sheep x within r of them
with u(x) = g Σ_k e(x − h_k) 1{|x − h_k| ≤ r}, where e(z) = z/|z| and g > 0. Let two sheep
x, y see the same set S of dogs. Then the dogs' contribution to d/dt |x − y|² is ≥ 0.

*Proof.* Restricted to S, u = g∇F with F(z) = Σ_{k∈S} |z − h_k|. F is convex, and the
gradient of a convex function is monotone: (x − y)·(∇F(x) − ∇F(y)) ≥ 0. Hence
d/dt |x − y|² = 2(x − y)·(u(x) − u(y)) ≥ 0 from the dog term. This holds at every instant,
whatever the dogs are doing. ∎

**Corollaries.**
- Every compactness measure built from pairwise distances (the centroid second moment
  M2 = (1/2N²) Σ_ij |x_i − x_j|², the diameter, the mean nearest-neighbour distance) gets a
  non-negative contribution from the push over pairs that see the same dogs.
- With independent isotropic noise of per-axis diffusivity D for each sheep, E|x − y|² grows
  by an extra 8D per unit time (4D per sheep in two dimensions, two independent sheep).
- Only pairs *on opposite sides of a notice circle* can be drawn together by the dogs. With a
  finite r, F becomes Σ min(|z − h_k|, r), which is not convex.

The earlier divergence argument (∇·u ≥ 0 off the notice circles) is the local, weaker form
of this: it constrains area change, not distances.

What Proposition 1 does *not* say: it concerns the push's contribution in a fixed state. In
the closed loop the dog also changes the other terms: the polarization, hence the dispersion,
and the effective cohesive response (§10). So it gives no lower bound, by itself, on the steady
size the nonlinear system settles to.

### 3.2 Scope

Proposition 1 needs the push to *add* to a velocity. The full Jadhav rule normalizes the
heading sum and carries a heading memory, so the dog's effect on a pair depends on everything
else in S. It also re-draws neighbours at random and freezes sheep out of range. For it,
"the direct push never shortens a same-visibility pair" is a **conjecture**. The evidence
below (§3.4: shape against standoff at nearly fixed polarization and noise) is consistent with
it but does not prove it. Moving, asymmetric or feedback dogs are *not* excluded for the full
rule.

### 3.3 Three kinds of tightening

| Task | Achievable? | Proof | Evidence |
|---|---|---|---|
| Recover a straggler | yes, through the notice edge (published rule: the flock out of range stands while the straggler is pushed) or through cohesion | impossible by push alone if straggler and flock see the same dogs (Prop. 1, additive) | Strömbom collect episodes shrink the spread by 0.09 m/step (continuification_driving.md §9.2) |
| Let a loose flock re-cohere | yes, by cohesion. The dog's role is to keep the sheep active (published rule) without spreading them | none needed for "possible"; the dog cannot *help* by push (Prop. 1) | driven flocks settle to a stable shape at every usable policy (§4) |
| Compress an equilibrated flock beyond its own state, and hold it | the push itself contributes no compression in the additive model (Prop. 1). In the Jadhav model the driven flock *is* tighter with strong cohesion and alignment, through the net of lower dispersion and a weaker restoring response | Prop. 1 covers the push term only; no statement about the closed-loop steady size | §3.4 and §10: M2 3.80 → 2.10 at w_att 3, w_ali 3 (dispersion down 4.4×, restoring κγ down 2.9×); no reduction at published weights. The notice edge shortens the flock along the drive but widened it more across in the one case tested (M2 2.12 → 2.69) |

### 3.4 Direct and total effects, separately

*Direct*: vary the standoff d at nearly fixed polarization and noise (r_dog 20, d ≥ 5 m: flock
speed within ±0.03, D⊥ within ±10%). *Total*: driven against dog-free, same rule (r_dog 20).
Shape: sd along / across the drive, and M2 = sd∥² + sd⊥² (m²).

| Flock (η 0.3) | dog-free: sd∥ / sd⊥, M2, D⊥ | dog at 10 m | 7.5 m | 5 m | 3 m |
|---|---|---|---|---|---|
| w_att 1.5, w_ali 1.3 (published) | 2.01 / 2.00, 8.01, 0.246 | 1.54 / 2.37, 7.99, 0.103 | 1.53 / 2.58, 9.01 | 1.66 / 3.00, 11.8 | 2.05 / 5.09, 30.1 (20% split) |
| w_att 3, w_ali 3 | 1.38 / 1.37, 3.80, 0.197 | 1.16 / 0.88, 2.11, 0.064 | 1.14 / 0.89, 2.10 | 1.08 / 0.97, 2.12 | 1.09 / 1.16, 2.54 |
| w_att 3, w_ali 0 | 0.72 / 0.71, 1.02, 0.263 | 0.94 / 0.85, 1.61, 0.209 | 0.95 / 0.88, 1.67 | 0.93 / 0.95, 1.77 | 0.93 / 1.13, 2.14 |
| w_att 1.5, w_ali 3 | 2.18 / 2.27, 9.90, 0.182 | 1.56 / 2.38, 8.08, 0.060 | 1.63 / 2.56, 9.20 | 1.58 / 3.22, 12.9 | 1.69 / 3.84, 17.6 |

- **Direct effect.** A nearer dog widens the flock across the drive and raises M2, in every
  row. Along the drive, the driven flock is shorter than the dog-free one wherever there is
  alignment.
- **Total effect.** The dog lowers D⊥ by a factor of 1.3 (no alignment) to 3.1 (w_ali = 3).
  M2 falls below the dog-free value only with strong cohesion and alignment at every
  standoff, or (1.5, 3) at the largest standoff. At the published weights the best the dog
  does is break even (7.99 vs 8.01 at 10 m).
- **Why it is not simply "lower diffusion".** §10 decomposes the change exactly. The dog
  lowers the heading dispersion and also weakens the effective restoring response. Which one
  decides the stationary size cannot be read off the ratio of the two, because κ depends on
  the state (§10, item 6).

## 4. What one dog reaches in the Jadhav model (grid)

Fourteen sheep are driven toward a target 2 km away by the published dog, under the
published rule. Collecting is off, so standoff (3, 5, 7.5, 10 m) and sideways swing (0, 3 m
sd) are the only knobs. The grid covers η, w_att, w_ali and r_dog, with 10 runs per setting,
over steps 150–300. A policy is *usable* when at most 20% of runs split and the flock moves
at more than 0.5 v.

| Flock (η 0.3) | no dog (modified rule): sd along / across | r_dog 12: usable policies, sd along / across, speed | r_dog 20 |
|---|---|---|---|
| w_att 0.75, w_ali 1.3 | 2.90 / 2.90 m | none | 1 of 8: 2.74 / 3.12 m, 0.91 |
| w_att 1.5, w_ali 1.3 (published) | 2.01 / 2.00 | 5 of 8: 1.34–1.79 / 2.71–3.19, 0.74–0.84 | 8 of 8: 1.52–2.05 / 2.32–5.09, 0.71–0.85 |
| w_att 3, w_ali 0 | 0.72 / 0.71 | 8 of 8: 0.91–1.16 / 0.87–1.70, 0.52–0.62 | 8 of 8: 0.92–0.95 / 0.85–1.13, 0.52–0.61 |
| w_att 3, w_ali 3 | 1.38 / 1.37 | 8 of 8: 0.80–1.48 / 0.91–3.34, 0.61–0.91 | 8 of 8: 1.06–1.16 / 0.82–1.16, 0.83–0.91 |

**Reading.**

- **Usable flocks** need w_att ≥ 1.5 at the published noise. At η = 2 they need alignment or
  cohesion of about 3.
- **Notice edge.** At r_dog 12 and standoff 7.5–10 m, the edge shortens the flock along the
  drive by 12–30% but widens it across. It splits once coverage falls to about 0.6.
- **Alignment** raises the driven speed (0.52–0.62 → 0.83–0.91 at w_att 3) and lowers D_ind.
- **Turning test** (modified rule). After a straight drive the target moved 90° to the side.
  Every setting turned within a median of 7–43 steps, flocks without alignment slowest. This
  test does *not* show that alignment is free in turning. The turn time there is set by the
  dog's repositioning (§5, §6). With a real actuator, alignment does raise the smallest
  trackable radius (§6, §7).
- **Stopping test** (modified rule only). After the dog leaves, the flock runs on 1.4–3.6 m
  in 100 steps without alignment, 7.6–10.8 m at w_ali = 1.3, and 32–51 m at w_ali = 3. Under
  the published rule it freezes.

## 5. The heading law, measured with an ideal actuator

Script: `experiments/steering_gain.py`. A flock driven for 150 steps is handed to a dog held
at a fixed bearing φ and standoff d in the flock's own frame. The dog is *re-placed every
step*: an ideal actuator with no speed limit. The grid is 9 bearings × d 5, 7.5, 10 m ×
w_ali 0, 1.3, 3 × w_att 1.5, 3, with 16 runs each.

**Convention.** θ is the flock's heading, c its centroid, h the dog. β = arg(h − c) and
φ = β − θ − π, so φ = 0 is directly behind and φ > 0 is behind on the right, which turns the
flock left.

**Model.** In a flock turning together, each sheep's own heading (w_inertia = 0.5) and the
heading it copies (w_ali) both lie along θ, and attraction and repulsion average out along it.
So

    θ_{t+1} − θ_t = δ(φ_t) + ε_t,     δ(φ) = atan2(w_dog sin φ, w_inertia + w_ali + w_dog cos φ)

| w_ali | linear gain K, measured (predicted) | fastest steady turn, ideal actuator | smallest radius, ideal actuator | P at φ = 0 → 60° |
|---|---|---|---|---|
| 0 | 0.67–0.80 (0.67) | 48–77°/step | 0.4–0.6 m | 0.61 → 0.55 |
| 1.3 (published) | 0.32–0.44 (0.36) | 22–29°/step | 1.2–2.0 m | 0.85 → 0.71 |
| 3 | 0.17–0.29 (0.22) | 10–17°/step | 2.2–4.9 m | 0.93 → 0.86 |

**What this supports.** For polarized flocks and |φ| ≤ 60°, a first-order discrete heading
model is approximately valid. The model error is:
- 13% of δ (median), 29% at the 90th percentile, in open loop;
- 5–12°/step one-step RMS in closed loop (§7), mostly the noise of a 14-sheep mean heading.

The turn rate reaches 63% of its steady value within 0–1 steps of a bearing change, so no
additional lag was resolved at one-step resolution. That is not a proof of "no lag".

**Validity limits.**
- Beyond about 90° the flock reverses and loses polarization instead of turning.
- Steering raises D⊥: 0.110 → 0.160 m²/step from 0° to 60° at the published weights, with
  τ_int unchanged.
- M2 stays within 10% (8.7–9.3 against 9.15) up to 60°. The flock gets shorter across and
  longer along.

**Alignment is the heading's inertia:** it lowers the gain threefold from w_ali = 0 to 3.

## 6. The actuator: what a dog of finite speed can hold

The dog sits at h = c + d e_β with β = θ + π + φ. With ċ = V e_θ (V = v P) and e_β^⊥ the unit
vector 90° from e_β:

    ḣ = (ḋ − V cos φ) e_β + (V sin φ + d(θ̇ + φ̇)) e_β^⊥,      φ̇ = e_β^⊥·(ḣ − ċ)/d − θ̇

Under ‖ḣ‖ ≤ v_dog this gives two constraints.

**Steady turns** (ḋ = φ̇ = 0, θ̇ = ω = δ(φ)):

    V² + 2 V d ω sin φ + d² ω² ≤ v_dog²

The dog must keep up with the flock *and* circle it on the outside of the turn. A path of
radius R is feasible in steady state only if some |φ| ≤ 60° gives V(φ)/δ(φ) ≤ R while
satisfying this.

**Changing the bearing** at fixed standoff:

    φ̇ ∈ [(−√(v_dog² − V²cos²φ) − V sin φ)/d − ω,  (√(v_dog² − V²cos²φ) − V sin φ)/d − ω]

At φ = 0 the margin is √(v_dog² − V²)/d, below the naive v_dog/d.

From the measured δ(φ) and P(φ), at d = 7.5 m and w_att = 1.5:

| w_ali | v_dog 1.5: feasible φ, turn rate, R_min | v_dog 3: feasible φ, turn rate, R_min | ideal actuator R_min | bearing-rate margin at φ = 0 (v_dog 1.5 / 3) |
|---|---|---|---|---|
| 0 | 14°, 9.6°/step, 3.5 m | 27°, 20.6°/step, 1.6 m | 0.6 m | 0.18 / 0.39 rad/step |
| 1.3 | 20°, 7.7°/step, 6.0 m | 47°, 18.3°/step, 2.4 m | 1.7 m | 0.16 / 0.38 |
| 3 | 32°, 6.1°/step, 8.5 m | 60°, 11.5°/step, 4.2 m | 4.2 m | 0.16 / 0.38 |

The ideal-actuator radii of §5 are lower bounds.

**What R_min is.** It is a *steady-turn estimate at fixed standoff and fixed bearing*. It
leaves out standoff changes and the transient at an arc entry. It is also a continuous-time
formula applied to a one-step model. Its one-step version uses the centroid moving along the
new heading and the dog along the chord of its circle, and gives R_min 0–4% smaller in every
cell (5.78 vs 6.02 m at w_ali 1.3, v_dog 1.5).

At the published dog speed the actuator roughly triples the smallest radius at w_ali = 1.3.
Alignment raises it from 3.5 to 8.5 m.

## 7. A real dog on a straight line and an arc

Script: `experiments/path_tracking.py`. The sheep model is unchanged and uses the published
rule. The dog moves at most v_dog per step. Its controller is a cascade:
- **Guidance:** θ_d = θ_path − atan(e/6 m), with e the signed cross-track error.
- **Turn command:** ω = Vκ + 0.3·(θ_d − θ).
- **Bearing:** φ_d = δ^{-1}(ω), inverted from the measured table and capped at 60°.
- **Dog:** moves toward c + d e_β(φ_d) plus one flock step, at most v_dog.

The path is 25 m straight, then a 90° left arc of radius R ∈ {3, 5, 8, 12, 20} m, then 40 m
straight. The grid covers w_ali ∈ {0, 1.3, 3} and v_dog ∈ {1.5, 3}, at d = 7.5, w_att = 1.5,
with 16 runs each, starting from flocks driven for 150 steps. Ten of the 480 runs had split
during that warm-up drive (which has no collecting) and are excluded. Everything below is
conditional on the remaining 470 runs. Errors are medians over runs, with the 90th
percentile in brackets.

| w_ali, v_dog (R_min) | arc radius: predicted outward excess max(0, R_min − R) → measured max cross-track error, median (90th pct) |
|---|---|
| 3, 1.5 (8.5 m) | 3 m: 5.5 → 6.5 (8.0); 5 m: 3.5 → 5.0 (6.0); 8 m: 0.5 → 1.5 (4.0); 12, 20 m: 0 → 1.2 |
| 1.3, 1.5 (6.0 m) | 3 m: 3.0 → 3.3 (4.9); 5 m: 1.0 → 2.1 (3.5); 8–20 m: 0 → 1.4–1.7 |
| 0, 1.5 (3.5 m) | 3 m: 0.5 → 1.9; 5–20 m: 0 → 2.0–2.1 |
| 3, 3 (4.2 m) | 3 m: 1.2 → 2.4 (4.2); 5–20 m: 0 → 1.1–1.3 |
| 1.3, 3 (2.4 m) and 0, 3 (1.6 m) | all radii: 0 → 1.4–1.6 and 1.9–2.2 |

**Checks of the effective model along the way:**

- **Tracking.** Every run completes the path, and no flock splits during tracking. The
  maximum error is a baseline of 1.1–2.2 m (set by heading noise, larger without alignment)
  plus about R_min − R when the arc is infeasible. That is an *empirical* relation for these
  runs, not a derived bound.
- **Heading law in closed loop.**
  - One-step RMS error is 5.3–5.6°/step at w_ali = 3, 7.2–7.5 at 1.3, and 11.5–11.9 at 0.
  - Accumulated turn on the arc is 0.85–1.19 of the prediction at w_ali = 3, but 0.68–0.96
    at 1.3 and 0.54–0.97 at 0. In noisy flocks the controller spends bearing on heading
    noise that reverts on its own, so the summed prediction overshoots. This is model error
    to keep, and it is larger when the flock is noisier.
- **Polarization closure.** The arc-averaged P is within 0.01–0.05 of P_eq(φ); per-step RMS
  is 0.03–0.10.
- **Shape moments** (medians):
  - M2 is 9.4–11.2 m² on the straight and 9.2–10.5 on the arc (medians over runs of the
    segment means; ranges over settings).
  - Each run's largest M2 after the arc starts has a median of 12.3–15.0, a 90th percentile of
    13.8–17.3, and a maximum over all runs of 14.2–19.7.
  - The flock is rounder on the arc: sd along 1.3–1.6 → 1.6–2.1 m, across 2.6–3.0 →
    2.1–2.7 m.
  - The ideal-actuator table at the arc's mean bearing predicts sd along 1.86–2.09 and across
    2.16–2.37. With the real dog the flock is up to 0.5 m wider across than that.
- **Validity.** On 95–100% of steps the flock stays where the laws were measured
  (|φ| ≤ 60°, every sheep within r_dog). The dog's speed is saturated on up to 33% of steps
  where predicted (w_ali 3, v_dog 1.5).

## 8. The effective model, assembled, and how it maps onto Di Lorenzo's analysis

State (c, θ, P, q): centroid, heading, polarization, and shape in the co-moving frame.

- **Centroid** (kinematic, exact given the definitions up to fluctuations):
  c_{t+1} = c_t + v P_t e_θ.
- **Heading** (§5, measured with error ε): θ_{t+1} = θ_t + δ(φ_t) + ε_t.
- **Polarization** (§2 R3, quasi-static, measured): P_t ≈ P_eq(φ), relaxing in 1–3 steps.
- **Shape** (not closed). A Fokker–Planck equation in the co-moving frame with D⊥(P) and
  drift b_collective + u_dog. The measured observables are sd∥, sd⊥ and M2. Prop. 1 bounds
  the push's direct contribution in the additive model only.
- **Actuator** (§6, exact): ‖h_{t+1} − h_t‖ ≤ v_dog, with h = c + d e_β(θ + π + φ).

**The mapping.** Di Lorenzo's analysis has two parts, and both have counterparts here.

*Feasibility.* His question is whether some non-negative herder density of admissible mass
produces the required velocity field. Here, a target (path c*(s), shape q*) is feasible if:
- the path curvature satisfies the actuator inequality of §6 at some |φ| ≤ 60°, i.e.
  R ≥ R_min (the counterpart of his non-negativity and mass window);
- q* lies in the set the dog can reach. By Prop. 1 (additive) that set does not extend below
  what cohesion and the attainable D⊥(P) hold, except through the notice edge.

*Stability.* His target-density L² estimate becomes an ISS estimate for the heading and
centroid loop. That is standard for a discrete integrator with a sector-bounded input map,
bounded noise and rate-limited actuation. Under it sits a shape estimate that would be the
analogue of his Theorem 2, with D(P) and a moving frame, and that is not available yet.

## 9. Candidate proposition

**Setting.**
- Sheep: the published Jadhav rule, N = 14, w_att = 1.5.
- Dog: speed ≤ v_dog at standoff d = 7.5, controlled by the cascade of §7.
- Path: piecewise-constant curvature κ(s) with radii R_i.

**Assumptions** (each measured, not proved):
- (A1) **Heading law.** For P_t ≥ 0.8 P_eq(0) and |φ_t| ≤ 60°,
  θ_{t+1} = θ_t + δ(φ_t) + ε_t with δ the measured map. The noise ε_t is zero-mean, roughly
  independent, with RMS σ_ε of 5–12°/step depending on w_ali. A multiplicative bias of
  0.55–1.2 appears in closed loop.
- (A2) **Polarization closure.** P_t = P_eq(φ) + ξ_t, relaxing in ≤ 7 steps.
- (A3) **Coverage and cohesion.** Every sheep is within r_dog, and no split occurs.
- (A4) **Actuator.** ‖h_{t+1} − h_t‖ ≤ v_dog. This one is exact.

**Derived** (from A1, A2 and A4):
- (D1) **Steady feasibility (an estimate).** At fixed standoff and fixed bearing, a κ-arc
  admits a steady bearing iff some |φ| ≤ 60° gives δ(φ) = V(φ)κ with
  V² + 2Vdδ sin φ + d²δ² ≤ v_dog². That defines R_min(v_dog, d, w_ali). It is a continuous-time
  formula on a one-step model (the chord version is 0–4% smaller), and it leaves out standoff
  changes and arc-entry transients.
- (D2) **Bearing-rate bounds** as in §6. The curvature jump at an arc entry costs a
  transient of about Δφ/(margin) steps.
- (D3) **Infeasible arcs (heuristic).** If R < R_min, the flock cannot turn faster than
  δ(φ_max), so its path radius stays near R_min. Two circles tangent at the arc entry then
  separate by about R_min − R over a 90° arc. §7 observes this relation. It is geometric
  reasoning for a deterministic steady turn, not a bound derived for the discrete stochastic
  loop.

**Claim to prove** (heading and centroid, in mean square). A deterministic limsup cannot
follow from a noise RMS, so the claim is stochastic. Write the heading error
e_t = θ_t − θ_d,t of the cascade as

    e_{t+1} = (1 − k_θ g_t) e_t + ε_t + β_t + Δ_t,

where:
- g_t ∈ [g_min, g_max] is the realized fraction of the commanded turn (the closed-loop bias of
  §7: 0.55–1.2);
- ε_t is zero-mean with E[ε_t² | past] ≤ σ_ε²;
- β_t is a bounded bias, |β_t| ≤ β̄;
- Δ_t is the jump of the desired heading at curvature changes, nonzero only at finitely many
  times t_j.

If every R_i ≥ R_min with margin, so that the bearing leaves saturation within T_j steps of
each jump, then with ρ = max |1 − k_θ g| < 1 the candidate bound is

    E e_t² ≤ 4 [ ρ^{2t} e_0² + ( Σ_{t_j ≤ t} ρ^{t − t_j − T_j} |Δ_j| )² + σ_ε² / (1 − ρ²) + β̄² / (1 − ρ)² ]

It has decaying initial and jump terms, plus a noise floor and a bias floor. The cross-track
error inherits a mean-square bound through the line-of-sight guidance, which is ISS in the
heading error while |e| < π/2. A deterministic ISS version with bounded disturbances exists,
but the only sure bound on ε_t is π, which makes it vacuous; the mean-square form is the
useful one.

*Status:* written out in [jadhav_dilorenzo_bridge.md §9](jadhav_dilorenzo_bridge.md), which
replaces the scalar sketch above with three changes:
- a two-state recursion in (heading error, cross-track error) with exact discrete geometry, in
  place of a continuous-time LOS argument;
- no factor 4;
- a closed-loop gain of 1.03 relative to the *command*. The 0.55–1.2 above is relative to the
  table at the achieved bearing.

At baseline the bound is about 1.4 m on the first straight, and 3.0, 2.0 and 1.8 m at R = 8, 12
and 20 m. It holds at every step of the published runs, at least 1.4× above the measured error.

**Shape claim.** §10 replaces the earlier conjecture with a conditional statement. While
every sheep moves, the flock stays polarized (P_μ ≥ P*) and the weighted inward relative
heading stays at least κ*, E[ΔM₂ | state] ≤ −a√M₂ + b. What is missing for the turn is a proof
that the process stays in that set. In §7 every sheep stays in range on 95–100% of steps,
P stays at 0.59–0.93, and each run's largest M2 is at most 19.7 m² over all runs.

**The weaker conclusion the evidence supports today.** For this flock size and these
parameters, a finite-speed dog with this cascade tracks a straight-then-arc path. Over the
470 runs that started as one flock, the centroid error is bounded whenever R ≥ R_min; below
that it misses by about R_min − R, empirically. The second moment stays below 20 m² and no
flock splits. This is an empirical statement for one flock size and parameter set, with an
estimated feasibility boundary and a proved conditional drift bound for the shape. It is not
a theorem.

## 10. Shape dynamics from the exact second-moment identity

Script: `experiments/shape_drift.py`; numbers in `results/shape_drift.json`; figure
`results/figures/shape_drift.png`.

**The identity.** Let ξ_i = x_i − c, M₂ = (1/N)Σ|ξ_i|², a_i the actual one-step displacements
and ā their mean. Then

    ΔM₂ = (2/N) Σ ξ_i·a_i + (1/N) Σ |a_i − ā|²

exactly (Σξ_i = 0). Checked to 1e-12 on 230 000 simulated steps.

**Under the full Jadhav rule.** a_i = v m_i h_i′.
- m_i ∈ {0, 1} is 1 when sheep i sees the dog or is repelled. That is the published stop
  rule, and it is a function of the current state.
- h_i′ = unit(S_i) is the new heading. S_i is the weighted sum of memory, repulsion, the
  dog's push, attraction to a random 4 of the 10 nearest, a copy of one of those, and noise.

Nothing is linearized. Two facts make the step tractable:

1. Moving sheep have |h_i′| = 1. So (1/N)Σ|a_i − ā|² = v²(f_m − |P̃|²) exactly, where f_m is
   the fraction moving and P̃ = (1/N)Σ m_i h_i′.
2. Given the full current state, each sheep's random draws (neighbour subsets, noise) are
   independent of the others'. Alignment copies *current* headings, so it does not couple the
   draws.

With μ_i = E[h_i′ | state] and M = (1/N)Σ m_i μ_i:

    E[ΔM₂ | state] = C + G,
    C = (2v/N) Σ m_i ξ_i·μ_i                                         (radial drift)
    G = v² [ f_m − |M|² − (1/N²) Σ m_i (1 − |μ_i|²) ] ≤ v² (f_m − f_m² P_μ²),   P_μ = |M| / f_m

**G is the heading dispersion.** It plays the role diffusion plays in a Fokker–Planck model.
It is set by how polarized the conditional mean headings are, not by any diffusion constant.

**C, rewritten.** With μ̄ = M/f_m and Σξ_i = 0:

    C = (2v/N) Σ m_i ξ_i·(μ_i − μ̄)  −  (2v/N) (Σ_frozen ξ_i)·μ̄

- For moving sheep the common heading cancels: only headings *relative to the flock's* change
  M₂.
- The second term exists only under the stop rule. Frozen sheep that sit ahead of the movers
  (the dog behind) contract the flock. This is the notice-edge compression, now an exact term.

When every sheep moves, let γ = (1/N)Σ|ξ_i| / √M₂ ∈ [1/√N, 1] be a shape factor and
κ = −Σ|ξ_i| ξ̂_i·(μ_i − μ̄) / Σ|ξ_i| the weighted inward relative heading. Then exactly

    C = −2v κ γ √M₂,      E[ΔM₂ | state] = −2v κ γ √M₂ + G.

**Approximations, used only to interpret.** Linearizing about a polarized flock gives each
sheep's relative heading as (I − êê)δS/|S|. Lateral relative headings are then first order.
Longitudinal ones are second order (the speed is fixed), and for sheep behind the centroid
they are bounded by the slack 1 − P. So κ∥ ≪ κ⊥ is expected. Unit-vector attraction makes an
outer sheep's inward relative heading roughly independent of its distance, so κ should
saturate as the flock spreads. That would give a drift ∝ −√M₂, not ∝ −M₂.

**Lemma (drift bound).** Let 𝒢(κ*, P*, γ*) be the set of states in which every sheep moves,
κ ≥ κ*, P_μ ≥ P* and γ ≥ γ*. On 𝒢,

    E[ΔM₂ | state] ≤ −a √M₂ + b,      a = 2v κ* γ*,   b = v² (1 − P*²).

*Proof.* Facts 1 and 2, all sheep moving, and the definitions. ∎

The increments are bounded: |ΔM₂| ≤ 2v max|ξ| + v². So Foster–Lyapunov arguments give finite
mean return times to {√M₂ ≤ b/a + ε}, and a bounded E√M₂ in stationarity. Both hold only *while
the process stays in 𝒢*. Nothing here shows that it stays.

**Measured.** Monte Carlo estimates of μ_i use K = 128 one-step draws per state. The states
are 30 observed states per setting, plus their dilations by 1.5, 2 and 3 about the centroid,
with headings unchanged. κ is given as median [10th percentile] over states in which every
sheep moves.

| Setting | G vs its bound v²(f_m − f_m²P²) at observed states | κ at 1× / 1.5× / 2× / 3× | κ∥ / κ⊥ at 2× | every sheep moves, 1× / 1.5× / 2× / 3× |
|---|---|---|---|---|
| published, r_dog 20, d 7.5 | 0.250 vs 0.259 | 0.08 [−0.02] / 0.16 [0.11] / 0.16 [0.12] / 0.15 [0.10] | 0.02 / 0.15 | 1.00 / 1.00 / 1.00 / 0.70 |
| published, r_dog 12, d 7.5 | 0.240 vs 0.249 | 0.05 [0.01] / 0.15 [0.11] / – / – | – | 0.97 / 0.27 / 0 / 0 |
| published, r_dog 12, d 10 | 0.458 vs 0.473 | (no state with every sheep moving above 1×) | – | 0.03 / 0 / 0 / 0 |
| strong (w_att 3, w_ali 3), r_dog 20, d 7.5 | 0.123 vs 0.132 | 0.05 [0.02] / 0.14 [0.07] / 0.20 [0.13] / 0.23 [0.15] | 0.05 / 0.15 | 1.00 at every dilation |
| paper-text parameters (n_att 5, e 0.5), r_dog 12, d 7.5 | 0.235 vs 0.244 | 0.05 [−0.02] / 0.13 [0.12] / – / – | – | 0.97 / 0.20 / 0 / 0 |
| dog-free, published weights (modified rule) | 0.892 vs 0.937 | 0.16 / 0.39 / 0.45 / 0.51 | 0.22 / 0.23 | 1.00 |
| dog-free, strong (modified rule) | 0.553 vs 0.588 | 0.14 / 0.32 / 0.38 / 0.42 | 0.16 / 0.19 | 1.00 |

**Reading.**

1. **G is the heading dispersion.** It is 0.94–0.97 of its bound v²(f_m − f_m²P²) in every
   setting; the remaining 3–6% is the dropped 1/N² term. The flock's "noise" is the spread of its headings, and the dog
   acts on it through polarization.
2. **κ saturates as the flock spreads** at the published weights (0.16, 0.16, 0.15 at 1.5×,
   2×, 3×) and grows slowly with strong cohesion. The restoring is constant-speed.
   −a√M₂ + b fits E[ΔM₂ | state] across dilations at least as well as −aM₂ + b in 12 of the
   14 settings that have states with every sheep moving. The two exceptions are strong flocks
   at r_dog 12, where the linear form wins marginally (residual 0.160 vs 0.173, and 0.355 vs
   0.359). **−aM₂ + b is not supported in general.**
3. **Longitudinal restoring is weak:** κ∥ ≈ 0.02 against κ⊥ ≈ 0.15 (published), and 0.05
   against 0.15 (strong). At fixed speed the back of the flock catches up only through the
   slack 1 − P.
4. **Where 𝒢 holds.**
   - With r_dog 20, every sheep moves in 93–100% of states at 1.5–2× the observed size, and
     in 27–100% at 3× (the dog 10 m back loses the far sheep). κ > 0 in every state with all
     sheep moving, with 10th percentiles of 0.06–0.19. On this family of states the
     drift is therefore negative with margin outside about 1.5× the stationary size. Taking
     κ* ≈ 0.06–0.1, γ* ≈ 0.9 and P* ≈ 0.83 gives a ≈ 0.1–0.18 and b ≈ 0.3 per step.
   - At the observed size, κ is small (0.03–0.08) and is ≤ 0 in 0–30% of states. The flock
     sits where restoring and dispersion balance, as a stationary flock must.
5. **Failure: loss of coverage.** With r_dog 12 and the dog 10 m back, almost no state has
   every sheep moving. Frozen sheep neither restore nor disperse, so E[ΔM₂ | state] ≈ 0 at
   every size up to 3×. Nothing brings an out-of-range sheep back except the dog moving. With
   the dog behind, partial coverage contracts instead: the frozen sheep are ahead, so the
   frozen term is negative. The stop rule therefore helps or stalls, depending on where the
   frozen sheep are. A fixed κ* cannot hold there, because κ is not even defined.
6. **What the dog changes.** Comparing driven flocks with dog-free ones (the same rule at
   r_dog 20), means at the observed states:

   | Flock | G, dog-free → driven | κγ, dog-free → driven | √M₂, dog-free → driven |
   |---|---|---|---|
   | strong (w_att 3, w_ali 3) | 0.567 → 0.128 (4.4× lower) | 0.156 → 0.053 (2.9× lower) | 1.96 → 1.47 m |
   | published | 0.868 → 0.262 (3.3× lower) | 0.140 → 0.062 (2.3× lower) | 2.78 → 2.95 m |

   The dog lowers the dispersion *and* weakens the effective restoring response. With
   normalization, its term and the now-aligned headings dilute attraction's share of each
   heading.

   **A correction to the first version of this item.** It argued from √M₂* ≈ G/(2vκγ) that
   the published flock's two changes nearly cancel. That contradicts the formula itself:
   G falling 3.3× and κγ falling 2.3× would shrink √M₂ by 1.45×, but it grew by 6%. The
   ratio formula is not valid. κ is not a constant of the flock: it depends on the state and
   rises with size (from about 0.05 at the observed size to 0.15 at 1.5×). It is therefore
   correlated with √M₂, and the stationary balance holds only for the expectations,
   E[2vκγ√M₂] = E[G] (with every sheep moving), not for a ratio of averages. These 30-state
   samples are also only roughly stationary: across settings the mean of C + G at observed
   states is −0.12 to +0.13 per step, up to half of G. So which change dominates the stationary
   size cannot be read off these numbers. What stands is qualitative: both terms fall under
   the dog, in opposite directions on size. Attributing the strong flock's tightening to lower
   diffusion alone was incomplete.

**Local drift and long-term guarantees, separately.**

- *Local, one step (proved).*
  - The identity and its conditional split into C + G, and the frozen-sheep term.
  - The drift lemma: on 𝒢, E[ΔM₂ | x] ≤ −a√M₂ + b.
- *Finite horizon, until exit (proved, given the lemma).* Fix ε > 0 and the ball
  B = {√M₂ < (b + ε)/a}. Let σ be the time the flock enters B and τ the time it leaves 𝒢.
  Before σ∧τ, E[ΔM₂ | x] ≤ −ε, so M₂(t∧σ∧τ) + ε (t∧σ∧τ) is a supermartingale. Hence:
  - E[σ∧τ] ≤ M₂(0)/ε;
  - P(M₂ reaches L before σ∧τ) ≤ M₂(0)/L, by the maximal inequality for non-negative
    supermartingales.

  This says that *either* the flock returns to the ball *or* it leaves 𝒢, within a bounded
  expected time, and that it is unlikely to grow far first. It does not say which of the two
  happens.
- *Long term (not established).* Recurrence to the ball and bounded moments in stationarity
  would need the drift condition on *every* reachable state outside a compact set, or proof
  that 𝒢 ∪ B is never left. §11 shows that neither holds in the form proved:
  - 𝒢 is left often, by polarization dips and by κ falling near zero at lateral splits;
  - there are reachable states with positive drift outside B (lateral splits under a short
    notice radius; any split of a 30-sheep flock).

  Every long-term statement in this document is therefore an observation on tested states.

**Measured, on observed states and their dilations:**
- 𝒢 holds outside about 1.5× the stationary size when coverage is full;
- the restoring is constant-speed;
- longitudinal restoring is about 7× weaker than lateral.

§11 extends this to other deformations and to trajectories.

**Towards a prescribed shape.** The exact stationary statement is E[C] = −E[G] under the
stationary law. There is no closed formula for the stationary M₂ in terms of the dog's knobs,
because κ is state-dependent. The set of stationary second moments the dog can produce, the
moment-level counterpart of Di Lorenzo's feasibility set, has to be measured. For the
published flock, §4 measured M₂ of 8–30 m² over standoffs, never below the dog-free 8.
Tracking a prescribed M₂*(t) needs M₂* inside that set, and changes slow against the
relaxation times observed in §11 (median 3–63 steps to re-enter the ball after a stress).
Densities beyond the second moment are not attempted.

## 11. Stress tests, trajectories, and a candidate sufficient condition

Scripts: `experiments/shape_stress.py` (one-step drift on deformed states) and
`experiments/shape_trajectories.py` (trajectories from them); numbers in
`results/shape_stress.json` and `results/shape_trajectories.json`; figures under the same
names. All use the published rule, with the dog left at its observed position (7.5 m behind)
for the one-step estimates.

**Uncertainty.**
- E[ΔM₂ | state] is a mean over K = 128 one-step draws, with standard error s.d./√K.
- κ's standard error comes from four batches of the draws.
- Family medians carry 95% bootstrap intervals over the 30 states.
- Typical per-state standard errors are 0.01–0.14 m² per step for E[ΔM₂] and 0.002–0.016 for
  κ. "Significantly positive" means more than two standard errors above zero.

### 11.1 One step, on deformed states

Observed states are deformed: stretched along or across the heading, sheared, split into two
halves by a gap, one sheep moved out, or headings randomized or half reversed. Medians over 30
states, with bootstrap 95% intervals.

| Deformation | published, r_dog 20: E[ΔM₂] [CI]; κ [CI]; in 𝒢 | published, r_dog 12: E[ΔM₂] [CI]; every sheep moves | strong, r_dog 20: E[ΔM₂] [CI]; κ [CI] |
|---|---|---|---|
| none (observed) | +0.02 [−0.25, +0.16]; 0.047 [0.017, 0.084]; 0.50 | −0.08 [−0.15, +0.04]; 1.00 | +0.01 [−0.02, +0.02]; 0.043 [0.041, 0.054] |
| stretch across ×3 | −1.63 [−2.04, −1.43]; 0.145; 0.90 | −0.05 [−0.21, +0.08]; 0.00 | −0.85 [−1.05, −0.74]; 0.194 |
| stretch along ×3 | −1.07 [−2.14, −0.81]; 0.115; 0.53 | −2.46 [−2.75, −2.26]; 0.03 | −0.46 [−0.56, −0.32]; 0.128 |
| split across, 8 m | −0.31 [−0.53, +0.33]; 0.042 [−0.006, 0.060]; 0.43 | −0.02 [−0.36, +0.38]; 0.27 | −1.71 [−1.85, −1.64]; 0.193 |
| split across, 12 m | **+0.18 [−0.34, +0.73]; 0.007 [−0.036, 0.036]; 0.20** | **+1.20 [+0.87, +1.68]; 0.10** | −2.36 [−2.60, −2.22]; 0.186 |
| split along, 12 m | −0.56 [−0.97, −0.22]; 0.044; 0.37 | −4.08 [−4.45, −3.74]; 0.10 | −1.13 [−1.21, −0.95]; 0.100 |
| outlier to the side, 14 m | −0.48 [−0.65, −0.41]; 0.111; 0.80 | −0.00 [−0.12, +0.12]; 0.00 | −0.86 [−0.89, −0.75]; 0.203 |
| outlier ahead, 10 m | −0.04 [−0.25, +0.13]; 0.048; 0.47 | −1.18 [−1.22, −1.09]; 0.00 | −0.16 [−0.19, −0.13]; 0.076 |
| heading noise (σ = 3) | +0.16 [+0.02, +0.46]; 0.091; 0.63 | +0.10 [−0.03, +0.29]; 1.00 | +0.43 [+0.34, +0.50]; 0.125 |
| half the headings reversed | +0.30 [−0.31, +0.61]; 0.093; 0.67 | +0.20 [−0.07, +0.58]; 1.00 | +0.48 [+0.39, +0.59]; 0.150 |

- **Lateral splits are the weak spot at the published weights.** κ falls to zero at a 12 m
  gap, and the drift's interval straddles zero. With r_dog 12, half the flock is out of range
  and the drift is significantly positive (+1.2) in 87% of states. The strong flock contracts
  from every deformation tested.
- **Heading perturbations give positive drift at every size.** They raise the dispersion G
  without moving anyone, and are size-preserving, so they start inside the ball (§11.2).
- **Ablations** set one weight to 0 in the same states and draws, averaged over the
  deformations. Attraction supplies κ (+0.16 to +0.52). The dog's push removes some
  (−0.01 to −0.32, more at the published weights), and so does alignment (−0.02 to −0.64).
  Alignment pulls relative headings toward the flock's and so reduces κ, while it also lowers
  G.
- **The per-draw cone certificate rarely holds.** It certifies 0–7% of lateral outer sheep at
  the published weights with r_dog 20, and up to 42% for the strong flock (lateral outlier).
  At r_dog 12 it reaches 36%, but mostly for sheep that are out of range and frozen, where it
  says nothing about motion. The inward relative
  heading is an *average* over draws, not something each draw guarantees.

**Topological test.** Synthetic compact flocks are built at roughly the observed density,
aligned, with every sheep in range (r_dog 40), and split into halves across the heading:

| Flock | Gap | Perceived neighbours in the other half | κ [CI] | E[ΔM₂] [CI] |
|---|---|---|---|---|
| 14 sheep | 4 m | 40% | 0.054 [0.034, 0.081] | −0.15 [−0.35, +0.02] |
| 14 sheep | 12 m | 40% | 0.011 [−0.010, 0.026] | +0.18 [−0.10, +0.41] |
| 30 sheep | 4 m | 3% | −0.127 [−0.150, −0.111] | +1.54 [+1.40, +1.78] |
| 30 sheep | 12 m | 0% | −0.238 [−0.266, −0.218] | +4.37 [+4.04, +4.96] |

*Fact (combinatorial, proved).* Split a flock into two parts A and B, with |A| ≤ |B|. Each
sheep perceives its K = 10 nearest others. If |A| ≤ K, each sheep of A perceives at least
K − (|A| − 1) ≥ 1 sheep of B. So the smaller part always sees the larger when N ≤ 2K + 1 = 21.
At N = 30 two parts of 15 can lose sight of each other completely, and then the measured κ
is negative: the dog's spreading is no longer opposed.

### 11.2 Trajectories: recovery or exit first, and what follows

From each deformed state the full model runs 150 steps, with the dog either driving only or
also collecting (Strömbom). Every 3 steps the script estimates κ, P and E[ΔM₂] from 32 draws.

- **Ball.** ρ = (b + 0.02)/a with κ* = 0.05 and P*, γ* the 10th percentiles at observed
  states. That gives 3.9, 4.5 and 2.5 m, against observed √M₂ of 3.05, 3.08 and 1.40 m.
- **Exit from 𝒢.** A sheep frozen, or κ or P below its threshold by more than two standard
  errors. The strict count without that margin gives more exits, mostly by polarization.
- **After an exit, observed to the end and classified:**
  - renewed cohesion: re-enters the ball;
  - freezing: most sheep stopped over the last 20 steps;
  - continued spreading: √M₂ up more than 1 m, or split.
- **Split:** the largest group linked at 6 m holds fewer than N − 1 sheep.

Counts are out of 30 trajectories per row. "In ball" means the run started inside the ball.

| Setting, deformation, dog | recovered before leaving 𝒢 / outside 𝒢 first, then recovered / then froze / started in ball | median [p90] steps to the ball | split at the end |
|---|---|---|---|
| published r_dog 20, stretch across ×3, drive | 16 / 14 / 0 / 0 | 29 [41] | 0 |
| published r_dog 20, split across 12 m, drive | 0 / 30 / 0 / 0 | 63 [97] | 0 |
| published r_dog 20, split along 12 m, drive | 0 / 30 / 0 / 0 | 54 [67] | 0 |
| published r_dog 20, outlier to the side 14 m, drive | 12 / 18 / 0 / 0 | 14 [22] | 0 |
| published r_dog 20, heading noise / half reversed | 0 / 0 / 0 / 30 (left the ball briefly in 0–7%) | – | 0 |
| published r_dog 12, dilated ×2, drive | 0 / 23 / 7 / 0 | 13 [21] | 0.07 (stragglers 0.20) |
| published r_dog 12, stretch across ×3, drive | 0 / 0 / **30** / 0 | never | **0.90** |
| published r_dog 12, split across 8 m, drive | 0 / 16 / 14 / 0 | 42 [56] | 0.40 |
| published r_dog 12, split across 12 m, drive | 0 / 0 / **30** / 0 | never | **1.00** |
| published r_dog 12, outlier to the side 14 m, drive | 0 / 2 / 26 / 2 | – | stragglers 0.97 |
| published r_dog 12, any of the above, **with collecting** | 0 / 28–30 / 0 / 0–2 | 12–64 [20–110] | 0 |
| strong r_dog 20, every deformation, either dog | all recover | 3–31 [7–37] | 0 |
| **30 sheep, compact start, drive only** | – | √M₂ 3.5 → 21 m by step 50 | **1.00**; all frozen by step 150 |
| **30 sheep, compact start, with collecting** | – | √M₂ 15–21 m from step 50 on | **1.00** (largest group 17 of 30) |

**Reading.**

1. **𝒢 is not invariant, but at full coverage that did not matter on these states.** At
   r_dog 20, most trajectories leave 𝒢 before re-entering the ball, or start outside it.
   - Causes: κ dips at lateral splits; polarization dips; frozen sheep at the far end of
     stretched flocks.
   - Every one of them re-enters the ball within 150 steps and none ends split, under either
     dog. Recovery from a 12 m lateral split, where the one-step drift is not significantly
     negative, takes a median of 63 steps.
2. **With a short notice radius, coverage decides, and the dog's feedback restores it.** At
   r_dog 12, a drive-only dog leaves laterally displaced sheep out of range. They freeze and
   stay apart (stretch across, 12 m split, lateral outlier: 90–100% fail). The collecting dog
   recovers every case.
3. **The strongest failure is a flock larger than twice the perception number.** Compact
   30-sheep flocks under the published weights fragment within 50 steps, and neither dog
   reunites them in 150 steps. The largest linked group is 17 of 30. The drive-only dog
   leaves all of them frozen.

### 11.3 A candidate sufficient condition, and its status

**Conjecture (published Jadhav rule, one dog).** From a state with every sheep moving, the
flock re-enters the ball B in finite expected time and does not split, provided:

- (a) *coverage is kept*: every sheep stays within r_dog of the dog, either because
  r_dog ≥ standoff + flock extent throughout, or because the dog collects sheep that leave it;
- (b) *in the lower-risk regime below the measured fission threshold*, N < N_c. The threshold
  is predicted by Φ(N) = 1 (§11.5): the dog's lateral push on each half against attraction across
  the gap, with cross share max(0, K + 1 − N/2)/K. Below N_c splits are less likely, not
  excluded. This replaces the first version's N ≤ 2K + 1, which §11.4 refutes;
- (c) *attraction beats the dog's spreading at the flock's edge on average*: κ ≥ κ* > 0 on
  the deformations the drive produces, with κ* measured as in §10. It holds with margin for
  w_att 3, w_ali 3, and is marginal for lateral splits at the published weights.

**Status, separated:**

| Piece | Status |
|---|---|
| Exact identity; the split E[ΔM₂ \| x] = C + G; the frozen-sheep term | proved |
| Drift lemma on 𝒢; the stopped-supermartingale corollary (return to B or exit from 𝒢 within E[σ∧τ] ≤ M₂(0)/ε) | proved |
| The counting fact (a separated part of at most K sheep perceives the rest) | proved, but it does not control fragmentation: as a condition, N ≤ 2K + 1 is **refuted** (§11.4) |
| (b) via the threshold Φ(N) = 1 | a mean-field balance with one fitted constant. It predicts the measured 50% split size of eight further settings with bias −0.6 to −0.8 sheep and worst error 1.3–1.4 (§11.5). Not a theorem; the bias is unexplained; below N_c is lower risk, not safe |
| The cone certificate (per-draw inward attraction) | proved, but it applies to few sheep, so it gives no κ bound |
| (a) to (c) imply recovery and no split | **conjecture**: it held in every tested trajectory satisfying them (3 settings × 10 deformations × 2 dogs × 30 states, 150 steps, N = 14) |
| Necessity of (a) | supported: without coverage and without collecting, 87–100% of runs end frozen and split, or with a stranded straggler, in 3 of the 10 deformations |
| Necessity of (b) | supported: above N_c the drive splits compact flocks into two halves in 93–100% of runs, with either dog |
| Necessity of (c) | not shown: published flocks recovered from 12 m lateral splits although κ ≈ 0 initially |

These are finite stress tests on chosen deformations of observed states, with N = 14 (and
16 from compact starts) for the positive results and horizons of 150–300 steps. They are not a proof over the state space.
States the families do not reach (three or more clusters, rotating flocks, the dog inside the
flock, long-horizon rare events) are untested.

### 11.4 The flock-size boundary: N ≤ 2K + 1 refuted, a fission threshold instead

Script: `experiments/size_sweep.py`; numbers in `results/size_sweep.json`; figure
`results/figures/size_sweep.png`. Compact aligned flocks of N = 14–30 are driven for 300 steps
under the published rule with r_dog 20, 30 runs each. Split means two or more sheep outside the
largest group linked at 6 m. Intervals are 95% Wilson intervals.

**The prediction, stated before the run.** Counting gives that a separated part of at most K = 10
sheep always perceives the rest. Two halves can therefore lose sight of each other only from
N = 22. If that counting controlled fragmentation, splits would be rare up to N = 21 and jump at
22.

**Result: refuted.**

| N | 14 | 16 | 17 | 18 | 19 | 21 | 22–30 |
|---|---|---|---|---|---|---|---|
| split after 300 steps, published weights, dog 7.5 m, drive only | 0/30 | 0/30 | 13/30 | 27/30 | 30/30 | 30/30 | 30/30 |
| same, with collecting | 0/30 | 0/30 | – | – | 28/30 | 30/30 | 30/30 |
| perceived neighbours across a 4 m split (one step) | 40% | 30% | – | – | 16% | 8% | 2–5% |
| κ across a 4 m split, median [CI] | +0.034 [0.014, 0.068] | +0.005 [−0.015, 0.019] | – | – | −0.049 [−0.088, −0.030] | −0.086 | −0.11 to −0.16 |

The boundary is between 16 and 18 sheep, not at 22. Every split is a *lateral fission* into
two near-halves (the largest group holds 52–58% of the flock). It starts after a median of
30–88 steps, earlier for larger flocks. The collecting dog does not prevent it.

Perceiving one sheep across the split is far too little. κ across the gap is already negative
at N = 19, where 16% of perceived neighbours are still across. What matters is the *share* of
cross-perception against the dog's lateral spreading.

**Where the boundary moves.** The same drive-only test, varying attraction and standoff:

| N | 14 | 16 | 17 | 18 | 19 | 21 | 24 |
|---|---|---|---|---|---|---|---|
| published weights, dog 7.5 m | 0/30 | 0/30 | 13/30 | 27/30 | 30/30 | 30/30 | 30/30 |
| published weights, dog 12 m | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 | 28/30 | 30/30 |
| w_att 3, dog 7.5 m | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 2/30 | 29/30 |

The transition is sharp, within 2–3 sheep. It moves up with weaker spreading (a farther dog;
the lateral push falls as 1/d) and with stronger attraction. Both are what the mechanism
predicts: fission when attraction across the flock's midline, weighted by the cross-perception
share, loses to the dog's lateral divergence across the half-width.

*Status.*
- *Observed:* the threshold N_c(w_att, d) at these three settings.
- *Not derived:* a formula for N_c.
- *Not proved:* the mechanism.
- The real flock (14 sheep) and the 14-sheep results of §10–§11 sit below N_c at every setting
  tested.

### 11.5 Deriving the fission threshold, with provenance

**Provenance.** Everything in this subsection and in §11.4 is simulation of the model, plus one
analytic formula. No real-trajectory data enter it; the real flock's 14 sheep appear only as a
reference size.

| | |
|---|---|
| Model | `shepherd/model.py` with the code's parameters: 4 attraction neighbours of K = 10 perceived (K varied where stated), noise 0.3, w_inertia 0.5, w_rep 2, w_dog 1, w_att 1.5, w_ali 1.3, r_rep 2 m, v_sheep 1, v_dog 1.5. The paper's Methods text gives n_att 5 and noise 0.5; those were not used here. Published rule: a sheep with no dog within r_dog stands still unless repelled |
| Initial condition | N sheep uniform in a disc of radius 0.9√N m, no pair closer than 1 m (rejection sampling); headings toward the target plus N(0, 0.1²) noise, normalized; the dog d m behind the disc centre |
| Dog | the published Strömbom dog in drive mode, collecting off (l_sep = 10⁶), standoff l_drive = d, r_dog = 20 m |
| Duration, runs | 300 steps; 30 runs per setting and size |
| Split | at step 300, two or more sheep outside the largest group linked by chains of distances ≤ 6 m |
| Calibration data | `results/size_sweep.json`, key `boundary`, case "published, dog 7.5 m" (`experiments/size_sweep.py`) |
| Validation data | `results/fission_threshold.json` (`experiments/fission_threshold.py`). This includes a rerun of the calibration setting with other seeds, and the cross share and lateral width measured in the same runs. A re-run of the script reproduced every split count exactly |
| Analysis | `results/fission_provenance.json` (`experiments/fission_provenance.py`; runs no simulations) |
| Figures | `results/figures/fission_curves.png` (full split curves), `fission_check.png` (approximation checks), `fission_threshold.png`, `size_sweep.png` |

**Model (analytic).** Take a polarized flock with the dog behind at distance d, split into left and
right halves with centroids at ±Y/2.
- The dog pushes each half outward with lateral component w_dog·(Y/2)/√(d² + (Y/2)²).
- Attraction pulls each half toward the other with mean lateral component w_att·λ·s. Here s is
  the share of a sheep's K perceived neighbours lying across the midline, and λ is the mean
  lateral component of a unit vector to one of them.
- Both terms sit in the same normalized heading sum, so normalization, heading memory and
  alignment set the rate of separation but not its sign.

Once a gap opens, each sheep's own half is nearer than the other. A half of n sheep then keeps
at least K + 1 − n cross neighbours, so s ≥ s_gap(N) = max(0, K + 1 − N/2)/K. Fission is
predicted where

    Φ(N) = w_dog (Y₀/2) / √(d² + (Y₀/2)²)  /  (w_att λ s_gap(N))  >  1

with:
- Y₀ = 1.6 σ⊥(N);
- σ⊥(N) = σ⊥(N_ref)·√(N/N_ref), where σ⊥(N_ref) is the median lateral standard deviation of
  driven flocks of N_ref = max(14, K + 1) sheep over steps 60–120 in unsplit runs, *measured in
  simulation* for each setting;
- λ = one constant, *fitted* to the calibration setting.

N_c is where Φ = 1.

**What 2K + 2 is.** At N = 2K + 2 two halves can each fill their neighbour lists internally.
Two sufficiently separated groups can then perceive nothing of each other, s_gap = 0, and the
model returns Φ = ∞. It is the first size at which that can happen. It is *not* a proven ceiling
on the actual fission threshold: a flock that never opens a gap need not split, and sizes above
2K + 2 with very strong attraction were not tested.

**How N_c and its uncertainty are estimated.**
- *Fit.* For each setting, the split counts k of n = 30 runs at each N are fitted by maximum
  likelihood with a logistic p(N) = 1/(1 + e^{−(N − N_c)/w}). N_c is its 50% point, with a 95%
  profile-likelihood interval (w profiled out, bounded below at 0.05 sheep, because
  0/30-then-30/30 data push it to 0).
- *Previous estimate.* The earlier figures used the linear interpolation between the two sizes
  bracketing 50%. It agrees within 0.3 sheep.
- *Why a single batch's interval is too narrow.* The calibration setting was run in three
  independent batches. Two of them disagree at N = 17 (13/30 vs 2/30 split, Fisher exact
  p = 0.002), so their 95% intervals, 17.16 [16.96, 17.36] and 17.62 [17.43, 17.80], do not
  overlap. Pooled, one logistic curve fits all 23 batch cells (deviance 15.5 on 21 df, no
  overdispersion overall), giving N_c = 17.38 [17.25, 17.51]. A practical uncertainty for one
  30-run batch is therefore about ±0.5 sheep: the batch-to-batch spread, not the single-batch
  interval.

**Split curves** (`fission_curves.png` shows all nine; simulated fractions with Wilson
intervals, the logistic fit, and the predicted N_c).

| Setting | role | measured N_c (logistic, 95% profile CI) | predicted N_c (λ = 0.795, as first published) | predicted, λ from pooled calibration (0.841) |
|---|---|---|---|---|
| published, dog 7.5 m | **calibration** (batch 1: 17.16 [16.96, 17.36]; rerun: 17.62 [17.43, 17.80]; pooled 17.38 [17.25, 17.51]) | – | 17.14 (fitted to batch 1) | 17.38 (fitted to the pool) |
| dog 12 m | validation | 20.17 [19.94, 20.39] | 19.07 | 19.22 |
| w_att 3 | validation | 22.17 [21.91, 22.45] | 21.03 | 21.08 |
| K = 7 | validation | 13.02 [12.93, 13.17] | 12.13 | 12.31 |
| K = 15 | validation | 24.29 [24.10, 24.49] | 24.25 | 24.63 |
| w_dog 0.5 | validation | 21.14 [20.88, 21.40] | 19.75 | 19.86 |
| w_ali 0 | validation | 17.20 [17.05, 17.38] | 17.00 | 17.25 |
| w_ali 3 | validation | 17.60 [17.39, 17.81] | 17.20 | 17.43 |
| dog 5 m | validation | 15.93 [15.63, 15.97] | 14.62 | 14.96 |

Over the eight validation settings, the first-published calibration gives a bias of −0.81 sheep
(RMS 0.95, worst 1.39). Calibrating on the pooled data gives −0.60 (RMS 0.81, worst 1.28).

**Check of the cross-neighbour approximation** (`fission_check.png`). The model uses the lower
bound s_gap. In the same runs, the actual cross share across the flock's lateral median line,
near N_c, is:
- **0.28–0.36 in intact flocks** (steps 10–30);
- **0.08–0.32 just before the first split**;
- **0–0.30 by the bound s_gap.**

So s_gap is a lower bound, sometimes far below (w_att 3 at N = 22: 0, against 0.08 just before
the split and 0.29 intact). Does it explain why the predictions run low? Re-predicting with the
measured share, recalibrated on the calibration setting only:

| Variant | λ | bias | RMS | worst |
|---|---|---|---|---|
| V0 s_gap (as published) | 0.795 | −0.81 | 0.95 | 1.39 |
| V2 measured share, intact flock | 0.574 | −1.17 | 1.60 | 3.25 |
| V3 measured share just before the split | 0.691 | −2.93 | 4.19 | 8.46 |
| V4 s_gap, lateral width measured at every N instead of √N-scaled | 0.922 | −0.78 | 0.89 | 1.31 |
| V5 s_gap, calibrated on the pooled calibration batches | 0.841 | −0.60 | 0.81 | 1.28 |

**Using the lower bound does not explain the underprediction. Using the measured share makes it
worse.** The intact flock's cross share changes too slowly with N to produce the threshold.
s_gap describes the state at the onset of a gap, which is what matters for fission. The √N
width scaling is also off: near N_c, flocks are 1.2–1.4× wider than scaled. But measuring the
width at every N leaves the bias unchanged, because λ absorbs the common factor. About a quarter
of the bias came from calibrating on the low batch (−0.81 → −0.60). **The remaining
underprediction of about 0.6 sheep is unexplained.** With eight settings and about ±0.5 sheep of
batch noise each, it is unlikely to be chance.

**What the evidence supports.**
- *Analytic:*
  - the bound s_gap;
  - the form of Φ;
  - its predictions that alignment does not move N_c (17.2 and 17.6 against 17.4 pooled), that
    K is the strongest lever (13.0 at K = 7, 24.3 at K = 15), and that N_c falls as the dog
    comes closer and rises as it moves away or pushes more weakly. All of these hold.
- *Fitted:* λ, once.
- *Measured in simulation:* σ⊥(N_ref) per setting, and every N_c.
- *Not established:*
  - Φ > 1 ⇒ fission is a mean-field balance tested on nine settings of this model;
  - the 0.6-sheep bias;
  - 2K + 2 as a ceiling.
- *How to read N < N_c.* Being below the measured 50% threshold is a **lower-risk regime, not a
  guarantee**. Below N_c splits still occur: 2/30 at N = 17 in the published rerun (13/30 in
  another batch), 1/30 at N = 16 with w_ali 3, and 4/30 at N = 21 with w_att 3.

## 12. Next

1. **Explain the 0.6-sheep underprediction** before adding settings. Two candidates stand out.
   One is the 300-step horizon: N_c is defined by splits by step 300, and slower fission near the
   threshold may need longer. The other is the use of the 50% point, where the model's balance
   may correspond to a different split probability. Then predict σ⊥ instead of measuring it.
2. **Attack (c) analytically.** Bound κ for lateral splits from the attraction geometry. Also
   explain why 14-sheep flocks recover from 12 m splits although κ ≈ 0 at the start.
3. **Widen the tested families:** three or more clusters, rotating flocks, the dog inside the
   flock, and longer horizons.
4. **Apply the sample identity to the real tracks:** compare the dispersion term with 1 − P²,
   and the radial term with the lateral/longitudinal split found here.
5. **Paused path work:** the S-curve test, and writing out the mean-square heading bound of §9.
