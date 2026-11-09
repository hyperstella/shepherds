# Di Lorenzo's convergence theorem and the Jadhav sheep: what survives

Purpose: decide whether the continuification convergence result of Di Lorenzo, Maffettone &
di Bernardo (arXiv:2411.04791, "A continuification-based control solution for large-scale
shepherding", 2024) says anything about real sheep and a real dog, given the assumptions of
Jadhav et al. (2024) about how they behave, and if not, what statement survives. Sections
1–8 redo his analysis for a *driven* flock with the kernels this project measured; sections
9–12 test the pieces in simulation; sections 13–15 test the three places where the first
pass was wrong or incomplete. Companion to `shepherd/feasibility.py` and to the Feasibility
section of REPORT.md.

## The answer

**His theorem, stated.** Targets on the periodic square [−π, π]² move by
dT = α Σ_j f(T − H_j) dt + √(2D) dB, with an odd soft-core push f (exponential, e^(−r/L),
off the torus) scaled by α = 1/(N_T + N_H). The herder density is computed once from the
goal (their Eqs. 9–12): the drift D∇ρ̄/ρ̄ is deconvolved numerically and shifted up by its
minimum. It is feedforward in the targets; a second layer (Theorem 1) steers the herders onto
it. Theorem 2: if ‖G‖∞ < 2 with G = ∇·(∇ρ̄/ρ̄), the target density's L² error decays as
exp(−D(2 − ‖G‖∞) t).

**Can it say anything about Jadhav's sheep and dog? Not as stated.** Every premise fails
under Jadhav's assumptions, each in a way that can be measured:

| His premise | What Jadhav's sheep and dog do instead | Consequence | § |
|---|---|---|---|
| a static goal | the flock is driven | harmless: in the co-moving frame the drive cancels, and his convergence holds at the same rate; his ‖G‖∞ < 2 is not even needed (his own example has ‖G‖∞ = 3.8 and still converges) | 2–5 |
| the herders supply all the drift | the sheep supply their own speed | a dog whose push must *be* the drive sits 3.0–3.3 m behind the centre for any push range from 2 to 8 m, under a metre behind the last sheep; real driving happens with the sheep 3–7 m from the dog. The drive is not a push | 6 |
| independent white noise | each sheep's own deviation is close to diffusive (persistence 0.5 s, Knudsen number 0.2), but most of what the first-order model leaves unexplained is shared by the whole flock and persistent | the flock's *shape* is a Fokker–Planck density; its *centroid* is a self-propelled body | 7, 13 |
| targets that ignore each other | cohesion | it reshapes the flock but cannot move it: pair forces cancel in the centroid, so the position is held by the dogs alone, with stiffness D/s² − k_coh | 5, 14 |
| a push that decays with distance (compressive beyond L) | a unit vector away from the dog within the notice radius (divergence 1/r > 0) | a static enclosure confines only through the notice-radius edge: under his ring the sheep settle on a circle of radius ρ*(r_dog), not at the goal, and start escaping at r_dog ≈ 1.3 × the ring radius (all of them by 1.5 ×) | 10, 15 |
| many weak herders, each scaled by 1/(N_T + N_H) | one dog of fixed strength | neither a discrete ring nor one dog touring it reproduces the designed density | 10, 11 |
| feedforward herders | one closed-loop dog that reads the flock | no hold: the published loop reaches the goal and overshoots by 20 m on average | 9 |

**What survives** is a statement about the centroid, not the density:

> With one Strömbom/Jadhav dog, if (i) the notice radius covers the flock from the drive
> point (r_dog > l_drive + the flock's depth; sharp: 9 m → 17% of runs reach, 10 m → 100%),
> (ii) attraction beats the dog's lateral spreading (w_att above 0.54–0.84 by the lateral
> inequality; swept, 0.6 → 25%, 0.9 → 78%, 1.2 → 100%), (iii) the flock does not split,
> and (iv) the dog is about as fast as the flock, then the centroid reaches the goal in
> finite time, linearly, t_reach ≈ t_dog + ρ₀/(v_sheep P∥) (to within one step on the
> median), and stays cohesive.
> The goal becomes an equilibrium only when the dog withdraws beyond the notice radius and
> the sheep stop (release: all 200 flocks freeze, after a median of 6 steps); for sheep that
> keep walking there is no hold.

His result survives *inside* this statement in two places. The flock's shape relative to
its moving centre is a Fokker–Planck density with D ≈ 0.12 m²/s, confined by cohesion and
the dog's curvature, and in the co-moving frame his estimate (in its weighted form)
applies to it. And his deconvolution, read kinetically, sets a useful single-dog standoff,
at the edge of the notice radius with no flanking. That dog drives the calibrated sheep as
well as a grid-searched one (§12).

What does not survive is exponential convergence of the density to a held goal. In
Jadhav's world the density is set by cohesion, the position by the dog, and the dog's only
confining handle is the edge of the sheep's notice disc.

## 1. The two microscopic pictures

Di Lorenzo-type, first order (what `feasibility.py` fits to the tracks):

    dx_i = [ Σ_j K_f(x_i − y_j) + Σ_k K_g(x_i − x_k) ] dt + √(2D) dW_i
    K_f(z) =  f(|z|) z/|z|,   f(r) = a e^(−r/L)      push away from a dog at y_j
    K_g(z) = −g(|z|) z/|z|                           toward a neighbour where g > 0

Mean field on the plane, with sheep density n and dog density m:

    ∂_t n + ∇·( n [ K_f*n_dogs + K_g*n ] ) = D Δn                                   (1)

Their paper has g = 0, a static goal, a periodic square, and forces scaled by 1/(N_T + N_H).

Jadhav/Strömbom: sheep move at a fixed speed when within r_dog of the dog and otherwise
stand; the heading relaxes toward a weighted sum of repulsion, attraction, alignment and
flight from the dog; the dog is one closed-loop agent that collects or drives. Measured
(Phase 4 of REPORT.md): metric attraction with a sign change at 1.25 m, alignment not
identifiable, a response to the dog's velocity, and a flock speed set from outside
(1.2–1.3 m/s regardless of dog distance from 0 to 8 m).

## 2. The translating goal in the co-moving frame

Goal profile n̄(ξ), ξ = x − ct, c the drive velocity. In the co-moving frame (1) is

    ∂_t n + ∇·( n [ K_f*m + K_g*n − c ] ) = D Δn                                    (2)

A steady driven flock satisfies ∇·( n̄ [v̄ − c] − D∇n̄ ) = 0. Take the zero-flux solution
n̄ (v̄ − c) = D∇n̄ (a nonzero divergence-free flux would be a steady circulation of sheep
inside the flock; this can be checked on the tracks). The dog-induced velocity the dogs
must produce wherever there are sheep is then

    u*(ξ) := (K_f*m̄)(ξ) = c + D ∇ln n̄(ξ) − (K_g*n̄)(ξ)                             (3)

Their Eq. 9 is the case c = 0, g = 0; the cohesion term is what `feasibility.py` added.
For the Gaussian goal used there, n̄ = N/(2πs²) e^(−|ξ|²/2s²) with
s = r_goal/√(−2 ln 0.05) ≈ 1.02 m for N = 14, the confinement term is D∇ln n̄ = −Dξ/s².

## 3. Potential form

Radial kernels are gradients: K_f = −∇Ψ with Ψ(r) = ∫_r^∞ f = aL e^(−r/L), and
K_g = −∇W with W(r) = ∫_0^r g (W rises where g attracts and is flat beyond the 3.6 m
cutoff). Up to a constant, (3) becomes a scalar matching problem on the support of n̄:

    (Ψ*m̄)(ξ) = Φ(ξ),      Φ(ξ) = D|ξ|²/(2s²) − c·ξ − (W*n̄)(ξ)                      (4)

Reading: the dogs must build, over the flock, a potential that is a bowl of curvature
D/s² (the diffusion balance), tilted by the drive (−c·ξ: higher behind the flock), and
flattened by cohesion (W*n̄ is itself a bowl wherever the flock lies within the attraction
range). How much it flattens is measured in §14: 4–28% of the curvature for 14 sheep at the
1 s diffusivity, depending on the assumed push range. The 30–45% fewer dogs of the REPORT
table is mostly where the solver put the ring, not this flattening. The sheep density is the
Gibbs state of the total potential,
n̄ ∝ exp(−(Φ + c·ξ + W*n̄)/D), which is the McKean–Vlasov self-consistency condition.

## 4. The obstruction, and the reformulation it forces

K_f is odd, so ∫K_f dz = 0 and the spatial mean of K_f*m vanishes for any dog density of
finite mass: exactly on a torus, and on the plane in the sense that K_f*m decays at
infinity. A constant c is therefore not in the range of m ↦ K_f*m. Equivalently, −c·ξ is
not periodic and Ψ*m is bounded. Di Lorenzo's deconvolution (their Eqs. 9–12 and Prop. 1:
a numerical left inverse of the discretized convolution, then a shift by the minimum) has no
translating-goal analogue: driving is necessarily a *local*
problem, matching u* only where the sheep are. The well-posed version is the convex program

    minimize over m ≥ 0:    ∫ n̄ |K_f*m − u*|² dξ  +  μ ∫ m dξ                        (5)

a non-negative least squares with an L¹ penalty, μ trading drift error against dog count.
Discretized on a grid or on rings this is what `minimum_dogs_local` does. Their
shift-by-min-H construction is the special case in which the residual can be made zero
everywhere. The weight n̄ is also what removed the periodic-box artefact (about 500 dogs
for 14 sheep) noted in REPORT.md.

## 5. Convergence in the co-moving frame

Suppose the dogs realize u* exactly (their Theorem 1 is about getting the herder density
there; take it as given). Then (2) is

    ∂_t n = ∇·( n ∇( D ln n + Φ̃ + W*n ) ),     Φ̃ := Φ + c·ξ = D|ξ|²/(2s²) − W*n̄     (6)

the Wasserstein gradient flow of the free energy

    F[n] = D ∫ n ln n + ∫ Φ̃ n + ½ ∫∫ W(ξ−ξ') n(ξ) n(ξ')                             (7)

and n̄ is its critical point. Consequences:

(a) **Translation drops out.** The tilt −c·ξ cancels against the frame change. Whatever
convergence holds for the static goal holds for the driven goal, at the same rate, as long
as the drift is realized.

(b) **g = 0, their setting.** Their herder density depends only on the goal (their text:
"feed-forward with respect to the targets' dynamics"), so given the herders the target
equation is *linear*, and not only after linearization: n∇ln(n/n̄) = n̄∇(n/n̄), so (6) is
exactly ∂_t n = D∇·(n̄ ∇(n/n̄)), the Ornstein–Uhlenbeck (Fokker–Planck) operator of n̄. It
is self-adjoint in L²(1/n̄), with eigenvalues −D(k₁+k₂)/s² on Hermite functions for the
Gaussian. The centroid error decays as e^(−Dt/s²) and the spread as e^(−2Dt/s²), globally
and with no condition on n̄. On the torus,
d/dt ∫e²/n̄ = −2D∫n̄|∇(e/n̄)|² ≤ −2Dλ(n̄)∫e²/n̄, with λ(n̄) the Poincaré constant of the
measure n̄, which is always positive. Since n̄ is bounded above and below on the torus, the
weighted and unweighted L² norms are equivalent, so their conclusion (global exponential
convergence in L²) holds unconditionally at rate 2Dλ(n̄).

Their rate D(2 − ‖G‖∞), with the condition ‖G‖∞ < 2 and G = ∇·(∇n̄/n̄) = Δ ln n̄, is what
the unweighted estimate gives: d/dt ½‖e‖² = −D‖∇e‖² − (D/2)∫G e², closed with the Poincaré
constant 1 of [−π, π]². The condition is an artefact of that estimate. **Their own example
violates it.** The product von Mises goal ln n̄ = k cos x₁ + k cos x₂ with k = 6/π (their
k = 3/r*, r* = π/2) has G = −k(cos x₁ + cos x₂) and ‖G‖∞ = 12/π ≈ 3.82 > 2. So Theorem 2
does not cover the simulation it accompanies, yet that simulation converges. The weighted
constant of this n̄ is λ = 1.59 (computed numerically), so the weighted error decays at
2Dλ = 0.032/s for their D = 0.01, a time constant of about 60 s for the norm against
their 200 s run. The condition is also what would bite a real flock: transplanted to a
periodic box the size of the field, it reads s > (field side)/(2π) ≈ 13 m for an 80 m field,
which a 1 m flock fails by an order of magnitude, although it converges fine.

(c) **g ≠ 0: cohesion holds the shape, not the position.** Hess Φ̃ + Hess(W*n̄) = (D/s²) I
holds by construction, but the curvature does not act on all modes alike. Pair forces are
antisymmetric, so they cancel in the centroid: d⟨ξ⟩/dt = ⟨u_dog⟩ + noise exactly, whatever
g is. A rigidly displaced flock is pulled back by the dogs alone, at the stiffness

    k_dog = −½ ⟨∇·u_dog⟩_n̄ = −⟨ξ·u_dog⟩_n̄ / (2s²) = D/s² − k_coh,     k_coh := −½ ⟨∇·(K_g*n̄)⟩_n̄     (10)

(the middle form is integration by parts against the Gaussian). Designing the dogs for a
cohesive flock therefore buys fewer dogs at the price of a softer centroid. The shape modes
(perturbations with zero mean displacement) feel cohesion and dogs together. So the linear
rate of the translation mode is k_dog, not D/s². In the first pass this item claimed the
rate was unchanged; that is wrong for exactly the mode driving cares about. §14 measures
k_coh/(D/s²). Globally, the standard sufficient condition for exponential convergence of (7)
(λ-convexity of Φ̃ and convexity of W, Carrillo–McCann–Villani) fails where g decreases,
i.e. at the 3.6 m cutoff where W goes flat. A sheep further than the cutoff from the flock
is not pulled back, and only *local* exponential convergence survives. This is the
mathematical content of "a sheep left behind stands still" and of the Strömbom
collect/drive switch: driving maintains the tilted bowl near the flock, collecting restores
confinement for a straggler outside the cohesion range.

(d) **Zero dogs: self-cohering, not self-confining.** With m = 0, (6) has a clustered
steady state iff the McKean–Vlasov equation with W alone does, which needs attraction to
beat diffusion at the flock's scale. In the repo's scaling (s ∝ √N at constant density) the
required confinement drift at the edge, about 2.45·D/s, falls like 1/√N, while the cohesion
drift, set by a fixed kernel and density, does not. So k_coh ≥ D/s² for large enough
flocks (the † entries of the REPORT table; with the individual diffusivity of §13, §14 puts
the crossing between 14 and 30 sheep at L = 5–8 m and beyond 50 at 3 m). But by (10) such a flock holds its shape only: with
k_dog = 0 its centroid has no restoring force and wanders, at D/N for independent noise
and, in the tracks, far faster (§13). Whether the pair kernel really superposes over many
neighbours is the empirical caveat REPORT.md already records.

## 6. What the first-order premise implies for a single trailing dog

Put one dog on the drive axis at distance d behind the flock centre and expand its push
over the flock (s ≪ L): along the axis, push(ξ) = a e^(−d/L) (1 − ξ/L + …). Matching (3)
with g ignored:

    constant:   a e^(−d/L) = c       ⇒   d  = L ln(a/c)
    slope:      c/L = D/s²           ⇒   L* = c s²/D

and the Hessian of the dog potential at the flock, with a e^(−d/L) = c,

    Hess(Ψ*m) = (c/L) r̂r̂ − (c/d) (I − r̂r̂)

The longitudinal curvature c/L confines and fixes L*. The transverse curvature −c/d is
anti-confining: a single static dog spreads the flock sideways at rate c/d, so cohesion
must supply D/s² + c/d transversally, or the dog must sweep an arc (flanking) to average
the transverse term out. With cohesion included, the slope requirement is smaller and L*
larger. With the stored fits (c = 1.2 m/s, s = 1.02 m, a(L) from `results/feasibility.json`):

| D used | L* = cs²/D | a(L*) | d = L ln(a/c) | gap to the last sheep, d − 2.45 s | c/d |
|---|---|---|---|---|---|
| D(1 s) = 0.25 m²/s | 5.0 m | 2.3 m/s | 3.3 m | 0.8 m | 0.36 /s |
| D(4 s) = 0.68 m²/s | 1.8 m | ≈5.5 m/s | 2.8 m | 0.3 m | 0.43 /s |

The constant-term condition alone fixes d, and it barely depends on the unidentified range:
with a(L) fitted at each L, d = L ln(a(L)/c) is 3.0 m at L = 2 m (a = 5.4), 3.3 m at 3 m
(3.57), 3.3 m at 5 m (2.3) and 3.2 m at 8 m (1.8). So whatever D is used for the slope
(§13 finds the D that spreads the shape is 0.12 m²/s, which puts L* near 11 m, beyond the
fitted range), the push that supplies c sits about 3.2 m behind the centre.
REPORT.md: "All driving happens 3–7 m from the dog." The first-order model wants the dog
under a metre behind the last sheep, because in that model the dog's force *is* the
flock's speed: c has to come from a e^(−d/L). Real sheep supply their own speed (Phase 4:
1.2–1.3 m/s at every dog distance) and the dog sets direction. So the translation term in
(3) must not be modelled as a push. This is where Leg A leaves Di Lorenzo's class of
models.

## 7. The kinetic repair

State p(ξ, θ, t) with heading θ and fixed speed v. Co-moving kinetic equation:

    ∂_t p + (v e(θ) − c)·∇_ξ p + ∂_θ( ω[p, m] p ) = D_r ∂²_θ p                       (8)

with ω the turning rate toward the heading rule's target (repulsion, attraction, flight
from the dog) and D_r = 1/τ for heading persistence τ. The moments n = ∫p dθ and
nP = ∫ e(θ) p dθ give

    ∂_t n + ∇·( n (vP − c) ) = 0                                                   (9)

A driven steady flock needs vP = c on the support of n̄: a polarization condition, not a
force balance. With v ≈ |c| this is P ≈ 1, which the real flock shows (P ≈ 0.95). The
dog's role in (8) is a heading bias (its term in ω), so a dog anywhere inside the notice
radius can drive at the sheep's own speed; the standoff is then set by the heuristic and
r_dog, not by force balance, consistent with 3–7 m. The analogue of the confinement term
D∇ln n̄ is the small inward heading bias needed at the flock's edge to balance heading
noise. The analogue of (5): find m whose heading-bias field produces P = c/v across the
flock.

**Diffusive limit.** (8) reduces to (1) with D = v_⊥² τ/2 only if τ ≪ s²/D, i.e. headings
decorrelate faster than the density relaxes. The windowed D(w) in `results/feasibility.json`
is exactly the signature to test this with: for an Ornstein–Uhlenbeck velocity residual
with persistence τ,

    D(w) = D_∞ [ 1 − (τ/w)(1 − e^(−w/τ)) ]

The three stored values (0.248, 0.434, 0.679 m²/s at w = 1, 2, 4 s) are fitted to within
1% by τ = 2.2 s and D_∞ = 1.26 m²/s. Read naively, s²/D_∞ ≈ 0.8 s < τ: the density would
relax faster than the headings forget, and the Fokker–Planck density would be the wrong
state variable. The residual speed √(2D_∞/τ) ≈ 1.1 m/s, as large as the flock speed,
already suggested that this "noise" is mostly coherent, unmodelled velocity. §13 tests
that directly by splitting the residual into the part all sheep share and each sheep's
own, in the flock frame. The two behave differently:

- **Each sheep's own deviation is diffusive.** τ ≈ 0.5 s along and across the heading,
  D ≈ 0.14 and 0.09 m²/s, persistence length 0.22–0.27 m against a flock radius of about
  0.9–1.0 m. That is a Knudsen number of 0.25, and in every one of the 20 drives at least
  20 s long it is below 0.4. The shape's relaxation time s²/D ≈ 7–9 s is 15 times τ. The
  flock's shape in the co-moving frame *is* a Fokker–Planck density, with D ≈ 0.12 m²/s,
  half the 1 s value used so far.
- **What the flock shares is not.** It carries most of the residual variance: 1.2–2.1
  times each sheep's own at 0.2 s, 7–10 times at 8 s. It is persistent: the centroid's
  residual along the heading has D ≈ 0.95 m²/s with τ ≈ 1.6 s (persistence length 1.2 m),
  and across the heading it is still ballistic at 8 s. Independent noise would give the
  centroid D/N ≈ 0.01 m²/s; it moves about a hundred times more.

So the reduction of (8) the data support is one level up from the one first proposed. The
kinetic variable is collective: a self-propelled centroid with a persistent heading, which
the dog steers. That centroid carries a shape that is diffusive at the flock's scale. The
convergence statement is still two-tier, but the tiers are different. The shape relaxes to
the cohesion-and-dog bowl at rates of order D/s² ≈ 0.13/s, and there Di Lorenzo's
machinery applies. The centroid's heading follows the dog on the 1–2 s scale of its
persistence, and its position is a pursuit problem, which §9 treats.

## 8. Computations this fixes

1. Fit τ and (D_∥, D_⊥) in the flock frame from the residuals, per drive. This decides how
   far the first-order theory can be pushed at all. *Done in §13: the shape can be pushed
   that far, the centroid cannot.*
2. Solve (5) for the translating Gaussian with measured f, g and D_⊥; report dog count,
   residual, and the shape of m̄ (expected: an arc behind the flock, densest on the axis).
3. Check the single-dog relations (L*, d, c/d) against the measured standoff and the dog's
   lateral-sweep amplitude; the transverse anti-confinement c/d predicts a minimum flanking
   amplitude for a given cohesion.
4. Compare m̄ from step 2 with the dog's time-averaged occupancy in the flock frame (Leg C).
5. Kinetic version of (5): with the heading rule's dog term as the bias, compute P over the
   flock for a dog at distance d and find the range of d that keeps vP = c; compare with
   r_dog and the tracks.

## 9. Closing the loop: what survives with the Jadhav sheep and dog (simulated)

Sections 2–7 kept Di Lorenzo's premise that the herders are a density that never looks at
the targets. The Jadhav model has one dog that reads the flock every step, and sheep that
turn toward a unit vector away from it, move at a fixed speed, and stand still when no dog
is within r_dog. This section asks what convergence statement survives, and tests each
piece on the published model. Script: `experiments/closed_loop_convergence.py`; numbers in
`results/closed_loop.json`; figure `results/figures/closed_loop.png`. Setup: published
parameters (N = 14, v_sheep = 1, v_dog = 1.5, r_dog = 12 m, w_att = 1.5, w_dog = 1,
noise 0.3, K = 10, n_att = 4, l_drive = 7.5 m, l_sep = 11.6 m), flock started 100 m from
the target, 300 steps, 200 runs; sweeps 100 runs at 400 steps. "Reached" means the centroid
came within 5 m.

### 9.1 What changes structurally

- **One dog cannot enclose.** The unit vector away from a point has divergence 1/r, so a
  single dog spreads the flock. Di Lorenzo's herders confine only by surrounding the
  targets. Here confinement must come from attraction; the dog supplies translation.
- **Freezing removes the hold problem and adds a coverage obligation.** A sheep outside
  r_dog is a fixed point, so the goal is an equilibrium set once the dog withdraws. The
  Fokker–Planck hold against diffusion, which needs herders forever, disappears. In
  exchange the dog's notice disc must sweep every sheep, or a straggler never returns.
- **The closed loop breaks linearity.** The dog's position is a function of the flock
  state, so the sheep equation is no longer linear given the herders, and the L² energy
  of Theorem 2 has no guaranteed sign. What can replace it is a cascade: the dog reaches
  its drive point regardless of the flock, the flock reaches the target given the dog is
  there, the flock stays cohesive meanwhile, and the collect/drive switching terminates.

### 9.2 The pieces, measured

**Pursuit (holds).** The drive point moves with the centroid, at 0.48 m/step during the
approach, and the dog closes at 0.95 m/step against the predicted v_dog − v_flock =
1.02. The dog is at the drive point by step 21 (median) in every run, then hovers within
0.8 m (median; 90th percentile 1.6 m). It spends 9% of the pre-arrival steps in the slow
mode, i.e. inside the flock, since the heuristic has no obstacle avoidance.

**Reaching (holds, and is linear).** With the dog behind the flock, the centroid's
distance to the target falls at −0.837 m/step against the predicted −v_sheep·P∥ = −0.833
(P∥ = 0.83 on drive steps). An exponential fit has 25 times the residual of a linear one.
The finite-time prediction t_reach = t_dog + (ρ(t_dog) − 5)/(v_sheep P∥) gives a median
of 133 steps against 134 observed. **The published loop is reach-only.** After reaching,
the drive point flips to the far side, the dog goes through the flock (23% slow steps) and
the centroid overshoots: mean distance 20 m, 90th percentile 37 m, about three axis crossings
in the remaining steps. There is no hold; convergence in Di Lorenzo's sense does not exist in the
closed loop, and what the real system does instead is release.

**Cohesion inequality (holds in aggregate, predicts thresholds, not steps).** For the
lateral-most sheep on drive steps (24 000 step-samples): lateral offset 4.9 m, dog 8.0 m
behind, so the dog's outward unit component is 0.52; κ (inward component of the
attraction unit vector) has median 0.98, 5th percentile 0.81, and is never negative. The
inequality w_att κ > w_dog·0.52 + noise needs κ > 0.55 and holds on 99.8% of steps. But
the per-step margin has correlation −0.01 with the sheep's realized lateral step; the
edge sheep moves inward on 63% of steps whatever the margin. Inertia, repulsion,
alignment and noise decide individual steps. The inequality is a statement about means,
and as a threshold predictor it works: it puts the attraction weight needed at 0.54 (no
noise) to 0.84 (worst-case noise), and the sweep shows success 25% at w_att = 0.6, 78% at
0.9, 100% from 1.2.

**Switching (untested in the published setting; in expectation only elsewhere).** With
the published weights the maximum spread never reaches l_sep, so no collect ever happens
in 200 runs. With w_att = 0.9 collects occur (6.7 episodes per run, median dwell 7 steps,
25% of steps). Then, with r_dog = 12, the centroid still gains 0.15 m/step toward the
target during collects and the spread shrinks 0.09 m/step, so V = ρ + λ·maxspread falls
in both modes on average (74% of collect steps for λ = 1). With r_dog = 20 the main flock
keeps seeing the dog during a collect and the centroid loses 0.20 m/step (ΔV ≤ 0 on only
40% of collect steps). The freezing mechanism of section 9.1 is real at the level of mean
steps. Yet the larger radius wins overall (93% vs 84% reached, 143 vs 200 steps), because
it makes collects rare (1.2 vs 6.7 episodes): keeping every sheep engaged prevents
stragglers in the first place. No pathwise Lyapunov function exists (ΔV ≤ 0 on about 77%
of all steps for any λ); the statement that can hold is in expectation, a supermartingale
argument with a dwell-time condition.

**Release (holds).** Taking each run at the moment of reaching and moving the dog 300 m
away, every run freezes: the last movement is at step 6 (median; 90th percentile 9), and
it is repulsion only, since the published flock is packed below r_rep. Mean displacement
1.7 m, maximum 4.2 m. The terminal state is an equilibrium after a short relaxation.

### 9.3 The four conditions, swept

| Condition | Claimed | Found |
|---|---|---|
| 1. Speed advantage v_dog > v_flock | Needed for pursuit | **Not needed for reaching.** v_dog = 0.8 < v_flock (0.83) still reaches in 70% of runs: sheep beyond r_dog freeze, the flock slows to P∥ = 0.57 and waits (every sheep sees the dog on 4.5% of steps). v_dog = 0.6: 19%, half the flocks split. Speed sets the time to reach (130 → 158 → 273 steps) and the splitting risk, and matters for repositioning, not for driving. |
| 2. Coverage r_dog > l_drive + front depth (+ hover) | Threshold 7.5 + 2.7 + 0.8 = 11.0 m | **Sharpest condition.** 9 m: 17% reached; 10 m: 100% but slowly (185 steps, every sheep sees the dog on 7% of steps); 11 m: 100% at full speed. Threshold between 9 and 10 m, a little below the prediction: a front sheep tolerates short losses of sight because repulsion from behind pushes it. |
| 3. Attraction weight (lateral inequality) | w_att > 0.54–0.84 | Transition 0.6 → 25%, 0.9 → 78%, 1.2 → 100%, bracketing the prediction. Above 1.2 the flock only gets tighter (final max spread 8.2 → 2.6 m). |
| 4. Non-splitting, "κ bounded below" | κ of the edge sheep | **Untested at the published parameters.** Flocks of 30 or more fail by coverage, not splitting: l_drive = 2√N grows past r_dog − depth, every sheep sees the dog on 0.3% of steps and the split rate stays at 1%, while κ's 5th percentile stays 0.64–0.69. Whether larger flocks split once the notice radius is wide enough to cover them is left for later. |

A side result: the published parameterization (r_dog = 12 m, l_drive = 2√N) can only
drive flocks of about 14 to 20 sheep for the trivial reason that the drive point leaves
the notice radius.

### 9.4 The theorem the simulation leaves standing

Under (i) coverage, r_dog larger than the drive distance plus the flock's depth, which
is sharp; (ii) an attraction weight above the lateral threshold of section 9.2, which the
inequality locates to within a factor of about 1.5; (iii) the flock not splitting under the dog's divergent push, which the sweeps at the
published parameters never exercised; and (iv) a dog at least about as fast as the flock, the
centroid reaches the target in finite time, t_reach ≈ t_dog + ρ₀/(v_sheep P∥), to within
a step, with the maximum spread below l_sep throughout. On release the flock freezes
within about ten steps. Without release the published loop overshoots and oscillates: the
closed loop has no stable hold, so "convergence to the goal" means reaching, not staying.

Compared with Di Lorenzo's result this is conditional and local rather than global,
finite-time rather than exponential, and in expectation rather than pathwise where
switching is involved. It holds for one dog and fourteen sheep rather than a continuum,
and its terminal state holds itself. Exponential rates survive only inside the pieces:
the dog's pursuit and the heading relaxation. The measured response to the dog's velocity
(Phase 4) is not in these simulations; the Phase 3d result that it improves steerability
suggests it acts as phase lead in the lateral loop, which remains to be tested the same way.

## 10. Di Lorenzo's target density inside the simulation: which components give the convergence

Script: `experiments/target_density.py`; numbers in `results/target_density.json`; figure
`results/figures/target_density.png`; three of its runs replay in `viz/closed_loop_replay.html`.

**Setup.** Their goal is a density. Here it is a Gaussian with 95% of the sheep within 5 m
(s = 2.04 m), the plane analogue of their von Mises profile; 5 m rather than the real flock's
2.5 m because the published model's 2 m repulsion radius cannot pack 14 sheep tighter. Their
herder arrangement is what the deconvolution of section 4 gives for this goal, computed with
`minimum_dogs_local` for the measured push a = 3.57 m/s, L = 3 m and D = 0.25 m²/s: a ring at
10.9 m carrying a total dog mass of 4.8, with a 2% drift error. The ladder uses 10 real dogs
on that ring. Convergence is measured their way, the relative L² error of a kernel density
estimate against n̄, and in the finite-N ways of Leg B: the fraction of sheep inside 5 m
(their confinement metric), the RMS distance to the goal against the target's √2·s = 2.89 m,
and an exponential rate fitted to the RMS excess. The L² floor for 14 samples is 0.54, so
their metric barely resolves anything at this flock size. 14 sheep, 50 runs, 200 s, two
starts: scattered uniformly inside the ring, and a tight cluster 7 m off the goal.

**The ladder**, one component per rung, from their targets to the Jadhav sheep under the
same ring (scattered start):

| Rung | rate (1/s) | excess at 10% by | inside 5 m | RMS (m) | L² error |
|---|---|---|---|---|---|
| L0 their targets: overdamped, exponential push summed over dogs, diffusion | 0.46 | 7 s | 1.00 | 2.03 | 0.65 |
| L1 + fixed speed, heading noise instead of diffusion | 0.12 | 20 s | 0.96 | 2.99 | 0.54 (floor) |
| L2 + Jadhav's push: unit vector away from each dog within r_dog | 0.50 | 4 s | 1.00 | 1.38 | 0.92 |
| L3 + freeze when no dog is within r_dog | 0.50 | 4 s | 1.00 | 1.38 | 0.92 |
| L4 + heading inertia | 0.49 | 4 s | 1.00 | 1.34 | 0.95 |
| L5 + social terms: the Jadhav sheep | 0.70 | 5 s | 0.99 | 2.31 | 0.61 |
| L5b the Jadhav sheep without the freeze | 0.70 | 5 s | 0.99 | 2.31 | 0.61 |
| L6 the published model with its own dog, started at the goal, no ring | none | never | 0.15 | 21.8 | 1.13 |

**Dog count.** L0 with the continuum mass 4.8 split over M dogs: M = 5 leaks (81% inside,
RMS 10.8 m), 10 gives 94%, and 20 or 40 give exactly the designed 95% with RMS 3.2 m and a
rate of 0.09/s against the Ornstein–Uhlenbeck 2D/s² = 0.12/s. L0 with M full dogs: 5 leaks
the same, 10 holds all with RMS 2.3 m, 20 and 40 over-confine to 1.5 and 1.0 m. The Jadhav
sheep hold 99% with RMS 2.3–2.4 m at every M from 5 to 40.

**What makes their convergence work, as the ladder shows it.**

1. **An additive push that decays with distance.** Additive: the drift is linear in the
   herder density, so the deconvolution can shape it into D∇ln n̄ and the continuum limit
   reproduces the designed coverage exactly. Decaying: a radial push f(r) r̂ has divergence
   f′(r) + f(r)/r, which is negative (compressive) where f falls faster than 1/r. For his
   exponential that is beyond r = L, and his ring sits at 10.9 m against L = 3 m, so the
   field at the goal is a sink. Jadhav's unit vector has neither property. It saturates (the
   sum is normalized), and its divergence is 1/r > 0 everywhere inside the notice disc, so
   with every dog in view the summed push points *outward* everywhere inside the ring. (It is
   the gradient of the summed distances to the dogs, which is smallest at the centre.) L2's
   fast collapse to an RMS of 1.4 m is therefore not compression. At r_dog = 12 m, a sheep
   1.1–1.2 m off centre loses sight of the dogs across the ring, and the push turns inward
   there. That circle is where the sheep settle (§15). The herder density has lost its handle
   on the shape; the notice radius has it instead.
2. **Many weak herders.** Few dogs leave gaps the size of the kernel range and the cloud
   leaks; full-strength dogs in numbers over-confine. Only their mean-field scaling of the
   push by 1/(N_T + N_H) lets the count grow while the total drift stays as designed. Real
   dogs do not scale down, so a discrete enclosure of dogs cannot be gap-free and correctly
   weighted at once.
3. **Ergodic motion, not necessarily overdamped.** The fixed-speed random walk of L1 still
   converges, four times slower, and to the right shape (L² at the floor). What the
   overdamped Fokker–Planck form buys is the rate's dependence on the bowl and the proof,
   not convergence itself.
4. **Static herders that surround the goal.** Every rung converges under the ring at the
   published r_dog = 12 m. The published model's own dog, started from the converged state,
   disperses it to 15% inside and an RMS of 22 m. For the Jadhav rungs, though, the property
   lives in the arrangement *and* the notice radius. 12 m happens to sit in a window just
   above the ring's 10.9 m; from 16 m on, every sheep of L2, L3 and L5 escapes (§15). His
   own exponential targets (L0, L1) do not depend on r_dog.
5. **No freeze inside the enclosure.** Freezing changes nothing while the notice radius
   covers the interior (L3 = L2, L5b = L5). It matters for release, section 9, not here.
6. **No social terms.** They are not needed to confine, but they fix the density: the
   Jadhav flock's RMS is 2.3 m for 5 dogs and for 40. The ring decides where the flock is,
   cohesion decides its shape. In the Jadhav world the density is not a control variable;
   the centroid is.

Items 1 and 2 are what the Jadhav sheep break; 3 survives; 4 survives only through the edge
of the notice disc. The convergence statement that carries over to real sheep is therefore
about the centroid. A static surrounding arrangement confines it in finite time when the
notice radius sits in the right window. The flock's density is cohesion's own. And one
closed-loop dog cannot hold what a ring holds.

## 11. Two single-dog ways to hold, tested

Script: `experiments/hold_strategies.py`; numbers in `results/hold_strategies.json`; figure
`results/figures/hold_strategies.png`; three runs replay in `viz/closed_loop_replay.html`.

### A. Reach-and-release

`model.Params.hold_radius` adds a hold mode to the published dog, off by default and
leaving the published model byte-identical (tests in `tests/test_model_ext.py`). Once the
centroid is within 5 m of the target the dog withdraws to 15 m from the centroid, which is
r_dog + 3, and waits with no heading noise; it re-engages when the centroid drifts beyond
10 m. Closed-loop task of section 9: start 100 m away, 450 steps, 200 runs; the published
sheep, which stop when no dog is in range, and the paper's modified sheep
(`social_always`), which keep moving.

| Sheep | Dog | post-reach time within 5 m | centroid distance, mean / 90th pct | sheep moving | hold episodes, re-engagements |
|---|---|---|---|---|---|
| published | published | 11% | 21.1 / 37.9 m | 99% | 0, 0 |
| published | hold rule | **100%** | **2.2 / 2.8 m** | **0%** | 1, 0 |
| always moving | published | 11% | 20.5 / 36.8 m | 100% | 0, 0 |
| always moving | hold rule | 34% | 11.1 / 27.5 m | 100% | 2.25, 1.25 |

For a flock that stops, the rule is an exact hold: the sheep freeze inside the goal and the
dog waits outside their notice radius. This is the equilibrium of section 9.1 made into a
controller, and it is the one thing the published loop lacked. For a flock that keeps
moving at full speed it is not a hold: the polarized flock walks off the moment the dog is
out of range, passes 10 m before a dog 15 m away on an arbitrary side can get behind it,
and the loop becomes release and recapture with excursions of tens of metres. It still
halves the mean excursion. Real released sheep graze at a fraction of this speed, so the
truth sits between the two rows. Untested improvements: wait behind the flock on the target
line rather than radially, and a narrower hysteresis band.

### B. One dog touring Di Lorenzo's ring

The herder density of section 10 is a ring at 10.9 m, uniform by symmetry, so a dog running
round it at constant speed is its time-multiplexed version. The Jadhav sheep with the dog's
path replayed, and Di Lorenzo's targets with the same single full-strength dog; 200 s,
50 runs, scattered start (the goal start gives the same picture). The static ring of 10
dogs holds 99% of the sheep and 100% of the targets.

| tour speed | lap | sheep inside / escaped / seeing the dog | targets inside / escaped |
|---|---|---|---|
| 0 (standing on the ring) | — | 0.32 / 0.00 / 0.00 | 0.07 / 0.54 |
| 1.5 m/s | 46 s | **0.00 / 1.00** / 0.04 | 0.23 / 0.52 |
| 3 m/s | 23 s | 0.37 / 0.13 / 0.61 | 0.30 / 0.41 |
| 6 m/s | 11 s | 0.45 / 0.06 / 0.59 | 0.34 / 0.34 |
| 12 m/s | 6 s | 0.25 / 0.17 / 0.49 | 0.33 / 0.41 |

- **The targets leak** at every speed, 30–54% beyond the ring, because the dog covers one
  point of the wall at a time and a diffusing cloud needs all of it. Averaging the herder
  field over a lap would need the lap to be short against the leakage time, and even a lap
  of 6 s, a third of the ring per step, is not.
- **The dog herds the sheep out.** At 1.5 m/s every sheep that sees the dog flees at full
  speed away from it, toward the unguarded far side of the ring, and nothing stops it there:
  all of them are outside within 40 s. At higher speeds a sheep sees a different point of
  the ring each step and takes a full 1 m step away from it, a random walk with a weak
  inward bias, 25–45% inside. This is component 1 of section 10 again: averaging a herder
  field over time needs small additive responses, and the sheep's response is a full step
  per sighting.

The ring tour, the Leg C idea of section 1, is refuted in this feedforward form for both
models. The many weak herders of Di Lorenzo's design cannot be replaced by one fast one,
for the targets because of leakage and for the sheep because of saturation. What one dog
can do at the goal is reach-and-release, a feedback rule with a stop.

## 12. A dog designed from the deconvolution, in the dog-control comparison

Script: `experiments/designed_dog.py`; design code in `shepherd/dog_design.py` (tests in
`tests/test_dog_design.py`); numbers in `results/designed_dog.json`; figure
`results/figures/designed_dog.png`. The task, starts, seeds and scores are those of the
existing comparison in `experiments/dog_control.py`: drive the flock 60 m to a target
within 300 s, 64 runs per setting. The baselines are the published Strömbom dog, the
data-derived collie, and the heuristic tuned on each flock and size over that script's
16-point grid on separate seeds. The flocks are the published sheep and the calibrated
sheep with the learned dog and pair kernels (M5τ, blend 2 of the evaluation), at 14 and
30 head.

**Design inputs.** The measured kernels (push 3.57 m/s at L = 3 m, pair kernel cut at
3.6 m, D = 0.248 m²/s), the goal density of the feasibility analysis (95% within
2.5·√(N/14) m), the flock speed of the comparison (1.0 m/s published, 1.29 m/s
calibrated), and the sheep's notice radius (12 m published, 6.7 m median of the
calibrated posterior).

**Two readings, two standoffs.**

- *First-order, his own.* The translating-goal deconvolution of section 4 with the
  flock's own drift. For 14 sheep it returns 7.8 dogs with 53% of their mass behind the
  flock and the mass-centre 0.3 m inside it: a ring, not a drive point, because the
  confinement drift at the flock's edge (0.6 m/s) is comparable to the drive (1 m/s). Only
  at 30 sheep does the arc become one-sided (92% behind, centre 4.1 m back). For one real
  dog, the best single position leaves 21–34% of the required drift unmatched and sits
  4.5 m behind the published flock, 3.5 m behind the calibrated one.
- *Direction-matched, the kinetic reading.* Score a position by the n̄-weighted mismatch
  between the unit vector away from the dog and the direction of the required drift. The
  mismatch falls monotonically with distance, so the notice radius binds: 9.0 m for the
  published sheep, 3.5 m for the calibrated ones (2.5 m at 30 head). The real collie's
  median working distance is 4.2 m. A lateral offset raises the mismatch at every offset,
  from 0.12 on the axis to 0.65 at 9 m aside, so the design does not flank.

**Results** (success rate / median time to target in seconds):

| Flock, N | published dog | data-derived | tuned on this flock | designed, first-order | designed, direction-matched |
|---|---|---|---|---|---|
| published sheep, 14 | 1.00 / 68 | 1.00 / 77 | 1.00 / 69 | 1.00 / 70 | 1.00 / 69 |
| published sheep, 30 | 0.00 | 0.06 / 252 | 0.14 / 265 | 0.00 | 0.02 / 283 |
| calibrated M5τ, 14 | 0.03 / 122 | 0.70 / 136 | **1.00 / 48** | **0.98 / 51** | **0.98 / 51** |
| calibrated M5τ, 30 | 0.00 | 0.03 / 253 | **0.95 / 51** | 0.89 / 56 | **0.94 / 55** |

Standoffs behind the centroid: published dog 7.5 and 11 m; data-derived 4.2 and 6.2 m
with 2.5 m of flanking; tuned 6 m (published, 14), 8.8 m with flanking (published, 30),
4.0 m and 3.7 m (calibrated); designed as above. The designs use the published dog's
speed, 1.5 m/s, and instant turning; the tuned settings on the calibrated flock use 3 m/s.

**Reading.**

1. **On the calibrated sheep the design reaches, without any search, what the grid
   search finds:** 98% against the tuned 100% at 14 head, 94% against 95% at 30, a few
   seconds slower at half the dog speed. The published dog (3%, 0%) and the data-derived
   collie (70%, 3%) fail on the same flock. The decisive quantity is the standoff relative
   to the notice radius. The tuning chose 4.0 and 3.7 m, the design 3.5 and 2.5 m, the
   coverage limits are 3.7 and 2.6 m; the published dog works at 7.5 and 11 m, beyond
   the 6.7 m the calibrated sheep notice, and they stand still. The data-derived collie is
   close enough on average but its 2.5 m flanking swings it out of sight (every sheep sees
   it on 31% of steps, against 84–89% for the tuned and designed dogs).
2. **On the published sheep at 14 head every dog succeeds** and the designs match the
   published dog's time and cohesion. **At 30 head every dog fails**, including the
   designs: that flock spreads to a mean cohesion of 9–15 m, far beyond the 3.7 m goal
   disc the design assumes, so no single standoff keeps it in sight. The design is only as
   good as its goal density; it designs for a flock as tight as the real one, which the
   calibrated sheep are (cohesion 1.4–1.6 m) and the published sheep at 30 are not.
3. **The two readings agree on the calibrated flock for different reasons.** The
   first-order one lands at 3.5 m because the measured push range puts the best single dog
   there; the kinetic one because coverage ends there. On the published sheep they
   disagree, 4.5 against 9 m, and both work, since that flock's 12 m notice radius leaves
   room.
4. **What Di Lorenzo's design contributes, operationally:** the standoff set by the
   notice radius, no flanking, and the count. The arc says a single dog is 3 to 7 dogs
   short of the confinement he would design; cohesion supplies the rest, which is why the
   calibrated flock under the designed dog is the tightest of the comparison.

Caveats: the calibrated flock at 30 head extrapolates a 14-sheep calibration; the designs
were not given the tuned dog's higher speed; and the first-order reading's single-dog
position is a least-squares compromise, not something that model endorses, since its own
answer is several dogs.

## 13. The residual: a diffusive shape and a self-propelled centroid

Script: `experiments/residual_persistence.py`; numbers in `results/residual_persistence.json`;
figure `results/figures/residual_persistence.png`.

**Method.** The first-order model of `shepherd/feasibility.py` (dog push and pair kernel,
fitted to all 29 drives) predicts each sheep's drift. Its residual displacement over a window
w is split into the flock-mean part, which all sheep share, and each sheep's deviation from
it. Both are resolved along and across the flock's heading at the start of the window. Each
apparent diffusivity D(w) = var/(2w) is fitted with an Ornstein–Uhlenbeck velocity plus
position noise, D(w) = D_∞[1 − (τ/w)(1 − e^(−w/τ))] + σ²/w. The individual parts are fitted
for w ≤ 4 s, because beyond that they sag: relative positions are bounded by cohesion that
the weak first-order pair kernel does not capture. As a check, the two parts recombine to the
stored D(w) of the feasibility analysis (0.251, 0.436, 0.673 m²/s at 1, 2, 4 s).

| Part of the residual | τ (s) | D_∞ (m²/s) | persistence length √(D_∞τ) | compared with |
|---|---|---|---|---|
| each sheep's own, along the heading | 0.50 | 0.14 | 0.27 m | s∥ = 0.98 m: Kn 0.27, s²/D = 6.7 s |
| each sheep's own, across | 0.55 | 0.09 | 0.22 m | s⊥ = 0.91 m: Kn 0.25, s²/D = 9.0 s |
| the centroid's, along | 1.6 | 0.95 | 1.2 m | independent noise would give D/N = 0.010 |
| the centroid's, across | still ballistic at 8 s | — | — | D/N = 0.007 |

Per drive (the 20 drives of at least 20 s), across the heading: τ has median 0.54 s (10th–90th
percentile 0.23–0.98 s), and Kn has median 0.20 (0.11–0.39), never above 1. Changing the fit
range (w up to 2, 3, 4, 6 or 8 s) moves the individual τ within 0.39–0.59 s and the
persistence length within 0.18–0.31 m. The shared part's variance, per sheep, relative to
each sheep's own is 2.1 (along) and 1.2 (across) at 0.2 s, 2.9 and 2.5 at 1 s, 5.3 and 6.2 at
4 s, and 7.2 and 10.0 at 8 s. The fitted position noise is at most 3 cm.

**Reading.**

1. **Di Lorenzo's noise model fits the shape.** Each sheep's deviation from the flock forgets
   itself in half a second and travels a quarter of the flock's radius doing so, so the
   density *relative to the centroid* obeys a Fokker–Planck equation at the flock's scale.
   The diffusivity that spreads the shape is D ≈ 0.12 m²/s (mean of the two axes), half the
   0.25 m²/s used in the feasibility table and a fifth of the 4 s value.
2. **The centroid does not.** Most of what the first-order model misses is shared by all
   sheep and persists for seconds along the heading, longer across it. This is the flock's
   own propulsion and turning, the flock speed "set from outside" of Phase 4, which a drift
   built from the dog's push and the pair kernel cannot produce. Under independent noise the
   centroid of a held flock would sit still to within √((D/N)/k_dog) ≈ 0.25 m. The real
   centroid moves like a persistent walker with a step of a metre or more.
3. **Caveat.** "Shared" means unexplained by this first-order class. A model with the
   measured response to the dog's velocity, or with alignment, would absorb part of it, and
   would no longer be a model of the type his theorem covers. That is the point: the class
   his theorem covers explains the shape and not the motion.

## 14. Cohesion holds the shape, the dogs hold the position

Script: `experiments/cohesion_stiffness.py` (functions `goal_points`, `cohesion_drift`,
`single_rings`, `centroid_stiffness` in `shepherd/feasibility.py`, tests in
`tests/test_feasibility.py`); numbers in `results/cohesion_stiffness.json`; figure
`results/figures/cohesion_stiffness.png`.

**What is computed.** For the feasibility table's goals (95% within 2.5·√(N/14) m) and its
kernels (push a e^(−r/L) with the pair kernel fitted jointly at each L, cut at 3.6 m), this
computes four things: cohesion's stiffness k_coh of (10); its share of the goal's curvature
D/s²; the NNLS designs of `minimum_dogs_local` with and without cohesion, with the stiffness
k_dog each realizes; and the least-squares mass and drift error of a *single* uniform ring at
each radius, so designs can be compared at the same radius.

| Push range L (14 sheep) | k_coh/(D/s²), at D(1 s) / at the individual D of §13 | NNLS dogs, none → measured cohesion (mass-weighted radius) | same ring: mass ratio | cheapest single ring within 5% drift error, none → measured | k_dog (1/s), none → measured |
|---|---|---|---|---|---|
| 3 m | 0.04 / 0.08 | 10.0 (8.3 m) → 6.8 (6.6 m) | ×0.96 | 6.45 → 6.00 | 0.238 → 0.229 |
| 5 m | 0.18 / 0.38 | 22.8 (13.1 m) → 12.6 (10.0 m) | ×0.81 | 14.2 → 11.6 | 0.238 → 0.195 |
| 8 m | 0.28 / 0.58 | 30.3 (14.5 m) → 21.6 (14.5 m) | ×0.71 | 29.7 → none | 0.238 → 0.172 |

Larger flocks, share at D(1 s) for L = 3, 5, 8 m: 0.26, 0.57, 0.79 at 30 sheep and 0.43,
0.87, 1.16 at 50. At the individual D these double: 0.54, 1.21, 1.65 and 0.90, 1.82, 2.43.

**Reading.**

1. **How much of the REPORT's 30–45% is cohesion.** On a common ring, cohesion saves 4%,
   19% and 29% at L = 3, 5 and 8 m. That is k_coh/(D/s²), as it should be. The rest of the
   NNLS saving at L = 3 and 5 m comes from the solver moving the ring inward (8.3 → 6.6 m,
   13.1 → 10.0 m) and accepting 4% drift error. Without cohesion, a single ring at 6.25 m
   does the same job within 5% error with 6.45 dogs, fewer than the cohesive NNLS design's
   6.8. NNLS minimizes the drift error, not the count, so two of its solutions compare two
   points on different trade-off curves.
2. **The share is not identified by the driving data.** The pair kernel is fitted jointly
   with the push at a fixed L. A longer push range varies less across the flock, so the
   converging part of the sheep's motion is credited to cohesion instead. Since L is
   unidentified, so is cohesion's share of the real flock's confinement: anywhere from 4%
   to 58% across the two diffusivities and three ranges.
3. **Whatever the share, it comes off the centroid.** k_dog = D/s² − k_coh holds for every
   design to within its drift error (0.229, 0.195 and 0.172 against 0.229, 0.195 and 0.171).
   Pair forces cannot hold the flock at a goal they know nothing about. A flock whose share
   reaches 1 needs no dog for its shape and has no restoring force for its position. With the
   individual D that happens between 14 and 30 sheep at L = 8 m, between 14 and 30 at 5 m,
   and beyond 50 at 3 m.

## 15. The notice radius is the Jadhav dog's only compressive part

Script: `experiments/target_density.py` (the notice-radius sweep at its end); numbers under
`notice` in `results/target_density.json`; figure `results/figures/target_density_notice.png`.

**Prediction.** A Jadhav dog pushes a sheep with the unit vector away from itself if the
sheep is within r_dog, and not at all otherwise. The divergence of that field is 1/r > 0
inside the notice disc and a negative line source on its edge, where the push drops from one
to zero. The edge is its only compressive part. A sum of such pushes from dogs all in view is
the gradient of the summed distances to the dogs, which for a surrounding ring is smallest
at the centre, so it points outward everywhere inside the ring. A ring of Jadhav dogs can
therefore confine only where the set of dogs in view changes. Two regimes follow. For
r_dog < r_ring the centre sees no dog, and the ring is a wall around a neutral disc of radius
r_ring − r_dog. For r_ring < r_dog, a sheep sees every dog near the centre and is pushed out,
until far dogs leave its notice disc; the push turns inward on a circle ρ*(r_dog). For the
ten-dog ring of §10 (radius 10.9 m), ρ* is computed on a dog's bearing and midway between
dogs. It exists from r_dog = 11 to 16 m and not from 18 m on.

**Result** (L2, L3 and L5 of the ladder under the same ring, scattered start, 50 runs, last
20 s; goal: median radius 2.4 m, 11% within 1 m):

| r_dog (m) | predicted | L2 unit push: median radius / inside 5 m / escaped | L3 + freeze: median radius | L5 Jadhav sheep: median radius / inside / centroid off goal |
|---|---|---|---|---|
| 6 | blind centre, wall at 4.9 m | 3.6 m / 0.90 / 0 | 4.4 m | 4.3 m / 0.94 / 0.3 m |
| 9 | blind centre, wall at 1.9 m | 1.4 m / 1.00 / 0 | 1.4 m | 2.2 m / 1.00 / 0.4 m |
| 11 | ρ* = 0.13–0.14 m | 0.5 m / 1.00 / 0 | 0.5 m | 1.9 m / 0.99 / 1.0 m |
| 12 (published) | ρ* = 1.13–1.19 m | 1.2 m / 1.00 / 0 | 1.2 m | 1.9 m / 0.99 / 0.9 m |
| 13 | ρ* = 2.13–2.22 m | 2.4 m / 1.00 / 0 | 2.4 m | 2.5 m / 0.98 / 1.1 m |
| 14 | ρ* = 3.26–3.67 m | 3.7 m / 0.84 / 0.15 | 3.7 m | 3.5 m / 0.89 / 2.5 m |
| 16 | ρ* = 5.9–7.0 m | all escape | all escape | all escape |
| 18, 22 | none | all escape | all escape | all escape |

**Reading.**

1. **The prediction holds.** Between 12 and 14 m the sheep settle within 0.1–0.4 m of ρ*.
   At 11 m the circle is smaller than a sheep's 1 m step, which sets a floor of about 0.5 m.
2. **The goal is hollow.** At 13–14 m no sheep is within 1 m of the centre, against 11%
   under the goal density. The radial profile peaks on the circle, not at the goal (figure,
   right). The cohesive flock (L5) sits as a whole with its centroid 1 m off centre at
   11–13 m, and 2.5 m off at 14 m, on that circle.
3. **Escape.** At 14 m, 15% of the L2 sheep leave. At 16 m everyone does, though a circle
   still exists: its basin is narrower than a full step plus heading noise. From 18 m there
   is nothing to hold them. Cohesion does not rescue the flock.
4. **Below the ring radius the ring is a wall.** L2 sheep random-walk in the blind disc.
   L3 sheep stand wherever the wall left them, with a median radius of 4.4 m at r_dog = 6 m
   against the goal's 2.4 m. Freezing matters here and only here, consistent with item 5 of
   §10.

So the component of his design that carries convergence, the push that compresses beyond
its range, has a single counterpart in Jadhav's world: the edge of the notice disc. Where
his deconvolution picks the ring radius from the push range L, for Jadhav's sheep the only
lever is r_dog relative to the ring, and it buys a point or a circle, never a designed
density. The same edge is why coverage is the sharp condition of the closed loop in §9:
outside the notice disc the dog does nothing, and at its edge is the only place it confines.
