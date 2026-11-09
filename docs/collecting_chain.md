# The collecting chain: from a disturbed flock back to the driving region and the ball

**Question.** [jadhav_dilorenzo_bridge.md](jadhav_dilorenzo_bridge.md) §10.6 listed what would connect the published
dog's collecting to the driving-region results:
1. a collecting drift;
2. escape from the coverage gap;
3. a handover condition;
4. no endless switching.

Bridge §12.6 then found that the 3-step drift D₃ fails where the 3-step window crosses into collecting. This document
takes the four links in turn at the baseline (bridge §0): the published sheep rule with its stop rule, and the
published Strömbom dog. Nothing is changed in either.

Evidence labels are as in the bridge: [E] exact, [P] proved, [D] derived under a stated approximation, [S] simulated.
"Certified" means a bound that holds at a state with no Monte Carlo, or with a stated confidence where Monte Carlo is
used.

Files:
- `shepherd/collecting.py`: the decision rule and the drift bound of §3. Tests are in `tests/test_collecting.py`.
- `experiments/collecting_chain.py`, writing `results/collecting_chain.json` and
  `results/figures/collecting_chain.png`. The run takes about 4 minutes.

## 0. The answer

| Link | Result | Status |
|---|---|---|
| 1. Collecting drift | A one-step drift for the farthest sheep's distance R alone is false: E[ΔR] > 0 exactly at 29% of collecting states, and at 66% of those where the target sheep does not yet see the dog. A two-phase potential Ψ = R + (distance from the dog to that sheep − 4 m)₊ has an exactly computable bound on its expected change. That bound is ≤ −0.1 m per step at 98.9% of the collecting states visited during recovery, between target switches. | [P] bound, certified per state; [S] where it is evaluated |
| 2. The gap | A sheep frozen while the dog drives lies ahead of the dog's notice disc, so the advancing dog reaches it after at most about 7 m of flock advance. On recovery paths 84% of frozen sheep are woken, in a median of 1 step. The 9.4–11.6 m lateral gap of §10.6 is not a trap. | [D] lemma; [S] |
| 3. Handover | The dog's rule has no hysteresis, so the analysis puts one in: a collecting cycle ends only once R < l_sep − 3 m. At those handovers D₃ ≤ −0.1 m² is certified at 136 of 141. The probability of reaching the ball before the next collecting cycle is at least 0.51 at every sampled handover, and its median is 0.999. | [S], probability 0.999 |
| 4. Switching | Cycles are geometric. From a handover, E[τ_ball] ≤ (L_C + L_D)/p. The worst sampled bounds give at most 214 steps, against 7.4 observed on average. Inside a collecting cycle, each target switch costs at most 4v + 2R′ in Ψ (proved), so the cycle length reduces to the number of switches. That number has no geometric bound: two stragglers on opposite sides take turns with no ratchet. With no rival beyond 9 m there are no switches across the flock. | [P] cycle and jump bounds; [S] the switch counts |

So the chain closes on the states sampled from recovery paths, with explicit constants. It is not a uniform theorem:
- Links 1 and 3 are certified on the states the closed loop visits, not over the state space.
- Target switches inside a collecting cycle are bounded in cost but not in number. With two stragglers on opposite
  sides the rule takes turns, and nothing forces progress between turns (§3.5).

## 1. The rule, and the phases [E]

The dog reads the state before the step (`shepherd/model.py`):
- **slow**, if it is within r_rep = 2 m of a sheep;
- **collect**, if the farthest sheep is at least l_sep = 2N^(2/3) = 11.62 m from the centroid;
- **drive**, otherwise.

`dog_decision` reproduces the model's recorded mode on every step of 40 runs (test).

Collecting aims at A = x_k + l_collect e_k: 2 m beyond the farthest sheep k, on the ray from the centroid, where
e_k = ξ_k/|ξ_k|. The dog moves 1.5 m per step along unit(u + 0.3 e(ψ)), with u the direction to A, so its heading
is within arcsin 0.3 = 17.5° of u. In the slow mode it creeps 0.05 m along its last heading.

R = max_i |ξ_i| is the collecting variable: collecting is on exactly while R ≥ l_sep, apart from slow steps. The
switch has no hysteresis.

## 2. What collecting does on recovery paths [S]

**Paths.** As in bridge §12.6:
1. Take an independent baseline snapshot.
2. Deform it by one of 11 deformations, chosen uniformly.
3. Run 200 closed-loop steps.

τ is the first entry into the ball (√M₂ < 4.46 m). The run uses 3,000 paths with seed 61. The potential's
constants (§3) and the band (§5) were chosen on a separate exploration seed.

- **Recovery.** 1,106 paths start outside the ball, and all of them enter it. τ has mean 20.7 steps, median 12,
  99th percentile 105, maximum 188.
- **Time split.** Before τ, the dog drives 65% of the time, collects 33% and is slow 2%. Collecting cycles occur on
  408 paths, at most 2 per path.
- **Collecting is mostly travelling** (exploration paths). The dog's median distance to its aim during collecting
  was 12.4 m. It got within 3 m of the aim in only 15% of episodes.
- **Waking, not pushing** (exploration paths). Most collecting ends because the dog comes within r_dog of the
  straggler, which then moves. Once it sees the dog, its expected radial heading points inward at 95% of states,
  even though the dog is usually between it and the flock (median away·e = +0.63, a push *outward*). The reason is
  that attraction (weight 1.5, and for the farthest sheep every neighbour lies inward) outweighs the dog (weight 1).
  On the reported paths the same shows in §3.1: E[ΔR] > 0 at only 6% of states where the target sees the dog.

## 3. Link 1: a collecting drift

### 3.1 A drift for R alone fails [E]

R′ ≥ |ξ_k′| for the current farthest sheep k. Also E|ξ_k′| ≥ |E ξ_k′| ≥ R + E[e_k·Δ], with Δ = ξ_k′ − ξ_k. So

    E[ΔR | x] ≥ E[e_k·Δ],

and the right-hand side is exact (shepherd/expected_heading.py). It is positive at 29% of the 7,630 collecting
states before τ. Split by whether the target sees the dog:

| Target | E[ΔR] > 0 exactly |
|---|---|
| Frozen (does not see the dog) | 66% |
| Awake (sees the dog) | 6% |

Drawing one collecting state per path at random gives P(E[ΔR] > 0) ≥ 0.195, with probability 0.999. While the dog
travels toward a frozen straggler, the sheep it passes flee, and that shifts the centroid away from the straggler.
As with C4 (bridge §10.4), a uniform one-step condition is false. The second component has to be the travel.

### 3.2 Proposition C1 (a two-phase collecting drift) [P]

For the current target k, let

    Ψ_k(x) = |ξ_k| + λ (|D − x_k| − d₀)₊,   λ = 1, d₀ = 4 m (twice l_collect).

At every state with R ≥ l_sep there is an explicit U(x) with E[Ψ_k(x′) − Ψ_k(x) | x] ≤ U(x), the same k on both
sides (`collecting_drift_bound`):

**The sheep part.** With d_j = v h_j′ for a moving sheep (0 otherwise), Δ = Σ_j w_j d_j, where w_k = v(1 − 1/N) and
w_j = −v/N. Since |ξ + Δ|² = (R + e·Δ)² + |Δ_⊥|²,

    |ξ_k + Δ| ≤ R + e·Δ + |Δ_⊥|² / (2(R − |Δ|_max)),

    E|ξ_k′| − R ≤ E[e·Δ] + (E|Δ|² − (E e·Δ)²) / (2(R − |Δ|_max)),

    E Δ = Σ_j w_j μ_j,   E|Δ|² = |E Δ|² + Σ_j w_j² ([moves_j] − |μ_j|²).

Both moments are exact, because the d_j are independent given the state.

**The dog part.**
- *Every draw.* The heading is within φ = θ + arcsin 0.3 of the direction to x_k, where θ is the angle between the
  aim and the sheep as seen from the dog. So

      |D′ − x_k′| ≤ √(d² − 3d cos φ + 2.25) + v[k moves],

  and in the slow mode |D′ − x_k′| ≤ d + 0.05 + v[k moves].
- *In expectation.* The dog's noise average has the same closed form as the sheep's: E[heading] = m(0.3) u. The
  dog's step is independent of the sheep's, so E|D′ − x_k′|² is exact. Where the distance cannot fall below d₀
  within the step (d − 1.5 − v ≥ d₀), E(|D′ − x_k′| − d₀)₊ ≤ √(E|D′ − x_k′|²) − d₀ by Jensen.

The smaller of the two dog bounds is used. ∎

**Checks.**
- At 40 random collecting states, one-step Monte Carlo (8,192 replicates each) never exceeds U by more than 4
  standard errors; the largest z is 0.83.
- The slack U − (Monte Carlo mean) has median 0.007 and 90th percentile 0.05. The bound is nearly tight.
- Along the paths, the realized change of Ψ_k minus U averages −0.044 ± 0.005 over 6,887 steps.
- The unit test checks the bound with the dog far away, inside the flock, behind the target and in the slow mode.

### 3.3 Where it holds [certified per state, S for the sample]

All 7,630 collecting states before τ:

| Target | States | U ≤ 0 | U ≤ −0.1 | Median U |
|---|---|---|---|---|
| Frozen (dog beyond r_dog) | 2,977 | 98.8% | 98.8% | −1.40 |
| Awake | 4,621 | 99.5% | 99.2% | −1.15 |
| Dog in slow mode | 32 | 81% | 75% | −0.39 |
| All | 7,630 | 99.2% | 98.9% | |

- **A random collecting state.** With one state drawn per path (408), P(U > −0.1) ≤ 0.036 with probability 0.999.
- **The 81 exceptions.** 37 have a frozen target, 36 an awake one and 8 are slow-mode states. Across them, the
  target's own expected radial step is outward (median E[e·Δ] = +0.70) and the dog is near r_dog (median 11.7 m).
  The dog has not yet reached a straggler that is drifting outward under alignment and inertia.
- **The step at 6.5 m in the figure.** It marks where the dog bound switches from the exact-moment form to the
  every-draw form (d < d₀ + 2.5). It comes from the bound, not from the dynamics.

### 3.4 What the drift does not cover: target switches [S]

When another sheep becomes the farthest, the potential jumps from Ψ_k to Ψ_{k′}.
- **How often.** A switch happens on 9.0% of consecutive collecting steps.
- **How large.** The jump has a median of 7.1 m, a 90th percentile of 22 m and a maximum of 29 m. One jump in five is
  negative.
- **Chattering.** Two stragglers on opposite sides make the dog cross the flock, and some cycles switch target 31
  times. A collecting cycle has a median of 1 switch and a 90th percentile of 9.

**Corollary** [P, given C1 on the states visited]. Take a collecting episode, a maximal run of states with
R ≥ l_sep. If U ≤ −c on every state of the episode, then optional stopping gives

    E[episode length] ≤ (Ψ(start) − (l_sep − 2v) + E[Σ jumps]) / c.

R falls by at most 2v per step, so R ≥ l_sep − 2v when the episode ends. The switches enter only through the expected
sum of jumps. That fits the measured lengths: the expected length of a collecting cycle (§6) correlates with Ψ at its
start by only −0.15. The number of stragglers decides it, not how far away the first one is.

### 3.5 Bounding the switches

`experiments/collecting_chain.py`, part G. The numbers come from the fresh paths. The two hypotheses tested and
dropped below were formed on the exploration seed.

**Lemma S1 (the jump is bounded)** [P]. At a switch from target k to k′,

    J = Ψ_{k′}(x′) − Ψ_k(x′) ≤ 4v + 2λR′.

*Proof.*
1. Before the step |ξ_k| ≥ |ξ_{k′}|, and each moves by at most 2v. So |ξ′_{k′}| − |ξ′_k| ≤ 4v.
2. The dog term changes by at most λ|x′_{k′} − x′_k| ≤ λ(|ξ′_{k′}| + |ξ′_k|) ≤ 2λR′. ∎

It held at all 618 switches, with J at most 86% of the bound (median 24%). With the corollary of §3.4:

    E[episode length] ≤ (Ψ(start) − l_sep + 2v + E[N_sw] (4v + 2λ R_max)) / c.

So bounding the length of a collecting episode reduces to bounding E[N_sw], the number of switches.

**Lemma S2 (a switch needs a second straggler)** [P]. Inside a collecting episode, a switch to sheep j at step t
needs |ξ_j(t+1)| = R(t+1) ≥ l_sep. While j is out of the dog's range and not repelled it does not move, so
|Δ|ξ_j|| = |ΔG| ≤ v n_moving/N per step. A sheep at distance r from the centroid can therefore become the target only
after the centroid has moved by at least l_sep − r in total. ∎

**What decides the number of switches** [S].
- **Which switches.** 90% of the 618 switches are across the flock, between targets on opposite sides, about 25 m
  apart on the exploration paths. 64% happen while the old target is being served, i.e. it sees the dog. The rest come while the dog is
  still travelling.
- **The rival's start distance** (the farthest sheep on the opposite side, when the episode starts). Clopper–Pearson
  bounds are at probability 0.999 over the bins.

  | Rival at the start | Episodes | Any across switch | Two or more | Mean length (steps) |
  |---|---|---|---|---|
  | < 9 m | 234 | 0 (≤ 3.9%) | 0 | 5.8 |
  | 9–10 m | 74 | 1.4% (≤ 15%) | 0 | 4.2 |
  | 10–11 m | 162 | 15% | 12% | 8.1 |
  | 11 m – l_sep | 134 | 47% | 31% | 13.5 |
  | ≥ l_sep (a second straggler) | 139 | 60% | 21% | 20.6 |

- **The cost.** Episodes without an across switch last 6.0 steps on average. With 1 or 2 they last 17. With 3 or
  more (57 of 743 episodes) they last 40, and the most is 20 across switches in one episode.

**Two mechanisms tested and dropped** (on the exploration seed, then confirmed on these paths).
1. *A self-defeating push.* The idea was that the dog, crossing the flock toward its target, pushes the centroid
   toward that target, and so hands the label to the rival. While the target is still frozen, though, the expected
   centroid step points toward it in only 36% of states.
2. *A ratchet.* The idea was that each service leaves R a little lower, so the turns run out. But R at successive
   across switches is not lower in 57% of cases (386 pairs; the median change is a 0.06 m rise, and the 10th–90th
   percentiles are −0.65 to +0.72 m). While a target is served the frozen rival does drift inward (78% of states,
   median 0.11 m per step), but that gain is lost while the dog crosses.

**So the number of switches has no geometric bound under this rule.** Two stragglers at nearly equal distances on
opposite sides take turns:
1. The served one is brought in until it ties the other.
2. The label passes to the other, and the dog crosses the flock.
3. Nothing in the rule makes the next tie lower than the last.

The episode ends when the noise happens to leave both below l_sep. What can be stated is conditional:
- with no second sheep beyond 9 m, there are no across switches (P ≤ 0.04);
- with a second straggler, each turn costs one flock crossing: after an across switch the dog starts a median
  17 m from its new target (exploration paths).

This is a property of farthest-sheep targeting without hysteresis in the choice of target. It is recorded as a
limitation; the rule is not changed.

## 4. Link 2: the coverage gap

**Lemma G** [D: the dog at the drive point, within ε; the centroid advancing straight]. While the dog drives, a
frozen sheep (out of the dog's range, not repelled) lies **ahead** of the dog's notice disc. It is reached once the
centroid has advanced at most ≈ 7.1 + ε m.

*Derivation.* Take drive coordinates, with s along the direction of motion and y across it. The dog stands
l_drive = 7.5 m behind the centroid, and the frozen sheep k is fixed at (s, y).
1. The sheep is out of range iff (s + 7.5)² + y² > 144.
2. If it were behind the disc, s + 7.5 < −√(144 − y²), so |ξ_k|² > (7.5 + √(144 − y²))² + y² ≥ 200. That puts it
   beyond l_sep, and the dog would be collecting. The same holds if |y| > 12.
3. So in driving, s + 7.5 > √(144 − y²). After the centroid advances by ℓ the dog is ℓ further on, and the sheep
   wakes once ℓ ≥ s + 7.5 − √(144 − y²).
4. Over |ξ_k| < l_sep this is at most 7.5 + l_sep − 12 = 7.1 m, the worst case being straight ahead. A hover error
   ε adds at most ε.

At 0.83 m per step (continuification_driving §9.2) and ε = 1.6 m (the 90th percentile of the hover), that is about
10 steps. ∎

**This corrects bridge §10.6.** That section said a sheep 9.4–11.6 m to the side "freezes, and is collected only once
the flock has moved far enough that it exceeds 11.6 m". In fact the dog, driving from behind, passes it and wakes it.

**Measured** [S]. Of 15,040 frozen-sheep events while the dog drove:

| | Events | Steps frozen: median / 90th pct / max |
|---|---|---|
| Woken | 12,689 (84%) | 1 / 4 / 22 |
| Collecting started first | 1,129 (8%) | 1 / 4 / 12 |
| Flock entered the ball first | 1,222 (8%) | |
| Lateral offset ≥ 9.4 m: woken | 1,519 | 1 / 5 / 11 |
| Lateral offset ≥ 9.4 m: collecting | 476 | 2 / 5 / 10 |

The longest waits, up to 22 steps, exceed the lemma's 10. They happen when the flock slows or the dog is out of
position, which are the lemma's two assumptions.

## 5. Link 3: the handover

**Hysteresis in the analysis.** Because the switch has none, a flock handed over with R just below l_sep can go
straight back. On the exploration seed, with cycles cut at l_sep itself, one handover reached the ball before
collecting again with probability only 0.004. So the analysis ends a collecting cycle only once R < l_sep − h; the
dog's rule is unchanged.
- **The choice of h.** h = 3 m was chosen on the exploration seed among 0–5 m. At 4–5 m almost every collecting cycle
  ends in the ball, so the cycles no longer separate the two modes.
- **What it buys.** With R < 8.6 m every sheep lies inside the drive point's lateral reach of 9.4 m (§10.6).

**Measured at handovers** [S, probability 0.999 per claim]. One handover was drawn per path that has one, 141 in all.
48 of them have full coverage.

| Quantity | Result |
|---|---|
| D₃ ≤ −0.1 m² (certified as in bridge §12.6) | 136 of 141; D₃ > 0 certified at 3. P(D₃ > −0.1) ≤ 0.124 over handovers |
| p(x) = P(ball before the next collecting cycle) | estimate: minimum 0.58, 10th percentile 0.92, median 0.999. Lower bound: ≥ 0.513 at every sampled handover |
| E[driving cycle length] | median 5 steps, largest 25; empirical-Bernstein upper bound ≤ 30.7 at every sampled handover |

The 1,024 replicates per state give the lower bound on p, by Clopper–Pearson with a union over states. None was
truncated at 150 steps.

## 6. Link 4: switching, and the cycle bound

**Proposition C2** [P]. Suppose that at every state where a collecting cycle can start, its expected length is at
most L_C. Suppose too that at every handover the driving cycle that follows has expected length at most L_D and ends
in the ball with probability at least p > 0. Then from any handover

    E[τ_ball] ≤ (L_C + L_D) / p,

and from a collecting start, E[τ_ball] ≤ L_C + (L_C + L_D)/p.

*Proof.* Cycle starts are stopping times. By the strong Markov property and the uniform bounds, the n-th driving
cycle starts with probability at most (1 − p)^(n−1), and each cycle pair has expected length at most L_C + L_D.
Summing the geometric series gives the bound. ∎

**Numbers.**
- **Collecting starts.** 408 sampled, with 4,096 replicates each and none truncated at 600 steps. The expected cycle
  length has median 31 steps and largest 71, and every one is ≤ 79.0 with probability 0.999.
- **Handovers** (§5). L_D ≤ 30.7 and p ≥ 0.513.
- **The sample-uniform bound.** From a handover, E[τ] ≤ (79.0 + 30.7)/0.513 = 214 steps; from a collecting start,
  ≤ 293. With the sample medians instead it would be 36. The observed time from a handover to the ball averages 7.4
  steps, with a 90th percentile of 17 and a maximum of 40.

The bound is loose because its three worst cases occur at different states. It is finite and explicit, and it rests
only on per-state quantities with confidence bounds.

## 7. Status

- **Proved.**
  - Proposition C1: the collecting drift bound, and its corollary for the cycle length given the drift and the jumps.
  - Proposition C2: the cycle bound.
  - Lemmas S1 and S2: a switch's jump is at most 4v + 2λR′, and a switch needs a second sheep at l_sep, which a
    frozen sheep can reach only through the centroid's motion.
  - The exact lower bound E[ΔR] ≥ E[e·Δ], which shows that R alone has no uniform one-step drift.
- **Derived.** Lemma G: in driving, frozen sheep lie ahead of the dog's disc and are woken within about 7 m of
  advance.
- **Certified on visited states.**
  - C1's bound ≤ −0.1 at 98.9% of collecting states; P(U > −0.1) ≤ 0.036 for a random one.
  - At handovers: D₃ ≤ −0.1 at 136 of 141, p ≥ 0.51 and L_D ≤ 31 at every sampled one.
  - At collecting starts: L_C ≤ 79 at every sampled one.
- **Not established.**
  - Uniformity over the state space.
  - A bound on the number of target switches inside a collecting cycle (§10.6 item 4). It is now located
    precisely: two stragglers on opposite sides that take turns. There is no ratchet: R at successive across
    switches is not lower in 57% of cases. Without a rival beyond 9 m there are no across switches (P ≤ 0.04).
  - The tail of collecting cycles beyond 600 steps, which no replicate reached.
- **Limitations, not pursued.**
  - Target switching across the flock: farthest-sheep targeting has no hysteresis in the choice of target.
  - The slow mode: the dog has no obstacle avoidance, and C1 fails at a quarter of slow-mode states.
  - Lateral splits during driving (bridge §12.6), which appear here as the handovers with lower p.
