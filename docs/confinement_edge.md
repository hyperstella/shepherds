# Can confinement in the follower swap be proved?

Setting: Jadhav sheep under 14 Di Lorenzo herders (`bridge_simulation.md` §4.9), 24 m torus, goal circle r* = 6 m, step
v = 1 m. Code: `experiments/confinement_edge.py`; numbers: `results/confinement_edge.json` (50 seeds, 450–600 s).
Tags as in `jadhav_dilorenzo_bridge.md`: **[P]** proved given stated assumptions, **[S]** measured in simulation.

**Result.** Confinement in the 6 m circle is not proved. A conditional exit lemma is proved below, but at 6 m it says
no more than counting sheep that come near the edge. The obstacle is specific: a sheep can cross from about 5.2 m to
beyond 6 m in one step, and the one-step inward drift needed below 5.5 m does not always hold.

## 1. No sure one-step barrier [S]

The herder term F(x) = Σ_{visible herders} ê_away is not normalized, and the other terms of the heading sum total at
most 0.5 + 2 + 1.5 + 1.3 + 0.3 = 5.6. If the inward herder push p = −F·n exceeded that, plus a small curvature term,
no sheep could step outward.

Sampled every 30 s over the late window, the largest p anywhere on the 6 m circle was 4.87, and the lowest was 1.02. The
best margin of the sure-barrier condition was −1.6. So the condition fails at every point sampled. No sure invariance
statement of this kind is possible for these herders, consistent with the four observed one-step exits.

## 2. Conditional exit lemma [P]

Let Y_i(t) be sheep i's wrapped distance from the goal centre. With no positional noise each step has length at most v,
so |ΔY_i| ≤ v. Fix r_a < r_b, λ > 0, and a stopping time τ.

**Hypothesis H(λ).** For every sheep i and every t < τ with r_a ≤ Y_i(t) < r_b,

    E[exp(λ ΔY_i(t)) | history at t] ≤ 1.

An *excursion* of sheep i starts at time s if Y_i(s) ≥ r_a and either s = t₀ or Y_i(s−1) < r_a. Let S be the number of
excursions started in [t₀, T∧τ), and assume Y_i(t₀) < r_a + v for all i.

**Lemma.** P(some sheep reaches r_b at a time ≤ T∧τ) ≤ E[S] · exp(−λ (r_b − r_a − v)).

*Proof.*
1. A sheep reaching r_b does so during some excursion, and an excursion starts with Y_i(s) < r_a + v.
2. Stop the excursion when it falls below r_a, reaches r_b, or at τ or T. Under H(λ), exp(λ Y_i) is a nonnegative
   supermartingale up to that time.
3. By optional stopping and Markov's inequality, P(reach r_b during this excursion | history at s) ≤
   exp(−λ (r_b − Y_i(s))) ≤ exp(−λ (r_b − r_a − v)).
4. Sum over all possible starts (i, s), using the tower property. ∎

**Corollary (mean drift suffices).** If E[ΔY_i | history] ≤ −c on the same event, then H(2c/v²) holds, by Hoeffding's
lemma on [−v, v]. The bound is then E[S] exp(−2c (r_b − r_a − v)/v²).

τ is needed. H cannot hold on all states: with outward headings and repulsion at the weakest point of the field
(p ≈ 1), inertia, alignment and repulsion alone contribute up to 3.8 outward. So the lemma says what happens *until the
hypothesis first fails*.

## 3. Testing the hypothesis on visited states [S]

Conditional on a stored state, the next step's law is finite apart from the noise angle: 210 attraction subsets, 4
alignment choices, 48-point rule for the angle. So drift and exponential moments are enumerated, not sampled. Check of
the enumeration: predicted mean Δr −0.462, realized −0.468 (s.e. 0.005), and 0 of 15,420 realized steps fell outside the
enumerated range.

| radius (m) | states | mean E[Δr] | worst E[Δr] | share with E[Δr] ≤ 0 |
|---|---|---|---|---|
| 3.5–4.0 | 10,298 | −0.41 | +0.90 | 82% |
| 4.0–4.5 | 3,870 | −0.54 | +0.86 | 89% |
| 4.5–5.0 | 1,016 | −0.61 | +0.82 | 93% |
| 5.0–5.5 | 214 | −0.72 | +0.44 | 98.6% |
| 5.5–6.0 | 19 | −0.89 | −0.69 | 100% |
| 6.0–7.0 | 3 | −0.98 | −0.97 | 100% |

For r_a = 5.5 (22 states), the largest E[exp(λΔr)] is below 1 at every λ tried up to the grid's top value of 3 (0.171 at
λ = 3). For r_a = 5.0, 4.5 and 4.0 it exceeds 1 at every λ tried, because some states have positive drift.

**The exits.** Three sheep crossed 6 m in the late windows:

| seed | from r (m) | to r (m) | E[Δr] at that state | P(Δr > 0) |
|---|---|---|---|---|
| 5036 | 5.16 | 6.16 | +0.05 | 0.55 |
| 5048 | 5.28 | 6.15 | +0.44 | 0.83 |
| 5049 | 5.16 | 6.06 | −0.28 | 0.25 |

Two of the three start from the only three states at r ≥ 5 with positive drift. Three exits are too few for a rate.

## 4. What the lemma gives

With r_a = 5.5 and λ = 3, the gap r_b − r_a − v is zero at r_b = 6, so the bound is just E[S] = 0.44 per 150 s window
(observed: 3 of 50 windows). That is only the count of sheep passing 5.5 m.

To get real decay at 6 m, r_a must be well below 5 m, and there the hypothesis fails on 7–18% of visited states. Larger
r_b does not rescue it: states above 6.16 m were never visited, so the hypothesis is untested there, and the bound
values for r_b = 7, 8 and 10 in the JSON should not be quoted.

## 5. What is and is not established

- **Proved:** the conditional lemma, for any herder layout.
- **Measured:** the hypothesis on visited states at r ≥ 5.5 m, from only 22 states.
- **Not proved:** confinement in the 6 m circle; that H holds up to any useful τ; any statement about how the flock
  first gets inside.
- **Possibly useful next:** the herders do not read the sheep and barely move late (largest step 0.013 m), so the field
  F can be treated as nearly fixed. An untried route is a potential that includes heading memory (as with W in
  `jadhav_dilorenzo_bridge.md` §11), so that outward-heading states stop violating the drift condition. A memory term
  also widens each step's range, which shrinks the gap, so it may not help at 6 m.
