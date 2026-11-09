"""One scorecard for comparing hybrids: reached, stayed, how fast, stayed together, at what cost.

Every quantity is read from a `bridge.BridgeResult` against that run's own (fixed) goal circle
(`target.centre`, `target.r_goal`), sampled every `sample_s` seconds.

  reached     occupancy(t): fraction of sheep inside the goal circle
  how fast    time to success: first t with occupancy ≥ `success` for at least `sustain` s (None: never)
  stayed      mean occupancy over the final `final_frac` of the run; after success, escapes = episodes
              with occupancy < `escape` lasting ≥ `escape_min` s (count, rate per 100 s, durations,
              whether the last one recovered)
  together    RMS radius about the centroid; largest-group fraction (sheep linked within `link` m);
              splits = episodes with largest group < `split_share` lasting ≥ `sample_s`; unrecovered =
              still split at the end
  cost        number of dogs, configured speed limit, measured mean and peak speed, total distance travelled
              (from step-to-step positions: a lower bound for herders that move within a step)
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

DEFAULTS = dict(success=0.8, sustain=30.0, final_frac=0.25, escape=0.5, escape_min=5.0, link=5.0,
                split_share=0.9, sample_s=1.0)


def _episodes(flag, t, min_len):
    """(start, duration, recovered) for runs of True lasting ≥ min_len."""
    out, k = [], 0
    while k < len(flag):
        if flag[k]:
            j = k
            while j < len(flag) and flag[j]:
                j += 1
            end = t[j] if j < len(flag) else t[-1]  # an episode still open at the end has not recovered
            if end - t[k] >= min_len:
                out.append((float(t[k]), float(end - t[k]), j < len(flag)))
            k = j
        else:
            k += 1
    return out


def occupancy_series(r, every):
    c = np.asarray(r.target.centre, float)
    idx = np.arange(0, len(r.t), every)
    occ = np.array([np.mean(np.linalg.norm(r.domain.wrap(r.pos_s[k] - c), axis=-1) <= r.target.r_goal) for k in idx])
    return r.t[idx], occ, idx


def score(r, **kw):
    p = {**DEFAULTS, **kw}
    dt_store = float(r.t[1] - r.t[0]) if len(r.t) > 1 else 1.0
    every = max(int(round(p["sample_s"] / dt_store)), 1)
    t, occ, idx = occupancy_series(r, every)
    dom, N = r.domain, r.pos_s.shape[1]
    T_end = float(t[-1])

    # how fast
    t_success = None
    for k in range(len(t)):
        j = np.searchsorted(t, t[k] + p["sustain"] - 1e-9)
        if j < len(t) and (occ[k:j + 1] >= p["success"]).all():
            t_success = float(t[k])
            break
    # stayed
    final = t >= T_end * (1 - p["final_frac"])
    esc = []
    if t_success is not None:
        after = t >= t_success
        esc = _episodes(occ[after] < p["escape"], t[after], p["escape_min"])
    post = T_end - t_success if t_success is not None else 0.0
    # together
    rms, largest = [], []
    for k in idx:
        x = r.pos_s[k]
        G = dom.centroid(x) if dom.periodic else x.mean(0)
        rms.append(float(np.sqrt((np.linalg.norm(dom.wrap(x - G), axis=-1) ** 2).mean())))
        d = np.linalg.norm(dom.wrap(x[:, None] - x[None]), axis=-1)
        _, lab = connected_components(d < p["link"], directed=False)
        largest.append(np.bincount(lab).max() / N)
    rms, largest = np.array(rms), np.array(largest)
    splits = _episodes(largest < p["split_share"], t, p["sample_s"])
    # cost
    M = r.pos_d.shape[1]
    if M:
        steps = np.linalg.norm(dom.wrap(np.diff(r.pos_d, axis=0)), axis=-1)  # (S−1, M), every stored step
        dist = steps.sum(0)
        speed = steps / dt_store
        cost = dict(dogs=M, speed_limit=float(r.herders.v), distance_total=float(dist.sum()),
                    distance_per_dog=float(dist.mean()), speed_mean=float(dist.mean() / max(T_end, 1e-9)),
                    speed_peak=float(speed.max()))
    else:
        cost = dict(dogs=0, speed_limit=None, distance_total=0.0, distance_per_dog=0.0, speed_mean=0.0, speed_peak=0.0)
    return dict(
        t=t.tolist(), occupancy=occ.tolist(),
        time_to_success=t_success,
        final_occupancy=float(occ[final].mean()),
        escapes=len(esc), escape_rate_per_100s=float(100 * len(esc) / post) if post > 0 else None,
        escape_duration_mean=float(np.mean([e[1] for e in esc])) if esc else 0.0,
        escape_unrecovered=int(sum(not e[2] for e in esc)),
        rms_spread_final=float(rms[final].mean()), largest_group_final=float(largest[final].mean()),
        largest_group_min=float(largest.min()), splits=len(splits), splits_unrecovered=int(sum(not s[2] for s in splits)),
        **cost,
    )
