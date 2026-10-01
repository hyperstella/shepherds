"""Robustness of the leadership result for the learned-kernel model.

    .venv/bin/python experiments/hierarchy_robustness.py
Writes results/hierarchy_robustness.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from shepherd import abc as ABC, data, kernels as K, protocol as PR, scenario as SC  # noqa: E402
from build_table import FIT_STATS, HYBRID_PRIORS  # noqa: E402

TABLES, OUT = ROOT / "results" / "tables", ROOT / "results"
COLS = [PR.SUMMARY_KEYS.index(k) for k in FIT_STATS]
N_DRAW = 16


def learned_posterior(n=N_DRAW, seed=0):
    hyb = dict(np.load(TABLES / "hybrid.npz"))
    th = {k: hyb[f"th_all_{k}"] for k in HYBRID_PRIORS}
    sims = ABC.aggregate(ABC.sanitize(hyb["tab_all"])[..., COLS], hyb["meta_durations"])
    obs = ABC.aggregate(ABC.sanitize(hyb["meta_obs"])[None, :, COLS], hyb["meta_durations"])[0]
    post = ABC.fit(th, sims, obs, ABC.mad_scale(sims), HYBRID_PRIORS, 0.03)
    pick = np.random.default_rng(seed).choice(len(post.samples["E"]), n, replace=False)
    g = {k: post.samples[k][pick] for k in HYBRID_PRIORS}
    fit = np.load(TABLES / "kernel_fit.npz")
    km = K.KernelModel(terms=tuple(str(t) for t in fit["terms"]))
    gains = lambda dog: {"E": g["E"], "D": dog * g["dog"], "DV": dog * g["dog"], "DL": dog * g["dog"]}
    return km, hyb["coef_all"], g, gains


def stats(segs, key="session"):
    recs = PR.hierarchy(segs, 2.0, 0.5, True, min_duration_s=8.0)
    st = PR.hierarchy_stats(recs)
    ci = PR.cluster_bootstrap(recs, lambda r: PR.hierarchy_stats(r)["spearman"], 300, 0, key)["ci95"]
    return {"rho": st["spearman"], "ci95": ci, "dog_rank": st.get("dog_rank")}


def polarization(segs):
    return float(np.nanmean([np.nanmean(PR.summaries(s)["P_med"]) for s in segs]))


def flip_front_back(p0, v0):
    """Mirror the starting flock along its direction of motion (front <-> back)."""
    g, h = p0.mean(0), v0.mean(0)
    h = h / (np.linalg.norm(h) + 1e-12)
    along = (p0 - g) @ h
    return p0 - 2 * along[:, None] * h[None, :], v0


def common_heading(p0, v0):
    """All sheep start with the flock's mean velocity (no individual heading information)."""
    return p0, np.repeat(v0.mean(0, keepdims=True), len(v0), 0)


def main():
    drives = data.load_drives(data.load_sessions())
    km, c, g, gains = learned_posterior()
    out = {}

    def run(name, dog=1.0, sigma=None, **kw):
        coef = K.apply_gains(km, c, gains(dog))
        segs = SC.replay_windows_kernel(drives, km, coef, g["sigma"] if sigma is None else np.full(N_DRAW, sigma),
                                        N_DRAW, 7, 0.2, **kw)
        flat = SC.unbatch(segs)
        out[name] = {**stats(flat), "P": polarization(segs)}
        print(f"{name:52s} rho {out[name]['rho']:+.3f} CI {np.round(out[name]['ci95'], 3)}  P {out[name]['P']:.2f}")

    real15 = SC.window_segments(drives, dt=0.2)
    real30 = SC.window_segments(drives, dt=0.2, horizon=30.0, burn=5.0)
    out["data, 15 s windows"] = {**stats(real15), "P": polarization(real15)}
    out["data, 30 s windows"] = {**stats(real30), "P": polarization(real30)}
    for k in ("data, 15 s windows", "data, 30 s windows"):
        print(f"{k:52s} rho {out[k]['rho']:+.3f} CI {np.round(out[k]['ci95'], 3)}  P {out[k]['P']:.2f}")

    run("learned (posterior noise)")
    run("learned, front/back flipped at start", init_fn=flip_front_back)
    run("learned, common starting heading", init_fn=common_heading)
    run("learned, 30 s windows", horizon=30.0, burn=5.0)
    # dog vs no dog with the flock's mean velocity imposed, across noise levels
    for sigma in (0.03, 0.06, 0.1, 0.2):
        for dog in (1.0, 0.0):
            run(f"velocity replayed, sigma {sigma:g}, {'dog' if dog else 'no dog'}", dog=dog, sigma=sigma,
                velocity_replay=True)
    # The learned alignment kernel is unidentified (near zero), which caps P at about 0.8.
    # Replace it by a weak uniform alignment (within 5 m) to reach the data's P ~ 0.89.
    sl, peaks = km.slices()["A"], km.pair.peaks
    for a in (0.005, 0.01, 0.02, 0.05):
        for dog in (1.0, 0.0):
            name = f"velocity replayed, alignment {a:g}/s, {'dog' if dog else 'no dog'}"
            coef = K.apply_gains(km, c, gains(dog))
            coef[:, sl] = np.where(peaks < 5.0, a, 0.0)
            segs = SC.replay_windows_kernel(drives, km, coef, np.full(N_DRAW, 0.1), N_DRAW, 7, 0.2, velocity_replay=True)
            out[name] = {**stats(SC.unbatch(segs)), "P": polarization(segs)}
            print(f"{name:52s} rho {out[name]['rho']:+.3f} CI {np.round(out[name]['ci95'], 3)}  P {out[name]['P']:.2f}")
    (OUT / "hierarchy_robustness.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
