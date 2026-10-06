"""Write viz/presets.json: calibrated presets for the viewer.

Sheep parameters are posterior medians (weights as median shares x 10; only
their ratios matter). The dog is the policy measured from the real collie.
Notes quote the closed-loop long-drive check (experiments/multi_horizon.py).

    .venv/bin/python viz/make_presets.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from common import draws, load, posterior  # noqa: E402
from build_table import V_SHEEP, WEIGHTS, measured_dog  # noqa: E402
from multi_horizon import long_check  # noqa: E402
from shepherd import abc as ABC, data  # noqa: E402

JS = {"w_inertia": "wInertia", "w_rep": "wRep", "w_dog": "wDog", "w_att": "wAtt", "w_ali": "wAli", "noise": "noise",
      "r_rep": "rRep", "r_dog": "rDog", "tau_s": "tauS", "l_drive": "lDrive", "v_dog": "vDog", "tau_d": "tauD",
      "flank_sigma": "flankSigma", "flank_tau": "flankTau", "dog_speed_gain": "dogSpeedGain", "noise_dog": "noiseDog",
      "l_sep": "lSep", "l_collect": "lCollect"}


def preset(name, label, drives, dog, dog_kernel=None, g_dogk=None):
    if not (ROOT / "results" / "tables" / f"{name}.npz").exists():
        return None  # table not built yet
    tab = load(name)
    if "tab2" not in tab:
        return None
    dt = float(tab["dt"])
    post = posterior(tab, name)
    sh = ABC.shares(post.samples, list(WEIGHTS))
    params = {JS[k]: round(10 * sh[k]["median"], 4) for k in WEIGHTS}
    for k in ("r_rep", "r_dog", "tau_s"):
        if k in post.samples:
            v = float(np.median(post.samples[k]))
            params[JS[k]] = None if (k == "tau_s" and v <= dt) else round(v, 3)
    params.update({JS[k]: (round(v, 3) if isinstance(v, float) else v) for k, v in dog.items()})
    params.update({"dt": dt, "vSheep": V_SHEEP[dt], "outrun": True, "socialAlways": False})
    extra = {}
    if dog_kernel is not None:
        knots, D, V, L = dog_kernel
        params.update({"dogKernel": {"knots": list(knots), "D": list(D), "V": list(V), "L": list(L)}, "gDogK": g_dogk})
        extra = {"dog_kernel": dog_kernel, "g_dogk": g_dogk}
    check = long_check(drives, draws(post, 16), dt, dog, **extra)
    return {"label": label, "params": params, "check": check,
            "shares": {k: sh[k]["median"] for k in WEIGHTS}}


def main():
    drives = data.load_drives(data.load_sessions())
    dog = measured_dog()
    from dog_response import learned_dog_kernel

    out = {}
    m4 = preset("M3tau", "Calibrated", drives, dog, dog_kernel=learned_dog_kernel(), g_dogk=2.0)
    m3t = preset("M3tau", "Without learned dog response", drives, dog)
    m2 = preset("M2", "Short-horizon fit", drives, dog)
    real = {"C_med": 1.13, "P_med": 0.95}
    f = lambda x: "<0.1%" if x < 0.001 else f"{100 * x:.1f}%"
    pct = lambda d: f"{f(d['w_ali'])} copying a neighbour, {f(d['w_rep'])} avoiding collisions, " \
                    f"{f(d['w_att'])} pull toward neighbours"
    if m4:
        c = m4["check"]
        m4["note"] = (f"Sheep rules fitted to the real drives on two horizons (15 s windows with the real dog replayed, and "
                      f"whole drives re-run in closed loop), at a 0.2 s step: {pct(m4['shares'])}. The response to the dog "
                      f"comes from kernels learned directly from the tracks (flight from the dog, reaction to its velocity "
                      f"and to how fast it closes in), at gain 2, the best fit. The dog is the policy measured from the "
                      f"collie. On the real drives: cohesion {c['C_med']:.1f} m, alignment {c['P_med']:.2f} (real flock "
                      f"{real['C_med']} m, {real['P_med']}). The learned dog response lets the dog turn the flock (a 90° turn "
                      f"succeeds in 75% of runs, against 11% without it), but over many back-and-forth trips the flock "
                      f"still spreads out, because its attraction to neighbours stays weak.")
        m4["params"]["scenario"] = "drives"
        out["calibrated"] = m4
    if m3t:
        c = m3t["check"]
        m3t["note"] = (f"The same fitted sheep rules without the learned dog response: fleeing the dog gets "
                       f"{f(m3t['shares']['w_dog'])} of the heading decision, because real drives start with the flock "
                       f"already moving the right way. On the real drives it looks fine (cohesion {c['C_med']:.1f} m), "
                       f"but switch to back and forth and the dog cannot turn the flock at the posts.")
        m3t["params"]["scenario"] = "drives"
        out["nodogk"] = m3t
    if m2:
        c = m2["check"]
        m2["note"] = (f"Fitted only on 15 s windows, at a 1 s step: {pct(m2['shares'])}. Nothing in short windows "
                      f"constrains cohesion, so over whole drives the flock comes apart (cohesion {c['C_med']:.1f} m, "
                      f"{100 * c['frac_far']:.0f}% of sheep more than 5 m from the centre).")
        m2["params"]["scenario"] = "drives"
        out["short"] = m2
    for v in out.values():
        v.pop("shares")
    (Path(__file__).parent / "presets.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {"params": v["params"], "check": v["check"]} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
