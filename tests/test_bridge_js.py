"""The JavaScript port (viz/bridge_sim.js) against the Python reference.

Deterministic parts must agree numerically: the Di Lorenzo design (herder mass, designed density,
drift error), the feasibility scan, the density error and the finite-population floor. The
dynamics use different random streams, so they are compared statistically on the published model.
Skipped when node is not installed.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shepherd import bridge as B

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_js(script):
    code = f"const Bridge = require({json.dumps(str(ROOT / 'viz' / 'bridge_sim.js'))});\n" + script
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_design_and_feasibility_match():
    F, H, T, D, _ = B.preset("dilorenzo_reference")
    F = replace(F, g_rep=0.5, l_rep=1.0, g_att=0.3, l_att=2.0)  # exercise the follower term too
    H = replace(H, n=200, grid=32)
    d = B.design_herders(F, H, T, D)
    js = run_js("""
      const {F, H, T, D} = Bridge.resolve({followers: {rule: 'first_order', n: 720, dt: 0.01, D: 0.01, a_dog: 1, L_dog: Math.PI,
        population_scaling: true, g_rep: 0.5, l_rep: 1.0, g_att: 0.3, l_att: 2.0},
        herders: {controller: 'dilorenzo', n: 200, v: Infinity, K_H: 10, bandwidth: 0.4, grid: 32},
        target: {kind: 'vonmises', k: 6 / Math.PI, r_goal: Math.PI / 2, bandwidth: 0.3}, domain: {periodic: true, size: 2 * Math.PI, images: 1}});
      const d = Bridge.designHerders(F, H, T, D);
      const sc = Bridge.feasibilityScan(F, T, D, [0.1, 0.3, 0.5, 0.7], 32);
      console.log(JSON.stringify({mass_needed: d.mass_needed, feasible: d.feasible, drift_error: d.drift_error,
        rho_H: Array.from(d.rho_H), needed: sc.needed}));
    """)
    assert abs(js["mass_needed"] - d.mass_needed) < 1e-9
    assert js["feasible"] == d.feasible
    assert abs(js["drift_error"] - d.drift_error) < 1e-9
    np.testing.assert_allclose(np.array(js["rho_H"]).reshape(32, 32), d.rho_H, atol=1e-9)
    sc = B.feasibility_scan(F, T, D, masses=[0.1, 0.3, 0.5, 0.7], grid=32)
    np.testing.assert_allclose(js["needed"], sc["needed"], atol=1e-9)


@pytest.mark.parametrize("key", ["dilorenzo_metres", "dilorenzo_jadhav_sheep"])
def test_dilorenzo_scenario_presets_match(key):
    """The page's metre-scale presets equal bridge.preset: same followers, same herders' design, same flat
    control, and the sub-stepped herder flow moves the herders as the Python one does (it reads no sheep)."""
    F, H, T, D, _ = B.preset(key)
    d = B.design_herders(F, H, T, D)
    flat = B.design_herders(F, replace(H, design_model=replace(H.design_model, D=0.0)), T, D)
    r = B.simulate_bridge(F, H, T, D, n_steps=3, seed=0, init="uniform")
    js = run_js(f"""
      const s = Bridge.createSim(Bridge.PRESETS.{key}.cfg, 1, 'uniform');
      const p = {json.dumps(r.pos_d[0].tolist())};
      for (let m = 0; m < s.M; m++) {{ s.dx[m] = p[m][0]; s.dy[m] = p[m][1]; }}
      for (let k = 0; k < 3; k++) Bridge.step(s);
      const fl = Bridge.resolve(Bridge.PRESETS.dilorenzo_jadhav_flat.cfg);
      console.log(JSON.stringify({{rule: s.F.rule, n: s.N, M: s.M, size: s.D.size, substeps: s.H.substeps,
        mass_needed: s.design.mass_needed, rho_H: Array.from(s.design.rho_H), flat: Bridge.designHerders(fl.F, fl.H, fl.T, fl.D).mass_needed,
        dogs: Array.from(s.dx).map((x, m) => [x, s.dy[m]])}}));
    """)
    assert js["rule"] == F.rule and js["n"] == F.n and js["M"] == H.n and js["substeps"] == H.substeps
    assert abs(js["size"] - D.size) < 1e-12
    assert abs(js["mass_needed"] - d.mass_needed) < 1e-9 and abs(js["flat"] - flat.mass_needed) < 1e-12
    np.testing.assert_allclose(np.array(js["rho_H"]).reshape(64, 64), d.rho_H, atol=1e-9)
    np.testing.assert_allclose(js["dogs"], r.pos_d[-1], atol=1e-6)


def test_metrics_match_on_plane_and_torus():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 2.5, (25, 2))
    T = B.Target("gaussian", sigma=2.5, bandwidth=1.0)
    plane = B.Metrics(T, B.Domain(False), 25)
    torus = B.Metrics(T, B.Domain(True, 40.0), 25, B.Grid(40.0, 32))
    js = run_js(f"""
      const xs = {json.dumps(pos[:, 0].tolist())}, ys = {json.dumps(pos[:, 1].tolist())};
      const T = Object.assign({{}}, Bridge.TARGET, {{sigma: 2.5, bandwidth: 1.0, r_goal: 2.5 * Math.sqrt(-2 * Math.log(0.05))}});
      const a = new Bridge.Metrics(T, Object.assign({{}}, Bridge.DOMAIN, {{periodic: false}}), 25);
      const b = new Bridge.Metrics(T, Object.assign({{}}, Bridge.DOMAIN, {{periodic: true, size: 40}}), 25, new Bridge.Grid(40, 32));
      console.log(JSON.stringify({{plane: a.all(xs, ys, 25), torus: b.all(xs, ys, 25), floor_p: a.floor, floor_t: b.floor}}));
    """)
    for k, v in plane.all(pos).items():
        assert abs(js["plane"][k] - v) < 1e-9, k
    for k, v in torus.all(pos).items():
        assert abs(js["torus"][k] - v) < 1e-9, k
    assert abs(js["floor_p"] - plane.floor) < 1e-12 and abs(js["floor_t"] - torus.floor) < 1e-9


def test_published_model_statistics_agree():
    """Same model, different random streams: the 60-step cohesion and dog–flock distance agree in distribution."""
    F, H, T, D, _ = B.preset("jadhav_reference")
    py = []
    for s in range(40):
        r = B.simulate_bridge(F, H, T, D, n_steps=60, seed=s, init="jadhav", box_length=40.0, store_every=60)
        py.append([r.metrics["cohesion"][-1], r.metrics["centroid_error"][-1]])
    js = run_js("""
      const out = [];
      for (let s = 0; s < 40; s++) {
        const sim = Bridge.createSim(Bridge.PRESETS.jadhav_reference.cfg, 1000 + s, 'jadhav', 40.0);
        for (let t = 0; t < 60; t++) Bridge.step(sim);
        const m = Bridge.measure(sim); out.push([m.cohesion, m.centroid_error]);
      }
      console.log(JSON.stringify(out));
    """)
    py, js = np.array(py), np.array(js)
    for c in range(2):
        se = np.sqrt(py[:, c].var() / 40 + js[:, c].var() / 40)
        assert abs(py[:, c].mean() - js[:, c].mean()) < 3.5 * se + 1e-9


def test_feedback_placement_and_surplus_match():
    rng = np.random.default_rng(4)
    pos = rng.normal(0, 3, (20, 2)); pos[:5] += [9.0, 0.0]
    dogs = np.array([[-20.0, 5.0], [20.0, -5.0], [3.0, 14.0]])
    for periodic, size in ((False, 80.0), (True, 40.0)):
        dom = B.Domain(periodic, size)
        T = B.Target("gaussian", sigma=2.5)
        F = B.Followers(n=20)
        H = B.Herders("density_feedback", n=3).resolve(F, dom, T)
        met = B.Metrics(T.resolve(dom), dom, 20, B.Grid(size, 32) if periodic else None)
        sur = met.surplus(pos)
        aims, mode, _ = B.feedback_aims(pos, dogs, np.zeros(2), sur, H, dom, False, F.r_dog, met.peak)
        js = run_js(f"""
          const xs = {json.dumps(pos[:, 0].tolist())}, ys = {json.dumps(pos[:, 1].tolist())};
          const dx = {json.dumps(dogs[:, 0].tolist())}, dy = {json.dumps(dogs[:, 1].tolist())};
          const r = Bridge.resolve({{followers: {{n: 20}}, herders: {{controller: 'density_feedback', n: 3}}, target: {{sigma: 2.5}}, domain: {{periodic: {str(periodic).lower()}, size: {size}}}}});
          const met = new Bridge.Metrics(r.T, r.D, 20, {"new Bridge.Grid(%g, 32)" % size if periodic else "null"});
          const sur = met.surplus(xs, ys, 20);
          const fa = Bridge.feedbackAims(xs, ys, 20, dx, dy, 3, [0, 0], sur, r.H, r.D, false, r.F.r_dog, met.peak, false);
          console.log(JSON.stringify({{sur: Array.from(sur), ax: Array.from(fa.ax), ay: Array.from(fa.ay), mode: Array.from(fa.mode)}}));
        """)
        np.testing.assert_allclose(js["sur"], sur, atol=1e-9)
        np.testing.assert_allclose(np.stack([js["ax"], js["ay"]], -1), aims, atol=1e-9)
        assert js["mode"] == mode.tolist()


@pytest.mark.parametrize("key", ["dilorenzo_reference", "dilorenzo_interacting", "hybrid_jadhav_sheep_dilorenzo_dogs",
                                 "dilorenzo_jadhav_sheep"])
def test_dilorenzo_count_is_the_smallest_feasible(key):
    """The page's 'Di Lorenzo's count' button: Python's design is feasible at that count and not one below."""
    js = run_js(f"console.log(JSON.stringify(Bridge.dilorenzoCount(Bridge.PRESETS.{key}.cfg)));")
    cfg = run_js(f"""const r = Bridge.resolve(Bridge.PRESETS.{key}.cfg);
      console.log(JSON.stringify({{F: r.F, H: Object.assign({{}}, r.H, {{v: null}}), T: r.T, D: r.D}}));""")
    fields = lambda cls, d: {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
    F = B.Followers(**fields(B.Followers, cfg["F"]))
    H = B.Herders(**{**fields(B.Herders, cfg["H"]), "v": np.inf,
                     "design_model": B.Followers(**fields(B.Followers, cfg["H"]["design_model"])) if cfg["H"]["design_model"] else None})
    T, D = B.Target(**fields(B.Target, cfg["T"])), B.Domain(**fields(B.Domain, cfg["D"]))
    n = js["n"]
    assert n is not None and n >= 1
    assert B.design_herders(F, replace(H, n=n), T, D).feasible
    if n > 1:
        assert not B.design_herders(F, replace(H, n=n - 1), T, D).feasible


def test_dilorenzo_count_reports_why_it_is_undefined():
    js = run_js("""console.log(JSON.stringify([Bridge.dilorenzoCount(Bridge.PRESETS.jadhav_reference.cfg),
      Bridge.dilorenzoCount(Bridge.PRESETS.dilorenzo_jadhav_flat.cfg)]));""")
    assert all(r["n"] is None and r["reason"] for r in js)
