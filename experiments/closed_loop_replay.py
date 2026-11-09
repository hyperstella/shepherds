"""Replay page for the closed-loop convergence scenarios.

Re-simulates one representative run of each scenario in
experiments/closed_loop_convergence.py and writes a self-contained page,
viz/closed_loop_replay.html, that replays it with the diagnostics the argument
uses: distance of the centroid to the target, maximum spread against l_sep, the
dog's mode, the drive point, the notice radius, and which sheep see the dog.

    .venv/bin/python experiments/closed_loop_replay.py
Serve viz/ (the "viz" launch config) and open closed_loop_replay.html.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import closed_loop_convergence as C  # noqa: E402
from shepherd.model import REPLAY, Result, simulate  # noqa: E402

OUT = ROOT / "viz" / "closed_loop_replay.html"

SCENARIOS = [
    dict(key="published", label="Published model: reach, then overshoot",
         note="N = 14, r_dog = 12 m, v_dog = 1.5. The dog reaches the drive point by about step 20, the centroid's distance "
              "to the target falls linearly, and after reaching the drive point flips, the dog goes through the flock and the "
              "centroid overshoots. The loop is reach-only.",
         kw=dict(), seed=C.SEED, pick="first"),
    dict(key="release", label="Same run, dog withdraws on arrival",
         note="The published run up to the step the centroid comes within 5 m, then the dog walks straight away at its own "
              "speed instead of following the drive rule. Once no sheep is within r_dog the flock stops where it is: the goal "
              "is an equilibrium only because the dog leaves. This is the release test of section 9.",
         kw=dict(), seed=C.SEED, pick="release"),
    dict(key="coverage", label="Coverage failure: r_dog = 9 m",
         note="The drive point is 7.5 m behind the centroid and the flock is about 3 m deep, so the front sheep do not see "
              "the dog (grey) and stand still. Only repulsion from behind moves them. 17% of runs reach within 400 steps.",
         kw=dict(r_dog=9.0, n_steps=400), seed=C.SEED + 2, pick="first"),
    dict(key="slow_dog", label="Slow dog: v_dog = 0.8 < flock speed",
         note="The dog is slower than the flock would be, yet 70% of runs reach: sheep that get more than r_dog ahead stop and "
              "wait for the dog, so the flock cannot outrun it. Speed sets the time to reach, not whether it reaches.",
         kw=dict(v_dog=0.8, n_steps=400), seed=C.SEED + 2, pick="success"),
]


def with_release(res, r, n_after=80):
    """Run r of `res` up to its reaching step, then the dog walks away at v_dog and the sheep follow their rules."""
    g = C.geometry(res)
    tr = int(C.reach_times(g)[r])
    p = g["p"]
    pos_s, vel_s = res.pos_s[r, :, :, tr], res.vel_s[r, :, :, tr]
    pos_d = res.pos_d[r, :, tr]
    u = pos_d - pos_s.mean(0)
    u = u / np.linalg.norm(u)
    path = pos_d[:, None] + u[:, None] * p.v_dog * np.arange(n_after)[None]
    r2 = simulate(C.base(), n_runs=1, seed=sc_seed(res) + 1, init={"pos_s": pos_s[None], "vel_s": vel_s[None]}, dog_path=path[None])
    cat = lambda a, b: np.concatenate([a[r : r + 1, ..., : tr + 1], b[..., 1:]], axis=-1)
    mode = np.concatenate([res.mode[r : r + 1, : tr + 1], np.full((1, n_after - 1), REPLAY, dtype=res.mode.dtype)], axis=1)
    return Result(res.params, cat(res.pos_s, r2.pos_s), cat(res.vel_s, r2.vel_s), cat(res.pos_d, r2.pos_d), cat(res.vel_d, r2.vel_d), mode)


def sc_seed(res):
    return C.SEED


def run_scenario(sc, n_runs=8):
    res = simulate(C.base(**sc["kw"]), n_runs=n_runs, seed=sc["seed"])
    if sc["pick"] == "release":
        res = with_release(res, 0)
    g = C.geometry(res)
    t_reach = C.reach_times(g)
    r = 0
    if sc["pick"] == "success":
        ok = np.flatnonzero(t_reach < g["rho"].shape[1])
        r = int(ok[0]) if len(ok) else 0
    p = g["p"]
    pos, _ = res.run(r)  # (N+1, 2, T), dog last
    T = pos.shape[-1]
    # rotate so that the line from the initial centroid to the target is the +x axis
    G0 = pos[:p.n_sheep, :, 0].mean(0)
    ang = np.arctan2(p.target[1] - G0[1], p.target[0] - G0[0])
    c, sn = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -sn], [sn, c]])
    tgt = np.asarray(p.target, float)
    pos = np.einsum("ij,njt->nit", rot, pos - tgt[None, :, None]) + tgt[None, :, None]
    return dict(
        key=sc["key"], label=sc["label"], note=sc["note"],
        N=int(p.n_sheep), r_dog=float(p.r_dog), l_sep=float(p.l_sep), l_drive=float(p.l_drive), v_dog=float(p.v_dog),
        v_sheep=float(p.v_sheep), arrive=C.ARRIVE, target=[float(p.target[0]), float(p.target[1])],
        t_reach=int(t_reach[r]) if t_reach[r] < T else None,
        pos=np.round(np.moveaxis(pos, -1, 0), 2).tolist(),  # (T, N+1, 2)
        mode=res.mode[r].astype(int).tolist(),
        rho=np.round(g["rho"][r], 2).tolist(),
        spread=np.round(g["maxspread"][r], 2).tolist(),
        ppar=np.round(g["P_par"][r], 3).tolist(),
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-loop replay</title>
<style>
  :root { --bg: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e3e3e6; --panel: #f6f6f8;
          --blue: #2a78d6; --orange: #eb6834; --green: #1baf7a; --red: #d63a3a; --grey: #b4b4ba; --purple: #7b61c4; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141416; --ink: #ececf0; --muted: #a0a0a8; --line: #2c2c31; --panel: #1d1d21; --grey: #55555c; } }
  :root[data-theme="dark"] { --bg: #141416; --ink: #ececf0; --muted: #a0a0a8; --line: #2c2c31; --panel: #1d1d21; --grey: #55555c; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 12px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .tabs button { font: 600 13px/1.2 inherit; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
  .tabs button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .note { color: var(--muted); margin: 0 0 10px; min-height: 2.8em; }
  canvas { width: 100%; display: block; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
  #field { aspect-ratio: 2 / 1; }
  #chart { aspect-ratio: 5 / 1; margin-top: 8px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; margin: 10px 0; }
  .controls button { font: 600 13px/1.2 inherit; padding: 7px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
  .controls label { color: var(--muted); display: flex; align-items: center; gap: 6px; }
  input[type=range] { width: min(100%, 420px); }
  .readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 6px 14px; font-variant-numeric: tabular-nums; margin: 6px 0 0; }
  .readout div { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; }
  .readout b { display: block; font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: 12px; margin-top: 8px; }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; background: var(--c); }
</style>
</head>
<body>
<main>
  <h1>Closed-loop herding: the scenarios behind experiments/closed_loop_convergence.py</h1>
  <p class="sub">One representative run each. The first scenarios re-simulate the published model (experiments/closed_loop_convergence.py); the last three put Di Lorenzo's goal density and herder ring into the same simulation (experiments/target_density.py). Scrub or play; the chart tracks the centroid's distance to the target and the maximum spread.</p>
  <div class="tabs" id="tabs" role="group" aria-label="Scenario"></div>
  <p class="note" id="note"></p>
  <canvas id="field"></canvas>
  <div class="controls">
    <button id="play" type="button">Pause</button>
    <label>speed <select id="speed"><option value="2">2 steps/s</option><option value="6" selected>6 steps/s</option><option value="15">15 steps/s</option><option value="40">40 steps/s</option></select></label>
    <label><input type="checkbox" id="follow" checked> follow the flock</label>
    <label><input type="checkbox" id="trails" checked> trails</label>
    <label style="flex: 1 1 260px;">step <input type="range" id="scrub" min="0" value="0" step="1"></label>
  </div>
  <div class="readout">
    <div><b>step</b><span id="r-step"></span></div>
    <div><b>distance to target</b><span id="r-rho"></span></div>
    <div><b>max spread / l_sep</b><span id="r-spread"></span></div>
    <div><b>dog mode</b><span id="r-mode"></span></div>
    <div><b>polarization P∥</b><span id="r-ppar"></span></div>
    <div><b>sheep seeing the dog</b><span id="r-sees"></span></div>
  </div>
  <canvas id="chart"></canvas>
  <div class="legend">
    <span style="--c: var(--blue)">sheep that sees the dog</span>
    <span style="--c: var(--grey)">sheep that does not (stands still)</span>
    <span style="--c: var(--orange)">dog, with its notice radius</span>
    <span style="--c: var(--purple)">drive point</span>
    <span style="--c: var(--green)">target, with the 5 m arrival circle; shaded disc: goal density (95%)</span>
  </div>
</main>
<script>
const DATA = /*__DATA__*/;
const MODES = { 0: ["start / static herders", "var(--grey)"], 1: ["drive", "var(--green)"], 2: ["collect", "var(--orange)"], 3: ["slow (in the flock)", "var(--red)"], 4: ["withdrawn (dog walks away)", "var(--purple)"], 5: ["hold (dog withdrawn, waiting)", "var(--purple)"], 6: ["touring the ring", "var(--blue)"] };
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S = { sc: DATA[0], t: 0, playing: true, sps: 6, follow: true, trails: true, acc: 0, last: 0 };
const field = $("#field"), chart = $("#chart");

function fit(canvas) {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) { canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr); }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return [ctx, r.width, r.height];
}

function bounds(sc) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const fr of sc.pos) for (const [x, y] of fr) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  x0 = Math.min(x0, sc.target[0] - 8); x1 = Math.max(x1, sc.target[0] + 8); y0 = Math.min(y0, sc.target[1] - 8); y1 = Math.max(y1, sc.target[1] + 8);
  return { x0, x1, y0, y1 };
}

function camera(sc, t, W, H) {
  let b = sc._b || (sc._b = bounds(sc));
  if (S.follow && !sc.static) {
    const fr = sc.pos[t]; let gx = 0, gy = 0;
    for (let i = 0; i < sc.N; i++) { gx += fr[i][0]; gy += fr[i][1]; }
    gx /= sc.N; gy /= sc.N; const half = Math.max((sc.r_dog || 0) + 6, (sc.l_sep || 0) + 4, 18);
    b = { x0: gx - half * 2, x1: gx + half * 2, y0: gy - half, y1: gy + half };
  }
  const pad = 12, s = Math.max(1e-3, Math.min((W - 2 * pad) / (b.x1 - b.x0), (H - 2 * pad) / (b.y1 - b.y0)));
  const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
  return (x, y) => [W / 2 + (x - cx) * s, H / 2 - (y - cy) * s, s];
}

const tx0 = (P, sc) => P(sc.target[0], sc.target[1])[0], ty0 = (P, sc) => P(sc.target[0], sc.target[1])[1];
function drawField() {
  const sc = S.sc, t = S.t, fr = sc.pos[t];
  const [ctx, W, H] = fit(field);
  if (W < 60 || H < 60) return;
  ctx.clearRect(0, 0, W, H);
  const P = camera(sc, t, W, H), s = P(0, 0)[2];
  // centroid, drive point
  let gx = 0, gy = 0; for (let i = 0; i < sc.N; i++) { gx += fr[i][0]; gy += fr[i][1]; } gx /= sc.N; gy /= sc.N;
  const dx = sc.target[0] - gx, dy = sc.target[1] - gy, dn = Math.hypot(dx, dy) || 1;
  const nd = sc.n_dogs || 1, dogs = fr.slice(sc.N, sc.N + nd);
  const px = gx - (sc.l_drive || 0) * dx / dn, py = gy - (sc.l_drive || 0) * dy / dn;
  const dog = dogs[0];
  if (sc.r_goal) {  // the goal density: 95% disc, shaded
    ctx.fillStyle = css("--green"); ctx.globalAlpha = 0.12; ctx.beginPath(); ctx.arc(tx0(P, sc), ty0(P, sc), sc.r_goal * s, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1;
  }
  // target
  const [tx, ty] = P(sc.target[0], sc.target[1]);
  ctx.strokeStyle = css("--green"); ctx.lineWidth = 1.5; ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(tx, ty, sc.arrive * s, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(tx - 6, ty); ctx.lineTo(tx + 6, ty); ctx.moveTo(tx, ty - 6); ctx.lineTo(tx, ty + 6); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "600 11px sans-serif"; ctx.fillText("target", tx + 8, ty - 6);
  if (tx < 0 || tx > W || ty < 0 || ty > H) {  // off-screen target: arrow at the edge with the distance
    const ax = Math.min(W - 60, Math.max(60, tx)), ay = Math.min(H - 30, Math.max(30, ty));
    const ang = Math.atan2(ty - ay, tx - ax);
    ctx.strokeStyle = css("--green"); ctx.fillStyle = css("--green"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + 22 * Math.cos(ang), ay + 22 * Math.sin(ang)); ctx.stroke();
    ctx.beginPath(); ctx.arc(ax + 22 * Math.cos(ang), ay + 22 * Math.sin(ang), 3.5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = css("--muted"); ctx.fillText(`target ${sc.rho[t].toFixed(0)} m`, ax - 20, ay - 10);
  }
  // l_sep circle around the centroid
  const [cgx, cgy] = P(gx, gy);
  if (sc.l_sep) {
    ctx.strokeStyle = css("--line"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(cgx, cgy, sc.l_sep * s, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.font = "10px sans-serif"; ctx.fillText("l_sep", cgx + sc.l_sep * s * 0.71 + 3, cgy - sc.l_sep * s * 0.71);
  }
  // notice radius around the dog(s)
  const [dgx, dgy] = P(dog[0], dog[1]);
  if (sc.r_dog) {
    ctx.strokeStyle = css("--orange"); ctx.setLineDash([3, 5]); ctx.globalAlpha = nd > 1 ? 0.25 : 0.7;
    for (const d of dogs) { const [qx, qy] = P(d[0], d[1]); ctx.beginPath(); ctx.arc(qx, qy, sc.r_dog * s, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }
  // trails
  if (S.trails) {
    const t0 = Math.max(0, t - 25);
    ctx.lineWidth = 1; ctx.globalAlpha = 0.35;
    for (let i = 0; i < sc.N + nd; i++) {
      ctx.strokeStyle = i >= sc.N ? css("--orange") : css("--blue"); ctx.beginPath();
      for (let k = t0; k <= t; k++) { const [x, y] = P(sc.pos[k][i][0], sc.pos[k][i][1]); if (k === t0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  // sheep
  let sees = 0;
  const rr = Math.max(2.5, Math.min(6, 0.55 * s));
  for (let i = 0; i < sc.N; i++) {
    const d = sc.r_dog ? dogs.some((q) => Math.hypot(fr[i][0] - q[0], fr[i][1] - q[1]) <= sc.r_dog) : true; if (d) sees++;
    const [x, y] = P(fr[i][0], fr[i][1]);
    ctx.fillStyle = d ? css("--blue") : css("--grey"); ctx.beginPath(); ctx.arc(x, y, rr, 0, 2 * Math.PI); ctx.fill();
  }
  // drive point (closed-loop dog only)
  if (!sc.static && sc.l_drive) {
    const [ppx, ppy] = P(px, py);
    ctx.strokeStyle = css("--purple"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(ppx - 5, ppy - 5); ctx.lineTo(ppx + 5, ppy + 5); ctx.moveTo(ppx - 5, ppy + 5); ctx.lineTo(ppx + 5, ppy - 5); ctx.stroke();
  }
  // dog(s)
  for (const d of dogs) {
    const [qx, qy] = P(d[0], d[1]);
    ctx.fillStyle = css("--orange"); ctx.beginPath(); ctx.arc(qx, qy, rr + 1.5, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  }
  // scale bar
  const bar = s * 10; ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(14, H - 14); ctx.lineTo(14 + bar, H - 14); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px sans-serif"; ctx.fillText("10 m", 14, H - 18);
  // readout
  $("#r-step").textContent = `${t} / ${sc.pos.length - 1}` + (sc.t_reach != null ? `  (reached at ${sc.t_reach})` : "  (never reached)");
  $("#r-rho").textContent = `${sc.rho[t].toFixed(1)} m`;
  $("#r-spread").textContent = sc.l_sep ? `${sc.spread[t].toFixed(1)} / ${sc.l_sep.toFixed(1)} m` : `${sc.spread[t].toFixed(1)} m`;
  const m = MODES[sc.mode[t]] || ["?", "var(--muted)"]; $("#r-mode").textContent = m[0]; $("#r-mode").style.color = m[1];
  $("#r-ppar").textContent = sc.ppar[t].toFixed(2);
  $("#r-sees").textContent = `${sees} / ${sc.N}`;
}

function drawChart() {
  const sc = S.sc, [ctx, W, H] = fit(chart);
  ctx.clearRect(0, 0, W, H);
  const T = sc.pos.length, L = 44, R = 10, top = 8, bot = 22, h = H - top - bot;
  const ymax = Math.max(Math.max(...sc.rho), Math.max(...sc.spread), sc.l_sep || 0, sc.r_goal || 0) * 1.05;
  const X = (t) => L + (t / (T - 1)) * (W - L - R), Y = (v) => top + h - (v / ymax) * h;
  // mode strip
  for (let t = 0; t < T - 1; t++) { ctx.fillStyle = (MODES[sc.mode[t]] || ["", "var(--grey)"])[1]; ctx.globalAlpha = 0.35; ctx.fillRect(X(t), H - bot + 4, Math.max(1, X(t + 1) - X(t)), 6); }
  ctx.globalAlpha = 1;
  // grid + labels
  ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted"); ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (const v of [0, Math.round(ymax / 2), Math.round(ymax)]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillText(`${v} m`, 4, Y(v) + 3); }
  ctx.strokeStyle = css("--green"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(L, Y(sc.arrive)); ctx.lineTo(W - R, Y(sc.arrive)); ctx.stroke();
  if (sc.l_sep) { ctx.strokeStyle = css("--orange"); ctx.beginPath(); ctx.moveTo(L, Y(sc.l_sep)); ctx.lineTo(W - R, Y(sc.l_sep)); ctx.stroke(); ctx.fillText("l_sep", W - R - 28, Y(sc.l_sep) - 3); }
  ctx.setLineDash([]);
  ctx.fillText("arrive", W - R - 30, Y(sc.arrive) - 3);
  const line = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); for (let t = 0; t < T; t++) { const x = X(t), y = Y(arr[t]); if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } ctx.stroke(); };
  line(sc.rho, css("--blue")); line(sc.spread, css("--orange"));
  ctx.fillStyle = css("--blue"); ctx.fillText("distance to target", L + 6, top + 10);
  ctx.fillStyle = css("--orange"); ctx.fillText("max spread", L + 110, top + 10);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(S.t), top); ctx.lineTo(X(S.t), H - bot + 10); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.fillText("mode: drive green, collect orange, slow red, withdrawn/hold purple, touring blue", L + 6, H - 2);
}

function draw() { drawField(); drawChart(); $("#scrub").value = S.t; }
function seek(t) { S.t = Math.max(0, Math.min(S.sc.pos.length - 1, t)); draw(); }
function setScenario(key) {
  S.sc = DATA.find((d) => d.key === key); S.t = 0; S.acc = 0;
  $("#scrub").max = S.sc.pos.length - 1; $("#note").textContent = S.sc.note;
  document.querySelectorAll("[data-key]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
  draw();
}
function tick(now) {
  if (S.playing) {
    const dt = Math.min(0.25, (now - S.last) / 1000); S.acc += dt * S.sps;
    while (S.acc >= 1) { S.acc -= 1; S.t = S.t + 1 >= S.sc.pos.length ? 0 : S.t + 1; }
    draw();
  }
  S.last = now; requestAnimationFrame(tick);
}

for (const sc of DATA) { const b = document.createElement("button"); b.type = "button"; b.dataset.key = sc.key; b.textContent = sc.label; b.addEventListener("click", () => setScenario(sc.key)); $("#tabs").appendChild(b); }
$("#play").addEventListener("click", () => { S.playing = !S.playing; $("#play").textContent = S.playing ? "Pause" : "Play"; });
$("#speed").addEventListener("change", (e) => (S.sps = +e.target.value));
$("#follow").addEventListener("change", (e) => { S.follow = e.target.checked; draw(); });
$("#trails").addEventListener("change", (e) => { S.trails = e.target.checked; draw(); });
$("#scrub").addEventListener("input", (e) => { S.playing = false; $("#play").textContent = "Play"; seek(+e.target.value); });
window.addEventListener("resize", draw);
setScenario(DATA[0].key); S.last = performance.now(); requestAnimationFrame(tick);
</script>
</body>
</html>
"""


TD_NOTES = {
    "l0_ring": "Di Lorenzo's targets: overdamped, pushed by the summed exponential kernel, diffusing. Under their ring the "
               "cloud relaxes toward the goal density (shaded) and stays. Start: scattered inside the ring.",
    "l5_ring": "The Jadhav sheep under the same ring: unit-vector flight within r_dog, fixed speed, freeze without a dog in "
               "range, social terms. Start: scattered inside the ring.",
    "l6_published_at_goal": "The published model started with the sheep sampled from the goal density and its own dog on the "
                            "ring. The drive rule has no stop, so the dog disperses what was already converged.",
    "hold_rule": "The published model with a reach-and-release rule: once the centroid is within 5 m of the target the dog "
                 "withdraws to 15 m from the centroid and waits (purple), re-engaging only if the centroid drifts beyond 10 m. "
                 "Same start as the first scenario.",
    "tour_jadhav": "One dog running round Di Lorenzo's ring at 1.5 m/s (a lap every 46 s), the Jadhav sheep inside. The "
                   "sheep that see it flee inward; the rest stand still. A time-multiplexed version of the ring.",
    "tour_targets": "The same touring dog with Di Lorenzo's diffusing targets. One dog covers one point of the ring at a "
                    "time, and the cloud leaks through the rest.",
}


def target_density_scenarios():
    out = []
    rows = []
    for name in ("target_density_replay.json", "hold_replay.json", "hold_calibrated_replay.json"):
        path = ROOT / "results" / name
        if path.exists():
            rows += json.loads(path.read_text())
    for d in rows:
        d.setdefault("note", TD_NOTES.get(d["key"], ""))
        d.setdefault("target", [0.0, 0.0])
        d.setdefault("arrive", C.ARRIVE)
        d.setdefault("t_reach", None)
        d.setdefault("v_dog", None)
        d.setdefault("v_sheep", 1.0)
        T = len(d["pos"])
        d.setdefault("mode", [0] * T if d.get("static") else [1] * T)
        d.setdefault("ppar", [0.0] * T)
        out.append(d)
    return out


def main():
    data = [run_scenario(sc) for sc in SCENARIOS] + target_density_scenarios()
    for d in data:
        lsep = f" / l_sep {d['l_sep']:.1f}" if d.get("l_sep") else ""
        print(f"{d['key']:22s} N={d['N']} dogs={d.get('n_dogs', 1)} steps={len(d['pos'])} reached={d['t_reach']} final rho {d['rho'][-1]:.1f} max spread {d['spread'][-1]:.1f}{lsep}")
    html = PAGE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
