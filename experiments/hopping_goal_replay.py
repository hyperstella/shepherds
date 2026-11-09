"""Replay page for the flock-level hybrid: Jadhav sheep, Di Lorenzo herders, a hopping goal centre.

The published Jadhav sheep keep every mechanism (attraction, repulsion, alignment, directional noise,
the standing rule). Their interactions set the flock's held shape S; the target density ρ̄ can then
only be reached as a time average, ⟨ρ⟩_t = S ∗ p_c, where p_c is where the flock's centroid spends
its time. So p_c is Di Lorenzo's deconvolution one level up: p_c = ρ̄ ⊘ S (Tikhonov, clipped at 0;
feasible when that is non-negative, i.e. the held flock is narrower than the target). Every `dwell`
seconds the goal centre jumps to a point drawn from p_c; Di Lorenzo's herders (their design and flow,
unchanged) re-form around it and the flock follows.

Writes viz/hopping_goal.html: the live run (hopping goal), the running time-averaged density for the
hopping and for a fixed goal, the target, and the time-averaged error of both against time.

    .venv/bin/python experiments/hopping_goal_replay.py [--seed 0] [--dwell 20] [--t-end 600]
Serve viz/ (the "viz" launch configuration) and open hopping_goal.html.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402

OUT = ROOT / "viz" / "hopping_goal.html"


def setup(substeps=20):
    F, H, T, D, _ = B.dilorenzo_scenario("jadhav")
    H = replace(H, substeps=substeps)  # 20 sub-steps: same end state as 100 in the page tests, 5× faster
    T = T.resolve(D)
    g = B.Grid(D.size, 64)
    return F, H, T, D, g


def flock_shape(F, H, T, D, g, seeds=(100, 101, 102), steps=300, t0=100):
    """The held flock's density about its own centroid (unit mass on the grid)."""
    acc, n = np.zeros((g.n, g.n)), 0
    for s in seeds:
        r = B.simulate_bridge(F, H, T, D, seed=s, init="uniform", n_steps=steps)
        for k in range(t0, steps + 1, 2):
            x = r.pos_s[k]
            acc += g.deposit(D.wrap(x - D.centroid(x)), np.full(F.n, 1.0 / F.n))
            n += 1
    S = acc / n
    return S / (S.sum() * g.h ** 2)


def centre_distribution(S, T, D, g, reg=1e-2):
    """p_c = ρ̄ ⊘ S by Tikhonov deconvolution; returns (p_c clipped and normalized, negative-mass share)."""
    h2 = g.h ** 2
    rho_t, _ = g.target(T, D)
    rho_t = rho_t / (rho_t.sum() * h2)
    Sh = np.fft.fft2(np.fft.ifftshift(S)) * h2
    P = np.real(np.fft.ifft2(np.fft.fft2(rho_t) * np.conj(Sh) / (np.abs(Sh) ** 2 + reg * np.abs(Sh).max() ** 2)))
    neg = float(-P[P < 0].sum() / P[P > 0].sum())
    P = np.clip(P, 0, None)
    return P / P.sum(), neg


def simulate_hopping(F, H, T, D, g, P, seed, dwell, t_end):
    """Segments of `dwell` s, each with the goal centre drawn from P (dwell = 0: the fixed goal)."""
    rng = np.random.default_rng(seed)
    cells = np.stack([g.X.ravel(), g.Y.ravel()], -1)
    st, t = None, 0
    pos_s, pos_d, centre, hop, designs = [], [], [], [], []
    while t < t_end:
        steps = int(min(dwell or t_end, t_end - t))
        c = cells[rng.choice(len(cells), p=P.ravel())] + (rng.random(2) - 0.5) * g.h if dwell else np.zeros(2)
        Tc = replace(T, centre=(float(c[0]), float(c[1])))
        r = B.simulate_bridge(F, H, Tc, D, seed=int(rng.integers(1 << 30)), init="uniform", init_state=st, n_steps=steps)
        first = 0 if st is None else 1
        pos_s.append(r.pos_s[first:]), pos_d.append(r.pos_d[first:])
        centre += [c] * (len(r.t) - first)
        hop += [len(designs)] * (len(r.t) - first)
        designs.append(r.design.rho_H)
        st = (r.pos_s[-1], r.hd_s[-1], r.pos_d[-1], np.zeros((H.n, 2)))
        t += steps
    return dict(pos_s=np.concatenate(pos_s), pos_d=np.concatenate(pos_d), centre=np.array(centre),
                hop=np.array(hop), designs=designs, t=np.arange(len(np.concatenate(pos_s)), dtype=float) * F.dt)


def running_average(run, F, T, D, g, t0, every):
    """Running time average of the KDE density from t0, its L² error against the smoothed target."""
    met = B.Metrics(T, D, F.n, g)
    h2 = g.h ** 2
    acc, m, err, frames = np.zeros((g.n, g.n)), 0, [], []
    for k, x in enumerate(run["pos_s"]):
        if run["t"][k] >= t0:
            acc += g.convolve(met.Kb, g.deposit(x, np.full(F.n, 1.0 / F.n)))
            m += 1
        avg = acc / m if m else None
        err.append(float(((avg - met.rho_b) ** 2).sum() * h2 / met.norm) if m else None)
        if k % every == 0:
            frames.append(avg)
    return err, frames, met


def down(field, f=2):
    n = field.shape[0] // f
    return field.reshape(n, f, n, f).mean((1, 3))


def to_bytes(field, vmax):
    """Display grid: rows top to bottom (y descending), 0–255."""
    return np.clip(np.round(255 * field.T[::-1] / vmax), 0, 255).astype(int).ravel().tolist()


def main(seed=0, dwell=20, t_end=600, t0=60, frame_every=5):
    F, H, T, D, g = setup()
    S = flock_shape(F, H, T, D, g)
    P, neg = centre_distribution(S, T, D, g)
    h2 = g.h ** 2
    print(f"held-shape RMS {np.sqrt((S * (g.X**2 + g.Y**2)).sum() * h2):.2f} m, p_c RMS {np.sqrt((P * (g.X**2 + g.Y**2)).sum()):.2f} m, "
          f"negative mass {neg*100:.1f}%", flush=True)
    runs = {"hopping": simulate_hopping(F, H, T, D, g, P, seed, dwell, t_end),
            "fixed": simulate_hopping(F, H, T, D, g, P, seed, 0, t_end)}
    avg = {}
    for k, run in runs.items():
        avg[k] = running_average(run, F, T, D, g, t0, frame_every)
        print(f"{k}: final time-averaged error {avg[k][0][-1]:.3f}", flush=True)
    met = avg["hopping"][2]
    target = down(met.rho_b)
    # colour scale from the target and the hopping average; the fixed-goal clump (about 3× the target's peak) saturates
    vmax = 1.15 * max(target.max(), max(down(f).max() for f in avg["hopping"][1][len(avg["hopping"][1]) // 3:] if f is not None))
    hop = runs["hopping"]
    design_max = max(d.max() for d in hop["designs"])
    data = dict(
        box=D.size, r_goal=T.r_goal, r_dog=F.r_dog, n_sheep=F.n, n_dogs=H.n, seed=seed, dwell=dwell, t0=t0,
        frame_every=frame_every, t=hop["t"].tolist(), grid=g.n // 2, design_grid=g.n,
        sheep=np.round(hop["pos_s"], 2).tolist(), dogs=np.round(hop["pos_d"], 2).tolist(),
        centre=np.round(hop["centre"], 2).tolist(), hop=hop["hop"].tolist(),
        designs=[to_bytes(d, design_max) for d in hop["designs"]],
        fixed_sheep=np.round(runs["fixed"]["pos_s"], 2).tolist(), fixed_dogs=np.round(runs["fixed"]["pos_d"], 2).tolist(),
        target=to_bytes(target, vmax), pc=to_bytes(down(P / h2), (P / h2).max()),
        avg={k: [to_bytes(down(f), vmax) if f is not None else None for f in avg[k][1]] for k in avg},
        err={k: [None if e is None else round(e, 4) for e in avg[k][0]] for k in avg},
        floor=float(met.floor), vmax=float(vmax), neg=neg,
        S_rms=float(np.sqrt((S * (g.X ** 2 + g.Y ** 2)).sum() * h2)),
        pc_rms=float(np.sqrt((P * (g.X ** 2 + g.Y ** 2)).sum())),
        target_rms=float(np.sqrt((g.target(T, D)[0] * (g.X ** 2 + g.Y ** 2)).sum() / g.target(T, D)[0].sum())),
    )
    OUT.write_text(PAGE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hopping-goal hybrid</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#d9d6cf;--card:#fff;--sheep:#2f6db5;--dog:#c0392b;--goal:#2e8b57;--hot:236,120,40;--heat:47,109,181}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#3a3936;--card:#201f1e;--sheep:#6ea8ef;--dog:#ef6f5e;--goal:#5cc38a;--hot:240,150,80;--heat:110,168,239}}
:root[data-theme="dark"]{--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#3a3936;--card:#201f1e;--sheep:#6ea8ef;--dog:#ef6f5e;--goal:#5cc38a;--hot:240,150,80;--heat:110,168,239}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1320px;margin:0 auto;padding:20px 16px 40px}h1{font-size:20px;margin:0 0 4px}p.lede{color:var(--mut);margin:0 0 12px;max-width:980px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:10px 0 16px}
button{font:inherit;padding:6px 14px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
input[type=range]{flex:1;min-width:180px}label{color:var(--mut);display:flex;gap:6px;align-items:center}
.grid{display:grid;grid-template-columns:1.25fr 1fr 1fr 1fr;gap:12px}
@media (max-width:1000px){.grid{grid-template-columns:1fr 1fr}}@media (max-width:560px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.card h2{font-size:14px;margin:0 0 2px}.card .note{color:var(--mut);font-size:12px;min-height:36px;margin:0 0 6px}
canvas.field{width:100%;aspect-ratio:1;display:block;border:1px solid var(--line);border-radius:6px}
.stats{font-variant-numeric:tabular-nums;margin-top:6px;font-size:13px}.stats b{font-weight:600}
.trace{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:12px}
canvas#trace{width:100%;height:150px;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:12.5px;margin-top:8px}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
</style></head><body><main>
<h1>Jadhav's sheep, Di Lorenzo's herders, a hopping goal</h1>
<p class="lede" id="lede"></p>
<div class="bar"><button id="play">Pause</button><input id="scrub" type="range" min="0" value="0"><span id="time" style="font-variant-numeric:tabular-nums;min-width:90px"></span>
<label>speed <select id="speed"><option>1</option><option selected>4</option><option>10</option><option>25</option></select></label>
<label><input type="checkbox" id="heat" checked> current herder design</label><label><input type="checkbox" id="pc"> centre distribution p_c</label>
<label><input type="checkbox" id="fixed"> show the fixed-goal run</label></div>
<div class="grid">
  <div class="card"><h2 id="live-title">Live: hopping goal</h2><p class="note" id="live-note"></p><canvas class="field" id="live"></canvas><div class="stats" id="live-stats"></div></div>
  <div class="card"><h2>Time-averaged density, hopping goal</h2><p class="note">Average of the flock's density from t = <span class="t0"></span> s up to now.</p><canvas class="field" id="avg-hop"></canvas><div class="stats" id="hop-stats"></div></div>
  <div class="card"><h2>Time-averaged density, fixed goal</h2><p class="note">Same sheep, same herders, goal never moves: the flock is pinned and the average stays a clump (its peak saturates the colour scale).</p><canvas class="field" id="avg-fix"></canvas><div class="stats" id="fix-stats"></div></div>
  <div class="card"><h2>Target</h2><p class="note">Di Lorenzo's von Mises goal (smoothed like the averages), r* dashed.</p><canvas class="field" id="target"></canvas><div class="stats" id="tgt-stats"></div></div>
</div>
<div class="trace"><canvas id="trace"></canvas>
<div class="legend"><span><span class="sw" style="background:var(--sheep)"></span>time-averaged error, hopping goal</span><span><span class="sw" style="background:var(--mut)"></span>time-averaged error, fixed goal</span><span><span class="sw" style="border-top:2px dotted var(--mut);height:0"></span>finite-N floor of one snapshot</span><span>log scale; relative L² error against the smoothed target</span></div></div>
</main><script>
const DATA = /*__DATA__*/null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = id => document.getElementById(id);
$('lede').textContent = `${DATA.n_sheep} published Jadhav sheep (attraction, repulsion, alignment, noise 0.3, standing rule) and ${DATA.n_dogs} Di Lorenzo herders (their design and flow) `+
  `in a ${DATA.box.toFixed(0)} m periodic box. The flock's own shape (RMS ${DATA.S_rms.toFixed(1)} m) is narrower than the target (RMS ${DATA.target_rms.toFixed(1)} m), so the target is reachable as a time average: `+
  `every ${DATA.dwell} s the goal centre jumps to a point drawn from p_c, the target deconvolved by the flock's shape (RMS ${DATA.pc_rms.toFixed(1)} m; ${(100*DATA.neg).toFixed(1)} % negative mass clipped). `+
  `The herders re-form around the new goal and the flock follows. Seed ${DATA.seed}.`;
document.querySelectorAll('.t0').forEach(e => e.textContent = DATA.t0);
$('live-note').textContent = `Cross: current goal centre; trail: the flock centroid over the last 60 s.`;
const S = DATA.t.length, scrub = $('scrub'); scrub.max = S - 1;
let k = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, acc = 0, last = performance.now();
$('play').textContent = playing ? 'Pause' : 'Play';
function fit(cv){const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1, w = Math.round(r.width*d), h = Math.round(r.height*d);
  if (cv.width !== w || cv.height !== h){cv.width = w; cv.height = h;} return [w, h, d];}
const imgCache = new Map();
function heatImage(key, bytes, n, color, alpha){
  const ck = key + color; if (imgCache.has(ck)) return imgCache.get(ck);
  const c = document.createElement('canvas'); c.width = c.height = n; const ctx = c.getContext('2d'), im = ctx.createImageData(n, n);
  const [r, g, b] = color.split(',').map(Number);
  for (let i = 0; i < n*n; i++){im.data[4*i] = r; im.data[4*i+1] = g; im.data[4*i+2] = b; im.data[4*i+3] = Math.round(bytes[i]*alpha);}
  ctx.putImageData(im, 0, 0); if (imgCache.size > 400) imgCache.clear(); imgCache.set(ck, c); return c;}
function circle(ctx, X, Y, s, d, color){ctx.setLineDash([6*d, 5*d]); ctx.strokeStyle = color; ctx.lineWidth = 1.5*d; ctx.beginPath(); ctx.arc(X(0), Y(0), DATA.r_goal*s, 0, 2*Math.PI); ctx.stroke(); ctx.setLineDash([]);}
function frame(cv){const [w, h, d] = fit(cv), ctx = cv.getContext('2d'), L = DATA.box, s = w / L; ctx.clearRect(0, 0, w, h);
  return {ctx, w, h, d, s, X: x => (x + L/2)*s, Y: y => (L/2 - y)*s};}
const wrap = v => ((v + DATA.box/2) % DATA.box + DATA.box) % DATA.box - DATA.box/2;
function centroid(pts){let cx = 0, sx = 0, cy = 0, sy = 0; const w = 2*Math.PI/DATA.box;
  for (const [x, y] of pts){cx += Math.cos(w*x); sx += Math.sin(w*x); cy += Math.cos(w*y); sy += Math.sin(w*y);} return [Math.atan2(sx, cx)/w, Math.atan2(sy, cy)/w];}
function drawAgents(f, sheep, dogs, cs){const {ctx, X, Y, d} = f;
  ctx.fillStyle = cs.sheep; for (const [x, y] of sheep){ctx.beginPath(); ctx.arc(X(x), Y(y), 3.2*d, 0, 2*Math.PI); ctx.fill();}
  ctx.fillStyle = cs.dog; for (const [x, y] of dogs){const px = X(x), py = Y(y), r = 5*d; ctx.beginPath(); ctx.moveTo(px, py-r); ctx.lineTo(px+r*0.87, py+r*0.5); ctx.lineTo(px-r*0.87, py+r*0.5); ctx.closePath(); ctx.fill();}}
function draw(){
  const cs = {sheep: css('--sheep'), dog: css('--dog'), goal: css('--goal'), mut: css('--mut'), fg: css('--fg'), line: css('--line'), hot: css('--hot'), heat: css('--heat')};
  const fixed = $('fixed').checked;
  $('live-title').textContent = fixed ? 'Live: fixed goal' : 'Live: hopping goal';
  // live
  let f = frame($('live'));
  if (!fixed && $('heat').checked){const hi = DATA.hop[k]; f.ctx.drawImage(heatImage('d' + hi, DATA.designs[hi], DATA.design_grid, cs.hot, 0.45), 0, 0, f.w, f.h);}
  if ($('pc').checked) f.ctx.drawImage(heatImage('pc', DATA.pc, DATA.grid, cs.heat, 0.6), 0, 0, f.w, f.h);
  circle(f.ctx, f.X, f.Y, f.s, f.d, cs.goal);
  const sheep = fixed ? DATA.fixed_sheep : DATA.sheep, dogs = fixed ? DATA.fixed_dogs : DATA.dogs;
  // centroid trail (last 60 s), broken across the seam
  f.ctx.strokeStyle = cs.mut; f.ctx.lineWidth = 1.5*f.d; f.ctx.beginPath(); let prev = null;
  for (let j = Math.max(0, k - 60); j <= k; j++){const c = centroid(sheep[j]);
    if (prev && Math.hypot(c[0]-prev[0], c[1]-prev[1]) < DATA.box/2) f.ctx.lineTo(f.X(c[0]), f.Y(c[1])); else f.ctx.moveTo(f.X(c[0]), f.Y(c[1])); prev = c;}
  f.ctx.stroke();
  drawAgents(f, sheep[k], dogs[k], cs);
  const gc = fixed ? [0, 0] : DATA.centre[k], gx = f.X(gc[0]), gy = f.Y(gc[1]), r = 7*f.d;
  f.ctx.strokeStyle = cs.goal; f.ctx.lineWidth = 2.5*f.d; f.ctx.beginPath(); f.ctx.moveTo(gx-r, gy-r); f.ctx.lineTo(gx+r, gy+r); f.ctx.moveTo(gx+r, gy-r); f.ctx.lineTo(gx-r, gy+r); f.ctx.stroke();
  const c = centroid(sheep[k]);
  $('live-stats').innerHTML = `goal hop <b>${fixed ? '—' : DATA.hop[k] + 1}</b> · flock centre ${Math.hypot(wrap(c[0]), wrap(c[1])).toFixed(1)} m from the target's centre`;
  // averages
  const fi = Math.min(Math.floor(k / DATA.frame_every), DATA.avg.hopping.length - 1);
  for (const [id, key, st] of [['avg-hop', 'hopping', 'hop-stats'], ['avg-fix', 'fixed', 'fix-stats']]){
    f = frame($(id)); const bytes = DATA.avg[key][fi];
    if (bytes) f.ctx.drawImage(heatImage(key + fi, bytes, DATA.grid, cs.heat, 1), 0, 0, f.w, f.h);
    circle(f.ctx, f.X, f.Y, f.s, f.d, cs.goal);
    const e = DATA.err[key][k]; $(st).innerHTML = e == null ? `averaging starts at t = ${DATA.t0} s` : `time-averaged error <b>${e.toFixed(3)}</b>`;}
  f = frame($('target')); f.ctx.drawImage(heatImage('target', DATA.target, DATA.grid, cs.heat, 1), 0, 0, f.w, f.h); circle(f.ctx, f.X, f.Y, f.s, f.d, cs.goal);
  $('tgt-stats').innerHTML = `one snapshot's finite-N floor: <b>${DATA.floor.toFixed(2)}</b>`;
  // trace
  const tv = $('trace'), [tw, th, d] = fit(tv), tc = tv.getContext('2d'); tc.clearRect(0, 0, tw, th);
  const lo = Math.log(0.01), hi = Math.log(3), tx = i => i/(S-1)*tw, ey = e => th - (Math.log(Math.min(Math.max(e, 0.01), 3)) - lo)/(hi - lo)*th;
  tc.strokeStyle = cs.line; tc.lineWidth = d; tc.strokeRect(0.5, 0.5, tw-1, th-1);
  tc.fillStyle = cs.mut; tc.font = `${11*d}px system-ui`; for (const v of [0.01, 0.1, 1]){tc.fillText(String(v), 4*d, ey(v) - 3*d); tc.beginPath(); tc.moveTo(0, ey(v)); tc.lineTo(tw, ey(v)); tc.stroke();}
  tc.setLineDash([2*d, 3*d]); tc.strokeStyle = cs.mut; tc.beginPath(); tc.moveTo(0, ey(DATA.floor)); tc.lineTo(tw, ey(DATA.floor)); tc.stroke(); tc.setLineDash([]);
  for (const [key, col] of [['fixed', cs.mut], ['hopping', cs.sheep]]){tc.strokeStyle = col; tc.lineWidth = 2*d; tc.beginPath(); let on = false;
    DATA.err[key].forEach((e, i) => {if (e == null) return; if (on) tc.lineTo(tx(i), ey(e)); else {tc.moveTo(tx(i), ey(e)); on = true;}}); tc.stroke();}
  tc.strokeStyle = cs.fg; tc.lineWidth = d; tc.beginPath(); tc.moveTo(tx(k), 0); tc.lineTo(tx(k), th); tc.stroke();
  scrub.value = k; $('time').textContent = `t = ${DATA.t[k].toFixed(0)} s`;
}
function loop(now){const dt = (now - last)/1000; last = now;
  if (playing){acc += dt*Number($('speed').value); while (acc >= 1){acc -= 1; k = (k + 1) % S;}}
  draw(); requestAnimationFrame(loop);}
$('play').onclick = e => {playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';};
scrub.oninput = () => {k = Number(scrub.value); playing = false; $('play').textContent = 'Play';};
requestAnimationFrame(loop);
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=int, default=20)
    ap.add_argument("--t-end", type=int, default=600)
    a = ap.parse_args()
    main(a.seed, a.dwell, a.t_end)
