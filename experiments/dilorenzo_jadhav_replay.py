"""Replay page for Di Lorenzo's scenario with Jadhav's sheep (bridge.dilorenzo_scenario).

Simulates one seed three ways, with identical sheep and herder starts, and writes a self-contained page,
viz/dilorenzo_jadhav_sheep.html:

  their followers   Di Lorenzo's first-order followers in metres (the reference: design and sheep agree)
  Jadhav sheep      the published heading sheep, same herders, same design
  flat design       Jadhav sheep, same herders, design computed with D = 0 (dogs present, nothing designed)

    .venv/bin/python experiments/dilorenzo_jadhav_replay.py [--seed 0] [--steps 300]
Serve viz/ (the "viz" launch config) and open dilorenzo_jadhav_sheep.html.
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

OUT = ROOT / "viz" / "dilorenzo_jadhav_sheep.html"


def panel(label, note, cfg, seed, steps, dt_out):
    F, H, T, D, kw = cfg
    n = int(round(steps / F.dt))
    every = max(int(round(dt_out / F.dt)), 1)
    r = B.simulate_bridge(F, H, T, D, seed=seed, init=kw["init"], n_steps=n, store_every=every)
    rho = r.design.rho_H
    rho = (255 * (rho - rho.min()) / max(rho.max() - rho.min(), 1e-12)).astype(int)
    return dict(label=label, note=note, t=np.round(r.t, 2).tolist(),
                sheep=np.round(r.pos_s, 2).tolist(), dogs=np.round(r.pos_d, 2).tolist(),
                occupancy=np.round(r.metrics["occupancy"], 3).tolist(),
                error=np.round(r.metrics["density_error"], 3).tolist(),
                cohesion=np.round(r.metrics["cohesion"], 2).tolist(),
                rho=rho.T[::-1].ravel().tolist(), grid=int(rho.shape[0]),
                feasible=bool(r.design.feasible), mass_needed=float(r.design.mass_needed),
                floor=float(r.floor))


def main(seed=0, steps=300):
    jad = B.dilorenzo_scenario("jadhav")
    ref = B.dilorenzo_scenario("first_order")
    F, H, T, D, kw = jad
    flat = (F, replace(H, design_model=replace(H.design_model, D=0.0)), T, D, kw)
    panels = [
        panel("Their followers (reference)",
              "Di Lorenzo's first-order followers in metres: additive push e^(−r/12 m) from every herder, Brownian D. "
              "The design is derived for exactly these followers.", ref, seed, steps, 1.0),
        panel("Jadhav sheep",
              "Published heading sheep (1 m/s, repulsion, 4-of-10 attraction, alignment, noise 0.3, flee herders within "
              "12 m, stand still otherwise). Same herders, same design.", jad, seed, steps, 1.0),
        panel("Jadhav sheep, flat design",
              "Same sheep and the same 14 herders running the same flow, but designed with D = 0: the target asks for no "
              "drift, so the herders stay spread over the box.", flat, seed, steps, 1.0),
    ]
    data = dict(box=D.size, r_goal=T.resolve(D).r_goal, r_dog=F.r_dog, n_sheep=F.n, n_dogs=H.n, seed=seed, panels=panels)
    OUT.write_text(PAGE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Di Lorenzo herders, Jadhav sheep</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#d9d6cf;--card:#fff;--sheep:#2f6db5;--dog:#c0392b;--goal:#2e8b57;--heat:47,109,181}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#3a3936;--card:#201f1e;--sheep:#6ea8ef;--dog:#ef6f5e;--goal:#5cc38a;--heat:110,168,239}}
:root[data-theme="dark"]{--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#3a3936;--card:#201f1e;--sheep:#6ea8ef;--dog:#ef6f5e;--goal:#5cc38a;--heat:110,168,239}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1280px;margin:0 auto;padding:20px 16px 40px}h1{font-size:20px;margin:0 0 4px}p.lede{color:var(--mut);margin:0 0 14px;max-width:900px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:10px 0 16px}
button{font:inherit;padding:6px 14px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
input[type=range]{flex:1;min-width:180px}label{color:var(--mut);display:flex;gap:6px;align-items:center}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:15px;margin:0 0 2px}.card .note{color:var(--mut);font-size:12.5px;min-height:54px;margin:0 0 8px}
canvas.field{width:100%;aspect-ratio:1;display:block;border:1px solid var(--line);border-radius:6px}
canvas.trace{width:100%;height:70px;display:block;margin-top:8px}
.stats{display:flex;gap:14px;font-variant-numeric:tabular-nums;margin-top:6px;font-size:13px}.stats b{font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:12.5px;margin-top:14px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle;margin-right:5px}
</style></head><body><main>
<h1>Di Lorenzo's herders, Jadhav's sheep</h1>
<p class="lede" id="lede"></p>
<div class="bar"><button id="play">Pause</button><input id="scrub" type="range" min="0" value="0"><span id="time" style="font-variant-numeric:tabular-nums;min-width:70px"></span>
<label>speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
<label><input type="checkbox" id="discs"> herders' 12 m notice discs</label><label><input type="checkbox" id="heat" checked> designed herder density</label></div>
<div class="grid" id="panels"></div>
<div class="legend"><span><span class="dot" style="background:var(--sheep)"></span>sheep</span><span><span class="dot" style="background:var(--dog)"></span>herders</span>
<span><span class="dot" style="border:1.5px dashed var(--goal)"></span>r* (goal disc)</span><span>shading: designed herder density ρ̄_H</span>
<span>traces: fraction within r* (solid), relative L² density error (dashed, log, floor dotted)</span></div>
</main><script>
const DATA = /*__DATA__*/null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
document.getElementById('lede').textContent =
  `Di Lorenzo et al. 2024 (Sec. IV-A) rescaled to metres: ${DATA.box.toFixed(0)} m periodic box, von Mises goal with r* = ${DATA.r_goal.toFixed(1)} m, `+
  `${DATA.n_sheep} followers and ${DATA.n_dogs} herders (their 28 % herder fraction). The herders keep everything of theirs: the feedforward `+
  `design deconvolved from their follower model and the Theorem-1 density flow (K_H = 10, uncapped). Only the followers change. Seed ${DATA.seed}; identical starts in every panel.`;
const host = document.getElementById('panels');
const P = DATA.panels.map(p => {
  const c = document.createElement('div'); c.className = 'card';
  c.innerHTML = `<h2></h2><p class="note"></p><canvas class="field"></canvas><div class="stats"></div><canvas class="trace"></canvas>`;
  c.querySelector('h2').textContent = p.label; c.querySelector('.note').textContent = p.note; host.appendChild(c);
  const heat = document.createElement('canvas'); heat.width = heat.height = p.grid;
  return {p, field: c.querySelector('.field'), trace: c.querySelector('.trace'), stats: c.querySelector('.stats'), heat, heatColor: null};
});
const S = DATA.panels[0].t.length; const scrub = document.getElementById('scrub'); scrub.max = S - 1;
let k = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, acc = 0, last = performance.now();
document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
function fit(cv){const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; const w = Math.round(r.width*d), h = Math.round(r.height*d);
  if (cv.width !== w || cv.height !== h){cv.width = w; cv.height = h;} return [w, h, d];}
function paintHeat(o){const col = css('--heat'); if (o.heatColor === col) return; o.heatColor = col;
  const g = o.p.grid, ctx = o.heat.getContext('2d'), im = ctx.createImageData(g, g), [r, gg, b] = col.split(',').map(Number);
  for (let i = 0; i < g*g; i++){im.data[4*i] = r; im.data[4*i+1] = gg; im.data[4*i+2] = b; im.data[4*i+3] = Math.round(o.p.rho[i]*0.55);}
  ctx.putImageData(im, 0, 0);}
function draw(){
  const L = DATA.box, half = L/2, showDiscs = document.getElementById('discs').checked, showHeat = document.getElementById('heat').checked;
  const cs = {sheep: css('--sheep'), dog: css('--dog'), goal: css('--goal'), line: css('--line'), mut: css('--mut'), fg: css('--fg')};
  for (const o of P){
    const [w, h, d] = fit(o.field), ctx = o.field.getContext('2d'), s = w / L, X = x => (x + half)*s, Y = y => (half - y)*s;
    ctx.clearRect(0, 0, w, h);
    if (showHeat){paintHeat(o); ctx.imageSmoothingEnabled = true; ctx.drawImage(o.heat, 0, 0, w, h);}
    ctx.setLineDash([6*d, 5*d]); ctx.strokeStyle = cs.goal; ctx.lineWidth = 1.6*d;
    ctx.beginPath(); ctx.arc(X(0), Y(0), DATA.r_goal*s, 0, 2*Math.PI); ctx.stroke(); ctx.setLineDash([]);
    const dogs = o.p.dogs[k];
    if (showDiscs){ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip(); ctx.fillStyle = cs.dog; ctx.globalAlpha = 0.05;
      for (const [x, y] of dogs) for (const ox of [-L, 0, L]) for (const oy of [-L, 0, L]){ctx.beginPath(); ctx.arc(X(x+ox), Y(y+oy), DATA.r_dog*s, 0, 2*Math.PI); ctx.fill();}
      ctx.restore();}
    ctx.fillStyle = cs.sheep; for (const [x, y] of o.p.sheep[k]){ctx.beginPath(); ctx.arc(X(x), Y(y), 3.2*d, 0, 2*Math.PI); ctx.fill();}
    ctx.fillStyle = cs.dog; for (const [x, y] of dogs){const px = X(x), py = Y(y), r = 5*d; ctx.beginPath(); ctx.moveTo(px, py-r); ctx.lineTo(px+r*0.87, py+r*0.5); ctx.lineTo(px-r*0.87, py+r*0.5); ctx.closePath(); ctx.fill();}
    o.stats.innerHTML = `<span>within r* <b>${(100*o.p.occupancy[k]).toFixed(0)}%</b></span><span>density error <b>${o.p.error[k].toFixed(2)}</b></span><span>cohesion <b>${o.p.cohesion[k].toFixed(1)} m</b></span>`;
    // traces
    const [tw, th] = fit(o.trace), tc = o.trace.getContext('2d'); tc.clearRect(0, 0, tw, th);
    const tx = i => i/(S-1)*tw, lo = Math.log(0.1), hi = Math.log(5), ey = e => th - (Math.log(Math.min(Math.max(e, 0.1), 5)) - lo)/(hi - lo)*th;
    tc.strokeStyle = cs.line; tc.lineWidth = d; tc.strokeRect(0.5, 0.5, tw-1, th-1);
    tc.setLineDash([2*d, 3*d]); tc.strokeStyle = cs.mut; tc.beginPath(); tc.moveTo(0, ey(o.p.floor)); tc.lineTo(tw, ey(o.p.floor)); tc.stroke();
    tc.setLineDash([5*d, 3*d]); tc.strokeStyle = cs.dog; tc.beginPath(); o.p.error.forEach((e, i) => i ? tc.lineTo(tx(i), ey(e)) : tc.moveTo(0, ey(e))); tc.stroke();
    tc.setLineDash([]); tc.strokeStyle = cs.sheep; tc.lineWidth = 1.6*d; tc.beginPath(); o.p.occupancy.forEach((v, i) => i ? tc.lineTo(tx(i), th - v*th) : tc.moveTo(0, th - v*th)); tc.stroke();
    tc.strokeStyle = cs.fg; tc.lineWidth = d; tc.beginPath(); tc.moveTo(tx(k), 0); tc.lineTo(tx(k), th); tc.stroke();
  }
  scrub.value = k; document.getElementById('time').textContent = `t = ${DATA.panels[0].t[k].toFixed(0)} s`;
}
function loop(now){const dt = (now - last)/1000; last = now;
  if (playing){acc += dt*12*Number(document.getElementById('speed').value); while (acc >= 1){acc -= 1; k = (k + 1) % S;}}
  draw(); requestAnimationFrame(loop);}
document.getElementById('play').onclick = e => {playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';};
scrub.oninput = () => {k = Number(scrub.value); playing = false; document.getElementById('play').textContent = 'Play';};
requestAnimationFrame(loop);
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=300.0, help="simulated seconds")
    a = ap.parse_args()
    main(a.seed, a.steps)
