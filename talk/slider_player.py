"""A time slider for one simulation, shown as a notebook output (used by notebooks/simulations.ipynb).

`player(...)` embeds the trajectories as numbers and draws them on a canvas with a Play button, a slider and buttons
that jump to named moments. It needs no running kernel and no external library, so a saved notebook plays offline.
Play always runs the whole simulation in `play_seconds`, however long the simulated run is.
"""

import json
import uuid

import numpy as np
from IPython.display import HTML

SHEEP, HERDER, GOAL, INK = "#C2185B", "#1565C0", "#2E7D32", "#424242"  # the deck's colours


def _frames(a, decimals):
    return [np.round(x, decimals).ravel().tolist() for x in np.asarray(a, float)]


def _series(y, decimals=3):
    return [None if not np.isfinite(v) else round(float(v), decimals) for v in np.asarray(y, float)]


def strip(title, series, ymin, ymax, hlines=()):
    """A small time-series chart under the slider. series: (label, colour, values per frame); hlines: (y, label)."""
    return dict(title=title, ymin=float(ymin), ymax=float(ymax), hlines=[dict(y=float(y), label=lab) for y, lab in hlines],
                series=[dict(label=lab, color=col, y=_series(y)) for lab, col, y in series])


def panel(title, sheep, dogs, xlim, ylim, goal=None, caption=None, heat=None, view=None, trails=(), headings=None,
          heading_len=1.5, dog_ring=None, highlight=None, scale=None, scale_label="", sheep_px=3.5, dog_px=7.0, decimals=2):
    """One view of the run. sheep and dogs: (frames, agents, 2), or None to reuse the first panel's.

    goal: dict(x, y, r, fill=False, label="")     heat: dict(grid=2D array with row 0 at the bottom, extent=(x0, x1, y0, y1))
    view: dict(cx=[...], cy=[...], half=h), a window that follows a point, in place of xlim/ylim
    trails: (points (frames, 2), colour, line width)     headings: (frames, agents, 2) unit vectors drawn at the sheep
    highlight: index of one sheep per frame to ring     caption: one string per frame
    """
    out = dict(title=title, xlim=[float(v) for v in xlim], ylim=[float(v) for v in ylim], sheep_px=float(sheep_px), dog_px=float(dog_px), heading_len=float(heading_len))
    if sheep is not None:
        out["sheep"], out["dogs"] = _frames(sheep, decimals), _frames(dogs, decimals)
    if goal is not None:
        out["goal"] = dict(x=float(goal["x"]), y=float(goal["y"]), r=float(goal["r"]), fill=bool(goal.get("fill", False)), label=goal.get("label", ""))
    if caption is not None:
        out["caption"] = list(caption)
    if heat is not None:
        g = np.asarray(heat["grid"], float)
        values = np.round(150 * np.clip(g, 0, None) / max(float(g.max()), 1e-12)).astype(int)
        out["heat"] = dict(nx=int(g.shape[1]), ny=int(g.shape[0]), values=values.ravel().tolist(), extent=[float(v) for v in heat["extent"]])
    if view is not None:
        out["view"] = dict(cx=_series(view["cx"], 2), cy=_series(view["cy"], 2), half=float(view["half"]))
    if len(trails):
        out["trails"] = [dict(x=_series(np.asarray(xy)[:, 0], 2), y=_series(np.asarray(xy)[:, 1], 2), color=col, width=float(w)) for xy, col, w in trails]
    if headings is not None:
        out["headings"] = _frames(headings, 2)
    if dog_ring is not None:
        out["dog_ring"] = float(dog_ring)
    if highlight is not None:
        out["highlight"] = [int(i) for i in highlight]
    if scale is not None:
        out["scale"] = float(scale)
        out["scale_label"] = scale_label
    return out


_SCRIPT = r"""
(function () {
  const root = document.getElementById("/*ID*/");
  if (!root) return;
  root.querySelector(".sp-untrusted").remove();
  const ALL = /*DATA*/;
  const cv = root.querySelector("canvas"), c = cv.getContext("2d");
  const slider = root.querySelector(".sp-time"), play = root.querySelector(".sp-play"), clock = root.querySelector(".sp-clock");
  const jumps = root.querySelector(".sp-jumps"), note = root.querySelector(".sp-note");
  const choice = root.querySelector(".sp-choice"), choiceValue = root.querySelector(".sp-choice-value");
  const INK = "#424242", GRID = "#E0E0E0", MUTED = "#757575", EVENT = "#2E7D32";
  const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
  let D = null, n = 0, t0 = 0, t1 = 0, paceT = [], paceS = [];
  let k = 0, playing = false, clock0 = 0, second0 = 0, W = 0, H = 0, rects = null;

  function interp(x, xs, ys) {
    for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1]);
    return ys[ys.length - 1];
  }

  function prepare() {
    for (const p of D.panels) {
      if (!p.sheep) { p.sheep = D.panels[0].sheep; p.dogs = D.panels[0].dogs; }
      if (!p.heat || p.heat.img) continue;
      const h = p.heat, off = document.createElement("canvas");
      off.width = h.nx; off.height = h.ny;
      const ctx = off.getContext("2d"), img = ctx.createImageData(h.nx, h.ny);
      for (let j = 0; j < h.ny; j++) for (let i = 0; i < h.nx; i++) {
        const o = 4 * (i + (h.ny - 1 - j) * h.nx);
        img.data[o] = img.data[o + 1] = img.data[o + 2] = 90;
        img.data[o + 3] = h.values[j * h.nx + i];
      }
      ctx.putImageData(img, 0, 0);
      h.img = off;
    }
  }

  function jumpButtons() {
    jumps.textContent = "";
    jumps.style.display = D.events.length ? "flex" : "none";
    if (!D.events.length) return;
    jumps.append("Jump to: ");
    for (const e of D.events) {
      const b = document.createElement("button");
      b.className = "sp-jump"; b.style.cssText = ALL.button; b.textContent = e.label;
      b.addEventListener("click", () => { k = e.frame; setPlaying(false); draw(); });
      jumps.append(b);
    }
  }

  // Switch to run i and keep the same simulated time (and keep playing, if it was).
  function use(i) {
    const now = D ? D.t[k] : null;
    D = ALL.variants[i];
    n = D.t.length; t0 = D.t[0]; t1 = D.t[n - 1];
    paceT = D.pace.map(p => p[0]); paceS = D.pace.map(p => p[1]);
    prepare();
    slider.max = n - 1;
    if (now !== null) k = Math.min(n - 1, Math.round((now - t0) / Math.max(t1 - t0, 1e-9) * (n - 1)));
    jumpButtons();
    note.textContent = D.note;
    if (choice) choiceValue.textContent = ALL.choice.values[i];
    if (playing) { second0 = interp(D.t[k], paceT, paceS); clock0 = performance.now(); }
    W = 0;
    layout();
  }

  function layout() {
    const w = cv.clientWidth;
    if (!w || w === W) return;
    W = w;
    const nP = D.panels.length, nS = D.strips.length, gap = 14, head = 22, foot = 24;
    rects = { panels: [], strips: [] };
    if (nP === 1 && nS) {
      const side = Math.min(W * 0.52, 480), x = side + 22;
      rects.panels.push({ x: 0, y: head, w: side, h: side });
      H = head + side + foot;
      const h = (H - 10 * (nS - 1)) / nS;
      D.strips.forEach((_, i) => rects.strips.push({ x: x, y: i * (h + 10), w: W - x, h: h }));
    } else {
      const pw = (W - gap * (nP - 1)) / nP, ph = Math.min(pw, 420), sh = nS ? 132 : 0;
      const sw = (W - gap * (nS - 1)) / Math.max(nS, 1);
      D.panels.forEach((_, i) => rects.panels.push({ x: i * (pw + gap), y: head, w: pw, h: ph }));
      D.strips.forEach((_, i) => rects.strips.push({ x: i * (sw + gap), y: head + ph + foot + 8, w: sw, h: sh }));
      H = head + ph + foot + (nS ? sh + 8 : 0);
    }
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + "px";
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function text(s, x, y, font, color, align) {
    c.font = font + " " + FONT; c.fillStyle = color; c.textAlign = align || "left"; c.fillText(s, x, y);
  }

  function drawPanel(p, r) {
    text(p.title, r.x + r.w / 2, r.y - 8, "600 13px", INK, "center");
    const v = p.view ? [p.view.cx[k] - p.view.half, p.view.cx[k] + p.view.half, p.view.cy[k] - p.view.half, p.view.cy[k] + p.view.half]
                     : [p.xlim[0], p.xlim[1], p.ylim[0], p.ylim[1]];
    const s = Math.min(r.w / (v[1] - v[0]), r.h / (v[3] - v[2]));
    const bw = s * (v[1] - v[0]), bh = s * (v[3] - v[2]), ox = r.x + (r.w - bw) / 2, oy = r.y + (r.h - bh) / 2;
    const X = x => ox + (x - v[0]) * s, Y = y => oy + (v[3] - y) * s;
    c.save();
    c.beginPath(); c.rect(ox, oy, bw, bh); c.clip();
    c.fillStyle = "#FAFAFA"; c.fillRect(ox, oy, bw, bh);
    if (p.heat) { const e = p.heat.extent; c.imageSmoothingEnabled = true; c.drawImage(p.heat.img, X(e[0]), Y(e[3]), (e[1] - e[0]) * s, (e[3] - e[2]) * s); }
    if (p.goal) {
      const g = p.goal;
      c.beginPath(); c.arc(X(g.x), Y(g.y), g.r * s, 0, 2 * Math.PI);
      if (g.fill) { c.fillStyle = "rgba(46,125,50,0.3)"; c.fill(); }
      else { c.strokeStyle = "#2E7D32"; c.lineWidth = 2; c.setLineDash([7, 5]); c.stroke(); c.setLineDash([]); }
      if (g.label) text(g.label, X(g.x), Y(g.y) - g.r * s - 5, "12px", "#2E7D32", "center");
    }
    for (const tr of p.trails || []) {
      c.beginPath();
      for (let i = 0; i <= k; i++) { i ? c.lineTo(X(tr.x[i]), Y(tr.y[i])) : c.moveTo(X(tr.x[i]), Y(tr.y[i])); }
      c.strokeStyle = tr.color; c.lineWidth = tr.width; c.stroke();
    }
    const dogs = p.dogs[k], sheep = p.sheep[k];
    if (p.dog_ring) {
      c.strokeStyle = "rgba(21,101,192,0.7)"; c.lineWidth = 1; c.setLineDash([5, 4]);
      for (let i = 0; i < dogs.length; i += 2) { c.beginPath(); c.arc(X(dogs[i]), Y(dogs[i + 1]), p.dog_ring * s, 0, 2 * Math.PI); c.stroke(); }
      c.setLineDash([]);
    }
    if (p.headings) {
      const h = p.headings[k];
      c.beginPath();
      for (let i = 0; i < sheep.length; i += 2) {
        c.moveTo(X(sheep[i]), Y(sheep[i + 1]));
        c.lineTo(X(sheep[i] + p.heading_len * h[i]), Y(sheep[i + 1] + p.heading_len * h[i + 1]));
      }
      c.strokeStyle = "#C2185B"; c.lineWidth = 1.5; c.stroke();
    }
    c.fillStyle = "#C2185B"; c.beginPath();
    for (let i = 0; i < sheep.length; i += 2) { const x = X(sheep[i]), y = Y(sheep[i + 1]); c.moveTo(x + p.sheep_px, y); c.arc(x, y, p.sheep_px, 0, 2 * Math.PI); }
    c.fill();
    if (p.highlight) {
      const i = 2 * p.highlight[k];
      c.beginPath(); c.arc(X(sheep[i]), Y(sheep[i + 1]), p.sheep_px + 5, 0, 2 * Math.PI); c.strokeStyle = INK; c.lineWidth = 1.5; c.stroke();
    }
    c.fillStyle = "#1565C0"; c.strokeStyle = "#FFFFFF"; c.lineWidth = 0.7;
    for (let i = 0; i < dogs.length; i += 2) {
      const x = X(dogs[i]), y = Y(dogs[i + 1]), d = p.dog_px;
      c.beginPath(); c.moveTo(x, y - d); c.lineTo(x + 0.87 * d, y + 0.5 * d); c.lineTo(x - 0.87 * d, y + 0.5 * d); c.closePath(); c.fill(); c.stroke();
    }
    if (p.scale) {
      const x = ox + 12, y = oy + bh - 12;
      c.strokeStyle = INK; c.lineWidth = 2; c.beginPath(); c.moveTo(x, y); c.lineTo(x + p.scale * s, y); c.stroke();
      text(p.scale_label, x, y - 6, "11px", INK);
    }
    c.restore();
    c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(ox, oy, bw, bh);
    if (p.caption) text(p.caption[k], r.x + r.w / 2, r.y + r.h + 16, "12.5px", INK, "center");
  }

  function drawStrip(s, r) {
    const a = { l: r.x + 36, r: r.x + r.w - 8, t: r.y + 20, b: r.y + r.h - 16 };
    const tx = t => a.l + (t - t0) / Math.max(t1 - t0, 1e-9) * (a.r - a.l);
    const yy = v => a.b - (v - s.ymin) / (s.ymax - s.ymin) * (a.b - a.t);
    text(s.title, r.x, r.y + 11, "600 11.5px", INK);
    let lx = r.x + r.w;
    for (const q of s.series.slice().reverse()) {
      if (s.series.length < 2) break;
      text(q.label, lx, r.y + 11, "11px", q.color, "right");
      lx -= c.measureText(q.label).width + 10;
    }
    c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(a.l, a.t, a.r - a.l, a.b - a.t);
    text(String(+s.ymax.toPrecision(3)), a.l - 5, a.t + 8, "10.5px", MUTED, "right");
    text(String(+s.ymin.toPrecision(3)), a.l - 5, a.b, "10.5px", MUTED, "right");
    text(Math.round(t0) + " " + D.unit, a.l, a.b + 12, "10.5px", MUTED);
    text(Math.round(t1) + " " + D.unit, a.r, a.b + 12, "10.5px", MUTED, "right");
    c.save();
    c.beginPath(); c.rect(a.l, a.t, a.r - a.l, a.b - a.t); c.clip();
    for (const h of s.hlines) {
      c.strokeStyle = MUTED; c.lineWidth = 1; c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(a.l, yy(h.y)); c.lineTo(a.r, yy(h.y)); c.stroke(); c.setLineDash([]);
      text(h.label, a.r - 4, yy(h.y) - 3, "10.5px", MUTED, "right");
    }
    for (const e of D.events) {
      c.strokeStyle = EVENT; c.lineWidth = 1; c.setLineDash([2, 3]);
      c.beginPath(); c.moveTo(tx(e.t), a.t); c.lineTo(tx(e.t), a.b); c.stroke(); c.setLineDash([]);
    }
    for (const q of s.series) {
      c.strokeStyle = q.color; c.lineWidth = 1.6; c.beginPath();
      let on = false;
      q.y.forEach((v, i) => {
        if (v === null) { on = false; return; }
        on ? c.lineTo(tx(D.t[i]), yy(v)) : c.moveTo(tx(D.t[i]), yy(v));
        on = true;
      });
      c.stroke();
    }
    c.strokeStyle = INK; c.lineWidth = 1.2; c.beginPath(); c.moveTo(tx(D.t[k]), a.t); c.lineTo(tx(D.t[k]), a.b); c.stroke();
    for (const q of s.series) {
      if (q.y[k] === null) continue;
      c.fillStyle = q.color; c.beginPath(); c.arc(tx(D.t[k]), yy(q.y[k]), 3.5, 0, 2 * Math.PI); c.fill();
    }
    c.restore();
  }

  function draw() {
    if (!rects) return;
    c.fillStyle = "#FFFFFF"; c.fillRect(0, 0, W, H);
    D.panels.forEach((p, i) => drawPanel(p, rects.panels[i]));
    D.strips.forEach((s, i) => drawStrip(s, rects.strips[i]));
    slider.value = k;
    clock.textContent = "t = " + Math.round(D.t[k]) + " / " + Math.round(t1) + " " + D.unit;
  }

  function setPlaying(on) {
    playing = on;
    play.textContent = on ? "❚❚ Pause" : (k >= n - 1 ? "↻ Replay" : "▶ Play");
    if (on) {
      if (k >= n - 1) k = 0;
      second0 = interp(D.t[k], paceT, paceS); clock0 = performance.now();
      requestAnimationFrame(tick);
    }
  }

  function tick(now) {
    if (!playing) return;
    const s = second0 + (now - clock0) / 1000;
    if (s >= D.play_seconds) { k = n - 1; setPlaying(false); }
    else k = Math.round((interp(s, paceS, paceT) - t0) / (t1 - t0) * (n - 1));
    draw();
    if (playing) requestAnimationFrame(tick);
  }

  play.addEventListener("click", () => setPlaying(!playing));
  slider.addEventListener("input", () => { k = +slider.value; setPlaying(false); draw(); });
  if (choice) choice.addEventListener("input", () => use(+choice.value));
  new ResizeObserver(layout).observe(cv);
  use(ALL.start);
})();
"""

_BUTTON = "padding:3px 10px;border:1px solid #BDBDBD;border-radius:5px;background:#FFFFFF;color:#424242;cursor:pointer;font:inherit;"


def variant(t, panels, strips=(), events=(), note="", unit="s", play_seconds=15.0, pace=()):
    """One run's data for a player. events: (time, label) moments to jump to.

    pace: (simulated time, playback second) points that give part of the run more playback time; the whole run
    still ends at play_seconds. Without it the run plays at a uniform rate.
    """
    t = np.asarray(t, float)
    span = float(t[-1] - t[0])
    points = [(float(t[0]), 0.0), *((float(a), float(b)) for a, b in pace), (float(t[-1]), float(play_seconds))]
    pace_t, pace_s = zip(*points)
    jumps, marks = [], []
    for when, label in events:
        k = int(np.argmin(np.abs(t - when)))
        into = float(np.interp(t[k], pace_t, pace_s))
        jumps.append(dict(t=round(float(t[k]), 2), label=label, frame=k))
        marks.append(f"{label}: t = {t[k]:.0f} {unit}, {into:.1f} s into playback")
    timing = f"Play runs the whole {span:.0f} {unit} in {play_seconds:.0f} s"
    if len(pace) > 1:
        timing += "; the speed varies so the key moments get more of that time."
    else:
        timing += "." if not pace else f"; the first {pace[0][0]:.0f} {unit} take {pace[0][1]:.0f} s of that."
    text = " ".join(x for x in (timing, "; ".join(marks) + ("." if marks else ""), note) if x)
    return dict(t=_series(t, 2), panels=list(panels), strips=list(strips), events=jumps, play_seconds=float(play_seconds), unit=unit,
                pace=[list(p) for p in points], note=text)


def player(title, t, panels, strips=(), events=(), note="", unit="s", play_seconds=15.0, pace=()):
    """The HTML output: panels and strips over a shared time slider."""
    return _html(title, [variant(t, panels, strips, events, note, unit, play_seconds, pace)])


def choice_player(title, label, choices, start=0):
    """A player with a second slider that switches between runs. choices: (value text, variant(...))."""
    return _html(title, [v for _, v in choices], dict(label=label, values=[text for text, _ in choices]), start)


def _html(title, variants, choice=None, start=0):
    uid = "sp" + uuid.uuid4().hex[:10]
    data = dict(variants=variants, start=int(start), choice=choice, button=_BUTTON)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    choice_row = "" if choice is None else f"""
  <div style="display:flex;gap:12px;align-items:center;margin:2px 0 8px;">
    <span style="font-weight:600;font-size:13px;white-space:nowrap;min-width:92px;">{choice["label"]}</span>
    <input class="sp-choice" type="range" min="0" max="{len(choice["values"]) - 1}" value="{int(start)}" step="1" style="flex:1;" aria-label="{choice["label"]}">
    <span class="sp-choice-value" style="min-width:128px;text-align:right;font-size:13px;font-weight:600;"></span>
  </div>"""
    return HTML(f"""
<div id="{uid}" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#FFFFFF;color:#424242;padding:10px 14px;border:1px solid #E0E0E0;border-radius:8px;box-sizing:border-box;width:1180px;max-width:100%;">
  <div style="font-weight:600;font-size:15px;margin-bottom:4px;">{title}</div>
  <div class="sp-untrusted" style="margin:6px 0;padding:6px 10px;background:#FFF3E0;border-radius:5px;font-size:13px;">This slider is inactive because the notebook is not trusted on this computer. Run this cell, or run: jupyter trust simulations.ipynb</div>
  {choice_row}
  <canvas style="display:block;width:100%;"></canvas>
  <div style="display:flex;gap:12px;align-items:center;margin-top:8px;">
    <button class="sp-play" style="{_BUTTON}min-width:92px;">&#9654; Play</button>
    <input class="sp-time" type="range" min="0" max="0" value="0" step="1" style="flex:1;" aria-label="Simulation time">
    <span class="sp-clock" style="min-width:128px;text-align:right;font-size:13px;font-variant-numeric:tabular-nums;"></span>
  </div>
  <div class="sp-jumps" style="margin-top:7px;display:none;flex-wrap:wrap;gap:6px;align-items:center;font-size:12.5px;"></div>
  <div class="sp-note" style="margin-top:6px;font-size:12px;color:#757575;"></div>
</div>
<script>{_SCRIPT.replace("/*ID*/", uid).replace("/*DATA*/", payload)}</script>
""")
