// Herding model of Jadhav et al. (2024), ported from shepherd/model.py
// (itself a port of model/herding_model.m), including the extensions used for
// calibration: time step dt with heading relaxation (tauS, tauD), separate
// dog noise, random sideways flanking of the drive point (Ornstein-Uhlenbeck,
// flankSigma/flankTau), a dog speed law (dogSpeedGain) and "social always on".
// With the defaults this is the published model. `outrun` (the dog circles the
// flock instead of cutting through it when the target switches) is an
// addition for the viewer's back-and-forth scenario only.

const DEFAULTS = {
  n: 14, rRep: 2, rDog: 12, K: 10, nAtt: 4, nAli: 1,
  vSheep: 1, vDog: 1.5, wInertia: 0.5, wRep: 2, wDog: 1, noise: 0.3,
  wAtt: 1.5, wAli: 1.3, slowStep: 0.05,
  lSep: null, lDrive: null, lCollect: null, // null = paper's defaults from rRep and n
  dt: 1, tauS: null, tauD: null, noiseDog: null, // null = no relaxation / same noise as sheep
  flankSigma: 0, flankTau: 2, dogSpeedGain: null, socialAlways: false, outrun: false,
  dogKernel: null, gDogK: 1, // learned dog response: {knots, D, V, L} (see shepherd/model.py)
};

const MODE = { INIT: 0, DRIVE: 1, COLLECT: 2, SLOW: 3 };

function resolveParams(p) {
  const q = Object.assign({}, DEFAULTS, p);
  if (q.lSep == null) q.lSep = q.rRep * Math.pow(q.n, 2 / 3);
  if (q.lDrive == null) q.lDrive = q.rRep * Math.sqrt(q.n);
  if (q.lCollect == null) q.lCollect = q.rRep;
  if (q.noiseDog == null) q.noiseDog = q.noise;
  q.K = Math.min(q.K, q.n - 1);
  q.nAtt = Math.min(q.nAtt, q.K);
  q.nAli = Math.min(q.nAli, q.nAtt);
  return q;
}

function norm2(x, y) { const m = Math.hypot(x, y); return m > 0 ? [x / m, y / m] : [0, 0]; }

// Hat-function kernel value at r (0 beyond the last knot), as in shepherd/kernels.py.
function hat(knots, coef, r) {
  let v = 0;
  for (let m = 0; m < knots.length - 1; m++) {
    const peak = knots[m], left = m === 0 ? knots[0] - (knots[1] - knots[0]) : knots[m - 1], right = knots[m + 1];
    const b = Math.min((r - left) / (peak - left), (right - r) / (right - peak));
    if (b > 0) v += b * coef[m];
  }
  return v;
}

function gauss(rng) { // Box-Muller
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Turn part of the way from the old heading toward the desired one (a = dt / tau).
function relax(ox, oy, nx, ny, tau, dt) {
  if (tau == null || tau <= dt) return [nx, ny];
  const a = dt / tau;
  return norm2((1 - a) * ox + a * nx, (1 - a) * oy + a * ny);
}

// start: {x, y} flock start centre, spread (m); dogOffset: extra distance of dog start
function createSim(params, start, target, rng = Math.random) {
  const p = resolveParams(params);
  const N = p.n;
  const s = {
    p, N, t: 0, rng,
    x: new Float64Array(N), y: new Float64Array(N),
    vx: new Float64Array(N), vy: new Float64Array(N),
    dx: 0, dy: 0, dvx: 1, dvy: 0,
    target: { x: target.x, y: target.y },
    mode: MODE.INIT, aim: { x: 0, y: 0 }, flank: 0, dogVx: 0, dogVy: 0,
  };
  const spread = 3 * p.rRep * Math.max(1, Math.sqrt(N / 14));
  for (let i = 0; i < N; i++) {
    s.x[i] = start.x - spread * rng();
    s.y[i] = start.y - spread * rng();
    const th = 2 * Math.PI * rng();
    s.vx[i] = Math.cos(th); s.vy[i] = Math.sin(th);
  }
  const dogSpread = start.dogSpread != null ? start.dogSpread : 3 * p.rDog;
  s.dx = start.x - dogSpread * rng();
  s.dy = start.y - dogSpread * rng();
  const td = 2 * Math.PI * rng();
  s.dvx = Math.cos(td); s.dvy = Math.sin(td);
  s.dogVx = s.dvx * p.vDog; s.dogVy = s.dvy * p.vDog;
  return s;
}

const _scratch = { idx: null, dist: null };

function step(s) {
  const { p, N, rng } = s;
  const dt = p.dt;
  const x0 = Float64Array.from(s.x), y0 = Float64Array.from(s.y);
  const vx0 = Float64Array.from(s.vx), vy0 = Float64Array.from(s.vy);
  const dx0 = s.dx, dy0 = s.dy;
  if (!_scratch.idx || _scratch.idx.length !== N) {
    _scratch.idx = new Array(N); _scratch.dist = new Float64Array(N);
  }
  const idx = _scratch.idx, dist = _scratch.dist;

  // ---- sheep: all read state at t-1 ----
  for (let i = 0; i < N; i++) {
    let rx = 0, ry = 0, nRep = 0;
    for (let j = 0; j < N; j++) {
      const ex = x0[j] - x0[i], ey = y0[j] - y0[i];
      const d = Math.hypot(ex, ey);
      dist[j] = d; idx[j] = j;
      if (j !== i && d < p.rRep) { rx += ex / d; ry += ey / d; nRep++; }
    }
    let rep = [0, 0];
    if (nRep) rep = norm2(-rx, -ry);

    const tdx = dx0 - x0[i], tdy = dy0 - y0[i];
    const dd = Math.hypot(tdx, tdy);
    const seesDog = dd <= p.rDog;
    let nvx, nvy;
    if (!seesDog && !p.socialAlways) {
      // grazing: only collision avoidance moves the sheep
      if (nRep) {
        [nvx, nvy] = norm2(p.wInertia * vx0[i] + p.wRep * rep[0], p.wInertia * vy0[i] + p.wRep * rep[1]);
        [nvx, nvy] = relax(vx0[i], vy0[i], nvx, nvy, p.tauS, dt);
        s.x[i] = x0[i] + p.vSheep * dt * nvx; s.y[i] = y0[i] + p.vSheep * dt * nvy;
        s.vx[i] = nvx; s.vy[i] = nvy;
      }
      continue;
    }
    const wD = seesDog ? p.wDog : 0; // with socialAlways, the dog only acts within rDog
    // K nearest (index 0 after a stable sort is the sheep itself)
    idx.sort((a, b) => dist[a] - dist[b] || a - b);
    const knn = idx.slice(1, p.K + 1);
    // choose nAtt of the K at random, then nAli of those
    for (let k = 0; k < p.nAtt; k++) {
      const r = k + Math.floor(rng() * (knn.length - k));
      const tmp = knn[k]; knn[k] = knn[r]; knn[r] = tmp;
    }
    const att = knn.slice(0, p.nAtt);
    let ax = 0, ay = 0;
    for (const j of att) {
      const ex = x0[j] - x0[i], ey = y0[j] - y0[i];
      const d = dist[j] + 2.220446049250313e-16;
      ax += ex / d; ay += ey / d;
    }
    [ax, ay] = norm2(ax, ay);
    for (let k = 0; k < p.nAli; k++) {
      const r = k + Math.floor(rng() * (att.length - k));
      const tmp = att[k]; att[k] = att[r]; att[r] = tmp;
    }
    let lx = 0, ly = 0;
    for (let k = 0; k < p.nAli; k++) { lx += vx0[att[k]]; ly += vy0[att[k]]; }
    [lx, ly] = norm2(lx, ly);
    const th = 2 * Math.PI * rng();
    let kx = 0, ky = 0;
    if (p.dogKernel && seesDog) {
      const kn = p.dogKernel.knots, ex = -tdx / dd, ey = -tdy / dd; // unit vector dog -> sheep
      const closing = Math.max(0, (s.dogVx - vx0[i] * p.vSheep) * ex + (s.dogVy - vy0[i] * p.vSheep) * ey);
      const fD = hat(kn, p.dogKernel.D, dd), fV = hat(kn, p.dogKernel.V, dd), fL = hat(kn, p.dogKernel.L, dd);
      const W = p.wInertia + p.wRep + p.wDog + p.wAtt + p.wAli + p.noise;
      const k = (p.gDogK * W * dt) / p.vSheep;
      kx = k * (fD * ex + fV * s.dogVx + fL * closing * ex);
      ky = k * (fD * ey + fV * s.dogVy + fL * closing * ey);
    }
    [nvx, nvy] = norm2(
      p.wInertia * vx0[i] + p.wRep * rep[0] - wD * tdx / dd + p.wAtt * ax + p.noise * Math.cos(th) + p.wAli * lx + kx,
      p.wInertia * vy0[i] + p.wRep * rep[1] - wD * tdy / dd + p.wAtt * ay + p.noise * Math.sin(th) + p.wAli * ly + ky,
    );
    [nvx, nvy] = relax(vx0[i], vy0[i], nvx, nvy, p.tauS, dt);
    s.x[i] = x0[i] + p.vSheep * dt * nvx; s.y[i] = y0[i] + p.vSheep * dt * nvy;
    s.vx[i] = nvx; s.vy[i] = nvy;
  }

  // ---- dog: reads sheep state at t-1 ----
  let minD = Infinity, gx = 0, gy = 0;
  for (let i = 0; i < N; i++) {
    minD = Math.min(minD, Math.hypot(x0[i] - dx0, y0[i] - dy0));
    gx += x0[i]; gy += y0[i];
  }
  gx /= N; gy /= N;
  if (p.flankSigma > 0) {
    const af = dt / p.flankTau;
    s.flank = s.flank * (1 - af) + p.flankSigma * Math.sqrt(2 * af) * gauss(rng);
  }
  if (minD <= p.rRep) {
    s.dx = dx0 + p.slowStep * dt * s.dvx; s.dy = dy0 + p.slowStep * dt * s.dvy;
    s.mode = MODE.SLOW;
  } else {
    let far = 0, farD = -1;
    for (let i = 0; i < N; i++) {
      const d = Math.hypot(x0[i] - gx, y0[i] - gy);
      if (d > farD) { farD = d; far = i; }
    }
    let aimX, aimY;
    if (farD >= p.lSep) {
      const k = (farD + p.lCollect) / farD;
      aimX = gx + k * (x0[far] - gx); aimY = gy + k * (y0[far] - gy);
      s.mode = MODE.COLLECT;
    } else {
      const ux = gx - s.target.x, uy = gy - s.target.y, g = Math.hypot(ux, uy);
      aimX = s.target.x + (g + p.lDrive) * ux / g;
      aimY = s.target.y + (g + p.lDrive) * uy / g;
      if (p.flankSigma > 0) { aimX += s.flank * (-uy / g); aimY += s.flank * (ux / g); }
      s.mode = MODE.DRIVE;
    }
    s.aim.x = aimX; s.aim.y = aimY;
    let [ox, oy] = norm2(aimX - dx0, aimY - dy0);
    if (p.outrun) [ox, oy] = outrunDirection(s, gx, gy, aimX, aimY, dx0, dy0, ox, oy);
    const th = 2 * Math.PI * rng();
    let [vx, vy] = norm2(ox + p.noiseDog * Math.cos(th), oy + p.noiseDog * Math.sin(th));
    [vx, vy] = relax(s.dvx, s.dvy, vx, vy, p.tauD, dt);
    let vd = p.vDog;
    if (p.dogSpeedGain != null) vd = Math.min(vd, p.dogSpeedGain * Math.hypot(aimX - dx0, aimY - dy0));
    s.dx = dx0 + vd * dt * vx; s.dy = dy0 + vd * dt * vy;
    s.dvx = vx; s.dvy = vy;
  }
  s.dogVx = (s.dx - dx0) / dt; s.dogVy = (s.dy - dy0) / dt;
  s.t += dt;
  return s;
}

// If the aim point lies on the far side of the flock, run around it on a
// circle of radius ~lDrive instead of straight through the sheep.
function outrunDirection(s, gx, gy, aimX, aimY, dx, dy, ox, oy) {
  const R = s.p.lDrive;
  const thD = Math.atan2(dy - gy, dx - gx), thA = Math.atan2(aimY - gy, aimX - gx);
  let dth = thA - thD;
  dth = Math.atan2(Math.sin(dth), Math.cos(dth));
  if (Math.abs(dth) < Math.PI / 3) return [ox, oy];
  const r = Math.hypot(dx - gx, dy - gy) || 1e-9;
  const radX = (dx - gx) / r, radY = (dy - gy) / r;
  const sg = Math.sign(dth) || 1;
  const tanX = -sg * radY, tanY = sg * radX; // counterclockwise when sg > 0
  const k = (R - r) / R; // push out when inside the circle, pull in when outside
  return norm2(tanX + 1.5 * k * radX, tanY + 1.5 * k * radY);
}

// Observables in the paper's definitions (sheep headings are unit vectors).
function observables(s) {
  const N = s.N;
  let gx = 0, gy = 0, mx = 0, my = 0;
  for (let i = 0; i < N; i++) { gx += s.x[i]; gy += s.y[i]; mx += s.vx[i]; my += s.vy[i]; }
  gx /= N; gy /= N; mx /= N; my /= N;
  let C = 0;
  for (let i = 0; i < N; i++) C += Math.hypot(s.x[i] - gx, s.y[i] - gy);
  C /= N;
  const P = Math.hypot(mx, my);
  const a = Math.atan2(my, mx), c = Math.cos(a), sn = Math.sin(a);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < N; i++) {
    const rx = s.x[i] - gx, ry = s.y[i] - gy;
    const along = c * rx + sn * ry, left = -sn * rx + c * ry;
    xmin = Math.min(xmin, -left); xmax = Math.max(xmax, -left);
    ymin = Math.min(ymin, along); ymax = Math.max(ymax, along);
  }
  const rdx = s.dx - gx, rdy = s.dy - gy;
  const dAlong = c * rdx + sn * rdy, dLeft = -sn * rdx + c * rdy;
  return {
    C, P, E: (ymax - ymin) / (xmax - xmin),
    dBD: Math.hypot(rdx, rdy), yRD: ymin - dAlong, xD: -dLeft,
    gx, gy, heading: a,
  };
}

if (typeof module !== "undefined") module.exports = { DEFAULTS, MODE, resolveParams, createSim, step, observables };
