// JavaScript port of shepherd/bridge.py: the Jadhav ↔ Di Lorenzo bridge simulator.
// Field names are the Python dataclass fields, so a configuration reads the same in both.
// Deterministic parts (design, metrics, feasibility) match the Python numerically
// (tests/test_bridge_js.py); the dynamics use their own seeded generator.

const Bridge = (() => {
  // ------------------------------------------------------------------ random numbers
  function mulberry32(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    const u = Math.max(rng(), 1e-12), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ------------------------------------------------------------------ defaults (= Python dataclasses)
  const FOLLOWERS = {
    rule: "heading", n: 14, dt: 1.0, v: 1.0,
    w_inertia: 0.5, w_rep: 2.0, w_att: 1.5, w_ali: 1.3, w_dog: 1.0, e_noise: 0.3,
    r_rep: 2.0, k_perceive: 10, n_att: 4, n_ali: 1, r_dog: 12.0,
    activation: "published", wander_speed: 0.0,
    a_dog: 1.0, L_dog: Math.PI, g_rep: 0.0, l_rep: 1.0, g_att: 0.0, l_att: 1.0,
    population_scaling: true, ali_fo: 0.0, r_ali: 3.0, D: 0.0,
  };
  const HERDERS = {
    controller: "strombom", n: 1, v: 1.5, slow_step: 0.05, l_sep: null, l_drive: null, l_collect: null,
    noise: null, spread_angle: 0.5, hold_radius: null, hold_release: null, hold_standoff: null,
    K_H: 10.0, bandwidth: null, grid: 64, reg: 1e-3, ring_radius: 15.0,
    design_model: null, // follower model the Di Lorenzo design assumes (null: the simulated followers)
    substeps: 1, // herder flow sub-steps per follower step
    fb_standoff: null, fb_exclusion: null, fb_drive_radius: null, fb_release: null, fb_engage: null, fb_threshold: 0.25,
  };
  const TARGET = { kind: "gaussian", centre: [0, 0], sigma: 2.5, k: 6 / Math.PI, r_goal: null, bandwidth: 1.0, floor: 1e-4 };
  const DOMAIN = { periodic: false, size: 80.0, images: 0 };

  function resolve(cfg) {
    const F = Object.assign({}, FOLLOWERS, cfg.followers || {});
    const H = Object.assign({}, HERDERS, cfg.herders || {});
    const T = Object.assign({}, TARGET, cfg.target || {});
    const D = Object.assign({}, DOMAIN, cfg.domain || {});
    F.k_perceive = Math.min(F.k_perceive, F.n - 1);
    F.n_att = Math.min(F.n_att, F.k_perceive);
    F.n_ali = Math.min(F.n_ali, F.n_att);
    if (H.controller === "none") H.n = 0;
    if (H.design_model) H.design_model = Object.assign({}, FOLLOWERS, H.design_model, { n: F.n });
    if (H.l_sep == null) H.l_sep = F.r_rep * Math.pow(F.n, 2 / 3);
    if (H.l_drive == null) H.l_drive = F.r_rep * Math.sqrt(F.n);
    if (H.l_collect == null) H.l_collect = F.r_rep;
    if (H.noise == null) H.noise = F.e_noise;
    if (H.bandwidth == null) H.bandwidth = 0.4 * D.size / (2 * Math.PI);
    if (T.r_goal == null) T.r_goal = T.kind === "gaussian" ? T.sigma * Math.sqrt(-2 * Math.log(0.05)) : D.size / 4;
    if (!D.periodic) D.images = 0;
    if (H.fb_standoff == null) H.fb_standoff = F.rule === "heading" ? Math.max(F.r_dog - 2, 1) : F.L_dog / 2;
    if (H.fb_exclusion == null) H.fb_exclusion = F.rule === "heading" ? F.r_dog : F.L_dog;
    if (H.fb_drive_radius == null) H.fb_drive_radius = T.r_goal;
    if (H.fb_engage == null) H.fb_engage = H.fb_release == null ? null : 1.5 * H.fb_release;
    return { F, H, T, D };
  }

  // ------------------------------------------------------------------ geometry
  const wrap1 = (d, L) => ((d + L / 2) % L + L) % L - L / 2;
  function wrap(d, dom) { return dom.periodic ? wrap1(d, dom.size) : d; }
  function centroid(xs, ys, n, dom) {
    if (!dom.periodic) {
      let sx = 0, sy = 0;
      for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
      return [sx / n, sy / n];
    }
    const w = 2 * Math.PI / dom.size;
    let cx = 0, sx = 0, cy = 0, sy = 0;
    for (let i = 0; i < n; i++) {
      cx += Math.cos(w * xs[i]); sx += Math.sin(w * xs[i]); cy += Math.cos(w * ys[i]); sy += Math.sin(w * ys[i]);
    }
    return [Math.atan2(sx, cx) / w, Math.atan2(sy, cy) / w];
  }
  const norm2 = (x, y) => { const m = Math.hypot(x, y); return m > 0 ? [x / m, y / m] : [0, 0]; };

  // ------------------------------------------------------------------ FFT (radix 2, in place)
  function fft1(re, im, n, inverse) {
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inverse ? 1 : -1);
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  // 2D FFT of an n×n complex array (row-major, index i*n+j), in place.
  function fft2(re, im, n, inverse) {
    const r = new Float64Array(n), m = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      fft1(re.subarray(i * n, (i + 1) * n), im.subarray(i * n, (i + 1) * n), n, inverse);
    }
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) { r[i] = re[i * n + j]; m[i] = im[i * n + j]; }
      fft1(r, m, n, inverse);
      for (let i = 0; i < n; i++) { re[i * n + j] = r[i]; im[i * n + j] = m[i]; }
    }
  }

  // ------------------------------------------------------------------ periodic grid
  class Grid {
    constructor(size, n) {
      this.size = size; this.n = n; this.h = size / n;
      const nn = n * n;
      this.X = new Float64Array(nn); this.Y = new Float64Array(nn);  // centred coordinates
      this.DX = new Float64Array(nn); this.DY = new Float64Array(nn); // wrapped displacements (FFT order)
      this.KX = new Float64Array(nn); this.KY = new Float64Array(nn); this.K2 = new Float64Array(nn);
      const half = n >> 1;
      for (let i = 0; i < n; i++) {
        const xi = (i - half) * this.h, di = (i < half ? i : i - n) * this.h, ki = 2 * Math.PI * (i < half ? i : i - n) / size;
        for (let j = 0; j < n; j++) {
          const q = i * n + j;
          const dj = (j < half ? j : j - n) * this.h, kj = 2 * Math.PI * (j < half ? j : j - n) / size;
          this.X[q] = xi; this.Y[q] = (j - half) * this.h;
          this.DX[q] = di; this.DY[q] = dj;
          this.KX[q] = ki; this.KY[q] = kj; this.K2[q] = ki * ki + kj * kj;
        }
      }
    }
    fftOf(a) { const re = Float64Array.from(a), im = new Float64Array(a.length); fft2(re, im, this.n, false); return { re, im }; }
    gaussianKernel(b) {
      const nn = this.n * this.n, K = new Float64Array(nn);
      let s = 0;
      for (let q = 0; q < nn; q++) {
        let v = 0;
        for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
          const dx = this.DX[q] + i * this.size, dy = this.DY[q] + j * this.size;
          v += Math.exp(-(dx * dx + dy * dy) / (2 * b * b));
        }
        K[q] = v; s += v;
      }
      for (let q = 0; q < nn; q++) K[q] /= s * this.h * this.h;
      return K;
    }
    vectorKernel(radial, images) {
      const nn = this.n * this.n, kx = new Float64Array(nn), ky = new Float64Array(nn);
      for (let q = 0; q < nn; q++) {
        for (let i = -images; i <= images; i++) for (let j = -images; j <= images; j++) {
          const zx = this.DX[q] + i * this.size, zy = this.DY[q] + j * this.size, r = Math.hypot(zx, zy);
          if (r > 0) { const m = radial(r) / r; kx[q] += m * zx; ky[q] += m * zy; }
        }
      }
      return { kx, ky };
    }
    // (kernel * density) via FFT; kernelF = this.fftOf(kernel). Returns a real array.
    convolveF(kernelF, density) {
      const n = this.n, nn = n * n, re = Float64Array.from(density), im = new Float64Array(nn);
      fft2(re, im, n, false);
      for (let q = 0; q < nn; q++) {
        const a = re[q], b = im[q];
        re[q] = a * kernelF.re[q] - b * kernelF.im[q]; im[q] = a * kernelF.im[q] + b * kernelF.re[q];
      }
      fft2(re, im, n, true);
      const h2 = this.h * this.h;
      for (let q = 0; q < nn; q++) re[q] *= h2;
      return re;
    }
    deposit(xs, ys, m, weight) {
      const n = this.n, h = this.h, rho = new Float64Array(n * n);
      for (let k = 0; k < m; k++) {
        const u = (xs[k] + this.size / 2) / h, v = (ys[k] + this.size / 2) / h;
        const i0 = Math.floor(u), j0 = Math.floor(v), fu = u - i0, fv = v - j0;
        const ia = ((i0 % n) + n) % n, ib = (ia + 1) % n, ja = ((j0 % n) + n) % n, jb = (ja + 1) % n;
        rho[ia * n + ja] += weight * (1 - fu) * (1 - fv); rho[ib * n + ja] += weight * fu * (1 - fv);
        rho[ia * n + jb] += weight * (1 - fu) * fv; rho[ib * n + jb] += weight * fu * fv;
      }
      for (let q = 0; q < n * n; q++) rho[q] /= h * h;
      return rho;
    }
    sample(field, x, y) {
      const n = this.n, h = this.h;
      const u = (x + this.size / 2) / h, v = (y + this.size / 2) / h;
      const i0 = Math.floor(u), j0 = Math.floor(v), fu = u - i0, fv = v - j0;
      const ia = ((i0 % n) + n) % n, ib = (ia + 1) % n, ja = ((j0 % n) + n) % n, jb = (ja + 1) % n;
      return field[ia * n + ja] * (1 - fu) * (1 - fv) + field[ib * n + ja] * fu * (1 - fv)
        + field[ia * n + jb] * (1 - fu) * fv + field[ib * n + jb] * fu * fv;
    }
    target(T, dom) {
      const nn = this.n * this.n, ln = new Float64Array(nn), gx = new Float64Array(nn), gy = new Float64Array(nn);
      const [cx, cy] = T.centre;
      let mx = -Infinity;
      for (let q = 0; q < nn; q++) {
        if (T.kind === "gaussian") {
          const dx = wrap(this.X[q] - cx, dom), dy = wrap(this.Y[q] - cy, dom);
          ln[q] = -(dx * dx + dy * dy) / (2 * T.sigma * T.sigma); gx[q] = -dx / (T.sigma * T.sigma); gy[q] = -dy / (T.sigma * T.sigma);
        } else {
          const w = 2 * Math.PI / dom.size;
          ln[q] = T.k * (Math.cos(w * (this.X[q] - cx)) + Math.cos(w * (this.Y[q] - cy)));
          gx[q] = -T.k * w * Math.sin(w * (this.X[q] - cx)); gy[q] = -T.k * w * Math.sin(w * (this.Y[q] - cy));
        }
        if (ln[q] > mx) mx = ln[q];
      }
      const rho = new Float64Array(nn);
      let s = 0;
      for (let q = 0; q < nn; q++) { rho[q] = Math.exp(ln[q] - mx); s += rho[q]; }
      for (let q = 0; q < nn; q++) { const fac = rho[q] / (rho[q] + T.floor); gx[q] *= fac; gy[q] *= fac; rho[q] /= s * this.h * this.h; }  // ∇ρ̂/(ρ̂ + floor·max): the drift is unconstrained where ρ̂ ≈ 0
      return { rho, gx, gy };
    }
  }

  const fKernel = (F) => (r) => F.a_dog * Math.exp(-r / F.L_dog);
  const gKernel = (F) => (r) => F.g_rep * Math.exp(-r / F.l_rep) - F.g_att * Math.exp(-r / F.l_att);
  const hasPair = (F) => F.g_rep !== 0 || F.g_att !== 0;

  // ------------------------------------------------------------------ Di Lorenzo design + feasibility
  function deconvolver(grid, F, dom, reg) {
    const Kf = grid.vectorKernel(fKernel(F), dom.images);
    const Fx = grid.fftOf(Kf.kx), Fy = grid.fftOf(Kf.ky), nn = grid.n * grid.n, h2 = grid.h * grid.h;
    let dmax = 0;
    const den = new Float64Array(nn);
    for (let q = 0; q < nn; q++) {
      Fx.re[q] *= h2; Fx.im[q] *= h2; Fy.re[q] *= h2; Fy.im[q] *= h2;
      den[q] = Fx.re[q] ** 2 + Fx.im[q] ** 2 + Fy.re[q] ** 2 + Fy.im[q] ** 2;
      if (den[q] > dmax) dmax = den[q];
    }
    const deconv = (vx, vy) => {
      const Vx = grid.fftOf(vx), Vy = grid.fftOf(vy), re = new Float64Array(nn), im = new Float64Array(nn);
      for (let q = 0; q < nn; q++) {
        const d = den[q] + reg * dmax;
        // conj(F) · V
        re[q] = (Fx.re[q] * Vx.re[q] + Fx.im[q] * Vx.im[q] + Fy.re[q] * Vy.re[q] + Fy.im[q] * Vy.im[q]) / d;
        im[q] = (Fx.re[q] * Vx.im[q] - Fx.im[q] * Vx.re[q] + Fy.re[q] * Vy.im[q] - Fy.im[q] * Vy.re[q]) / d;
      }
      re[0] = 0; im[0] = 0;
      fft2(re, im, grid.n, true);
      return re;
    };
    return { Kf, Fx, Fy, deconv };
  }

  function designHerders(F, H, T, dom, grid) {
    if (!dom.periodic) throw new Error("Di Lorenzo's design is a deconvolution on a periodic domain");
    grid = grid || new Grid(dom.size, H.grid);
    const nn = grid.n * grid.n, h2 = grid.h * grid.h;
    const { rho, gx, gy } = grid.target(T, dom);
    const N = F.n, M = H.n;
    F = H.design_model || F; // what the herders believe about the followers (counts stay the simulated ones)
    const mT = F.population_scaling ? N / (N + M) : N, mH = F.population_scaling ? M / (N + M) : M;
    const vx = new Float64Array(nn), vy = new Float64Array(nn);
    for (let q = 0; q < nn; q++) { vx[q] = F.D * gx[q]; vy[q] = F.D * gy[q]; }
    if (hasPair(F)) {
      const Kg = grid.vectorKernel(gKernel(F), dom.images);
      const cx = grid.convolveF(grid.fftOf(Kg.kx), rho), cy = grid.convolveF(grid.fftOf(Kg.ky), rho);
      for (let q = 0; q < nn; q++) { vx[q] -= mT * cx[q]; vy[q] -= mT * cy[q]; }
    }
    const dc = deconvolver(grid, F, dom, H.reg);
    const Hd = dc.deconv(vx, vy);
    let hmin = Infinity;
    for (let q = 0; q < nn; q++) if (Hd[q] < hmin) hmin = Hd[q];
    const rhoH = new Float64Array(nn);
    let needed = 0;
    for (let q = 0; q < nn; q++) { rhoH[q] = Hd[q] - hmin; needed += rhoH[q] * h2; }
    const feasible = mH >= needed;
    if (feasible) { const add = (mH - needed) / (dom.size * dom.size); for (let q = 0; q < nn; q++) rhoH[q] += add; }
    else { const sc = needed > 0 ? mH / needed : 0; for (let q = 0; q < nn; q++) rhoH[q] *= sc; }
    const ax = grid.convolveF(grid.fftOf(dc.Kf.kx), rhoH), ay = grid.convolveF(grid.fftOf(dc.Kf.ky), rhoH);
    let num = 0, denom = 0, rmax = 0;
    for (let q = 0; q < nn; q++) if (rho[q] > rmax) rmax = rho[q];
    for (let q = 0; q < nn; q++) {
      const w = rho[q] / rmax;
      num += w * ((ax[q] - vx[q]) ** 2 + (ay[q] - vy[q]) ** 2); denom += w * (vx[q] ** 2 + vy[q] ** 2);
    }
    return { grid, rho_hat: rho, rho_H: rhoH, H: Hd, mass_needed: needed, mass_available: mH, feasible,
      drift_error: Math.sqrt(num / Math.max(denom, 1e-300)), Kf: dc.Kf };
  }

  function feasibilityScan(F, T, dom, masses, gridN, reg) {
    const grid = new Grid(dom.size, gridN || 64), nn = grid.n * grid.n, h2 = grid.h * grid.h;
    const { rho, gx, gy } = grid.target(T, dom);
    const dc = deconvolver(grid, F, dom, reg == null ? 1e-3 : reg);
    const vx = new Float64Array(nn), vy = new Float64Array(nn);
    for (let q = 0; q < nn; q++) { vx[q] = F.D * gx[q]; vy[q] = F.D * gy[q]; }
    const HD = dc.deconv(vx, vy);
    let HG = null;
    if (hasPair(F)) {
      const Kg = grid.vectorKernel(gKernel(F), dom.images);
      HG = dc.deconv(grid.convolveF(grid.fftOf(Kg.kx), rho), grid.convolveF(grid.fftOf(Kg.ky), rho));
    }
    const needed = masses.map((mH) => {
      let hmin = Infinity, s = 0;
      const Hq = new Float64Array(nn);
      for (let q = 0; q < nn; q++) { Hq[q] = HD[q] - (HG ? (1 - mH) * HG[q] : 0); if (Hq[q] < hmin) hmin = Hq[q]; }
      for (let q = 0; q < nn; q++) s += (Hq[q] - hmin) * h2;
      return s;
    });
    const ok = masses.map((m, i) => m >= needed[i]);
    const feas = masses.filter((m, i) => ok[i]);
    return { mass: masses, needed, feasible: ok, min_feasible_mass: feas.length ? Math.min(...feas) : null,
      max_feasible_mass: feas.length ? Math.max(...feas) : null };
  }
  const herdersFromMass = (nSheep, mass) => nSheep * mass / (1 - mass);

  // The herder count Di Lorenzo's design asks for: the smallest N_H whose herder mass covers the minimum
  // mass M̂ (2024 Eqs. 11–12: N_H = N_T M̂/(1 − M̂) with their α; M̂ itself in dog units without it). With a
  // pair kernel M̂ depends on N_H (2026 Theorem 1), so the count is searched. Every candidate is checked with
  // the same design the simulation runs (the herders' follower model, grid, kernel images and reg).
  function dilorenzoCount(cfg, maxDogs) {
    const { F, H, T, D } = resolve(cfg);
    maxDogs = maxDogs || 5000;
    if (!D.periodic) return { n: null, reason: "Di Lorenzo's design is a deconvolution on a periodic domain; switch the domain to periodic." };
    const Fd = H.design_model || F;
    const at = (M) => designHerders(F, Object.assign({}, H, { n: M, controller: "dilorenzo" }), T, D);
    const probe = at(1);
    if (!(probe.mass_needed > 0)) return { n: null, reason: "The design is empty: the followers the herders design for have no diffusion or pair term to counter (D = 0)." };
    let M;
    if (!Fd.population_scaling) M = Math.ceil(probe.mass_needed);
    else if (!hasPair(Fd)) {
      if (probe.mass_needed >= 1) return { n: null, reason: `M̂ = ${probe.mass_needed.toFixed(3)} ≥ 1: no herder count suffices.` };
      M = Math.ceil(herdersFromMass(F.n, probe.mass_needed));
    } else {
      const masses = Array.from({ length: 199 }, (_, i) => 0.005 * (i + 1));
      const sc = feasibilityScan(Fd, T, D, masses, H.grid, H.reg);
      if (sc.min_feasible_mass == null) return { n: null, reason: "No herder mass is feasible for this pair kernel (2026 Theorem 1's interval is empty)." };
      M = Math.floor(herdersFromMass(F.n, Math.max(sc.min_feasible_mass - 0.005, 0)));
    }
    M = Math.max(M, 1);
    while (M <= maxDogs && !at(M).feasible) M++;
    if (M > maxDogs) return { n: null, reason: `No count up to ${maxDogs} is feasible.` };
    while (M > 1 && at(M - 1).feasible) M--;
    const d = at(M);
    return { n: M, mass_needed: d.mass_needed, mass_available: d.mass_available, scaled: Fd.population_scaling, own_model: !!H.design_model };
  }

  // ------------------------------------------------------------------ metrics
  class Metrics {
    constructor(T, dom, n, grid) {
      this.T = T; this.dom = dom; this.n = n;
      this.useGrid = dom.periodic || T.kind !== "gaussian";
      if (this.useGrid) {
        this.grid = grid || new Grid(dom.size, 64);
        this.KbF = this.grid.fftOf(this.grid.gaussianKernel(T.bandwidth));
        this.rho_b = this.grid.convolveF(this.KbF, this.grid.target(T, dom).rho);
        let s = 0; for (const v of this.rho_b) s += v * v;
        this.norm = s * this.grid.h * this.grid.h;
        const K2 = this.grid.gaussianKernel(T.bandwidth * Math.SQRT2);
        let k0 = -Infinity; for (const v of K2) if (v > k0) k0 = v;
        this.floor = (k0 - this.norm) / (n * this.norm);
        this.peak = Math.max(...this.rho_b);
      } else {
        this.floor = T.sigma * T.sigma / (n * T.bandwidth * T.bandwidth);
        this.peak = 1 / (2 * Math.PI * (T.sigma * T.sigma + T.bandwidth * T.bandwidth));
      }
    }
    densityError(xs, ys, N) {
      const b = this.T.bandwidth, [cx, cy] = this.T.centre;
      if (!this.useGrid) {
        const s = this.T.sigma;
        let pair = 0;
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const dx = xs[i] - xs[j], dy = ys[i] - ys[j];
          pair += Math.exp(-(dx * dx + dy * dy) / (4 * b * b));
        }
        pair /= 4 * Math.PI * b * b * N * N;
        const a2 = s * s + 2 * b * b;
        let cross = 0;
        for (let i = 0; i < N; i++) { const dx = xs[i] - cx, dy = ys[i] - cy; cross += Math.exp(-(dx * dx + dy * dy) / (2 * a2)); }
        cross /= 2 * Math.PI * a2 * N;
        const norm = 1 / (4 * Math.PI * (s * s + b * b));
        return (pair - 2 * cross + norm) / norm;
      }
      const rho = this.grid.convolveF(this.KbF, this.grid.deposit(xs, ys, N, 1 / N));
      let e = 0; for (let q = 0; q < rho.length; q++) e += (rho[q] - this.rho_b[q]) ** 2;
      return e * this.grid.h * this.grid.h / this.norm;
    }
    // ρ̂_b(x_i) − ρ̄_b(x_i): the sheep's kernel density estimate minus the smoothed target, both unit mass
    surplus(xs, ys, N) {
      const b = this.T.bandwidth, [cx, cy] = this.T.centre, out = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let rh = 0;
        for (let k = 0; k < N; k++) { const dx = wrap(xs[i] - xs[k], this.dom), dy = wrap(ys[i] - ys[k], this.dom); rh += Math.exp(-(dx * dx + dy * dy) / (2 * b * b)); }
        rh /= 2 * Math.PI * b * b * N;
        let rb;
        if (!this.useGrid) { const a2 = this.T.sigma ** 2 + b * b, dx = xs[i] - cx, dy = ys[i] - cy; rb = Math.exp(-(dx * dx + dy * dy) / (2 * a2)) / (2 * Math.PI * a2); }
        else rb = this.grid.sample(this.rho_b, wrap(xs[i], this.dom), wrap(ys[i], this.dom));
        out[i] = rh - rb;
      }
      return out;
    }
    all(xs, ys, N) {
      const [cx, cy] = this.T.centre, G = centroid(xs, ys, N, this.dom);
      let inside = 0, coh = 0;
      for (let i = 0; i < N; i++) {
        if (Math.hypot(wrap(xs[i] - cx, this.dom), wrap(ys[i] - cy, this.dom)) <= this.T.r_goal) inside++;
        coh += Math.hypot(wrap(xs[i] - G[0], this.dom), wrap(ys[i] - G[1], this.dom));
      }
      return { density_error: this.densityError(xs, ys, N), occupancy: inside / N, cohesion: coh / N,
        centroid_error: Math.hypot(wrap(G[0] - cx, this.dom), wrap(G[1] - cy, this.dom)) };
    }
  }

  // ------------------------------------------------------------------ simulation state
  const MODE = { INIT: 0, DRIVE: 1, COLLECT: 2, SLOW: 3, HOLD: 5, FLOW: 6, STATIC: 7, SHAPE: 8 };

  // Where the density-feedback dogs go (= bridge.feedback_aims). Deterministic.
  function feedbackAims(xs, ys, N, dx, dy, M, c, surplus, H, dom, holding, reach, peak, driving) {
    const G = centroid(xs, ys, N, dom);
    const awx = wrap(G[0] - c[0], dom), awy = wrap(G[1] - c[1], dom), gn = Math.hypot(awx, awy);
    const e = gn > 0 ? [awx / gn, awy / gn] : [1, 0];
    let spread = 0;
    for (let i = 0; i < N; i++) spread = Math.max(spread, Math.hypot(wrap(xs[i] - G[0], dom), wrap(ys[i] - G[1], dom)));
    const standoffHold = H.hold_standoff != null ? H.hold_standoff : spread + reach + 1;
    const ax = new Float64Array(M), ay = new Float64Array(M), mode = new Uint8Array(M).fill(MODE.HOLD);
    const holdAims = () => {
      for (let m = 0; m < M; m++) {
        const ox = wrap(dx[m] - G[0], dom), oy = wrap(dy[m] - G[1], dom), on = Math.hypot(ox, oy);
        const o = on > 1e-9 ? [ox / on, oy / on] : e;
        ax[m] = G[0] + standoffHold * o[0]; ay[m] = G[1] + standoffHold * o[1];
      }
    };
    driving = driving ? gn > 0.5 * H.fb_drive_radius : gn > H.fb_drive_radius;
    if (holding) { holdAims(); return { ax, ay, mode, driving }; }
    if (driving) {
      const relx = wrap(H.l_drive * e[0], dom), rely = wrap(H.l_drive * e[1], dom);
      for (let m = 0; m < M; m++) {
        const off = H.spread_angle * (m - (M - 1) / 2), cs = Math.cos(off), sn = Math.sin(off);
        ax[m] = G[0] + cs * relx - sn * rely; ay[m] = G[1] + sn * relx + cs * rely; mode[m] = MODE.DRIVE;
      }
      return { ax, ay, mode, driving };
    }
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => surplus[b] - surplus[a] || a - b);
    const targets = [];
    for (const i of order) {
      if (surplus[i] <= H.fb_threshold * peak || targets.length === M) break;
      if (targets.every((k) => Math.hypot(wrap(xs[i] - xs[k], dom), wrap(ys[i] - ys[k], dom)) >= H.fb_exclusion)) targets.push(i);
    }
    holdAims();
    const free = Array.from({ length: M }, (_, m) => m);
    for (const i of targets) {
      let best = 0, bd = Infinity;
      free.forEach((m, k) => { const d = Math.hypot(wrap(xs[i] - dx[m], dom), wrap(ys[i] - dy[m], dom)); if (d < bd) { bd = d; best = k; } });
      const m = free.splice(best, 1)[0];
      let ux = wrap(xs[i] - c[0], dom), uy = wrap(ys[i] - c[1], dom), un = Math.hypot(ux, uy);
      if (un < 1e-9) { ux = wrap(xs[i] - G[0], dom); uy = wrap(ys[i] - G[1], dom); un = Math.hypot(ux, uy); }
      if (un > 1e-9) { ux /= un; uy /= un; } else { ux = 1; uy = 0; }
      ax[m] = xs[i] + H.fb_standoff * ux; ay[m] = ys[i] + H.fb_standoff * uy; mode[m] = MODE.SHAPE;
    }
    return { ax, ay, mode, driving };
  }

  function createSim(cfg, seed, initHow, boxLength) {
    const { F, H, T, D } = resolve(cfg);
    const rng = mulberry32(seed == null ? 1 : seed);
    const N = F.n, M = H.n;
    const s = {
      F, H, T, D, rng, N, M, t: 0, step_count: 0,
      x: new Float64Array(N), y: new Float64Array(N), hx: new Float64Array(N), hy: new Float64Array(N),
      spd: new Float64Array(N), ldx: new Float64Array(N), ldy: new Float64Array(N), active: new Uint8Array(N),
      dx: new Float64Array(M), dy: new Float64Array(M), dvx: new Float64Array(M), dvy: new Float64Array(M),
      mode: new Uint8Array(M), holding: false, driving: false, design: null, grid: null,
    };
    const [cx, cy] = T.centre;
    initHow = initHow || (D.periodic ? "uniform" : "jadhav");
    boxLength = boxLength == null ? (D.periodic ? D.size : 250) : boxLength;
    if (initHow === "jadhav") {
      const th = 2 * Math.PI * rng(), sx = cx + boxLength * Math.cos(th), sy = cy + boxLength * Math.sin(th);
      for (let i = 0; i < N; i++) { s.x[i] = sx - 3 * F.r_rep * rng(); s.y[i] = sy - 3 * F.r_rep * rng(); }
      for (let m = 0; m < M; m++) { s.dx[m] = sx - 3 * F.r_dog * rng(); s.dy[m] = sy - 3 * F.r_dog * rng(); }
    } else if (initHow === "uniform") {
      const L = D.size;
      for (let i = 0; i < N; i++) { s.x[i] = (rng() - 0.5) * L; s.y[i] = (rng() - 0.5) * L; }
      const k = Math.ceil(Math.sqrt(M));
      for (let m = 0; m < M; m++) {
        const a = Math.floor(m / k), b = m % k;
        s.dx[m] = (a + 0.5) * L / k - L / 2 + 0.01 * L / k * gauss(rng); s.dy[m] = (b + 0.5) * L / k - L / 2 + 0.01 * L / k * gauss(rng);
      }
    } else { // spread
      for (let i = 0; i < N; i++) { s.x[i] = cx + boxLength * gauss(rng); s.y[i] = cy + boxLength * gauss(rng); }
      const off = rng();
      for (let m = 0; m < M; m++) { const a = 2 * Math.PI * (m + off) / Math.max(M, 1); s.dx[m] = cx + 2 * boxLength * Math.cos(a); s.dy[m] = cy + 2 * boxLength * Math.sin(a); }
    }
    for (let i = 0; i < N; i++) { const th = 2 * Math.PI * rng(); s.hx[i] = Math.cos(th); s.hy[i] = Math.sin(th); s.spd[i] = F.activation === "contagion" ? 0 : F.v; }
    for (let m = 0; m < M; m++) { const th = 2 * Math.PI * rng(); s.dvx[m] = Math.cos(th); s.dvy[m] = Math.sin(th); }
    if (D.periodic) { for (let i = 0; i < N; i++) { s.x[i] = wrap1(s.x[i], D.size); s.y[i] = wrap1(s.y[i], D.size); } for (let m = 0; m < M; m++) { s.dx[m] = wrap1(s.dx[m], D.size); s.dy[m] = wrap1(s.dy[m], D.size); } }
    if (H.controller === "dilorenzo" && M > 0) {
      s.design = designHerders(F, H, T, D);
      s.grid = s.design.grid;
      s.KbHF = s.grid.fftOf(s.grid.gaussianKernel(H.bandwidth));
      s.w_dog = (H.design_model || F).population_scaling ? 1 / (N + M) : 1;
      s.rhoHmax = Math.max(...s.design.rho_H);
    }
    if (H.controller === "static_ring") {
      for (let m = 0; m < M; m++) { const a = 2 * Math.PI * m / Math.max(M, 1); s.dx[m] = cx + H.ring_radius * Math.cos(a); s.dy[m] = cy + H.ring_radius * Math.sin(a); s.mode[m] = MODE.STATIC; }
    }
    s.metrics = new Metrics(T, D, N, s.grid);
    s.hold_release = H.hold_radius == null ? null : (H.hold_release == null ? 2 * H.hold_radius : H.hold_release);
    s.hold_standoff = H.hold_radius == null ? null : (H.hold_standoff == null ? F.r_dog + 3 : H.hold_standoff);
    return s;
  }

  const fKernelImages = (F, D) => {
    const f = fKernel(F), im = D.images;
    return (zx, zy, out) => { // sum over images into out[0], out[1]
      let ox = 0, oy = 0;
      for (let i = -im; i <= im; i++) for (let j = -im; j <= im; j++) {
        const xx = zx + i * D.size, yy = zy + j * D.size, r = Math.max(Math.hypot(xx, yy), 1e-9), m = f(r) / r;
        ox += m * xx; oy += m * yy;
      }
      out[0] = ox; out[1] = oy;
    };
  };

  function step(s) {
    const { F, H, T, D, rng, N, M } = s;
    const dt = F.dt, L = D.size;
    const x0 = Float64Array.from(s.x), y0 = Float64Array.from(s.y), hx0 = Float64Array.from(s.hx), hy0 = Float64Array.from(s.hy);
    const nx = new Float64Array(N), ny = new Float64Array(N), nhx = new Float64Array(N), nhy = new Float64Array(N);
    const tmp = [0, 0];

    // ------------------------------------------------ followers (read state at t−1)
    if (F.rule === "heading") {
      const dist = new Float64Array(N), idx = new Array(N), keys = new Float64Array(N);
      const K = F.k_perceive;
      for (let i = 0; i < N; i++) {
        let rx = 0, ry = 0, nRep = 0;
        for (let j = 0; j < N; j++) {
          const ex = wrap(x0[j] - x0[i], D), ey = wrap(y0[j] - y0[i], D), d = Math.hypot(ex, ey);
          dist[j] = d; idx[j] = j;
          if (j !== i && d < F.r_rep) { rx += ex / d; ry += ey / d; nRep++; }
        }
        const rep = nRep ? norm2(-rx, -ry) : [0, 0];
        let ax = 0, ay = 0, sees = false;
        for (let m = 0; m < M; m++) {
          const tx = wrap(s.dx[m] - x0[i], D), ty = wrap(s.dy[m] - y0[i], D), dd = Math.hypot(tx, ty);
          if (dd <= F.r_dog) { sees = true; ax -= tx / dd; ay -= ty / dd; }
        }
        // K nearest (stable), then n_att of them at random, then n_ali of those
        idx.sort((a, b) => dist[a] - dist[b] || a - b);
        const knn = idx.slice(1, K + 1);
        for (let k = 0; k < K; k++) keys[k] = rng();
        const order = knn.map((_, k) => k).sort((a, b) => keys[a] - keys[b]);
        const att = order.slice(0, F.n_att).map((k) => knn[k]);
        for (let k = 0; k < F.n_att; k++) keys[k] = rng();
        const ord2 = att.map((_, k) => k).sort((a, b) => keys[a] - keys[b]);
        const ali = ord2.slice(0, F.n_ali).map((k) => att[k]);
        let sx = 0, sy = 0;
        for (const j of att) { const ex = wrap(x0[j] - x0[i], D), ey = wrap(y0[j] - y0[i], D), d = Math.hypot(ex, ey) + Number.EPSILON; sx += ex / d; sy += ey / d; }
        const sAtt = norm2(sx, sy);
        let lx = 0, ly = 0;
        for (const j of ali) { lx += hx0[j]; ly += hy0[j]; }
        const sAli = norm2(lx, ly);
        const th = 2 * Math.PI * rng(), ex_ = Math.cos(th), ey_ = Math.sin(th);
        const actx = F.w_inertia * hx0[i] + F.w_rep * rep[0] + F.w_dog * ax + F.w_att * sAtt[0] + F.e_noise * ex_ + F.w_ali * sAli[0];
        const acty = F.w_inertia * hy0[i] + F.w_rep * rep[1] + F.w_dog * ay + F.w_att * sAtt[1] + F.e_noise * ey_ + F.w_ali * sAli[1];
        const grz = norm2(F.w_inertia * hx0[i] + F.w_rep * rep[0], F.w_inertia * hy0[i] + F.w_rep * rep[1]);
        let h, len;
        if (F.activation === "published") {
          h = sees ? norm2(actx, acty) : grz;
          let moves = sees || nRep > 0;
          len = moves ? F.v : 0;
          if (!moves && F.wander_speed > 0) { h = norm2(F.w_inertia * hx0[i] + F.e_noise * ex_, F.w_inertia * hy0[i] + F.e_noise * ey_); len = F.wander_speed; moves = true; }
          if (!moves) h = [hx0[i], hy0[i]];
        } else if (F.activation === "always") {
          h = norm2(actx, acty); len = F.v;
        } else { // contagion
          h = norm2(actx, acty);
          let sn = 0; for (const j of ali) sn += s.spd[j];
          len = sees ? F.v : sn / Math.max(ali.length, 1);
        }
        s.active[i] = sees ? 1 : 0;
        nhx[i] = h[0]; nhy[i] = h[1];
        nx[i] = x0[i] + len * dt * h[0]; ny[i] = y0[i] + len * dt * h[1];
        if (F.activation === "contagion") s.ldx[i] = len; // store the new speed (applied after the loop)
        if (F.D > 0) { const sd = Math.sqrt(2 * F.D * dt); nx[i] += sd * gauss(rng); ny[i] += sd * gauss(rng); }
      }
      if (F.activation === "contagion") for (let i = 0; i < N; i++) s.spd[i] = s.ldx[i];
    } else { // first order
      const Kf = fKernelImages(F, D), g = gKernel(F), im = D.images, pair = hasPair(F), sc = F.population_scaling ? 1 / (N + M) : 1;
      const ldx = Float64Array.from(s.ldx), ldy = Float64Array.from(s.ldy);
      for (let i = 0; i < N; i++) {
        let vx = 0, vy = 0, act = 0;
        for (let m = 0; m < M; m++) {
          const zx = wrap(x0[i] - s.dx[m], D), zy = wrap(y0[i] - s.dy[m], D);
          if (Math.hypot(zx, zy) <= F.L_dog) act = 1;
          Kf(zx, zy, tmp); vx += tmp[0]; vy += tmp[1];
        }
        if (pair) for (let j = 0; j < N; j++) {
          if (j === i) continue;
          const zx = wrap(x0[i] - x0[j], D), zy = wrap(y0[i] - y0[j], D);
          for (let a = -im; a <= im; a++) for (let b = -im; b <= im; b++) {
            const xx = zx + a * L, yy = zy + b * L, r = Math.max(Math.hypot(xx, yy), 1e-9), mm = g(r) / r;
            vx += mm * xx; vy += mm * yy;
          }
        }
        vx *= sc; vy *= sc;
        if (F.ali_fo > 0) {
          let ax = 0, ay = 0, c = 0;
          for (let j = 0; j < N; j++) { if (j !== i && Math.hypot(wrap(x0[i] - x0[j], D), wrap(y0[i] - y0[j], D)) <= F.r_ali) { ax += ldx[j] / dt; ay += ldy[j] / dt; c++; } }
          if (c) { vx += F.ali_fo * ax / c; vy += F.ali_fo * ay / c; }
        }
        const sd = Math.sqrt(2 * F.D * dt);
        nx[i] = x0[i] + vx * dt + sd * gauss(rng); ny[i] = y0[i] + vy * dt + sd * gauss(rng);
        s.ldx[i] = nx[i] - x0[i]; s.ldy[i] = ny[i] - y0[i];
        const hh = norm2(s.ldx[i], s.ldy[i]);
        if (hh[0] || hh[1]) { nhx[i] = hh[0]; nhy[i] = hh[1]; } else { nhx[i] = hx0[i]; nhy[i] = hy0[i]; }
        s.active[i] = act;
      }
    }

    // ------------------------------------------------ herders (read sheep at t−1)
    const ndx = Float64Array.from(s.dx), ndy = Float64Array.from(s.dy), ndvx = Float64Array.from(s.dvx), ndvy = Float64Array.from(s.dvy);
    const [cx, cy] = T.centre;
    if (M && H.controller === "strombom") {
      const G = centroid(x0, y0, N, D);
      let far = 0, farD = -1;
      const fgx = new Float64Array(N), fgy = new Float64Array(N);
      for (let i = 0; i < N; i++) { fgx[i] = wrap(x0[i] - G[0], D); fgy[i] = wrap(y0[i] - G[1], D); const d = Math.hypot(fgx[i], fgy[i]); if (d > farD) { farD = d; far = i; } }
      if (H.hold_radius != null) {
        const dT = Math.hypot(wrap(G[0] - cx, D), wrap(G[1] - cy, D));
        s.holding = s.holding ? dT <= s.hold_release : dT <= H.hold_radius;
      }
      const awx = wrap(G[0] - cx, D), awy = wrap(G[1] - cy, D), gn = Math.hypot(awx, awy);
      const e = gn > 0 ? [awx / gn, awy / gn] : [1, 0];
      const pdx = gn > 0 ? cx + (gn + H.l_drive) * awx / gn : G[0] + H.l_drive * e[0];
      const pdy = gn > 0 ? cy + (gn + H.l_drive) * awy / gn : G[1] + H.l_drive * e[1];
      const relx = wrap(pdx - G[0], D), rely = wrap(pdy - G[1], D);
      let collector = -1;
      if (farD >= H.l_sep && !s.holding) {
        let best = Infinity;
        for (let m = 0; m < M; m++) { const d = Math.hypot(wrap(x0[far] - s.dx[m], D), wrap(y0[far] - s.dy[m], D)); if (d < best) { best = d; collector = m; } }
      }
      for (let m = 0; m < M; m++) {
        let minD = Infinity;
        for (let i = 0; i < N; i++) { const d = Math.hypot(wrap(x0[i] - s.dx[m], D), wrap(y0[i] - s.dy[m], D)); if (d < minD) minD = d; }
        const slow = minD <= F.r_rep && !s.holding;
        const off = H.spread_angle * (m - (M - 1) / 2), c = Math.cos(off), sn = Math.sin(off);
        let aimx = G[0] + c * relx - sn * rely, aimy = G[1] + sn * relx + c * rely, mode = MODE.DRIVE;
        if (m === collector) { aimx = G[0] + (farD + H.l_collect) * fgx[far] / farD; aimy = G[1] + (farD + H.l_collect) * fgy[far] / farD; mode = MODE.COLLECT; }
        if (s.holding) {
          const ox = wrap(s.dx[m] - G[0], D), oy = wrap(s.dy[m] - G[1], D), on = Math.hypot(ox, oy);
          const o = on > 1e-9 ? [ox / on, oy / on] : e;
          aimx = G[0] + s.hold_standoff * o[0]; aimy = G[1] + s.hold_standoff * o[1]; mode = MODE.HOLD;
        }
        const tax = wrap(aimx - s.dx[m], D), tay = wrap(aimy - s.dy[m], D);
        const th = 2 * Math.PI * rng();
        const u = norm2(tax, tay);
        let dir = norm2(u[0] + H.noise * Math.cos(th), u[1] + H.noise * Math.sin(th));
        let v = H.v;
        if (s.holding) { const da = Math.hypot(tax, tay); dir = [tax / Math.max(da, 1e-9), tay / Math.max(da, 1e-9)]; v = Math.min(H.v, da / dt); }
        if (slow) { ndx[m] = s.dx[m] + H.slow_step * dt * s.dvx[m]; ndy[m] = s.dy[m] + H.slow_step * dt * s.dvy[m]; mode = MODE.SLOW; }
        else { ndx[m] = s.dx[m] + v * dt * dir[0]; ndy[m] = s.dy[m] + v * dt * dir[1]; ndvx[m] = dir[0]; ndvy[m] = dir[1]; }
        s.mode[m] = mode;
      }
    } else if (M && H.controller === "density_feedback") {
      if (H.fb_release != null) {
        const err = s.metrics.densityError(x0, y0, N);
        s.holding = s.holding ? err < H.fb_engage : err < H.fb_release;
      }
      const fa = feedbackAims(x0, y0, N, s.dx, s.dy, M, T.centre, s.metrics.surplus(x0, y0, N), H, D, s.holding, F.rule === 'heading' ? F.r_dog : F.L_dog, s.metrics.peak, s.driving);
      s.driving = fa.driving;
      for (let m = 0; m < M; m++) {
        let minD = Infinity;
        for (let i = 0; i < N; i++) { const d = Math.hypot(wrap(x0[i] - s.dx[m], D), wrap(y0[i] - s.dy[m], D)); if (d < minD) minD = d; }
        const slow = minD <= F.r_rep && fa.mode[m] !== MODE.HOLD;
        const tax = wrap(fa.ax[m] - s.dx[m], D), tay = wrap(fa.ay[m] - s.dy[m], D), da = Math.hypot(tax, tay);
        let sx, sy;
        if (slow) { const u = norm2(s.dvx[m], s.dvy[m]); sx = H.slow_step * dt * u[0]; sy = H.slow_step * dt * u[1]; }
        else { const v = Math.min(H.v, da / dt); sx = v * dt * tax / Math.max(da, 1e-9); sy = v * dt * tay / Math.max(da, 1e-9); }
        ndx[m] = s.dx[m] + sx; ndy[m] = s.dy[m] + sy; ndvx[m] = sx / dt; ndvy[m] = sy / dt;
        s.mode[m] = slow ? MODE.SLOW : fa.mode[m];
      }
    } else if (M && H.controller === "dilorenzo") {
      // Feedforward flow: sub-stepping inside one follower step is exact for their dynamics (substeps = 1: one Euler step).
      const grid = s.grid, nn = grid.n * grid.n, nSub = Math.max(Math.round(H.substeps) || 1, 1), hSub = dt / nSub;
      const yx = Float64Array.from(s.dx), yy = Float64Array.from(s.dy);
      for (let j = 0; j < nSub; j++) {
      if (j) for (let m = 0; m < M; m++) { yx[m] = wrap(yx[m], D); yy[m] = wrap(yy[m], D); }
      const rhoH = grid.convolveF(s.KbHF, grid.deposit(yx, yy, M, s.w_dog));
      const re = new Float64Array(nn), imA = new Float64Array(nn);
      for (let q = 0; q < nn; q++) re[q] = s.design.rho_H[q] - rhoH[q];
      fft2(re, imA, grid.n, false);
      // ŵ = i k K_H ê / |k|²
      const wxr = new Float64Array(nn), wxi = new Float64Array(nn), wyr = new Float64Array(nn), wyi = new Float64Array(nn);
      for (let q = 1; q < nn; q++) {
        if (grid.K2[q] === 0) continue;
        const f = H.K_H / grid.K2[q];
        wxr[q] = -grid.KX[q] * imA[q] * f; wxi[q] = grid.KX[q] * re[q] * f;
        wyr[q] = -grid.KY[q] * imA[q] * f; wyi[q] = grid.KY[q] * re[q] * f;
      }
      fft2(wxr, wxi, grid.n, true); fft2(wyr, wyi, grid.n, true);
      const floor = 1e-3 * s.rhoHmax;
      for (let q = 0; q < nn; q++) { const r = Math.max(rhoH[q], floor); wxr[q] /= r; wyr[q] /= r; }
      for (let m = 0; m < M; m++) {
        let ux = grid.sample(wxr, yx[m], yy[m]), uy = grid.sample(wyr, yx[m], yy[m]);
        const sp = Math.hypot(ux, uy);
        if (isFinite(H.v) && sp > H.v) { ux *= H.v / sp; uy *= H.v / sp; }
        yx[m] += ux * hSub; yy[m] += uy * hSub; ndvx[m] = ux; ndvy[m] = uy; s.mode[m] = MODE.FLOW;
      }
      }
      for (let m = 0; m < M; m++) {
        ndx[m] = yx[m]; ndy[m] = yy[m];
        if (nSub > 1) { ndvx[m] = wrap(yx[m] - s.dx[m], D) / dt; ndvy[m] = wrap(yy[m] - s.dy[m], D) / dt; } // mean velocity
      }
    }

    for (let i = 0; i < N; i++) { s.x[i] = wrap(nx[i], D); s.y[i] = wrap(ny[i], D); s.hx[i] = nhx[i]; s.hy[i] = nhy[i]; }
    for (let m = 0; m < M; m++) { s.dx[m] = wrap(ndx[m], D); s.dy[m] = wrap(ndy[m], D); s.dvx[m] = ndvx[m]; s.dvy[m] = ndvy[m]; }
    s.t += dt; s.step_count++;
  }

  function measure(s) {
    const m = s.metrics.all(s.x, s.y, s.N);
    let px = 0, py = 0, act = 0, dsp = 0;
    for (let i = 0; i < s.N; i++) { px += s.hx[i]; py += s.hy[i]; act += s.active[i]; }
    m.polarization = s.F.rule === "heading" ? Math.hypot(px, py) / s.N : NaN;
    m.active_fraction = act / s.N;
    for (let k = 0; k < s.M; k++) dsp += Math.hypot(s.dvx[k], s.dvy[k]) * (s.H.controller === "strombom" ? s.H.v : 1);
    m.dog_speed = s.M ? dsp / s.M : NaN;
    m.t = s.t; m.floor = s.metrics.floor;
    return m;
  }

  // ------------------------------------------------------------------ presets (= bridge.preset)
  const PRESETS = {
    jadhav_reference: {
      label: "Jadhav reference (published model)",
      provenance: "reference",
      note: "herding_model.m / simulation_hm.m: 14 sheep, one Strömbom dog, open plane, start 250 m from the target. Sheep outside r_dog stand still unless repelled.",
      cfg: { followers: {}, herders: { controller: "strombom", n: 1 }, target: { kind: "gaussian", sigma: 2.5 }, domain: { periodic: false } },
      run: { init: "jadhav", box_length: 250, n_steps: 370, dt_label: "1 s" },
    },
    jadhav_modified: {
      label: "Jadhav modified model (Supp. Sec. 3)",
      provenance: "reference",
      note: "herding_model_sim.m: every sheep applies the social rule; the dog term and speed reset act within r_dog = 3 m; speeds start at 0 and are copied from alignment neighbours (8 of 10 attract, 2 align, drive point 2 m behind).",
      cfg: { followers: { activation: "contagion", r_dog: 3.0, n_att: 8, n_ali: 2 }, herders: { controller: "strombom", n: 1, l_drive: 2.0 }, target: { kind: "gaussian", sigma: 2.5 }, domain: { periodic: false } },
      run: { init: "jadhav", box_length: 250, n_steps: 500 },
    },
    dilorenzo_reference: {
      label: "Di Lorenzo reference (arXiv:2411.04791 §IV-A, scaled 1/5)",
      provenance: "reference (N scaled)",
      note: "[−π, π]² torus, first-order followers, f = e^(−r/L) with L = π, α = 1/(N_T+N_H), D = 0.01, von Mises goal k = 6/π, r* = π/2, K_H = 10, KDE bandwidth 0.4. The paper uses 720 + 280 agents and dt = 0.01; this preset uses 144 + 56 and dt = 0.05 so the page stays interactive. The feasibility panel gives the paper's numbers.",
      cfg: { followers: { rule: "first_order", n: 144, dt: 0.05, D: 0.01, a_dog: 1.0, L_dog: Math.PI, population_scaling: true }, herders: { controller: "dilorenzo", n: 56, v: Infinity, K_H: 10, bandwidth: 0.4, grid: 64 }, target: { kind: "vonmises", k: 6 / Math.PI, r_goal: Math.PI / 2, bandwidth: 0.3 }, domain: { periodic: true, size: 2 * Math.PI, images: 1 } },
      run: { init: "uniform", n_steps: 4000 },
    },
    dilorenzo_interacting: {
      label: "Di Lorenzo interacting followers (arXiv:2604.11353, weak Morse)",
      provenance: "reference (2D analogue, N scaled)",
      note: "As the 2024 reference plus the follower–follower Morse kernel g(r) = (1/ℓ_r)e^(−r/ℓ_r) − (ζ/ℓ_a)e^(−r/ℓ_a), ℓ_r = π/2, ℓ_a = π, ζ = 1. The 2026 paper's feasibility now depends on the herder mass (see panel).",
      cfg: { followers: { rule: "first_order", n: 144, dt: 0.05, D: 0.01, a_dog: 1.0, L_dog: Math.PI, population_scaling: true, g_rep: 2 / Math.PI, l_rep: Math.PI / 2, g_att: 1 / Math.PI, l_att: Math.PI }, herders: { controller: "dilorenzo", n: 56, v: Infinity, K_H: 10, bandwidth: 0.4, grid: 64 }, target: { kind: "vonmises", k: 6 / Math.PI, r_goal: Math.PI / 2, bandwidth: 0.3 }, domain: { periodic: true, size: 2 * Math.PI, images: 1 } },
      run: { init: "uniform", n_steps: 4000 },
    },
    hybrid_jadhav_sheep_dilorenzo_dogs: {
      label: "Hybrid A: Jadhav sheep under Di Lorenzo herders (torus)",
      provenance: "hybrid",
      note: "Heading-rule sheep (always active, plus Brownian D = 0.5 m²/s so the design has a diffusion to confine) on an 80 m torus with their von Mises goal (r* = 10 m); herders from Di Lorenzo's design for a weak push 0.8 e^(−r/12 m) m/s, which these sheep do not have. Changes rule, domain, activation, noise and controller together. Switch the rule to first-order to see the followers the design is for.",
      cfg: { followers: { rule: "heading", n: 60, activation: "always", a_dog: 0.8, L_dog: 12.0, population_scaling: false, D: 0.5 }, herders: { controller: "dilorenzo", n: 16, v: 1.5, K_H: 1.0, grid: 64 }, target: { kind: "vonmises", k: 3 / (2 * Math.PI * 10 / 80), r_goal: 10.0, bandwidth: 1.5 }, domain: { periodic: true, size: 80.0 } },
      run: { init: "uniform", n_steps: 600 },
    },
    hybrid_first_order_sheep_strombom_dogs: {
      label: "Hybrid B: first-order sheep (metres) under a Strömbom dog",
      provenance: "hybrid",
      note: "Di Lorenzo-type followers in metres (push 2.3 e^(−r/5 m) m/s as fitted in feasibility.py, Morse cohesion, D = 0.12 m²/s) driven by Jadhav's closed-loop dog on the open plane.",
      cfg: { followers: { rule: "first_order", n: 50, dt: 0.1, D: 0.12, a_dog: 2.3, L_dog: 5.0, g_rep: 1.0, l_rep: 1.0, g_att: 0.2, l_att: 4.0, population_scaling: false }, herders: { controller: "strombom", n: 1, v: 1.5 }, target: { kind: "gaussian", sigma: 3.0 }, domain: { periodic: false } },
      run: { init: "spread", box_length: 8, n_steps: 2000 },
    },
  };

  // Di Lorenzo 2411.04791 §IV-A in metres (= bridge.dilorenzo_scenario): 12/π m and 1 s per unit, 24 m box,
  // r* = 6 m, 36 + 14 agents, herder flow at their dt = 0.01 (100 sub-steps of a 1 s step). The herders design
  // from their own follower model; only the simulated followers differ between the two presets.
  const DL_S = 12 / Math.PI;
  const DL_MODEL = { rule: "first_order", n: 36, dt: 0.1, D: 0.01 * DL_S * DL_S, a_dog: DL_S, L_dog: Math.PI * DL_S, population_scaling: true };
  const DL_HERDERS = (substeps) => ({ controller: "dilorenzo", n: 14, v: Infinity, K_H: 10, bandwidth: 0.4 * DL_S, grid: 64, design_model: DL_MODEL, substeps });
  const DL_TARGET = { kind: "vonmises", k: 6 / Math.PI, r_goal: Math.PI / 2 * DL_S, bandwidth: 0.3 * DL_S };
  const DL_DOMAIN = { periodic: true, size: 2 * Math.PI * DL_S, images: 1 };
  PRESETS.dilorenzo_metres = {
    label: "Di Lorenzo scenario in metres (their followers)",
    provenance: "reference, rescaled",
    note: "Their §IV-A setting at 12/π m and 1 s per unit, so the push range L = π is 12 m: 24 m torus, von Mises goal r* = 6 m, 36 followers and 14 herders (their 28 % herder fraction), K_H = 10, uncapped, flow at dt = 0.01. M̂ = 0.241, the same as in their units. Compare with the next preset, which changes only the followers.",
    cfg: { followers: Object.assign({}, DL_MODEL), herders: DL_HERDERS(10), target: DL_TARGET, domain: DL_DOMAIN },
    run: { init: "uniform", n_steps: 2000, spf: 10 },
  };
  PRESETS.dilorenzo_jadhav_sheep = {
    label: "Di Lorenzo scenario, Jadhav sheep (herders unchanged)",
    provenance: "one component swapped: the followers",
    note: "The same scenario and herders as the previous preset. The design still comes from Di Lorenzo's own follower model (push e^(−r/12 m), Brownian D), but the sheep are Jadhav's published baseline: 1 m/s, flee herders within 12 m, stand still otherwise. Over 5 seeds every sheep is inside r* by about 20 s, but packed about twice as tight as the target (cohesion 1.9 m against 4.2 m). The next preset removes the design and keeps everything else (docs/bridge_simulation.md §4.5).",
    cfg: { followers: { n: 36 }, herders: DL_HERDERS(100), target: DL_TARGET, domain: DL_DOMAIN },
    run: { init: "uniform", n_steps: 200, spf: 1 },
  };
  PRESETS.dilorenzo_jadhav_flat = {
    label: "Same, flat design (control)",
    provenance: "control for the previous preset",
    note: "Jadhav sheep and the same 14 herders running the same flow, but the herders' model has D = 0: the target asks for no drift, so the design is flat and the herders stay spread over the box. Over 5 seeds the flock clumps about 10 m from the goal (17 % inside r*): the designed ring, not the herders' presence, is what brings the sheep in.",
    cfg: { followers: { n: 36 }, herders: Object.assign(DL_HERDERS(100), { design_model: Object.assign({}, DL_MODEL, { D: 0 }) }), target: DL_TARGET, domain: DL_DOMAIN },
    run: { init: "uniform", n_steps: 200, spf: 1 },
  };

  PRESETS.feedback_dog_jadhav = {
    label: "Density-feedback dog over Jadhav sheep (heuristic)",
    provenance: "heuristic controller, published sheep",
    note: "The published sheep with one dog that reads the flock's density surplus and releases when the L² error falls below 1.0 (about twice the finite-N floor). Compare with the Jadhav reference plus the hold (Herders panel). Not from either paper.",
    cfg: { followers: {}, herders: { controller: "density_feedback", n: 1, fb_release: 1.0 }, target: { kind: "gaussian", sigma: 2.5 }, domain: { periodic: false } },
    run: { init: "jadhav", box_length: 60, n_steps: 600 },
  };

  // The convenience slider: λ = 0 is Jadhav's social parameters, λ = 1 removes them. It changes
  // only these fields of the heading rule, never the rule, activation, dog response or domain.
  const SLIDER_FIELDS = { w_att: [1.5, 0], w_ali: [1.3, 0], w_inertia: [0.5, 0], e_noise: [0.3, 0], D: [0, 0.12] };
  function sliderParams(lambda) {
    const out = {};
    for (const k in SLIDER_FIELDS) { const [a, b] = SLIDER_FIELDS[k]; out[k] = a + (b - a) * lambda; }
    return out;
  }

  return { FOLLOWERS, HERDERS, TARGET, DOMAIN, PRESETS, SLIDER_FIELDS, sliderParams, resolve, createSim, step, measure,
    Grid, Metrics, designHerders, feasibilityScan, herdersFromMass, dilorenzoCount, feedbackAims, mulberry32, fft2, centroid, wrap, MODE };
})();

if (typeof module !== "undefined") module.exports = Bridge;
