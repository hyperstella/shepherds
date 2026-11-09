"""Reproduce the figures and headline numbers of Jadhav et al. (2024).

    .venv/bin/python reproduce.py            # data + model, writes results/
    .venv/bin/python reproduce.py --runs 50  # quicker model run
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shepherd import data, metrics as M, network as NW
from shepherd.model import COLLECT, DRIVE, Params, simulate

OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"

# Reference palette (categorical slots 1-3) and neutrals.
C_DATA, C_MODEL, C_MODEL_DRIVE = "#2a78d6", "#eb6834", "#1baf7a"
C_SHEEP, C_TEXT, C_MUTED, C_GRID = "#b8b7b1", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
})

PAPER = {  # values stated in the paper text (Fig. 2 caption and Results)
    "mean_speed_barycenter": 1.3, "mean_speed_dog": 1.5, "peak_speed_barycenter": 0.9, "peak_speed_dog": 0.8,
    "cohesion_mean": 1.21, "cohesion_sd": 0.34, "polarization_mean": 0.85, "polarization_sd": 0.17,
    "elongation_mode": 0.8, "pearson_di_indegree": 0.85, "n_drives": 26,
}


def pdf(x, edges):
    x = np.asarray(x)
    h, e = np.histogram(x[np.isfinite(x)], bins=edges, density=True)
    return 0.5 * (e[:-1] + e[1:]), h


def hist_mode(x, edges):
    c, h = pdf(x, edges)
    return float(c[np.argmax(h)])


def label(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=C_TEXT)


# --------------------------------------------------------------------------- data

def analyse_data(sessions):
    N = data.N_SHEEP
    concat = data.concat_drives_by_session(sessions)
    drives = data.load_drives(sessions)
    out = {}

    # Fig 2f-j: pooled over concatenated drives, skipping sample 0 where the MATLAB code does
    sp_s, sp_b, sp_d, P, E, xD, yRD, C, dBD = ([] for _ in range(9))
    for c in concat:
        v = c.vel[:, :, 1:]
        sp_s.append(M.speeds(v[:N]).ravel())
        sp_b.append(np.linalg.norm(v[:N].mean(0), axis=0))
        sp_d.append(M.speeds(v)[data.DOG])
        P.append(M.polarization_from_phi(c.phi[:, 1:], N))
        e, x, y = M.frame_observables(c.pos[:, :, 1:], v, N)
        E.append(e), xD.append(x), yRD.append(y)
        C.append(M.cohesion(c.pos, N))
        dBD.append(M.dog_barycenter_distance(c.pos, N))
    out.update({k: np.concatenate(v) for k, v in dict(
        speed_sheep=sp_s, speed_bary=sp_b, speed_dog=sp_d, P=P, E=E, xD=xD, yRD=yRD, C=C, dBD=dBD).items()})

    # Fig 3: viewing angles and heading difference
    psi_bd, psi_db, phi_bd = [], [], []
    for c in concat:
        a, b, d = M.viewing_angles(c.pos[:, :, 1:], c.vel[:, :, 1:], N)
        psi_bd.append(a[:-1]), psi_db.append(b[:-1]), phi_bd.append(d[:-1])
    out.update(psi_BD=np.concatenate(psi_bd), psi_DB=np.concatenate(psi_db), phi_BD=np.concatenate(phi_bd))

    # Fig 4: turning over 3 s (30 samples) vs viewing angle
    turn = [M.turning_vs_viewing(c.pos, c.vel, N, lag=30) for c in concat]
    out["turn"] = {k: np.concatenate([t[k] for t in turn]) for k in turn[0]}

    # Fig 5: leader-follower networks per drive, sheep only (figure_5de.m) and with dog (figure_5c.m)
    indeg, di, indeg_dog, nets = [], [], [], []
    for d in drives:
        ux, uy = np.cos(d.phi[:, 1:]), np.sin(d.phi[:, 1:])
        k, _ = NW.leader_follower(ux[:N], uy[:N])
        indeg.append(k)
        di.append(NW.front_back_position(d.pos[:N, :, 1:], ux[:N], uy[:N], N))
        kd, edges_d = NW.leader_follower(ux[: N + 1], uy[: N + 1])
        indeg_dog.append(kd)
        nets.append((d, edges_d))
    out["indeg"], out["di"] = np.array(indeg), np.array(di)
    out["indeg_with_dog"] = np.array(indeg_dog)
    out["nets"] = nets
    out["concat"], out["drives"] = concat, drives
    return out


# -------------------------------------------------------------------------- model

def analyse_model(res):
    """Observables pooled over runs; also split by the dog's DRIVE mode."""
    p = res.params
    N, R, _ = p.n_sheep, *res.mode.shape
    keys = ["C", "P", "dBD", "E", "xD", "yRD", "psi_BD", "psi_DB", "phi_BD"]
    acc = {k: [] for k in keys}
    drive_mask = []
    for r in range(R):
        pos, vel = res.run(r)
        acc["C"].append(M.cohesion(pos, N))
        acc["P"].append(M.polarization_from_vel(vel, N))
        acc["dBD"].append(M.dog_barycenter_distance(pos, N))
        e, x, y = M.frame_observables(pos, vel, N)
        acc["E"].append(e), acc["xD"].append(x), acc["yRD"].append(y)
        a, b, d = M.viewing_angles(pos, vel, N)
        acc["psi_BD"].append(a), acc["psi_DB"].append(b), acc["phi_BD"].append(d)
        drive_mask.append(res.mode[r] == DRIVE)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["drive_mask"] = np.concatenate(drive_mask)
    out["mode_fraction"] = {
        "drive": float((res.mode[:, 1:] == DRIVE).mean()),
        "collect": float((res.mode[:, 1:] == COLLECT).mean()),
        "slow": float((res.mode[:, 1:] == 3).mean()),
    }

    # Fig 5e: model_network_ana.m (per run, |CCF|, sheep only, lags up to 20 steps)
    indeg, di = [], []
    for r in range(R):
        ux, uy = res.vel_s[r, :, 0, 1:], res.vel_s[r, :, 1, 1:]
        k, edges = NW.leader_follower(ux, uy, maxlag=20, cmin=0.5, use_abs=True)
        if not edges:
            continue
        indeg.append(k)
        di.append(NW.front_back_position(res.pos_s[r, :, :, 1:], ux, uy, N))
    out["indeg"], out["di"] = np.array(indeg), np.array(di)

    # Dog's place in the hierarchy (network over sheep + dog)
    dog_k, sheep_min = [], []
    for r in range(R):
        pos, vel = res.run(r)
        k, _ = NW.leader_follower(vel[:, 0, 1:], vel[:, 1, 1:], maxlag=20, cmin=0.5, use_abs=True)
        dog_k.append(k[N]), sheep_min.append(k[:N].min())
    out["dog_mean_indegree"] = float(np.mean(dog_k))
    out["frac_runs_dog_lowest"] = float(np.mean(np.array(dog_k) <= np.array(sheep_min)))
    return out


def analyse_null(n_runs, seed):
    """Paper's null model: no dog, sheep always interact and move."""
    res = simulate(Params(dog_present=False), n_runs=n_runs, seed=seed)
    edges, P = [], []
    for r in range(n_runs):
        ux, uy = res.vel_s[r, :, 0, 1:], res.vel_s[r, :, 1, 1:]
        edges.append(len(NW.leader_follower(ux, uy, maxlag=20, cmin=0.5, use_abs=True)[1]))
        P.append(M.polarization_from_vel(res.vel_s[r], res.params.n_sheep).mean())
    return {"edges_per_run_mean": float(np.mean(edges)), "polarization_mean": float(np.mean(P))}


# ------------------------------------------------------------------------ figures

def fig1_2_timeseries(D):
    """Fig 1b trajectory and Fig 2a-e time series for drive 5 of session 4."""
    s = next(x for x in D["sessions"] if x.number == 4)
    sl = s.drive_slices()[4]
    sl = slice(sl.start + 40, sl.stop)
    pos, vel, phi = s.pos[:, :, sl], s.vel[:, :, sl], s.phi[:, sl]
    N, T = data.N_SHEEP, pos.shape[2]
    t = np.arange(T) * data.DT
    marks = np.arange(0, T, 50)

    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    for i in range(N):
        ax.plot(pos[i, 0], pos[i, 1], color=C_SHEEP, lw=1)
        ax.plot(pos[i, 0, marks], pos[i, 1, marks], "o", ms=4, color=C_SHEEP)
    b = pos[:N].mean(0)
    vb = vel[:N].mean(0)
    ax.plot(b[0], b[1], color=C_DATA, lw=2, label="Barycenter")
    ax.quiver(b[0, marks], b[1, marks], vb[0, marks], vb[1, marks], color=C_DATA, scale=12, width=0.006)
    ax.plot(pos[data.DOG, 0], pos[data.DOG, 1], color=C_MODEL, lw=1.5, label="Dog")
    ax.plot(pos[data.DOG, 0, marks], pos[data.DOG, 1, marks], "o", ms=5, color=C_MODEL)
    ax.plot([], [], color=C_SHEEP, lw=1, label="Sheep (14)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.set_title("Trajectories, session 4 drive 5 (26 s)")
    ax.legend(loc="best")
    label(ax, "1b")

    e, xD, yRD = M.frame_observables(pos, vel, N)
    panels = [
        ("a", "Speed (m/s)", [(M.speeds(vel[:N]).T, C_SHEEP, None, 0.8), (np.linalg.norm(vb, axis=0), C_DATA, "v_B", 2),
                              (M.speeds(vel)[data.DOG], C_MODEL, "v_D", 2)]),
        ("c", "Polarization P", [(M.polarization_from_phi(phi, N), C_DATA, None, 2)]),
        ("d", "Distance (m)", [(yRD, C_MODEL_DRIVE, "ȳ_RD dog→rear sheep", 2),
                               (M.dog_barycenter_distance(pos, N), C_DATA, "d_BD dog→barycenter", 2)]),
        ("b", "Cohesion C (m) / Elongation E", [(M.cohesion(pos, N), C_DATA, "C", 2), (e, C_MODEL, "E", 2)]),
        ("e", "x̄_D (m) / φ_B (rad)", [(xD, C_MODEL, "x̄_D dog lateral", 2),
                                      (np.arctan2(vb[1], vb[0]), C_DATA, "φ_B flock heading", 2)]),
    ]
    spots = [gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 1], gs[1, 2]]
    for spot, (letter, ylab, series) in zip(spots, panels):
        ax = fig.add_subplot(spot)
        for y, col, lab, lw in series:
            ax.plot(t, y, color=col, lw=lw, label=lab)
        for m in marks:
            ax.axvline(m * data.DT, color=C_GRID, lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Time (s)"), ax.set_ylabel(ylab)
        if any(s[2] for s in series):
            ax.legend(fontsize=8, loc="best")
        label(ax, "2" + letter)
    fig.tight_layout()
    fig.savefig(FIG / "fig1b_fig2ae_timeseries.png")
    plt.close(fig)


def fig2_pdfs(D):
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4))
    se = np.arange(0, 4.01, 0.2)
    for x, col, lab in [(D["speed_bary"], C_DATA, "Barycenter"), (D["speed_dog"], C_MODEL, "Dog"),
                        (D["speed_sheep"], C_SHEEP, "Individual sheep")]:
        axs[0].plot(*pdf(x, se), color=col, label=lab)
    axs[0].set_xlabel("Speed (m/s)"), axs[0].legend(fontsize=8)
    axs[1].plot(*pdf(D["E"], np.arange(0.1, 7.01, 0.2)), color=C_DATA)
    axs[1].set_xlabel("Elongation E"), axs[1].set_xlim(0, 4)
    axs[2].plot(*pdf(D["P"], np.arange(0, 1.001, 0.04)), color=C_DATA)
    axs[2].set_xlabel("Polarization P")
    axs[3].plot(*pdf(D["C"], 31), color=C_DATA, label="Cohesion C")
    axs[3].plot(*pdf(D["dBD"], 31), color=C_MODEL, label="d_BD")
    axs[3].set_xlabel("Distance (m)"), axs[3].legend(fontsize=8)
    axs[4].plot(*pdf(D["yRD"], 60), color=C_MODEL_DRIVE, label="ȳ_RD")
    axs[4].plot(*pdf(D["xD"], 60), color=C_MODEL, label="x̄_D")
    axs[4].set_xlabel("Dog position rel. flock (m)"), axs[4].legend(fontsize=8)
    for ax, l in zip(axs, "fghij"):
        ax.set_ylabel("PDF"), label(ax, "2" + l)
    fig.suptitle("Fig. 2f–j  Observables pooled over all 29 drives", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig2fj_pdfs.png")
    plt.close(fig)


def fig3_4_angles(D):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    edges = np.linspace(-180, 180, 50)
    axs[0].plot(*pdf(np.degrees(D["psi_BD"]), edges), color=C_DATA, label="ψ_BD (barycenter sees dog)")
    axs[0].plot(*pdf(np.degrees(D["psi_DB"]), edges), color=C_MODEL, label="ψ_DB (dog sees barycenter)")
    axs[0].set_xlabel("Viewing angle (deg)"), axs[0].legend(fontsize=8), label(axs[0], "3a")
    axs[1].plot(*pdf(np.degrees(D["phi_BD"]), edges), color=C_DATA)
    axs[1].set_xlabel("Heading difference φ_BD (deg)"), label(axs[1], "3b")
    for ax in axs[:2]:
        ax.set_ylabel("PDF"), ax.set_xticks([-180, -90, 0, 90, 180])

    tr = D["turn"]
    for ax, x, y, col, xl, yl, l in [
        (axs[2], tr["psi_BD"], tr["dphi_B"], C_DATA, "ψ_BD (deg)", "Δφ_B over 3 s (deg)", "4a"),
        (axs[3], tr["psi_DB"], tr["dphi_D"], C_MODEL, "ψ_DB (deg)", "Δφ_D over 3 s (deg)", "4b"),
    ]:
        st = M.binned_stats(x, y)
        c = np.degrees(st["center"])
        ok = st["n"] >= 50  # sparse bins (dog rarely in front of flock) shown hollow, off the line
        q1, q3, med = (np.where(ok, np.degrees(st[k]), np.nan) for k in ("q1", "q3", "median"))
        ax.fill_between(c, q1, q3, color=col, alpha=0.18, lw=0, label="IQR")
        ax.plot(c, med, "o-", color=col, ms=6, label="Median (n ≥ 50)")
        ax.plot(c[~ok], np.degrees(st["median"])[~ok], "o", mfc="none", color=col, ms=6, label="Median (n < 50)")
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_xticks([-180, -90, 0, 90, 180])
        ax.legend(fontsize=8), label(ax, l)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fig4_angles_turning.png")
    plt.close(fig)


def fig5_network(D, Mo, stats_data, stats_model, stats_upstream):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    # 5c: network of one drive in the flock frame (session 4 drive 5, with dog)
    N = data.N_SHEEP
    d, edges = next((d, e) for d, e in D["nets"] if d.session == 4 and d.index == 4)
    pos, vel = d.pos[: N + 1], d.vel[: N + 1]
    xb, yb = M.flock_frame(pos, vel, N)
    x, y = xb.mean(1), yb.mean(1)
    indeg = np.zeros(N + 1, int)
    for leader, follower, _ in edges:
        indeg[leader] += 1
        axs[0].annotate("", xy=(x[leader], y[leader]), xytext=(x[follower], y[follower]),
                        arrowprops=dict(arrowstyle="-|>", color=C_SHEEP, lw=0.8, shrinkA=6, shrinkB=6))
    for i in range(N + 1):
        col = C_MODEL if i == N else C_DATA
        axs[0].scatter(x[i], y[i], s=40 + 25 * indeg[i], color=col, zorder=3, edgecolor="#fcfcfb", lw=1.5)
        axs[0].text(x[i], y[i] + 0.35, "Dog" if i == N else str(i + 1), ha="center", fontsize=8, color=C_TEXT)
    axs[0].annotate("", xy=(0, 1.2), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=2))
    axs[0].set_aspect("equal"), axs[0].set_xlabel("x̄ (m)"), axs[0].set_ylabel("ȳ (m, direction of motion ↑)")
    axs[0].set_title("One drive (session 4, drive 5)\narrows follower→leader, size ∝ indegree", fontsize=10)
    axs[0].text(-0.3, 1.1, "5c", transform=axs[0].transAxes, fontsize=12, fontweight="bold", color=C_TEXT)

    for ax, st, col, title, l in [
        (axs[1], stats_data, C_DATA, f"Data, 29 drives\nPearson r = {stats_data['pearson'][0]:.2f} (paper: 0.85)", "5d"),
        (axs[2], stats_model, C_MODEL, f"Model, our port\nSpearman ρ = {stats_model['spearman'][0]:.2f}", "5e"),
    ]:
        ax.errorbar(st["k"], st["mean"], yerr=st["se"], fmt="o", color=col, ms=7, capsize=0, lw=1.5, label="Mean ± s.e.")
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.set_xlabel("Indegree (number of followers)"), ax.set_ylabel("⟨d_i⟩  front(+) / back(−)  (m)")
        ax.set_title(title, fontsize=10), label(ax, l)
    axs[2].plot(stats_upstream["k"], stats_upstream["mean"], "s", mfc="none", color=C_MUTED, ms=8,
                label="Authors' MATLAB output")
    axs[1].legend(fontsize=8), axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_leader_follower.png")
    plt.close(fig)


def fig6_model_vs_data(D, Mo):
    m = Mo["drive_mask"]
    specs = [
        ("C", "Cohesion C (m)", np.arange(0, 6.01, 0.2)),
        ("E", "Elongation E", np.arange(0.1, 7.01, 0.2)),
        ("P", "Polarization P", np.arange(0, 1.001, 0.04)),
        ("yRD", "ȳ_RD dog→rear sheep (m)", np.arange(-10, 10.01, 0.5)),
        ("xD", "x̄_D dog lateral (m)", np.arange(-10, 10.01, 0.5)),
        ("psi_BD", "ψ_BD (deg)", np.linspace(-180, 180, 50)),
        ("psi_DB", "ψ_DB (deg)", np.linspace(-180, 180, 50)),
        ("phi_BD", "φ_BD (deg)", np.linspace(-180, 180, 50)),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    for ax, (k, xl, edges) in zip(axs.ravel(), specs):
        f = np.degrees if k.startswith(("psi", "phi")) else (lambda v: v)
        ax.plot(*pdf(f(D[k]), edges), color=C_DATA, label="Data (drives)")
        ax.plot(*pdf(f(Mo[k]), edges), color=C_MODEL, label="Model, all steps (as published)")
        ax.plot(*pdf(f(Mo[k][m]), edges), color=C_MODEL_DRIVE, ls="--", label="Model, drive steps only")
        ax.set_xlabel(xl), ax.set_ylabel("PDF")
        if k == "E":
            ax.set_xlim(0, 4)
        if k.startswith(("psi", "phi")):
            ax.set_xticks([-180, -90, 0, 90, 180])
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Fig. 6  Model vs data (model: 300 runs × 370 steps, N = 14, Δt = 1 s)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_model_vs_data.png")
    plt.close(fig)


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    sessions = data.load_sessions()
    D = analyse_data(sessions)
    D["sessions"] = sessions
    res = simulate(Params(), n_runs=args.runs, seed=args.seed)
    Mo = analyse_model(res)

    st_data = NW.indegree_vs_position(D["indeg"].ravel(), D["di"].ravel())
    st_model = NW.indegree_vs_position(Mo["indeg"].ravel(), Mo["di"].ravel(), min_count=100)
    up = np.loadtxt(data.DATA_DIR / "model_di_indeg.csv", delimiter="\t")
    st_up = dict(k=up[0], mean=up[1], se=up[2])

    fig1_2_timeseries(D)
    fig2_pdfs(D)
    fig3_4_angles(D)
    fig5_network(D, Mo, st_data, st_model, st_up)
    fig6_model_vs_data(D, Mo)

    tr = D["turn"]
    flank = (np.abs(tr["psi_BD"]) > np.radians(15)) & (np.abs(tr["psi_BD"]) < np.radians(165))
    kd = D["indeg_with_dog"]
    sheep_rank = kd[:, :14].mean(0)
    summary = {
        "data": {
            "n_sessions": len(sessions),
            "n_drives": len(D["drives"]),
            "drive_seconds_total": float(sum(d.pos.shape[2] for d in D["drives"]) * data.DT),
            "mean_speed_sheep": float(D["speed_sheep"].mean()),
            "mean_speed_barycenter": float(D["speed_bary"].mean()),
            "mean_speed_dog": float(D["speed_dog"].mean()),
            "peak_speed_barycenter": hist_mode(D["speed_bary"], np.arange(0, 4.01, 0.2)),
            "peak_speed_dog": hist_mode(D["speed_dog"], np.arange(0, 4.01, 0.2)),
            "cohesion_mean": float(D["C"].mean()), "cohesion_sd": float(D["C"].std(ddof=1)),
            "polarization_mean": float(D["P"].mean()), "polarization_sd": float(D["P"].std(ddof=1)),
            "polarization_mode": hist_mode(D["P"], np.arange(0, 1.001, 0.04)),
            "elongation_mode": hist_mode(D["E"], np.arange(0.1, 7.01, 0.2)),
            "elongation_median": float(np.nanmedian(D["E"])),
            "dBD_p10_p90": [float(v) for v in np.nanpercentile(D["dBD"], [10, 90])],
            "yRD_p10_p90": [float(v) for v in np.nanpercentile(D["yRD"], [10, 90])],
            "frac_dog_behind_rear_sheep": float((D["yRD"] > 0).mean()),
            "psiBD_frac_within_30deg_of_rear": float((np.abs(D["psi_BD"]) > np.radians(150)).mean()),
            "xD_p5_p95": [float(v) for v in np.nanpercentile(D["xD"], [5, 95])],
            "corr_psiBD_vs_turnB_on_flanks": float(np.corrcoef(tr["psi_BD"][flank], tr["dphi_B"][flank])[0, 1]),
            "corr_psiDB_vs_turnD": float(np.corrcoef(*[v[np.isfinite(v)] for v in (tr["psi_DB"], tr["dphi_D"])])[0, 1])
            if np.isfinite(tr["psi_DB"]).all() and np.isfinite(tr["dphi_D"]).all() else None,
            "pearson_di_indegree": float(st_data["pearson"][0]),
            "pearson_di_indegree_p": float(st_data["pearson"][1]),
            "indegree_levels": st_data["k"].tolist(),
            "dog_mean_indegree": float(kd[:, 14].mean()),
            "sheep_mean_indegree_range": [float(sheep_rank.min()), float(sheep_rank.max())],
            "dog_rank_by_mean_indegree_of_15": int(1 + (np.sort(kd.mean(0))[::-1] > kd[:, 14].mean()).sum()),
        },
        "model": {
            "runs": args.runs, "steps": res.params.n_steps, "seed": args.seed,
            "dog_mode_fraction": Mo["mode_fraction"],
            "cohesion_mean_all": float(np.nanmean(Mo["C"])),
            "cohesion_mean_drive": float(np.nanmean(Mo["C"][Mo["drive_mask"]])),
            "polarization_mean_all": float(np.nanmean(Mo["P"])),
            "polarization_mean_drive": float(np.nanmean(Mo["P"][Mo["drive_mask"]])),
            "elongation_mode_all": hist_mode(Mo["E"], np.arange(0.1, 7.01, 0.2)),
            "elongation_mode_drive": hist_mode(Mo["E"][Mo["drive_mask"]], np.arange(0.1, 7.01, 0.2)),
            "frac_dog_behind_rear_sheep_all": float((Mo["yRD"] > 0).mean()),
            "frac_dog_behind_rear_sheep_drive": float((Mo["yRD"][Mo["drive_mask"]] > 0).mean()),
            "runs_with_edges": int(len(Mo["indeg"])),
            "spearman_di_indegree": float(st_model["spearman"][0]),
            "pearson_di_indegree": float(st_model["pearson"][0]),
            "indegree_levels": st_model["k"].tolist(),
            "upstream_csv_spearman": float(__import__("scipy.stats").stats.spearmanr(up[0], up[1])[0]),
            "di_p5_p95": [float(v) for v in np.percentile(Mo["di"], [5, 95])],
            "dog_mean_indegree": Mo["dog_mean_indegree"],
            "frac_runs_dog_lowest": Mo["frac_runs_dog_lowest"],
            "yRD_p10_p90_drive": [float(v) for v in np.nanpercentile(Mo["yRD"][Mo["drive_mask"]], [10, 90])],
            "dBD_p10_p90": [float(v) for v in np.nanpercentile(Mo["dBD"], [10, 90])],
            "psiBD_frac_within_30deg_of_rear": float((np.abs(Mo["psi_BD"]) > np.radians(150)).mean()),
        },
        "null_model": analyse_null(args.runs, args.seed + 1),
        "paper": PAPER,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
