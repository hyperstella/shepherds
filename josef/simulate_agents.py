"""Reproduce the 2D agent simulation and confinement plot (actual trajectories).

Run: python josef/simulate_agents.py
The algorithm is the seminar group's ``part3_continuification.py/run_2d``,
vendored without modification in model.py. No internet or external datasets
are needed after the packages in requirements.txt are installed.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import i0

from model import run_2d, mcrit_2d

D = 0.01
L = np.pi
K_H = 10.0
K = 6.0 / np.pi
R_GOAL = np.pi / 2
N_SHEEP = 720
N_HERDERS = 277
GRID = 64
DELTA_T = 0.01
T_FINAL = 200.0
KDE_BW = 0.4
V_MAX = 3.0
SEED = 0


def exact_target_goal_fraction(k_value: float) -> float:
    """Probability mass of product-von-Mises target inside a centered disk."""
    Z = (2 * np.pi * i0(k_value))**2
    def radial_integrand(r, theta):
        return r * np.exp(k_value * (np.cos(r*np.cos(theta)) + np.cos(r*np.sin(theta))))
    inside, _ = quad(
        lambda r: quad(lambda theta: radial_integrand(r, theta), 0, 2*np.pi,
                       epsabs=1e-10)[0], 0, R_GOAL, epsabs=1e-10
    )
    return float(inside/Z)


def snapshot(ax, H, T, title, show_legend=False):
    ax.scatter(T[:, 0], T[:, 1], color='#C2185B', s=5, alpha=.8,
               linewidths=0, label='Sheep (720)')
    ax.scatter(H[:, 0], H[:, 1], color='#1565C0', s=11, alpha=.8,
               marker='D', linewidths=0, label='Herders (277)')
    ax.add_patch(Circle((0, 0), R_GOAL, fill=False, color='#2E7D32',
                        linestyle='--', lw=1.4, label='Goal disk'))
    ax.set(xlim=(-np.pi,np.pi), ylim=(-np.pi,np.pi))
    ax.set_aspect('equal')
    ax.set_xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
    ax.set_yticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='lower left', fontsize=7)


def main(output_dir: Path, quick: bool=False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    t_final = 2.0 if quick else T_FINAL
    result = run_2d(NT=N_SHEEP, NH=N_HERDERS, D=D, L=L,
                    KH=K_H, k=K, rstar=R_GOAL, M=GRID, dt=DELTA_T,
                    T=t_final, bw=KDE_BW, seed=SEED, vmax=V_MAX,
                    n_frames=0)
    target_exact = exact_target_goal_fraction(K)
    initial_percent = 100*float(result['chi'][0])
    final_percent = 100*float(result['chi'][-1])
    report = {
        'run_type': 'quick_smoke_test' if quick else 'full_slide_reproduction',
        'N_sheep': N_SHEEP, 'N_herders': N_HERDERS,
        'diffusion_D': D, 'kernel_range_L': L, 'target_k': K,
        'feedback_KH': K_H, 'goal_radius': R_GOAL,
        'dt': DELTA_T, 't_final': t_final, 'grid': GRID,
        'kde_bandwidth': KDE_BW, 'max_herder_speed': V_MAX, 'seed': SEED,
        'initial_sheep_in_goal': int(round(result['chi'][0]*N_SHEEP)),
        'final_sheep_in_goal': int(round(result['chi'][-1]*N_SHEEP)),
        'initial_goal_percent': initial_percent,
        'final_goal_percent': final_percent,
        'target_goal_percent': 100*target_exact,
        'critical_herder_mass': float(result['Mc_cf']),
        'actual_herder_mass': float(result['MH']),
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__
    }
    with (output_dir/'agent_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    with (output_dir/'reproduce_2d_agent_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['time', 'chi_percent'])
        writer.writerows(zip(result['times'], 100*result['chi']))

    plt.rcParams.update({'font.size':9, 'font.family':'DejaVu Sans'})
    fig, axes=plt.subplots(1,3,figsize=(14.6,3.15),
                          gridspec_kw={'width_ratios':[1,1,1.65]})
    H0,T0=result['snap0']
    snapshot(axes[0],H0,T0,f't=0: {initial_percent:.2f}%',True)
    snapshot(axes[1],result['H'],result['T'],
             f't={t_final:g}: {final_percent:.2f}%')
    axes[2].plot(result['times'],100*result['chi'],color='#C2185B',lw=1.8,
                 label='Agent simulation')
    axes[2].axhline(100*target_exact,color='#2E7D32',ls=':',
                     label=f'Von Mises target = {100*target_exact:.2f}%')
    axes[2].set(xlim=(0,t_final),ylim=(0,100),xlabel='Time (model units)',
                ylabel='Sheep inside goal (%)',title='Confinement')
    axes[2].legend(frameon=False,loc='lower right')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output_dir/f'reproduced_2d_agents.{ext}',
                    **({'dpi':180} if ext=='png' else {}))
    plt.close(fig)
    print('RESULTS:', json.dumps(report,indent=2))
    print(f'Saved image, CSV and JSON in {output_dir}')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,
                        default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--quick',action='store_true',
                        help='Fast 2-model-unit smoke test; NOT the published slide')
    args=parser.parse_args()
    main(args.output_dir,args.quick)
