"""Recreate 'Visualizing our density design' (three panels, analytic only).

Run from any working directory: python josef/analytic_density.py
Outputs: josef/output/josef_analytic_illustration_figure.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")  # Works without a display/GUI.
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np

from model import mcrit_2d


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Slide's analytic illustration uses 300 herders; simulation uses 277.
    D, L, k = 0.01, np.pi, 6 / np.pi
    NH, NT = 300, 720
    MH = NH / (NH + NT)
    G1 = 2 * np.pi * L**2 / (1 + L**2)**1.5
    B = D * k / (L * G1)
    A = MH / (4 * np.pi**2)

    x = np.linspace(-np.pi, np.pi, 401)
    X, Y = np.meshgrid(x, x)
    sheep = np.exp(k * (np.cos(X) + np.cos(Y)))
    sheep /= sheep.max()  # Color normalized for visualization, not mass density.
    herder = A - B * (np.cos(X) + np.cos(Y))
    herder /= herder.max()  # Color normalized for visualization.

    fig, axarr = plt.subplots(1, 3, figsize=(13.75, 3.66), dpi=190)
    fig.subplots_adjust(left=.02, right=.99, bottom=.095, top=.91, wspace=.19)
    mag = LinearSegmentedColormap.from_list('sheepmap', ['#fffafa', '#f8a2c5', '#b70b59'])
    blu = LinearSegmentedColormap.from_list('herdermap', ['#fafdff', '#a8c6ec', '#1465ae', '#08346e'])
    for a in axarr:
        a.set_aspect('equal')
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(-np.pi, np.pi)
        a.set_xticks([-np.pi, 0, np.pi])
        a.set_xticklabels([r'$-\pi$', '0', r'$\pi$'], fontsize=9)
        a.set_yticks([0, np.pi])
        a.set_yticklabels(['0', r'$\pi$'], fontsize=9)
        a.tick_params(length=2, width=.7, pad=2, color='#87919b')
        for s in a.spines.values():
            s.set_color('#adb7c1')
            s.set_linewidth(.65)
        a.add_patch(Circle((0, 0), np.pi/2, fill=False, ls=(0, (4, 3)),
                           lw=1.55, ec='#198044', zorder=4))
    axarr[0].imshow(sheep, origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                    cmap=mag, interpolation='bilinear', vmin=0, vmax=1)
    axarr[1].imshow(herder, origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                    cmap=blu, interpolation='bilinear', vmin=0, vmax=1)
    axarr[2].set_facecolor('#f7fbfa')
    axarr[2].add_patch(Circle((0,0), np.pi/2, fc='#e3f2e9', ec='none', zorder=0, alpha=.65))
    q = np.linspace(-2.72, 2.72, 11)
    Qx, Qy = np.meshgrid(q, q)
    # v = -D*k*(sin(x1),sin(x2)); D*k only scales all arrows and is
    # omitted in the quiver drawing to make their direction visible.
    U, V = -np.sin(Qx), -np.sin(Qy)
    axarr[2].quiver(Qx,Qy,U,V,angles='xy',scale_units='xy',scale=3.7,width=.007,
                     headlength=4.5,headwidth=4.2,headaxislength=4.2,
                     color='#236e55',pivot='mid',zorder=3)
    for a, title in zip(axarr, ['Desired sheep density', 'Required herder density',
                                'Induced sheep drift']):
        a.set_title(title, fontweight='bold', size=12, color='#24343a', pad=9)
    for ext in ('pdf', 'png'):
        kwargs = {'dpi': 220} if ext == 'png' else {}
        fig.savefig(output_dir / f'josef_analytic_illustration_figure.{ext}',
                    bbox_inches='tight', pad_inches=.06, **kwargs)
    plt.close(fig)
    print(f'Analytic illustration: N_H={NH}, N_T={NT}, '
          f'M_H={MH:.8f}, threshold={mcrit_2d(D,k,L):.8f}, '
          f'minimum reference density={A-2*B:.10f}')
    print(f'Saved two files to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'output')
    args = parser.parse_args()
    main(args.output_dir)
