"""One-command reproduction of both Part III figure panels.

From the repository root, run:
    python josef/reproduce_all.py
"""
from __future__ import annotations

from pathlib import Path
import argparse
from analytic_density import main as analytic_main
from simulate_agents import main as simulation_main


def main(output_dir: Path, quick: bool=False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Generating the analytic density/drift figure...', flush=True)
    analytic_main(output_dir)
    print('Running the stochastic agents (20,000 steps for full mode)...', flush=True)
    metrics = simulation_main(output_dir, quick=quick)
    if not quick:
        expected_start, expected_finish = 138, 577
        if (metrics['initial_sheep_in_goal'], metrics['final_sheep_in_goal']) != (expected_start, expected_finish):
            print('NOTE: Counts differ from our reference run. Check package versions, '
                  'seed, and platform; see agent_metrics.json.')
        else:
            print('Verified: 138 sheep initially and 577 sheep finally inside the goal.')
    print('All generated files:', output_dir.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path,
                   default=Path(__file__).resolve().parent / 'output')
    p.add_argument('--quick', action='store_true',
                   help='Smoke test only; does not reproduce the t=200 slide')
    a = p.parse_args()
    main(a.output_dir, a.quick)
