"""Plot every final training seed; do not imply intermediate learning curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    data = json.loads(Path(args.results).read_text())
    assert data['complete'] and len(data['task_rows']) == 12
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    phases = ['main_v1', 'dual_buffer_v2', 'pure_autonomous_main_v1']
    labels = ['Original defaults', 'Filtered dual buffer', 'Successful pure-autonomous protocol']
    colors = ['#0072B2', '#D55E00', '#009E73']
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(17, 6))
    for i, (phase, label, color) in enumerate(zip(phases, labels, colors)):
        offset = (i - 1) * .23
        means = [r[phase] for r in data['task_rows']]
        ax.bar(x + offset, means, width=.21, color=color, alpha=.72,
               label=f"{label} (macro {data['macro'][phase]:.3f})")
        for j, task in enumerate(data['task_rows']):
            values = sorted((r['seed'], r['flowdagger']) for r in data['studies'][phase] if r['task'] == task['task'])
            assert len(values) == 3
            ax.scatter(j + offset + np.linspace(-.045, .045, 3), [v for _, v in values],
                       color='black', s=12, alpha=.7, zorder=4)
    ax.scatter(x, [r['paper_flow'] for r in data['task_rows']], marker='D',
               facecolors='white', edgecolors='black', s=45, zorder=6, label='Paper Table 1 Flow')
    ax.set_xticks(x, [r['task'].removeprefix('metaworld_').replace('_', '-') for r in data['task_rows']], rotation=30, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Final success rate')
    ax.set_title('Same 50-rollout collection budget; all three training seeds retained')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, frameon=False)
    fig.text(.5, .01, 'Bars: seed means. Black dots: individual seeds (25 evaluations each). '
             'Actual update counts vary. Pure protocol expanded after a 3-task exploratory study.',
             ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.34, left=.055, right=.995, top=.91)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'main_protocol_comparison.{ext}', dpi=180)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
