"""Plot the frozen-policy deployment map from the 41 reported conditions."""
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/tables/table-ec-16.json'
OUT = ROOT / 'out/figures'


def observations():
    table = json.loads(INPUT.read_text())
    levels = defaultdict(dict)
    setting = None
    for panel in table['panels']:
        for row in panel:
            cells = row.get('cells', [])
            if len(cells) != 6 or not isinstance(cells[2], dict):
                continue
            setting = cells[0] or setting
            base, full = (Decimal(cell['mean']) for cell in cells[2:4])
            levels[setting][Decimal(cells[1])] = 100 * (full - base) / base
    return levels


def main():
    levels = observations()
    assert len(levels) == 9 and sum(map(len, levels.values())) == 41
    coordinates = []
    for name, values in levels.items():
        reference = Decimal(2 if name == 'Expedia30' else 1)
        coordinates.append((name, float(values[reference]), float(max(values.values()) - min(values.values()))))
    xmid = median(x for _, x, _ in coordinates)
    ymid = median(y for _, _, y in coordinates)
    fig, ax = plt.subplots(figsize=(9.25, 6.2), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .055, 1, .945))
    names = {'Head': 'E. Head', 'Regular': 'E. Regular', 'Cold': 'E. Cold'}
    offsets = {'Expedia30': (12, 13), 'Q-Cold': (-1, 16), 'Q-Head': (8, 13),
               'Q-Regular': (13, 11), 'Q-Unseen': (10, 12)}
    shown = set()
    for name, x, y in coordinates:
        family = 'KuaiSearch' if name == 'Overall' or name.startswith('Q-') else 'Expedia'
        color = '#b25c31' if family == 'KuaiSearch' else '#22567b'
        ax.scatter(x, y, color=color, s=60, edgecolor='white', linewidth=.6,
                   zorder=3, label=family if family not in shown else None)
        shown.add(family)
        ax.annotate(names.get(name, name), (x, y), xytext=offsets.get(name, (8, 11)),
                    textcoords='offset points', ha='center' if name == 'Q-Cold' else 'left',
                    va='bottom', fontsize=8.6, color='#233744',
                    bbox=dict(boxstyle='round,pad=.16', fc='white', ec='#d8e0e5', lw=.5),
                    arrowprops=dict(arrowstyle='-', color='#8396a3', lw=.8, shrinkB=7))
    ax.axvline(xmid, color='#95a7b3', lw=.9, ls='--')
    ax.axhline(ymid, color='#95a7b3', lw=.9, ls='--')
    for x, uplift, align in [(0.015, 'Low', 'left'), (.985, 'High', 'right')]:
        for y, sensitivity, anchor in [(.964, 'High', 'top'), (.089, 'Low', 'bottom')]:
            ax.text(x, y, f'{uplift} source uplift\n{sensitivity} resource sensitivity',
                    transform=ax.transAxes, ha=align, va=anchor, fontsize=8,
                    color='#526776', bbox=dict(fc='white', ec='none', pad=2))
    ax.set(xlim=(.9, 18.65), ylim=(.15, 19.2),
           xlabel='Source-condition uplift\nFull − Base (% of Base mean)',
           ylabel='Resource sensitivity\nRange of uplift across target resources (percentage points)')
    ax.set_title('Deployment map: source uplift and resource sensitivity', loc='left', weight='bold', pad=13)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.spines[['right','top']].set_visible(False)
    ax.grid(color='#e2e8ed', linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', bbox_to_anchor=(.99, .88), frameon=False, fontsize=9)
    fig.text(.12, .027, f'Dashed lines mark the medians: {xmid:.1f}% uplift and {ymid:.1f} percentage points of sensitivity.', color='#526776', fontsize=8.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'rq3_deployment_quadrants.{extension}', dpi=180)
    plt.close(fig)
    payload = [dict(setting=name, source_uplift_pct=x, sensitivity_pp=y) for name, x, y in coordinates]
    (OUT / 'deployment-map.json').write_text(json.dumps(payload, indent=2)+'\n')


if __name__ == '__main__':
    main()
