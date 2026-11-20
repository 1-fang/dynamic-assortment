"""Plot epoch-level validation means supplied in a CSV by the caller."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCENES = ('Expedia30', 'Expedia Head', 'Expedia Regular', 'Expedia Cold',
          'KuaiSearch Overall', 'Q-Head', 'Q-Regular', 'Q-Cold', 'Q-Unseen')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path, help='Columns: setting, epoch, base_mean, sr_mean')
    parser.add_argument('--out', type=Path, default=Path('out/figures/11_epoch_trajectories.pdf'))
    args = parser.parse_args()
    with args.csv_file.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    values = defaultdict(dict)
    for row in rows:
        scene, epoch = row['setting'], int(row['epoch'])
        if scene not in SCENES or epoch in values[scene]:
            raise ValueError(f'Unexpected setting or duplicate epoch: {scene}, {epoch}')
        values[scene][epoch] = (float(row['base_mean']), float(row['sr_mean']))
    if set(values) != set(SCENES):
        raise ValueError('Supply observed validation means for all nine settings')
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), layout='constrained')
    for ax, scene in zip(axes.flat, SCENES, strict=True):
        epochs = sorted(values[scene])
        base, sr = zip(*(values[scene][epoch] for epoch in epochs))
        ax.plot(epochs, base, 'o-', ms=3, label='A2C Transformer')
        ax.plot(epochs, sr, 's--', ms=3, label='A2C Transformer + S/R')
        metric = ('Revenue / day' if scene == 'Expedia30' else
                  'Revenue / search' if scene.startswith('Expedia ') else 'GMV / 1,000')
        ax.set(title=scene, xlabel='Epoch', ylabel=metric)
        ax.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == '__main__':
    main()
