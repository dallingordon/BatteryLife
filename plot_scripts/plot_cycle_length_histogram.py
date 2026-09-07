"""
Sorted per-cell view of real, usable cycle-sequence length (from cycle_length_stats.csv),
each bar split at the current training pipeline's early_cycle_threshold cutoff.

Every cell's FULL bar is drawn - the blue portion is what Dataset_original actually uses
(<= cutoff cycles), the red portion on top is real, physically measured cycle data that
sits past the cutoff and is never touched by training. This intentionally avoids the
"looks like it's not in the dataset" reading a plain histogram-with-a-line gives: the data
is there (full bar height = real usable_length), the split just shows how much of it goes
unused.

Usage:
    source plot_scripts/venv/bin/activate
    python plot_scripts/plot_cycle_length_histogram.py \
        --csv ../cycle_length_stats.csv --cutoff 100 --out ../figures/cycle_length_histogram.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_available = {f.name.lower() for f in fm.fontManager.ttflist}
for candidate in ['Arial', 'Liberation Sans', 'DejaVu Sans']:
    if candidate.lower() in _available:
        plt.rcParams['font.family'] = candidate
        break

SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
USED_COLOR = '#2a78d6'     # sequential blue - real data the model actually sees
UNUSED_COLOR = '#d03b3b'   # status critical - real data past the cutoff, never used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='cycle_length_stats.csv')
    ap.add_argument('--cutoff', type=int, default=100,
                     help='the early_cycle_threshold currently used by Dataset_original')
    ap.add_argument('--out', default='figures/cycle_length_histogram.png')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    s = df['usable_length'].dropna().sort_values().reset_index(drop=True)
    n = len(s)

    used = np.minimum(s, args.cutoff)
    unused = np.maximum(s - args.cutoff, 0)

    frac_cells_over = (s > args.cutoff).mean()
    frac_volume_thrown_away = unused.sum() / s.sum()

    print(f'n cells = {n}')
    print(f'{(s > args.cutoff).sum()} cells ({frac_cells_over * 100:.1f}%) have usable_length > {args.cutoff}')
    print(f'{frac_volume_thrown_away * 100:.1f}% of total real cycle-volume lies past the cutoff '
          f'and is never used by the current {args.cutoff}-cycle window')

    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.bar(x, used, width=1.0, color=USED_COLOR, linewidth=0, zorder=2,
           label=f'used by training (≤ {args.cutoff} cycles)')
    ax.bar(x, unused, width=1.0, bottom=used, color=UNUSED_COLOR, linewidth=0, zorder=2,
           label='real data past the cutoff (never used)')

    ax.axhline(args.cutoff, color=INK_PRIMARY, linestyle='--', linewidth=1.4, zorder=4)
    ax.text(n * 0.01, args.cutoff, f'  {args.cutoff}-cycle training cutoff', color=INK_PRIMARY,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

    annotation = (
        f'{frac_cells_over * 100:.0f}% of cells have more real cycles than the cutoff uses\n'
        f'{frac_volume_thrown_away * 100:.0f}% of all real cycle-data (red) is never seen by training'
    )
    ax.text(0.02, 0.97, annotation, transform=ax.transAxes, ha='left', va='top',
            fontsize=10.5, color=INK_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.8))

    ax.set_xlim(0, n)
    ax.set_ylim(0, s.max() * 1.03)
    ax.set_xlabel(f'cells, sorted by usable_length (n={n})', color=INK_SECONDARY)
    ax.set_ylabel('usable_length (real measured cycles)', color=INK_SECONDARY)
    ax.set_title('Real cycle data per cell: how much sits past the training cutoff',
                 color=INK_PRIMARY, fontsize=13, fontweight='bold', pad=14)

    ax.tick_params(colors=INK_MUTED)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(BASELINE)
    ax.grid(axis='y', color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(loc='lower right', frameon=True, facecolor=SURFACE, edgecolor=BASELINE, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    root, _ext = os.path.splitext(args.out)
    fig.savefig(root + '.pdf', facecolor=SURFACE)
    print(f'\nSaved figure to {args.out} and {root}.pdf')


if __name__ == '__main__':
    main()
