"""
Histogram of real, usable cycle-sequence length per cell (from dataset_cycle_length_stats.py's
cycle_length_stats.csv), with the current training pipeline's early_cycle_threshold cutoff
marked - to show how much real per-cycle data past that cutoff is never touched by
Dataset_original today.

Usage:
    source plot_scripts/venv/bin/activate   # (any env with pandas + matplotlib works)
    python plot_scripts/plot_cycle_length_histogram.py \
        --csv ../cycle_length_stats.csv --cutoff 100 --out ../figures/cycle_length_histogram.png

Run from plot_scripts/ (matching plot_dataloader_cycles.py's convention), or pass explicit
--csv / --out paths if you run it from the repo root instead.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font fallback: Arial usually isn't installed on the SCC - avoid noisy findfont warnings,
# same pattern as plot_dataloader_cycles.py.
_available = {f.name.lower() for f in fm.fontManager.ttflist}
for candidate in ['Arial', 'Liberation Sans', 'DejaVu Sans']:
    if candidate.lower() in _available:
        plt.rcParams['font.family'] = candidate
        break

# dataviz skill reference palette (light mode)
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
BAR_COLOR = '#2a78d6'      # sequential blue, step 450
CUTOFF_COLOR = '#d03b3b'   # status: critical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='cycle_length_stats.csv')
    ap.add_argument('--cutoff', type=int, default=100,
                     help='the early_cycle_threshold currently used by Dataset_original')
    ap.add_argument('--out', default='figures/cycle_length_histogram.png')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    s = df['usable_length'].dropna()
    n = len(s)

    over = s[s > args.cutoff]
    frac_cells_over = len(over) / n
    used_volume = np.minimum(s, args.cutoff).sum()
    total_volume = s.sum()
    frac_volume_thrown_away = 1 - used_volume / total_volume

    print(f'n cells = {n}')
    print(f'{len(over)} cells ({frac_cells_over * 100:.1f}%) have usable_length > {args.cutoff}')
    print(f'{frac_volume_thrown_away * 100:.1f}% of total real cycle-volume lies past the cutoff '
          f'and is never used by the current {args.cutoff}-cycle window')

    # log-spaced bins: this distribution spans ~1 to a few thousand and is heavily right-skewed
    lo = max(1, s.min())
    hi = s.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.hist(s, bins=bins, color=BAR_COLOR, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.set_xscale('log')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axvspan(args.cutoff, xlim[1], color=CUTOFF_COLOR, alpha=0.06, zorder=0)
    ax.axvline(args.cutoff, color=CUTOFF_COLOR, linestyle='--', linewidth=2, zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim[0], ylim[1] * 1.18)
    ax.text(args.cutoff, ylim[1] * 1.19, f'current training cutoff\n({args.cutoff} cycles)',
            color=CUTOFF_COLOR, fontsize=10, ha='center', va='top', fontweight='bold')

    annotation = (
        f'{frac_cells_over * 100:.0f}% of cells have more real cycles than the cutoff uses\n'
        f'{frac_volume_thrown_away * 100:.0f}% of all real cycle-data is never seen by training'
    )
    ax.text(0.98, 0.96, annotation, transform=ax.transAxes, ha='right', va='top',
            fontsize=10.5, color=INK_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.8))

    ax.set_xlabel('usable_length (real measured cycles, log scale)', color=INK_SECONDARY)
    ax.set_ylabel('number of cells', color=INK_SECONDARY)
    ax.set_title("How much real cycle data the 100-cycle cutoff throws away",
                 color=INK_PRIMARY, fontsize=13, fontweight='bold', pad=14)

    ax.tick_params(colors=INK_MUTED)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(BASELINE)
    ax.grid(axis='y', color=GRIDLINE, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    # also drop a PDF next to it, matching the repo's figures/ convention (jpg+pdf pairs)
    root, _ext = os.path.splitext(args.out)
    fig.savefig(root + '.pdf', facecolor=SURFACE)
    print(f'\nSaved figure to {args.out} and {root}.pdf')


if __name__ == '__main__':
    main()
