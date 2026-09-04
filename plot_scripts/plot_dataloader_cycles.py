'''
Visualize the first and last stored cycle of one processed battery file, exactly as
Dataset_original (data_provider/data_loader.py) normalizes it - same resampling,
same voltage/current/capacity normalization, no re-implementation.

Both cycles are drawn on one combined plot: voltage and Qi (capacity) are shown for
both cycle 1 and the last stored cycle (they're what actually shift with degradation),
each channel in a light/dark shade pair so the two cycles are easy to tell apart.
Current is checked for whether it's effectively identical across the two cycles (it
usually is - same fixed charge/discharge protocol every cycle) and drawn once if so,
or as two distinct lines if it isn't.

Mimics the styling conventions of plot_scripts/plt_MATR_sequences.py (Arial font when
available, editable pdf.fonttype=42, set_ax_linewidth helper, dual y-axes so current's
larger C-rate swings don't squash voltage/Qi).

Usage (must be able to see the real processed dataset at --root_path, e.g. on the SCC):
    python plot_scripts/plot_dataloader_cycles.py --dataset HUST --flag train
    python plot_scripts/plot_dataloader_cycles.py --dataset HUST --file_name HUST_1-1.pkl
'''
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Dataset_original hardcodes a relative path to data_provider/life_classes.json,
# so it must be instantiated with the repo root as the working directory.
os.chdir(REPO_ROOT)

from data_provider.data_loader import Dataset_original
from data_provider.data_split_recorder import split_recorder


def _pick_font(preferred=('Arial', 'Liberation Sans', 'DejaVu Sans')):
    '''plt_MATR_sequences.py hardcodes Arial (for Illustrator-editable figures), but that
    font isn't installed everywhere (e.g. the SCC). Use it if present, otherwise fall back
    to a metric-compatible or matplotlib-bundled font instead of findfont warning on every
    draw.'''
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return 'DejaVu Sans'  # ships with matplotlib itself, always present


FONT_NAME = _pick_font()
font = {'family': FONT_NAME}
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = FONT_NAME
matplotlib.rcParams['mathtext.it'] = FONT_NAME
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42  # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42

# Voltage: light -> dark blue (cycle 1 -> last cycle). Qi: light -> dark green. Current: gray.
VOLTAGE_COLORS = ('#9ecae1', '#08519c')
QI_COLORS = ('#a1d99b', '#006d2c')
CURRENT_COLOR = '#636363'
CURRENT_SPLIT_COLORS = ('#fdae6b', '#e6550d')  # used only if current actually differs between cycles


def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)


def get_source_dataset_name(file_name):
    '''Same prefix logic Dataset_original.read_data() uses to assign total_dataset_ids -
    i.e. which original source dataset (MATR, HUST, Tongji, ...) this file came from,
    independent of whichever --dataset split (e.g. MIX_large) it's being loaded under.'''
    if (file_name not in split_recorder.MICH_EXP_test_files
            and file_name not in split_recorder.MICH_EXP_train_files
            and file_name not in split_recorder.MICH_EXP_val_files):
        return file_name.split('_')[0]
    return 'MICH_EXP'


def build_dataset_args(root_path, dataset, seq_len, charge_discharge_length, early_cycle_threshold):
    ns = argparse.Namespace()
    ns.root_path = root_path
    ns.dataset = dataset
    ns.seq_len = seq_len
    ns.charge_discharge_length = charge_discharge_length
    ns.early_cycle_threshold = early_cycle_threshold
    ns.weighted_loss = False
    return ns


def plot_combined(ax, first_cycle, last_cycle, last_cycle_number, atol=1e-3):
    '''first_cycle, last_cycle: [3, fixed_len] normalized arrays (voltage, current, Qi) exactly
    as stored by the dataloader. Voltage and Qi both live in roughly [0, 1] normalized units so
    they share the primary axis (light shade = cycle 1, dark shade = last cycle); current (C-rate)
    swings much larger and gets its own twin axis. Current is only drawn twice if the two cycles'
    current profiles actually differ by more than `atol` - otherwise a single shared line is drawn,
    since most protocols apply the same fixed charge/discharge current every cycle.'''
    voltage1, current1, qi1 = first_cycle
    voltage2, current2, qi2 = last_cycle
    x = np.arange(first_cycle.shape[1])

    lines = []
    lines.append(ax.plot(x, voltage1, '-', color=VOLTAGE_COLORS[0], linewidth=1.5, label='Voltage - cycle 1')[0])
    lines.append(ax.plot(x, voltage2, '-', color=VOLTAGE_COLORS[1], linewidth=1.5, label=f'Voltage - cycle {last_cycle_number}')[0])
    lines.append(ax.plot(x, qi1, '-', color=QI_COLORS[0], linewidth=1.5, label='Qi - cycle 1')[0])
    lines.append(ax.plot(x, qi2, '-', color=QI_COLORS[1], linewidth=1.5, label=f'Qi - cycle {last_cycle_number}')[0])

    ax.set_xlabel('Resampled point index (charge + discharge)', fontsize=12)
    ax.set_ylabel('Voltage / Qi (normalized)', fontsize=12)
    set_ax_linewidth(ax)
    set_ax_font_size(ax, fontsize=10)

    ax2 = ax.twinx()
    current_matches = np.allclose(current1, current2, atol=atol)
    if current_matches:
        lines.append(ax2.plot(x, current1, '--', color=CURRENT_COLOR, linewidth=1.2, label='Current (C-rate) - same both cycles')[0])
    else:
        lines.append(ax2.plot(x, current1, '--', color=CURRENT_SPLIT_COLORS[0], linewidth=1.2, label='Current (C-rate) - cycle 1')[0])
        lines.append(ax2.plot(x, current2, '--', color=CURRENT_SPLIT_COLORS[1], linewidth=1.2, label=f'Current (C-rate) - cycle {last_cycle_number}')[0])
    ax2.set_ylabel('Current (C-rate)', color=CURRENT_COLOR, fontsize=12)
    ax2.tick_params('y', colors=CURRENT_COLOR)
    set_ax_linewidth(ax2)
    set_ax_font_size(ax2, fontsize=10)

    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, frameon=False, loc='best')
    return current_matches


def find_usable_file(dataset, requested_file_name):
    '''Return (file_name, charge_discharge_curves, eol) for a file with enough cycle life
    to actually be usable by the dataloader (Dataset_original silently skips files whose
    eol <= early_cycle_threshold).'''
    candidates = [requested_file_name] if requested_file_name else list(dataset.files)
    for file_name in candidates:
        charge_discharge_curves, attn_masks, labels, eol, _ = dataset.read_samples_from_one_cell(file_name)
        if charge_discharge_curves:
            return file_name, charge_discharge_curves[0], eol
    raise RuntimeError(
        f'No usable file found (requested={requested_file_name!r}). '
        f'Files are skipped if their cycle life <= early_cycle_threshold, or if the pkl is missing at root_path.'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the processed dataset (same meaning as run_main.py --root_path)')
    parser.add_argument('--dataset', type=str, default='HUST', help='dataset arg used to pick the split file list (same meaning as run_main.py --dataset)')
    parser.add_argument('--flag', type=str, default='train', choices=['train', 'val', 'test'], help='which split to pull the file from')
    parser.add_argument('--file_name', type=str, default=None, help='specific processed file name to plot; default: first usable file in the split')
    parser.add_argument('--seq_len', type=int, default=5, help='matches run_main.py default')
    parser.add_argument('--charge_discharge_length', type=int, default=100, help='matches run_main.py default')
    parser.add_argument('--early_cycle_threshold', type=int, default=100, help='matches run_main.py default')
    parser.add_argument('--out_dir', type=str, default=os.path.join(REPO_ROOT, 'plot_scripts', 'plots'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds_args = build_dataset_args(args.root_path, args.dataset, args.seq_len, args.charge_discharge_length, args.early_cycle_threshold)

    # val/test require a label_scaler fitted on train; build train first if needed (mirrors run_main.py)
    label_scaler = life_class_scaler = None
    if args.flag != 'train':
        print(f'Loading train split first to fit label_scaler (required for flag={args.flag}) ...')
        train_dataset = Dataset_original(ds_args, flag='train')
        label_scaler = train_dataset.return_label_scaler()
        life_class_scaler = train_dataset.return_life_class_scaler()

    print(f'Loading Dataset_original(dataset={args.dataset}, flag={args.flag}) from {args.root_path} ...')
    dataset = Dataset_original(ds_args, flag=args.flag, label_scaler=label_scaler, life_class_scaler=life_class_scaler)

    file_name, curves, eol = find_usable_file(dataset, args.file_name)
    curves = np.array(curves)  # [early_cycle_threshold, 3, fixed_len]
    source_dataset = get_source_dataset_name(file_name)

    first_cycle = curves[0]
    last_cycle = curves[-1]
    last_cycle_number = curves.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    current_matches = plot_combined(ax, first_cycle, last_cycle, last_cycle_number)
    ax.set_title(f'{file_name}  |  source: {source_dataset}  |  label (cycle life): {eol}', fontsize=12)
    fig.tight_layout()

    safe_name = file_name.replace('.pkl', '').replace('/', '_')
    out_jpg = os.path.join(args.out_dir, f'{safe_name}_first_last_cycle.jpg')
    out_pdf = os.path.join(args.out_dir, f'{safe_name}_first_last_cycle.pdf')
    plt.savefig(out_jpg, dpi=600)
    plt.savefig(out_pdf)

    print(f'File: {file_name} | Source dataset: {source_dataset} | Label (cycle life, cycles): {eol}')
    print(f'Current identical across cycle 1 and cycle {last_cycle_number}: {current_matches}')
    print(f'Saved: {out_jpg}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
