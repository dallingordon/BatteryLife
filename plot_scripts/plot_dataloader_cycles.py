'''
Visualize the first and last stored cycle of one processed battery file, exactly as
Dataset_original (data_provider/data_loader.py) normalizes it - same resampling,
same voltage/current/capacity normalization, no re-implementation.

Mimics the styling conventions of plot_scripts/plt_MATR_sequences.py (Arial font,
editable pdf.fonttype=42, seaborn color palette, set_ax_linewidth helper) but plots
the *normalized* [3, charge_discharge_length] curve arrays the dataloader actually
feeds to the models, rather than raw pkl records.

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
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Dataset_original hardcodes a relative path to data_provider/life_classes.json,
# so it must be instantiated with the repo root as the working directory.
os.chdir(REPO_ROOT)

from data_provider.data_loader import Dataset_original
from data_provider.data_split_recorder import split_recorder

font = {'family': 'Arial'}
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial'
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42  # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42

CHANNEL_NAMES = ('Voltage (V / V_max)', 'Current (C-rate)', 'Qi (Capacity, normalized)')


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


def plot_cycle(ax, curve, title):
    '''curve: [3, fixed_len] normalized array (voltage, current, Qi) exactly as stored by the dataloader'''
    x = np.arange(curve.shape[1])
    colors = sns.color_palette()
    for ch in range(curve.shape[0]):
        ax.plot(x, curve[ch], '-', color=colors[ch], label=CHANNEL_NAMES[ch], linewidth=1.5)
    ax.set_xlabel('Resampled point index (charge + discharge)', fontsize=12)
    ax.set_ylabel('Normalized value', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    set_ax_linewidth(ax)
    set_ax_font_size(ax, fontsize=10)


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_cycle(axes[0], first_cycle, 'Cycle 1 (first)')
    plot_cycle(axes[1], last_cycle, f'Cycle {curves.shape[0]} (last, end of early-cycle window)')

    fig.suptitle(f'{file_name}  |  source dataset: {source_dataset}  |  label (cycle life): {eol}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    safe_name = file_name.replace('.pkl', '').replace('/', '_')
    out_jpg = os.path.join(args.out_dir, f'{safe_name}_first_last_cycle.jpg')
    out_pdf = os.path.join(args.out_dir, f'{safe_name}_first_last_cycle.pdf')
    plt.savefig(out_jpg, dpi=600)
    plt.savefig(out_pdf)

    print(f'File: {file_name} | Source dataset: {source_dataset} | Label (cycle life, cycles): {eol}')
    print(f'Saved: {out_jpg}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
