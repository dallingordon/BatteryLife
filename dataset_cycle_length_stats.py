"""
Distribution statistics of real, usable cycle-sequence lengths across every processed
dataset in ./dataset - for sizing a variable-length-context SSM training setup (sample
subsequences from a minimum up to a maximum length, predict EOL at the cutoff).

Run this from the BatteryLife repo root, on the machine where ./dataset/<SOURCE>/*.pkl
and "./dataset/Life labels/*.json" actually live (the SCC, not a laptop checkout).

For each cell file this reproduces the same eol / valid_cycle_number logic that
data_provider/data_loader.py's Dataset_original.read_cell_df uses, without instantiating
the full Dataset class:
    eol                = the life label for that file (from "Life labels/<X>_labels.json")
    valid_cycle_number = len(data['cycle_data'])   (cycles actually recorded in the pkl)
    usable_length      = min(eol, valid_cycle_number)  <- the real subsequence ceiling,
                          since read_samples_from_one_cell never samples i >= eol anyway.

Known simplification vs. the full loader: a couple of dataset-specific filename quirks
(BIT2, the SMICH->MICH_EXP rename) aren't replicated here, so a handful of files may show
up as "no eol found." The script prints the match rate so you can tell whether that's
negligible or worth chasing down.

Note on runtime: each pkl stores the FULL un-resampled per-cycle time series (every
voltage/current/capacity/time sample ever recorded for that cell), not the compact
resampled curves the model trains on - so pickle.load() has to deserialize all of that
just to read len(cycle_data). That's why this can take a while on the larger sources
(ISU-ILCC, Stanford_2, MATR, ...); the progress bar below is there so it doesn't look hung.

Usage:
    cd BatteryLife
    python dataset_cycle_length_stats.py [--dataset-root ./dataset] [--out cycle_length_stats.csv]
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import pandas as pd

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

SKIP_DIRS = {'Life labels', 'READMEs', 'seen_unseen_labels'}


def load_all_life_labels(dataset_root):
    """Merge every Life labels/*.json into one {file_name: eol} lookup."""
    label_dir = os.path.join(dataset_root, 'Life labels')
    lookup = {}
    per_file_counts = {}
    for path in sorted(glob.glob(os.path.join(label_dir, '*.json'))):
        with open(path) as f:
            data = json.load(f)
        per_file_counts[os.path.basename(path)] = len(data)
        for k, v in data.items():
            lookup[k] = v
    return lookup, per_file_counts


def eol_lookup_key(source_dir, file_name):
    """Mirror the one dataset-specific rename read_cell_data_according_to_prefix applies
    before checking the label dict (Tongji labels use '-#', filenames on disk use '--')."""
    if source_dir.startswith('Tongji'):
        return file_name.replace('--', '-#')
    return file_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='./dataset')
    ap.add_argument('--out', default='cycle_length_stats.csv')
    ap.add_argument('--progress-every', type=int, default=25,
                     help='when tqdm is unavailable, print a line every N files')
    args = ap.parse_args()

    life_labels, per_file_counts = load_all_life_labels(args.dataset_root)
    print(f'Loaded {len(life_labels)} life labels from {len(per_file_counts)} label files:')
    for fname, n in sorted(per_file_counts.items()):
        print(f'  {fname}: {n}')

    source_dirs = sorted(
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d)) and d not in SKIP_DIRS
    )

    # Collect (source_dir, pkl_path) pairs up front so we know the total for a progress bar.
    all_files = []
    for source_dir in source_dirs:
        pkl_files = sorted(glob.glob(os.path.join(args.dataset_root, source_dir, '*.pkl')))
        print(f'  found {len(pkl_files):4d} .pkl files in {source_dir}/')
        all_files.extend((source_dir, p) for p in pkl_files)

    print(f'\nReading {len(all_files)} cell files (this is the slow part - each pkl holds '
          f'full raw per-cycle data, not the resampled curves)...\n')

    rows = []
    iterator = tqdm(all_files, unit='file') if HAVE_TQDM else all_files
    start = time.time()
    last_source = None
    for i, (source_dir, pkl_path) in enumerate(iterator):
        file_name = os.path.basename(pkl_path)

        if HAVE_TQDM:
            if source_dir != last_source:
                iterator.set_description(source_dir)
                last_source = source_dir
        else:
            if source_dir != last_source:
                print(f'-- starting {source_dir} --')
                last_source = source_dir
            if i % args.progress_every == 0:
                elapsed = time.time() - start
                print(f'  [{i}/{len(all_files)}] elapsed={elapsed:.0f}s  now on {source_dir}/{file_name}')

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            valid_cycle_number = len(data['cycle_data'])
        except Exception as e:
            print(f'  [WARN] could not read {pkl_path}: {e}', file=sys.stderr)
            continue

        key = eol_lookup_key(source_dir, file_name)
        eol = life_labels.get(key)

        rows.append({
            'source': source_dir,
            'file_name': file_name,
            'eol': eol,
            'valid_cycle_number': valid_cycle_number,
            'usable_length': min(eol, valid_cycle_number) if eol is not None else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nWrote {len(df)} rows to {args.out}')

    matched = df[df['eol'].notna()].copy()
    print(f'Matched an eol label for {len(matched)}/{len(df)} files '
          f'({100 * len(matched) / max(len(df), 1):.1f}%)')

    if len(matched) == 0:
        return

    def summarize(series, label):
        s = series.dropna()
        qs = s.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        print(f'\n{label}  (n={len(s)})')
        print(f'  min={s.min():.0f}  max={s.max():.0f}  mean={s.mean():.1f}  '
              f'median={s.median():.0f}  std={s.std():.1f}')
        print('  percentiles: ' + ', '.join(f'{int(p * 100)}%={v:.0f}' for p, v in qs.items()))

    summarize(matched['usable_length'], 'Overall usable_length (min(eol, valid_cycle_number))')

    print('\nPer-source breakdown (usable_length):')
    for source, g in matched.groupby('source'):
        s = g['usable_length'].dropna()
        if len(s) == 0:
            continue
        print(f'  {source:15s} n={len(s):4d}  min={s.min():6.0f}  median={s.median():6.0f}  '
              f'mean={s.mean():7.1f}  max={s.max():6.0f}')

    mismatch = matched[matched['valid_cycle_number'] < matched['eol']]
    if len(mismatch) > 0:
        print(f'\n{len(mismatch)} files have fewer recorded cycles than their eol label '
              f'(valid_cycle_number < eol) - usable_length is capped below the eol target for these:')
        print(mismatch[['source', 'file_name', 'eol', 'valid_cycle_number']]
              .sort_values('source').to_string(index=False))


if __name__ == '__main__':
    main()
