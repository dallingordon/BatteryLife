"""
Build the HierGen cycle cache (data_provider/data_loader_gen.py). Run once from the repo root (CPU):

    python precompute_gen_cells.py --root_path ./dataset --workers 4
    python precompute_gen_cells.py --limit 3 --check      # quick: 3 cells + check the grid against the baseline loader

Li-ion (MIX_large) train / val / test cells -> <root_path>/gen_cycle_cache. Re-running skips finished cells.
--check: for the first cells, the V / max(V), I, Q channels must equal Dataset_original.get_charge_discharge_curves.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from data_provider.data_loader_full import make_cell_reader
from data_provider.data_loader_gen import build_gen_cache, default_cache_dir, split_files, cache_paths

_reader = None


def _init(root, P):
    global _reader
    _reader = make_cell_reader(root, P)


def _work(job):
    f, split, cache_dir, overwrite = job
    t = time.time()
    return f, build_gen_cache(_reader, f, cache_dir, split, overwrite), time.time() - t


def check_against_baseline(reader, file_name, cache_dir, n_cycles=5):
    """Grid of the first cycles vs the baseline curves (which divide V by the cycle's max)."""
    import pandas as pd
    data, eol = reader.read_cell_data_according_to_prefix(file_name)
    nominal = 1.85 if file_name.startswith('RWTH') else 3.2 if file_name.startswith('SNL_18650_NCA_25C_20-80') \
        else data['nominal_capacity_in_Ah']
    dfs = []
    for idx, sub in enumerate(data['cycle_data'][:n_cycles]):
        df = pd.DataFrame({k: sub[k] for k in reader.need_keys})
        df['cycle_number'] = idx + 1
        df.loc[df['charge_capacity_in_Ah'] < 0] = np.nan
        df.loc[df['discharge_capacity_in_Ah'] < 0] = np.nan
        df.bfill(inplace=True)
        dfs.append(df)
    base = reader.get_charge_discharge_curves(file_name, pd.concat(dfs), n_cycles, nominal)   # [n, 3, P]
    mine = np.load(cache_paths(cache_dir, file_name)[0])[:n_cycles]
    v = mine[:, 0] / mine[:, 0].max(axis=1, keepdims=True)
    err = max(np.abs(v - base[:, 0]).max(), np.abs(mine[:, 1] - base[:, 1]).max(), np.abs(mine[:, 2] - base[:, 2]).max())
    return err


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', default='./dataset')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    cache_dir = a.cache_dir or default_cache_dir(a.root_path)
    jobs = []
    for s in a.splits:
        for f in split_files(s):
            jobs.append((f, s, cache_dir, a.overwrite))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{len(jobs)} cells -> {cache_dir} | workers {a.workers}', flush=True)
    reader = make_cell_reader(a.root_path, 300)
    merged = f'{a.root_path}/total_MICH/'
    if any(j[0].startswith('MICH') for j in jobs) and not (os.path.isdir(merged) and os.listdir(merged)):
        reader.merge_MICH(merged)
    metas, t0 = {}, time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.root_path, 300)) as pool:
        for n, (f, m, dt) in enumerate(pool.imap_unordered(_work, jobs), 1):
            metas[f] = m
            if n % 25 == 0 or n == len(jobs) or a.limit:
                print(f'  {n}/{len(jobs)} | {time.time() - t0:.0f}s | {f}: {m["status"]}, {m.get("n_cycles")} cycles '
                      f'({dt:.1f}s)', flush=True)
    print('status:', dict(Counter(m['status'] for m in metas.values())))
    trunc = [m for m in metas.values() if m.get('truncated_at_cycle')]
    print(f'caches that stop early at an unusable cycle: {len(trunc)}')
    for m in trunc[:20]:
        print(f'  {m["file_name"]}: cycle {m["truncated_at_cycle"]} of {m["valid_cycle_number"]} ({m["truncate_reason"]})')
    if a.check:
        for f, m in list(metas.items())[:5]:
            if m['status'] in ('ok', 'eol<=100'):
                err = check_against_baseline(reader, f, cache_dir)
                print(f'[check] {f}: max |grid - baseline curve| = {err:.2e} {"OK" if err < 1e-5 else "MISMATCH"}')
