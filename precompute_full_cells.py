"""
Build the cycle cache for the full-timescale loader (data_provider/data_loader_full.py). Run once, from the repo root:

    python precompute_full_cells.py --root_path ./dataset --workers 4

Writes one <cell>.npy ([n_cycles, 3, charge_discharge_length] float32, cycles 1..eol-1) and one <cell>.meta.json per
training cell into <root_path>/full_cycle_cache. Re-running skips finished cells (--overwrite to redo). By default
covers the training cells of all three split seeds (2021/42/2024), so any --split_seed can be used afterwards.
Cells with no pkl / no label are recorded in their meta and skipped by the loader, same as the baseline.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import Pool

from data_provider.data_loader import CHEMISTRIES, POOLED_SPLIT_SEEDS, pooled_split_files
from data_provider.data_loader_full import make_cell_reader, build_cell_cache, default_cache_dir

_reader = None


def _init_worker(root_path, charge_discharge_length):
    global _reader
    _reader = make_cell_reader(root_path, charge_discharge_length)


def _work(job):
    file_name, cache_dir, overwrite = job
    t = time.time()
    return file_name, build_cell_cache(_reader, file_name, cache_dir, overwrite), time.time() - t


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', default='./dataset')
    p.add_argument('--cache_dir', default=None, help='default <root_path>/full_cycle_cache')
    p.add_argument('--charge_discharge_length', type=int, default=300)   # same as the tuned-hyperparameter runs
    p.add_argument('--chemistries', nargs='+', default=CHEMISTRIES, choices=CHEMISTRIES)
    p.add_argument('--split_seeds', nargs='+', type=int, default=list(POOLED_SPLIT_SEEDS), choices=POOLED_SPLIT_SEEDS)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='debug: only the first N cells')
    a = p.parse_args()
    cache_dir = a.cache_dir or default_cache_dir(a.root_path)

    files = []
    for c in a.chemistries:
        for s in a.split_seeds:
            for f in pooled_split_files(c, 'train', s):
                if f not in files:
                    files.append(f)
    if a.limit:
        files = files[:a.limit]
    print(f'{len(files)} training cells -> {cache_dir} | workers={a.workers}')

    # MICH cells read from a merged folder; build it once here, before any worker starts
    reader = make_cell_reader(a.root_path, a.charge_discharge_length)
    merged = f'{a.root_path}/total_MICH/'
    if any(f.startswith('MICH') for f in files) and not (os.path.isdir(merged) and os.listdir(merged)):
        reader.merge_MICH(merged)

    jobs = [(f, cache_dir, a.overwrite) for f in files]
    metas, t0 = {}, time.time()
    with Pool(a.workers, initializer=_init_worker, initargs=(a.root_path, a.charge_discharge_length)) as pool:
        for n, (f, meta, dt) in enumerate(pool.imap_unordered(_work, jobs), 1):
            metas[f] = meta
            if n % 25 == 0 or n == len(jobs):
                print(f'  {n}/{len(jobs)} cells | {time.time() - t0:.0f}s elapsed | last: {f} ({dt:.1f}s)', flush=True)

    status = Counter(m['status'] for m in metas.values())
    ok = [m for m in metas.values() if m['status'] == 'ok']
    print(f'\nstatus: {dict(status)}')
    print(f'cycles stored: {sum(m["n_cycles"] for m in ok)} over {len(ok)} cells | longest: {max((m["n_cycles"] for m in ok), default=0)}')
    trunc = [m for m in ok if m['truncated_at_cycle'] is not None]
    print(f'cells whose cache stops early at an unusable cycle: {len(trunc)}')
    for m in trunc[:30]:
        print(f'  {m["file_name"]}: stopped at cycle {m["truncated_at_cycle"]} of {m["n_wanted"]} ({m["truncate_reason"]})')
    size = sum(os.path.getsize(os.path.join(cache_dir, x)) for x in os.listdir(cache_dir) if x.endswith('.npy'))
    print(f'cache size: {size / 1e9:.2f} GB')
