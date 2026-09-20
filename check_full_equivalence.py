"""
Check on real data that the full-timescale cache matches the baseline loader. Run from the repo root on SCC
after precompute_full_cells.py:

    python check_full_equivalence.py                        # ~2 min: CALB / Zn-ion / Na-ion scalers + counts, curves for a few cells of every chemistry

Checks
  1. first 100 cached cycles of a few cells per chemistry (incl. a Zn-ion charge-first cell) == Dataset_original.read_cell_df curves
  2. with max_cycles=100 and no cap, sample counts per chemistry and the label scaler == Dataset_pooled(flag='train')
     (small chemistries only; Li-ion would need loading the whole baseline set)
"""
import argparse
import numpy as np

from data_provider.data_loader import CHEMISTRIES, pooled_split_files
from data_provider.data_loader_pooled import Dataset_pooled
from data_provider.data_loader_full import Dataset_full_timescale, load_cell_meta, default_cache_dir, cell_cache_paths, _CellReader

p = argparse.ArgumentParser()
p.add_argument('--root_path', default='./dataset')
p.add_argument('--cache_dir', default=None)
p.add_argument('--charge_discharge_length', type=int, default=300)
p.add_argument('--cells_per_chem', type=int, default=3)
p.add_argument('--split_seed', type=int, default=2021)
a = p.parse_args()
a.seq_len, a.early_cycle_threshold, a.weighted_loss, a.dataset = 1, 100, False, 'POOLED'
cache_dir = a.cache_dir or default_cache_dir(a.root_path)
small = ['CALB', 'Zn-ion', 'Na-ion']

# baseline object over the small chemistries (fast) -- also gives us read_cell_df for any cell
base = Dataset_pooled(a, flag='train', chemistries=small, split_seed=a.split_seed)

# ---- 1. curves ----
print('\n1. cached curves vs Dataset_original.read_cell_df (first 100 cycles)')
worst = 0.0
charge_first = _CellReader(a.root_path, 1).ZN_coin_charge_first_file_names
for c in CHEMISTRIES:
    files = [f for f in pooled_split_files(c, 'train', a.split_seed) if (load_cell_meta(cache_dir, f) or {}).get('status') == 'ok']
    picks = files[:a.cells_per_chem] + files[-1:]
    if c == 'Zn-ion':
        picks += [f for f in files if f in charge_first][:1]
    for f in dict.fromkeys(picks):
        meta = load_cell_meta(cache_dir, f)
        cached = np.load(cell_cache_paths(cache_dir, f)[0], mmap_mode='r')
        ref = base.read_cell_df(f)[1]                          # [100, 3, len]
        n = min(100, meta['n_cycles'])
        diff = float(np.abs(np.asarray(cached[:n], dtype=np.float64) - ref[:n]).max())
        worst = max(worst, diff)
        print(f'  {c:6s} {f:45s} cycles cached {meta["n_cycles"]:5d}  max|diff| over {n} cycles = {diff:.2e}')
assert worst < 1e-5, f'cache differs from baseline curves (max diff {worst})'
print(f'  OK (worst {worst:.2e}; float32 storage rounding only)')

# ---- 2. counts + scaler ----
print('\n2. max_cycles=100, no cap: sample counts and label scaler vs Dataset_pooled train (CALB, Zn-ion, Na-ion)')
full = Dataset_full_timescale(a, chemistries=small, split_seed=a.split_seed, cache_dir=cache_dir, max_cycles=100, verbose=False)
bc, fc = base.chemistry_counts(), full.chemistry_counts()
for c in small:
    print(f'  {c:6s} baseline samples {bc[c]:6d} | full-timescale {fc[c]:6d}')
    assert bc[c] == fc[c], f'sample count mismatch for {c}'
assert np.allclose(base.label_scaler.mean_, full.label_scaler.mean_) and np.allclose(base.label_scaler.scale_, full.label_scaler.scale_)
print(f'  label scaler identical (mean {full.label_scaler.mean_[0]:.3f}, scale {full.label_scaler.scale_[0]:.3f})')
print('\nALL OK')
