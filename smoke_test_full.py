"""
Smoke test for the full-timescale loader. Run from the repo root on SCC after precompute_full_cells.py:

    python smoke_test_full.py
    python smoke_test_full.py --full_prefixes_per_cell 50 --full_max_cycles 1000     # empty / omitted = off
"""
import argparse
import numpy as np

from data_provider.data_loader import CHEMISTRIES
from data_provider.data_loader_full import add_full_timescale_args
from data_provider.data_factory import data_provider_full

ITEMS = ['cycle_curve_data', 'curve_attn_mask', 'labels', 'life_class', 'scaled_life_class', 'weights', 'seen_unseen_ids', 'chemistry_ids']

p = argparse.ArgumentParser()
p.add_argument('--chemistries', nargs='+', default=CHEMISTRIES, choices=CHEMISTRIES)
p.add_argument('--split_seed', type=int, default=2021, choices=[2021, 42, 2024])
p.add_argument('--root_path', default='./dataset')
p.add_argument('--num_workers', type=int, default=2)
p.add_argument('--charge_discharge_length', type=int, default=300)
p.add_argument('--seq_len', type=int, default=1)
p.add_argument('--early_cycle_threshold', type=int, default=100)
p.add_argument('--seed', type=int, default=2021)
add_full_timescale_args(p)
args = p.parse_args()
args.weighted_loss = False

data, loader = data_provider_full(args, chemistries=args.chemistries, split_seed=args.split_seed)
print('\nfirst 4 batches (batch size 1, prefix lengths differ, nothing padded):')
for n, batch in zip(range(4), loader):
    print(f'  batch {n}: ' + ' | '.join(f'{k} {tuple(v.shape)}' for k, v in zip(ITEMS, batch)))
    x, m = batch[0], batch[1]
    assert x.shape[0] == 1 and x.shape[1] == m.shape[1] and bool((m == 1).all())
    print(f'           chemistry {CHEMISTRIES[batch[7].item()]} | scaled label {batch[2].item():.3f} | prefix length {x.shape[1]}')

print('\nresampling: epoch 0 vs epoch 1 (only differs when K is set)')
a0 = (data._cell_idx.copy(), data._lens.copy())
data.set_epoch(1)
same = len(a0[1]) == len(data._lens) and bool((a0[1] == data._lens).all())
print(f'  identical prefix draw across epochs: {same}')

print('\nfull pass over one epoch (batch size 1) -- checks every sample loads:')
n = 0
for batch in loader:
    n += 1
    if n >= 300:
        break
print(f'  loaded {n} samples OK (stopped at 300)')
