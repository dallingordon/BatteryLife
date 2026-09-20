"""
Smoke test for the pooled multi-chemistry dataloader (data_provider/data_loader_pooled.py).
Loads real data, prints dataset sizes and batch shapes. Run from the repo root on SCC:

    python smoke_test_pooled.py                          # all four chemistries (Li-ion is slow to load)
    python smoke_test_pooled.py --chemistries CALB Zn-ion Na-ion    # fast first pass, skips Li-ion
"""
import argparse
import time
import numpy as np

from data_provider.data_loader import CHEMISTRIES
from data_provider.data_split_recorder import split_recorder
from data_provider.data_factory import data_provider_pooled

SPLIT_PREFIX = {'Li-ion': 'MIX_large', 'CALB': 'CALB', 'Zn-ion': 'ZNcoin', 'Na-ion': 'NAion_2021'}
BATCH_ITEMS = ['cycle_curve_data', 'curve_attn_mask', 'labels', 'life_class', 'scaled_life_class',
               'weights', 'seen_unseen_ids', 'chemistry_ids']

p = argparse.ArgumentParser()
p.add_argument('--chemistries', nargs='+', default=CHEMISTRIES, choices=CHEMISTRIES)
p.add_argument('--root_path', default='./dataset')
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--num_workers', type=int, default=4)
p.add_argument('--charge_discharge_length', type=int, default=300)   # matches the tuned-hyperparam runs
p.add_argument('--seq_len', type=int, default=1)
p.add_argument('--early_cycle_threshold', type=int, default=100)
args = p.parse_args()
args.weighted_loss = False
args.dataset = 'POOLED'   # Dataset_pooled sets this itself; here for completeness
chems = args.chemistries
print(f'chemistries: {chems}\n')


def show_batch(name, loader):
    batch = next(iter(loader))
    print(f'  first {name} batch:')
    for k, v in zip(BATCH_ITEMS, batch):
        print(f'    {k:18s} shape={tuple(v.shape)!s:22s} dtype={str(v.dtype):14s}', end='')
        if k == 'chemistry_ids':
            ids, cnt = np.unique(v.numpy(), return_counts=True)
            print('  ' + ', '.join(f'{CHEMISTRIES[i]}:{c}' for i, c in zip(ids, cnt)))
        elif k in ('labels', 'life_class'):
            print(f'  min={v.min().item():.3f} max={v.max().item():.3f}')
        else:
            print()
    return batch


def timed(label, fn):
    t = time.time(); out = fn(); print(f'  [{label}: {time.time() - t:.1f}s]'); return out


# ---- pooled train ----
print('== pooled TRAIN ==')
train_data, train_loader = timed('load train', lambda: data_provider_pooled(args, 'train', chemistries=chems))
counts = train_data.chemistry_counts()
n_cells = {c: len(getattr(split_recorder, f'{SPLIT_PREFIX[c]}_train_files')) for c in chems}
print(f'  files listed: {len(train_data.files)} (expected {sum(n_cells.values())}) | samples: {len(train_data)} | batches/epoch: {len(train_loader)}')
for c in CHEMISTRIES:
    if c in chems:
        cs = counts[c]
        print(f'    {c:7s} cells listed={n_cells[c]:4d}  samples={cs:6d}  share of samples={cs / len(train_data):.1%}')
    else:
        assert counts[c] == 0
show_batch('train', train_loader)
scaler = train_data.return_label_scaler()
print(f'  label_scaler (pooled, fit on per-cell life): mean={scaler.mean_[0]:.1f} std={np.sqrt(scaler.var_[0]):.1f}')
life_scaler = train_data.return_life_class_scaler()

# ---- per-chemistry val / test ----
for flag in ('val', 'test'):
    print(f'\n== per-chemistry {flag.upper()} ==')
    for c in chems:
        d, dl = timed(f'load {c} {flag}', lambda: data_provider_pooled(args, flag, scaler, life_scaler, chemistries=[c]))
        expected = getattr(split_recorder, f'{SPLIT_PREFIX[c]}_{flag}_files')
        assert d.files == expected, f'{c} {flag}: file list differs from the per-chemistry baseline split'
        assert set(d.total_chemistry_ids.tolist()) == {CHEMISTRIES.index(c)}
        print(f'  {c}: files={len(d.files)} samples={len(d)} batches={len(dl)}')
    # one batch from the last chemistry as a shape check
    show_batch(f'{chems[-1]} {flag}', dl)

print('\nSMOKE TEST OK')
