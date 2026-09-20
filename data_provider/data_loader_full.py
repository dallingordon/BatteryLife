"""
Full-timescale ("unified full-timescale") training dataset, for Mamba-style models.

Same idea as the pooled loader (data_loader_pooled.py): one model trained on Li-ion (MIX_large) + CALB + Zn-ion
+ Na-ion, every sample carries a chemistry id, same training cells, same exclusions (no life label, missing pkl,
eol <= early_cycle_threshold). What differs:

  * prefixes are not limited to 1..100. A cell with N usable cycles yields prefixes 1..min(N, eol-1)
    (optionally capped by `max_cycles`).
  * per-cell cap: each epoch, every cell contributes at most `prefixes_per_cell` (K) prefixes, drawn at random
    (different every epoch, chemistry-agnostic). Cells with fewer than K eligible prefixes contribute all of them.
  * NO padding. A sample is one prefix of its own length; use batch size one (my_collate_fn_full enforces it).

Config (both are "empty = off"):
  max_cycles          None -> use every stored cycle;  int -> prefix length never exceeds it
  prefixes_per_cell   None -> drop nothing;            int -> K prefixes per cell per epoch

Val / test are NOT built here. Keep using Dataset_pooled(flag='val'/'test', chemistries=[c]) so evaluation is
the published cells and prefixes 1..100, unchanged.

Cycle curves come from a disk cache built once by precompute_full_cells.py (separate step, no concurrent-job
races). The cache stores every cycle up to eol-1 for each cell; max_cycles is applied at load time, so changing
it never needs a rebuild. Curves are computed with Dataset_original.get_charge_discharge_curves, i.e. the exact
same per-cycle code as the baseline (check_full_equivalence.py verifies this on real data).

Nothing here changes Dataset_original / Dataset_pooled.
"""
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_provider.data_loader import Dataset_original, CHEMISTRIES, datasetName2ids, pooled_split_files
from data_provider.data_split_recorder import split_recorder

CACHE_VERSION = 1
_LIFE_CLASSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'life_classes.json')


def int_or_none(s):
    """argparse type: '' / 'none' / 'None' -> None, otherwise int. Lets an unset shell variable mean "off"."""
    if s is None:
        return None
    if isinstance(s, str) and s.strip().lower() in ('', 'none', 'null'):
        return None
    v = int(s)
    if v < 1:
        raise ValueError(f'expected a positive integer or empty, got {s}')
    return v


def add_full_timescale_args(parser):
    parser.add_argument('--full_max_cycles', type=int_or_none, default=None,
                        help='full-timescale loader: longest prefix (cycles) used in training. Empty = no max')
    parser.add_argument('--full_prefixes_per_cell', type=int_or_none, default=None,
                        help='full-timescale loader: at most K random prefixes per cell per epoch. Empty = drop nothing')
    parser.add_argument('--full_cache_dir', type=str, default=None,
                        help='full-timescale loader: cycle cache dir (default <root_path>/full_cycle_cache)')
    return parser


def default_cache_dir(root_path):
    return os.path.join(root_path, 'full_cycle_cache')


def cell_cache_paths(cache_dir, file_name):
    stem = os.path.join(cache_dir, file_name.replace('.pkl', ''))
    return stem + '.npy', stem + '.meta.json'


def load_cell_meta(cache_dir, file_name):
    _, meta_path = cell_cache_paths(cache_dir, file_name)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def file_dataset_id(file_name):
    """source-dataset id of a cell file, same rule as Dataset_original.read_data."""
    if (file_name in split_recorder.MICH_EXP_test_files or file_name in split_recorder.MICH_EXP_train_files
            or file_name in split_recorder.MICH_EXP_val_files):
        return datasetName2ids['MICH_EXP']
    return datasetName2ids[file_name.split('_')[0]]


# --------------------------------------------------------------------------------------------------------------
# cache building (used by precompute_full_cells.py)
# --------------------------------------------------------------------------------------------------------------
class _CellReader(Dataset_original):
    """Borrows Dataset_original's pkl reading and per-cycle curve extraction without loading any dataset."""

    def __init__(self, root_path, charge_discharge_length):   # deliberately does not call super().__init__
        self.root_path = root_path
        self.charge_discharge_len = charge_discharge_length
        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
        # must match Dataset_original.__init__ (check_full_equivalence.py covers a charge-first cell)
        self.ZN_coin_charge_first_file_names = ['ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_430-3_20231212185323_03_2.pkl']


def make_cell_reader(root_path, charge_discharge_length):
    return _CellReader(root_path, charge_discharge_length)


def _one_cycle_curve(reader, file_name, sub_cycle_data, cycle_number, nominal_capacity):
    """[3, charge_discharge_len] curve for one cycle, or None if the cycle has no usable rows.
    Mirrors Dataset_original.read_cell_df (per-cycle cleaning) + get_charge_discharge_curves (per-cycle body)."""
    cycle_df = pd.DataFrame()
    for key in reader.need_keys:
        cycle_df[key] = sub_cycle_data[key]
    cycle_df['cycle_number'] = cycle_number
    cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan   # same outlier handling as the baseline
    cycle_df.loc[cycle_df['discharge_capacity_in_Ah'] < 0] = np.nan
    cycle_df.bfill(inplace=True)
    cycle_df = cycle_df[cycle_df['cycle_number'] == cycle_number].copy()
    if len(cycle_df) == 0:
        return None
    cycle_df['cycle_number'] = 1
    return reader.get_charge_discharge_curves(file_name, cycle_df, 1, nominal_capacity)[0]


def build_cell_cache(reader, file_name, cache_dir, overwrite=False):
    """Write <cache_dir>/<cell>.npy ([n_cycles, 3, len] float32) and <cell>.meta.json (written last = "complete").
    Stores cycles 1..min(valid cycles, eol-1), stopping early at the first unusable cycle (recorded in the meta).
    Returns the meta dict."""
    npy_path, meta_path = cell_cache_paths(cache_dir, file_name)
    if not overwrite and os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {'version': CACHE_VERSION, 'file_name': file_name, 'charge_discharge_len': reader.charge_discharge_len}
    curves = []
    data, eol = reader.read_cell_data_according_to_prefix(file_name)
    if data is None:
        meta['status'] = 'no_pkl'
    elif eol is None:
        meta['status'] = 'no_label'
    else:
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
        cycle_data = data['cycle_data']
        n_want = min(len(cycle_data), math.ceil(eol) - 1)   # prefixes need cycles strictly before eol
        meta.update(eol=eol, valid_cycle_number=len(cycle_data), n_wanted=n_want, nominal_capacity=float(nominal_capacity),
                    truncated_at_cycle=None, truncate_reason=None)
        for idx in range(max(n_want, 0)):
            reason = None
            try:
                curve = _one_cycle_curve(reader, file_name, cycle_data[idx], idx + 1, nominal_capacity)
                if curve is None:
                    reason = 'no usable rows'
                elif not np.isfinite(curve).all():
                    reason = 'non-finite values'
            except Exception as e:   # e.g. a cycle with no charge/discharge current
                curve, reason = None, f'{type(e).__name__}: {e}'
            if reason is not None:
                meta['truncated_at_cycle'] = idx + 1
                meta['truncate_reason'] = reason
                break
            curves.append(curve)
        meta['n_cycles'] = len(curves)
        meta['status'] = 'ok' if curves else 'no_cycles'
    if curves:
        arr = np.stack(curves).astype(np.float32)
        tmp = npy_path + '.tmp'
        with open(tmp, 'wb') as f:
            np.save(f, arr)
        os.replace(tmp, npy_path)
    tmp = meta_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f)
    os.replace(tmp, meta_path)
    return meta


# --------------------------------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------------------------------
class Dataset_full_timescale(Dataset):
    """Training-only. Call set_epoch(e) before iterating each epoch to redraw the per-cell prefixes."""

    def __init__(self, args, chemistries=None, split_seed=2021, cache_dir=None, max_cycles=None,
                 prefixes_per_cell=None, seed=0, label_scaler=None, life_class_scaler=None, verbose=True):
        """
        :param chemistries: subset of CHEMISTRIES (default all four)
        :param split_seed: 2021 / 42 / 2024, picks the CALB / Zn-ion / Na-ion split (same as Dataset_pooled)
        :param max_cycles: None = no max
        :param prefixes_per_cell: K, None = drop nothing
        :param seed: base seed of the per-epoch prefix draw
        :param label_scaler / life_class_scaler: pass in to reuse; default fits on this dataset's cells, which is
                what Dataset_pooled(flag='train') fits on (same cells) -- hand these to the val/test loaders.
        """
        chemistries = list(chemistries) if chemistries else list(CHEMISTRIES)
        for c in chemistries:
            assert c in CHEMISTRIES, f'unknown chemistry {c}, expected one of {CHEMISTRIES}'
        assert max_cycles is None or max_cycles >= 1
        assert prefixes_per_cell is None or prefixes_per_cell >= 1
        if getattr(args, 'weighted_loss', False):
            raise NotImplementedError('--weighted_loss is not supported by the full-timescale loader')
        self.args = args
        self.chemistries = chemistries
        self.split_seed = split_seed
        self.max_cycles = max_cycles
        self.prefixes_per_cell = prefixes_per_cell
        self.seed = seed
        self.verbose = verbose
        self.cache_dir = cache_dir or default_cache_dir(args.root_path)
        self.early_cycle_threshold = args.early_cycle_threshold
        self.charge_discharge_len = args.charge_discharge_length
        self._mm = {}
        life_classes = json.load(open(_LIFE_CLASSES_PATH))

        train_files = {c: list(pooled_split_files(c, 'train', split_seed)) for c in chemistries}
        held_out = set(f for c in chemistries for k in ('val', 'test') for f in pooled_split_files(c, k, split_seed))
        assert not held_out & set(f for fs in train_files.values() for f in fs), 'train cell also in val/test'

        self.cells, missing, dropped = [], [], Counter()
        for c in chemistries:
            for f in train_files[c]:
                meta = load_cell_meta(self.cache_dir, f)
                if meta is None:
                    missing.append(f)
                    continue
                if meta['status'] != 'ok':
                    dropped[meta['status']] += 1                       # no_pkl / no_label / no_cycles
                    continue
                if meta['charge_discharge_len'] != self.charge_discharge_len:
                    raise ValueError(f'cache for {f} was built with charge_discharge_length='
                                     f'{meta["charge_discharge_len"]}, run uses {self.charge_discharge_len}')
                if meta['eol'] <= self.early_cycle_threshold:
                    dropped[f'eol<={self.early_cycle_threshold}'] += 1  # same exclusion as the baseline
                    continue
                n_stored = meta['n_cycles']
                life_class = None
                for class_label, (lo, hi) in life_classes.items():
                    if lo <= meta['eol'] < hi:
                        life_class = int(class_label)
                        break
                if life_class is None:
                    raise ValueError(f'{f}: eol {meta["eol"]} is in no life class of life_classes.json')
                self.cells.append(dict(file=f, chem=CHEMISTRIES.index(c), dataset_id=file_dataset_id(f), eol=meta['eol'],
                                       n_stored=n_stored, n_eligible=n_stored if max_cycles is None else min(n_stored, max_cycles),
                                       life_class=life_class, truncated_at=meta.get('truncated_at_cycle')))
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} training cells have no cache in {self.cache_dir} (e.g. {missing[:3]}). Build it first:\n'
                f'  python precompute_full_cells.py --root_path {args.root_path} --cache_dir {self.cache_dir} '
                f'--charge_discharge_length {self.charge_discharge_len}')
        assert self.cells, 'no training cells'

        eols = np.array([c['eol'] for c in self.cells], dtype=np.float64).reshape(-1, 1)
        classes = np.array([c['life_class'] for c in self.cells], dtype=np.float64).reshape(-1, 1)
        if label_scaler is None:
            self.label_scaler = StandardScaler().fit(eols)
            self.life_class_scaler = StandardScaler().fit(classes)
        else:
            self.label_scaler, self.life_class_scaler = label_scaler, life_class_scaler
        self._scaled_eol = self.label_scaler.transform(eols)          # [n_cells, 1]
        self.dropped = dict(dropped)
        self.cell_chemistry_ids = np.array([c['chem'] for c in self.cells], dtype=np.int64)  # per cell

        if verbose:
            n_trunc = sum(c['truncated_at'] is not None for c in self.cells)
            print(f'[full-timescale train] {len(self.cells)} cells | max_cycles={max_cycles} | prefixes_per_cell={prefixes_per_cell} '
                  f'| dropped: {self.dropped or "none"} | cells with a truncated cache: {n_trunc}')
        self.set_epoch(0)

    # ---- per-epoch prefix draw ----
    def set_epoch(self, epoch):
        """Redraw which prefixes are used this epoch. Call before creating the DataLoader iterator each epoch."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        K = self.prefixes_per_cell
        cell_idx, lens = [], []
        for ci, cell in enumerate(self.cells):
            n = cell['n_eligible']
            chosen = np.arange(1, n + 1) if (K is None or n <= K) else np.sort(rng.choice(n, size=K, replace=False)) + 1
            cell_idx.append(np.full(len(chosen), ci, dtype=np.int64))
            lens.append(chosen.astype(np.int64))
        self._cell_idx = np.concatenate(cell_idx)
        self._lens = np.concatenate(lens)
        self.epoch = int(epoch)
        if self.verbose:
            print(self.epoch_summary())

    def epoch_summary(self):
        chem_of_sample = self.cell_chemistry_ids[self._cell_idx]
        n_samples, n_steps = len(self._lens), int(self._lens.sum())
        lines = [f'[full-timescale train] epoch {self.epoch}: {n_samples} samples, {n_steps} cycle-steps '
                 f'(K={self.prefixes_per_cell}, max_cycles={self.max_cycles})',
                 f'  {"chemistry":8s} {"cells":>6s} {"samples":>9s} {"share":>7s} {"cycle-steps":>12s} {"share":>7s}']
        for i, name in enumerate(CHEMISTRIES):
            m = chem_of_sample == i
            if not (self.cell_chemistry_ids == i).any():
                continue
            s, st = int(m.sum()), int(self._lens[m].sum())
            lines.append(f'  {name:8s} {int((self.cell_chemistry_ids == i).sum()):6d} {s:9d} {100 * s / n_samples:6.1f}% '
                         f'{st:12d} {100 * st / n_steps:6.1f}%')
        return '\n'.join(lines)

    def chemistry_counts(self):
        """{chemistry: samples this epoch}"""
        chem_of_sample = self.cell_chemistry_ids[self._cell_idx]
        return {c: int((chem_of_sample == i).sum()) for i, c in enumerate(CHEMISTRIES)}

    def return_label_scaler(self):
        return self.label_scaler

    def return_life_class_scaler(self):
        return self.life_class_scaler

    # ---- torch Dataset ----
    def __len__(self):
        return len(self._lens)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mm'] = {}          # memmaps are reopened lazily in each worker
        return state

    def _cell_curves(self, ci):
        if ci not in self._mm:
            npy_path, _ = cell_cache_paths(self.cache_dir, self.cells[ci]['file'])
            self._mm[ci] = np.load(npy_path, mmap_mode='r')
        return self._mm[ci]

    def __getitem__(self, index):
        ci, L = int(self._cell_idx[index]), int(self._lens[index])
        cell = self.cells[ci]
        curves = torch.from_numpy(np.array(self._cell_curves(ci)[:L], dtype=np.float32))   # [L, 3, len], copied out of the memmap
        return {
            'cycle_curve_data': curves,
            'curve_attn_mask': torch.ones(L),          # no padding: everything is real
            'labels': self._scaled_eol[ci],            # shape (1,), same as the baseline
            'life_class': cell['life_class'],
            'scaled_life_class': cell['life_class'] - 1,
            'weight': 1.0,
            'dataset_id': cell['dataset_id'],
            'seen_unseen_id': 1,                       # training set: unused
            'chemistry_id': cell['chem'],
            'prefix_len': L,
        }


def my_collate_fn_full(samples):
    """Batch size one, no padding. Same 8-item layout as my_collate_fn_pooled:
    (cycle_curve_data [1,L,3,len], curve_attn_mask [1,L], labels [1,1], life_class [1], scaled_life_class [1],
     weights [1], seen_unseen_ids [1], chemistry_ids [1])."""
    assert len(samples) == 1, 'the full-timescale loader is batch size 1 (prefixes have different lengths, no padding)'
    s = samples[0]
    return (s['cycle_curve_data'].unsqueeze(0), s['curve_attn_mask'].unsqueeze(0), torch.Tensor([s['labels']]),
            torch.Tensor([s['life_class']]), torch.Tensor([s['scaled_life_class']]), torch.Tensor([s['weight']]),
            torch.Tensor([s['seen_unseen_id']]), torch.LongTensor([s['chemistry_id']]))
