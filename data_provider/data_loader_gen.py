"""
Data for the hierarchical generative model (HierGen, 10/18): models/HierGen.py, train_hiergen.py.

Per cell, every cycle on the usual 300-point grid (150 charge + 150 discharge points, evenly spaced by record index,
rests dropped -- the same segmentation and resampling as Dataset_original.get_charge_discharge_curves), but:
  * channels kept physical: V in volts (NOT divided by the cycle's max), I in C-rate, Q / nominal, t in seconds since
    the start of that half (for plots / later use; not a model input)
  * ALL usable cycles of the cell are stored (not just 1..100, not just up to eol-1), so training sees the knee and the
    crossing of 80%. Val / test cells are cached too: their cycles 1..100 condition the rollout, the rest is only used
    to plot the true trajectory.
  * per cycle, the label-consistent state of health from the RAW records (capacity_check.cycle_capacity: max
    discharge capacity / nominal / SOC span), i.e. the exact quantity the cycle-life label thresholds at 0.8.

Cache (built once by precompute_gen_cells.py) in <root_path>/gen_cycle_cache:
    <cell>.npy        [n_cycles, 4, 300] float32   channels V (V), I (C), Q (/nominal), t (s)
    <cell>.meta.json  eol (json label), nominal, soc_span, discharge_first, soh [n_cycles] (raw), n_cycles,
                      truncated_at_cycle / truncate_reason (cache stops at the first unusable cycle), split
Cells with no pkl / no label / eol <= 100 are recorded (status) and skipped by the dataset, same as the baseline.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.data_split_recorder import split_recorder

CACHE_VERSION = 1
GEN_CHANNELS = ('V', 'I', 'Q', 't')
DISCHARGE_FIRST_PREFIXES = ('RWTH', 'CALB_0', 'CALB_25', 'CALB_45')


def default_cache_dir(root_path):
    return os.path.join(root_path, 'gen_cycle_cache')


def cache_paths(cache_dir, file_name):
    stem = os.path.join(cache_dir, file_name.replace('.pkl', ''))
    return stem + '.npy', stem + '.meta.json'


def split_files(split):
    return list(getattr(split_recorder, f'MIX_large_{split}_files'))


def _prefix(file_name):
    p = file_name.split('_')[0]
    return '_'.join(file_name.split('_')[:2]) if p == 'CALB' else p


def is_discharge_first(file_name, zn_charge_first):
    p = _prefix(file_name)
    return p in DISCHARGE_FIRST_PREFIXES or (p == 'ZN-coin' and file_name not in zn_charge_first)


def resample_half(n_half, *arrays):
    """Same as Dataset_original.resample_charge_discharge_curves: n_half points on linspace(1, len+1) by record index."""
    n = len(arrays[0])
    raw = np.arange(1, n + 1)
    grid = np.linspace(1, n + 1, num=n_half, endpoint=True)
    return [np.interp(grid, raw, a) for a in arrays]


def cycle_grid(cycle_df, nominal, discharge_first, n_points=300):
    """One cycle -> [4, n_points] (V volts, I C-rate, Q / nominal, t s since half start). Mirrors the segmentation of
    Dataset_original.get_charge_discharge_curves line by line (only the V / max(V) normalisation is left out)."""
    v = cycle_df['voltage_in_V'].values
    i = cycle_df['current_in_A'].values
    i_c = i / nominal
    qc = cycle_df['charge_capacity_in_Ah'].values
    qd = cycle_df['discharge_capacity_in_Ah'].values
    t = cycle_df['time_in_s'].values
    charge_end = np.nonzero(i_c >= 0.01)[0][-1]
    discharge_end = np.nonzero(i_c <= -0.01)[0][-1]
    if discharge_first:
        dv, dq, di, dt = v[:discharge_end], qd[:discharge_end], i[:discharge_end], t[:discharge_end]
        cv, cq, ci, ct = v[discharge_end:], qc[discharge_end:], i[discharge_end:], t[discharge_end:]
        keep = np.abs(ci / nominal) > 0.01
        cv, cq, ci, ct = cv[keep], cq[keep], ci[keep], ct[keep]
    else:
        dv, dq, di, dt = v[charge_end:], qd[charge_end:], i[charge_end:], t[charge_end:]
        keep = np.abs(di / nominal) > 0.01
        dv, dq, di, dt = dv[keep], dq[keep], di[keep], dt[keep]
        cv, cq, ci, ct = v[:charge_end], qc[:charge_end], i[:charge_end], t[:charge_end]
    h = n_points // 2
    dv, di, dq, dt = resample_half(h, dv, di, dq, dt)
    cv, ci, cq, ct = resample_half(h, cv, ci, cq, ct)
    dt, ct = dt - dt[0], ct - ct[0]
    if discharge_first:   # order on the grid = order in the cycle (same as the baseline curves)
        V, I, Q, T = [np.concatenate(x) for x in ((dv, cv), (di, ci), (dq, cq), (dt, ct))]
    else:
        V, I, Q, T = [np.concatenate(x) for x in ((cv, dv), (ci, di), (cq, dq), (ct, dt))]
    return np.stack([V, I / nominal, Q / nominal, T]).astype(np.float32)


def build_gen_cache(reader, file_name, cache_dir, split, overwrite=False):
    """reader: data_loader_full._CellReader. Writes the cell's .npy + .meta.json (meta last = complete)."""
    from capacity_check import cycle_capacity
    npy_path, meta_path = cache_paths(cache_dir, file_name)
    if not overwrite and os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    os.makedirs(cache_dir, exist_ok=True)
    meta = dict(version=CACHE_VERSION, file_name=file_name, split=split, channels=list(GEN_CHANNELS),
                n_points=reader.charge_discharge_len)
    data, eol = reader.read_cell_data_according_to_prefix(file_name)
    grids = []
    if data is None:
        meta['status'] = 'no_pkl'
    elif eol is None:
        meta['status'] = 'no_label'
    else:
        _, _, soh_raw, nominal, span = cycle_capacity(data, file_name)
        dfirst = is_discharge_first(file_name, reader.ZN_coin_charge_first_file_names)
        meta.update(eol=float(eol), nominal=float(nominal), soc_span=float(span), discharge_first=bool(dfirst),
                    valid_cycle_number=len(data['cycle_data']), truncated_at_cycle=None, truncate_reason=None)
        for idx, sub in enumerate(data['cycle_data']):
            df = pd.DataFrame({k: sub[k] for k in reader.need_keys})
            df.loc[df['charge_capacity_in_Ah'] < 0] = np.nan     # same cleaning as Dataset_original.read_cell_df
            df.loc[df['discharge_capacity_in_Ah'] < 0] = np.nan
            df = df.bfill()
            reason = None
            try:
                g = cycle_grid(df, nominal, dfirst, reader.charge_discharge_len) if len(df) else None
                if g is None:
                    reason = 'no usable rows'
                elif not np.isfinite(g).all():
                    reason = 'non-finite values'
                elif not np.isfinite(soh_raw[idx]):
                    reason = 'non-finite raw capacity'
            except Exception as e:
                g, reason = None, f'{type(e).__name__}: {e}'
            if reason is not None:
                meta['truncated_at_cycle'], meta['truncate_reason'] = idx + 1, reason
                break
            grids.append(g)
        meta['n_cycles'] = len(grids)
        meta['soh'] = [float(x) for x in soh_raw[:len(grids)]]
        if not grids:
            meta['status'] = 'no_cycles'
        elif eol <= 100:
            meta['status'] = 'eol<=100'
        else:
            meta['status'] = 'ok'
    if grids:
        tmp = npy_path + '.tmp'
        with open(tmp, 'wb') as f:
            np.save(f, np.stack(grids).astype(np.float32))
        os.replace(tmp, npy_path)
    tmp = meta_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f)
    os.replace(tmp, meta_path)
    return meta


def load_meta(cache_dir, file_name):
    _, p = cache_paths(cache_dir, file_name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


class GenCells(Dataset):
    """One item = one cell's whole cached history: curves [n, 4, 300], soh [n], plus meta. split in train/val/test."""

    def __init__(self, root_path, split, cache_dir=None, max_cycles=None, min_cycles=101, verbose=True,
                 seen_unseen_json='cal_for_test.json'):
        self.cache_dir = cache_dir or default_cache_dir(root_path)
        self.split = split
        self.max_cycles = max_cycles
        su_path = os.path.join(root_path, 'seen_unseen_labels', seen_unseen_json)
        su = json.load(open(su_path)) if os.path.exists(su_path) else {}
        self.cells, missing, dropped = [], [], {}
        for f in split_files(split):
            m = load_meta(self.cache_dir, f)
            if m is None:
                missing.append(f)
                continue
            if m['status'] != 'ok':
                dropped[m['status']] = dropped.get(m['status'], 0) + 1
                continue
            if split == 'train' and m['n_cycles'] < min_cycles:
                dropped['short cache'] = dropped.get('short cache', 0) + 1
                continue
            n = m['n_cycles'] if max_cycles is None else min(m['n_cycles'], max_cycles)
            self.cells.append(dict(file=f, dataset=f.split('_')[0], n=n, eol=m['eol'], nominal=m['nominal'],
                                   soc_span=m['soc_span'], discharge_first=m['discharge_first'],
                                   soh=np.asarray(m['soh'][:n], dtype=np.float32),
                                   unseen=su.get(f, 'seen') == 'unseen'))
        if missing:
            raise FileNotFoundError(f'{len(missing)} {split} cells have no cache in {self.cache_dir} (e.g. {missing[:3]}). '
                                    f'Build it: python precompute_gen_cells.py --root_path {root_path}')
        self._mm = {}
        if verbose:
            ns = np.array([c['n'] for c in self.cells])
            print(f'[gen {split}] {len(self.cells)} cells | dropped {dropped or "none"} | cycles per cell p50 '
                  f'{np.median(ns):.0f} max {ns.max()} | total {ns.sum()}')

    def __len__(self):
        return len(self.cells)

    def __getstate__(self):
        s = self.__dict__.copy()
        s['_mm'] = {}
        return s

    def curves(self, idx):
        if idx not in self._mm:
            self._mm[idx] = np.load(cache_paths(self.cache_dir, self.cells[idx]['file'])[0], mmap_mode='r')
        return self._mm[idx][:self.cells[idx]['n']]

    def __getitem__(self, idx):
        c = self.cells[idx]
        return dict(curves=torch.from_numpy(np.array(self.curves(idx), dtype=np.float32)), soh=torch.from_numpy(c['soh']),
                    idx=idx, discharge_first=c['discharge_first'])

    def channel_stats(self, max_cells=None):
        """mean / std of V, I, Q over every grid point of (up to max_cells) cells: model input normalisation."""
        s = np.zeros(3)
        s2 = np.zeros(3)
        n = 0
        for i in range(len(self.cells) if max_cells is None else min(max_cells, len(self.cells))):
            a = np.asarray(self.curves(i)[:, :3], dtype=np.float64)       # [n, 3, P]
            s += a.sum(axis=(0, 2))
            s2 += (a ** 2).sum(axis=(0, 2))
            n += a.shape[0] * a.shape[2]
        mean = s / n
        return mean.astype(np.float32), np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12)).astype(np.float32)


def collate_cells(items):
    """Pad cycles to the longest cell: curves [B, N, 4, P], soh [B, N], mask [B, N] (True = real cycle), idx [B],
    discharge_first [B] bool."""
    B = len(items)
    N = max(it['curves'].shape[0] for it in items)
    C, P = items[0]['curves'].shape[1:]
    curves = torch.zeros(B, N, C, P)
    soh = torch.zeros(B, N)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, it in enumerate(items):
        n = it['curves'].shape[0]
        curves[b, :n] = it['curves']
        soh[b, :n] = it['soh']
        mask[b, :n] = True
    return dict(curves=curves, soh=soh, mask=mask, idx=torch.tensor([it['idx'] for it in items]),
                discharge_first=torch.tensor([bool(it['discharge_first']) for it in items]))
