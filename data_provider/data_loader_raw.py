"""
Raw-record pooled loader (10/17), for LongMamba --long_raw 1.

No resampling. Every record the tester logged in cycles 1..early_cycle_threshold goes in, in order, rests included,
with the time step to the previous record as tau. Nothing is interpolated and nothing is dropped.

Per cell (built once, shared by all of the cell's prefix samples):
    feats [N, 4] float32   per record: V (/ that cycle's max V, or volts with --raw_v_norm none), I / nominal (C-rate),
                           charge capacity / nominal, discharge capacity / nominal   (the tester's own counters)
    tau   [N]    float32   (t_n - t_{n-1}) / --dt_c, the real step to the previous record, ACROSS cycle boundaries too
                           (the first record of cycle k gets the real gap from the last record of cycle k-1).
                           First record of the cell = 1. Negative / non-finite steps = 1. Zero steps stay 0. Upper clip
                           --dt_clip (0 = none), meant for non-logging pauses, not for the logging interval itself.
                           Without --dt_c (TIME_DT const / off): all ones.
    cid   [N]    int16     0-based cycle index of each record
    starts [L+1] int64     record offset where cycle k+1 starts; a k-cycle prefix is records [:starts[k]].
                           A cycle the cell does not have contributes no records (the model still puts its token there).

Samples are exactly Dataset_pooled's (same cells, same prefixes 1..100, same labels / weights / seen-unseen ids); only
the inputs differ. The 300-point curves are still built by the base class (labels, NaN checks) but are not returned.

Rows: read_cell_df already set rows with negative capacity to NaN and back-filled them (same as the baseline). Any NaN
left at the end of a cycle is forward-filled, then 0.

Batches (collate): x [B, T, 4] (zero padded), curve_attn_mask [B, L], labels, life_class, scaled_life_class, weights,
seen_unseen_ids, chemistry_ids, tau [B, T] (pad 1), cid [B, T] long (pad -1)  -> 10 items; tau is item 9, cid item 10.

Lengths vary ~100x across datasets (SNL ~77 records/cycle, XJTU ~6900), so batches are built by
StepBudgetBatchSampler: up to --batch_size samples AND (samples x longest length) <= --raw_max_batch_steps
(0 = no step budget). Samples are length-sorted inside random pools so padding stays small; batch order is shuffled.
"""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from data_provider.data_loader_pooled import Dataset_pooled

RAW_CHANNELS = 4


def raw_cell_arrays(df, n_cycles, nominal_capacity, v_norm='cycle_max'):
    """df: read_cell_df's per-cell dataframe (cycle_number 1-based). -> feats [N,4] f32, times [N] f64, cid [N] i16, starts [L+1]."""
    if v_norm not in ('cycle_max', 'none'):
        raise ValueError(f'raw_v_norm must be cycle_max or none, got {v_norm}')
    feats, times, cids, counts = [], [], [], []
    cyc_col = df['cycle_number'].values
    for k in range(1, n_cycles + 1):
        sel = cyc_col == k
        n = int(sel.sum())
        counts.append(n)
        if n == 0:
            continue
        sub = df.loc[sel, ['voltage_in_V', 'current_in_A', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']]
        sub = sub.ffill().fillna(0.0)
        a = sub.to_numpy(dtype=np.float64)
        v = a[:, 0]
        if v_norm == 'cycle_max':
            vmax = np.max(v) if n else 0.0
            v = v / vmax if vmax > 0 else v
        f = np.stack([v, a[:, 1] / nominal_capacity, a[:, 2] / nominal_capacity, a[:, 3] / nominal_capacity], axis=1)
        feats.append(f.astype(np.float32))
        times.append(a[:, 4])
        cids.append(np.full(n, k - 1, dtype=np.int16))
    starts = np.zeros(n_cycles + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    if feats:
        return np.concatenate(feats), np.concatenate(times), np.concatenate(cids), starts
    return np.zeros((0, RAW_CHANNELS), np.float32), np.zeros(0), np.zeros(0, np.int16), starts


def raw_times_to_tau(times, c, clip=0.0):
    """times [N] seconds as logged -> tau [N] float32 (see module docstring). c <= 0 -> all ones."""
    times = np.asarray(times, dtype=np.float64)
    tau = np.ones(len(times), dtype=np.float64)
    if c and c > 0 and len(times) > 1:
        d = np.diff(times) / c
        bad = ~np.isfinite(d) | (d < 0)
        d[bad] = 1.0
        tau[1:] = d
        if clip and clip > 0:
            np.minimum(tau, clip, out=tau)
    return tau.astype(np.float32)


class Dataset_pooled_raw(Dataset_pooled):
    def __init__(self, args, flag='train', chemistries=None, split_seed=2021, **kwargs):
        self._dt_c = float(getattr(args, 'dt_c', 0) or 0)
        self._dt_clip = float(getattr(args, 'dt_clip', 0) or 0)
        self._v_norm = getattr(args, 'raw_v_norm', 'cycle_max')
        self._cells = []            # one dict per cell: feats, tau, cid, starts, name
        self._sample_cell = []      # cell index per sample
        self._pending = None
        super().__init__(args, flag=flag, chemistries=chemistries, split_seed=split_seed, **kwargs)
        assert len(self._sample_cell) == len(self.total_charge_discharge_curves), \
            f'raw / sample misalignment: {len(self._sample_cell)} vs {len(self.total_charge_discharge_curves)}'
        self.n_cycles = np.array([int(round(float(np.sum(m)))) for m in self.total_curve_attn_masks], dtype=np.int64)
        self.lengths = np.array([int(self._cells[ci]['starts'][n]) for ci, n in zip(self._sample_cell, self.n_cycles)],
                                dtype=np.int64)
        self._print_stats(flag)

    def _print_stats(self, flag):
        if not self._cells:
            return
        n_rec = np.array([c['starts'][-1] for c in self._cells])
        allt = np.concatenate([c['tau'] for c in self._cells])
        q = np.quantile(allt, [0.01, 0.5, 0.99, 0.999])
        frac_clip = float((allt >= self._dt_clip).mean()) if self._dt_clip > 0 else 0.0
        print(f'[raw {flag}] {len(self._cells)} cells | records per cell (<= {self.early_cycle_threshold} cycles) '
              f'p10 {np.percentile(n_rec, 10):.0f} p50 {np.median(n_rec):.0f} p90 {np.percentile(n_rec, 90):.0f} '
              f'max {n_rec.max()} | sample length p50 {np.median(self.lengths):.0f} max {self.lengths.max()}')
        if self._dt_c > 0:
            print(f'[raw {flag}] c={self._dt_c:g}s clip={self._dt_clip or "none"} | tau p1 {q[0]:.3g} p50 {q[1]:.3g} '
                  f'p99 {q[2]:.3g} p99.9 {q[3]:.3g} max {allt.max():.3g} | at clip {100 * frac_clip:.3f}% | '
                  f'zero {100 * (allt == 0).mean():.2f}%')
            by_ds = {}
            for c in self._cells:
                by_ds.setdefault(c['name'].split('_')[0], []).append(c['tau'])
            print(f'[raw {flag}] median tau by dataset: ' + ', '.join(
                f'{k} {np.median(np.concatenate(v)):.3g}' for k, v in sorted(by_ds.items())))

    # Dataset_original.read_cell_df calls this once per cell; keep the raw records of that cell.
    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity, return_times=False):
        if return_times:
            return super().get_charge_discharge_curves(file_name, df, early_cycle_threshold, nominal_capacity, return_times=True)
        feats, times, cid, starts = raw_cell_arrays(df, early_cycle_threshold, nominal_capacity, self._v_norm)
        if np.any(~np.isfinite(feats)):
            raise ValueError(f'{file_name}: non-finite raw features')
        self._pending = dict(feats=feats, tau=raw_times_to_tau(times, self._dt_c, self._dt_clip), cid=cid,
                             starts=starts, name=file_name)
        return super().get_charge_discharge_curves(file_name, df, early_cycle_threshold, nominal_capacity)

    def read_samples_from_one_cell(self, file_name):
        self._pending = None
        out = super().read_samples_from_one_cell(file_name)
        curves, eol = out[0], out[3]
        if eol is not None and curves is not None:     # read_data keeps exactly these samples, in this order
            if self._pending is None:
                raise RuntimeError(f'{file_name}: no raw records captured')
            self._cells.append(self._pending)
            self._sample_cell += [len(self._cells) - 1] * len(curves)
        return out

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        cell = self._cells[self._sample_cell[index]]
        end = int(self.lengths[index])
        sample['raw_x'] = torch.from_numpy(cell['feats'][:end])
        sample['raw_tau'] = torch.from_numpy(cell['tau'][:end])
        sample['raw_cid'] = torch.from_numpy(cell['cid'][:end].astype(np.int64))
        del sample['cycle_curve_data']
        return sample


def my_collate_fn_pooled_raw(samples):
    B = len(samples)
    T = max(1, max(len(s['raw_tau']) for s in samples))
    x = torch.zeros(B, T, samples[0]['raw_x'].shape[1], dtype=samples[0]['raw_x'].dtype)
    tau = torch.ones(B, T, dtype=samples[0]['raw_tau'].dtype)
    cid = torch.full((B, T), -1, dtype=torch.long)
    for b, s in enumerate(samples):
        n = len(s['raw_tau'])
        x[b, :n] = s['raw_x']
        tau[b, :n] = s['raw_tau']
        cid[b, :n] = s['raw_cid']
    curve_attn_mask = torch.vstack([torch.as_tensor(s['curve_attn_mask']).float().unsqueeze(0) for s in samples])
    labels = torch.Tensor([s['labels'] for s in samples])
    life_class = torch.Tensor([s['life_class'] for s in samples])
    scaled_life_class = torch.Tensor([s['scaled_life_class'] for s in samples])
    weights = torch.Tensor([s['weight'] for s in samples])
    seen_unseen_ids = torch.Tensor([s['seen_unseen_id'] for s in samples])
    chemistry_ids = torch.LongTensor([s['chemistry_id'] for s in samples])
    return x, curve_attn_mask, labels, life_class, scaled_life_class, weights, seen_unseen_ids, chemistry_ids, tau, cid


def make_budget_batches(lengths, batch_size, max_steps, shuffle, rng=None, pool_batches=50):
    """Index batches: <= batch_size samples and len(batch) * max(length) <= max_steps (max_steps <= 0: no budget).
    A sample longer than max_steps gets a batch of its own. shuffle: random pools of pool_batches * batch_size samples,
    length-sorted inside each pool, batch order shuffled. Otherwise: one global length sort."""
    lengths = np.asarray(lengths)
    n = len(lengths)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    pool = max(1, pool_batches * batch_size) if shuffle else max(1, n)
    batches = []
    for p in range(0, n, pool):
        chunk = idx[p:p + pool]
        chunk = chunk[np.argsort(-lengths[chunk], kind='stable')]
        cur, cur_max = [], 0
        for i in chunk:
            m = max(cur_max, int(lengths[i]))
            if cur and (len(cur) >= batch_size or (max_steps > 0 and (len(cur) + 1) * m > max_steps)):
                batches.append(cur)
                cur, m = [], int(lengths[i])
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
    if shuffle:
        order = rng.permutation(len(batches))
        batches = [batches[o] for o in order]
    return batches


class StepBudgetBatchSampler(Sampler):
    # batch_size is None on purpose: batches have variable size, and accelerate's BatchSamplerShard (multi-process
    # only) would otherwise hold back every batch shorter than batch_size. Runs here are single-process.
    def __init__(self, lengths, batch_size, max_steps, shuffle):
        self.lengths = np.asarray(lengths)
        self.max_batch = int(batch_size)
        self.batch_size = None
        self.max_steps = int(max_steps or 0)
        self.shuffle = shuffle
        self.drop_last = False
        self._n = len(make_budget_batches(self.lengths, self.max_batch, self.max_steps, False))

    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if self.shuffle else None
        batches = make_budget_batches(self.lengths, self.max_batch, self.max_steps, self.shuffle, rng)
        self._n = len(batches)
        return iter(batches)

    def __len__(self):
        return self._n


def data_provider_pooled_raw(args, flag, label_scaler=None, life_class_scaler=None, chemistries=None, split_seed=2021):
    """Same as data_factory.data_provider_pooled, with Dataset_pooled_raw, the 10-item collate and step-budget batches."""
    train = (flag == 'train')
    data_set = Dataset_pooled_raw(args, flag=flag, chemistries=chemistries, split_seed=split_seed,
                                  label_scaler=label_scaler, life_class_scaler=life_class_scaler)
    # +1 token per cycle and the CLS: what the model actually runs over
    steps = data_set.lengths + data_set.n_cycles + 1
    sampler = StepBudgetBatchSampler(steps, args.batch_size, getattr(args, 'raw_max_batch_steps', 0), shuffle=train)
    data_loader = DataLoader(data_set, batch_sampler=sampler, num_workers=args.num_workers,
                             collate_fn=my_collate_fn_pooled_raw)
    return data_set, data_loader
