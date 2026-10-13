"""
Pooled loader + per-point normalised time step tau (10/13), for LongMamba --long_time_dt real / --long_dt_input 1.

Dataset_pooled_dt is Dataset_pooled with one addition: every sample also carries
    cycle_tau [L, P]   (P = charge_discharge_length = 300; L = early_cycle_threshold cycles, same as the curves)
returned as its OWN tensor (batch item 9, after chemistry_ids). The curves stay 3 channels, so augmentation and every
other model are untouched, and with the dt flags off nothing here is used.

Where tau comes from: Dataset_original.get_charge_discharge_curves(return_times=True) interpolates time_in_s at the
same resampled points as V / I / Q (150 charge + 150 discharge points, evenly spaced in RECORD index, rest rows
|I| < 0.01C dropped). Within each half, tau of point j = (t_j - t_{j-1}) / c, c = --dt_c (seconds, ONE global constant:
the median interior step over the Li-ion train cells from dt_stats.py), upper-clipped at --dt_clip (0 = no clip).

Edge steps get tau = 1 (v1: rests left out, the step counts as one typical step):
  - point 0 of each half (cycle start; start of the second half, whose curve-order predecessor can be later in time
    for discharge-first cells, e.g. RWTH)
  - point 1 of each half: the baseline segmentation starts the chronologically second half with the last record of
    the first half, so that step spans the dropped rest (dt_stats.py reports these as first_tau). For simplicity the
    same position is set to 1 in both halves.
  - negative / non-finite steps, and cycles the cell does not have (all-zero rows).
Zero steps (repeated timestamps) stay 0: no time passed, the SSM state is unchanged and takes no input.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader_pooled import Dataset_pooled, my_collate_fn_pooled


def times_to_tau(times, c, clip=0.0):
    """times [L, P] seconds (charge half then discharge half) -> tau [L, P] float32, see the module docstring."""
    times = np.asarray(times, dtype=np.float64)
    L, P = times.shape
    H = P // 2
    tau = np.ones((L, P), dtype=np.float64)
    for s in (0, H):
        seg = times[:, s:s + H]
        d = np.diff(seg, axis=1)[:, 1:] / c              # steps into points s+2 .. s+H-1
        bad = ~np.isfinite(d) | (d < 0)
        d[bad] = 1.0
        tau[:, s + 2:s + H] = d
    tau[~np.any(times != 0, axis=1)] = 1.0               # cycles the cell does not have
    if clip and clip > 0:
        np.minimum(tau, clip, out=tau)
    return tau.astype(np.float32)


class Dataset_pooled_dt(Dataset_pooled):
    def __init__(self, args, flag='train', chemistries=None, split_seed=2021, **kwargs):
        c = float(getattr(args, 'dt_c', 0) or 0)
        if c <= 0:
            raise ValueError('Dataset_pooled_dt needs --dt_c > 0 (seconds; median interior step from dt_stats.py)')
        self._dt_c = c
        self._dt_clip = float(getattr(args, 'dt_clip', 0) or 0)
        self._tau_list = []
        self._pending_times = None
        super().__init__(args, flag=flag, chemistries=chemistries, split_seed=split_seed, **kwargs)
        assert len(self._tau_list) == len(self.total_charge_discharge_curves), \
            f'tau / sample misalignment: {len(self._tau_list)} tau vs {len(self.total_charge_discharge_curves)} samples'
        cells = list({id(t): t for t in self._tau_list}.values())
        if cells:
            allt = np.concatenate([t.reshape(-1) for t in cells])
            q = np.quantile(allt, [0.01, 0.5, 0.99])
            frac_clip = float((allt >= self._dt_clip).mean()) if self._dt_clip > 0 else 0.0
            print(f'[dt {flag}] {len(cells)} cells | c={c:g}s clip={self._dt_clip or "none"} | tau p1 {q[0]:.3g} '
                  f'p50 {q[1]:.3g} p99 {q[2]:.3g} max {allt.max():.3g} | at clip {100 * frac_clip:.2f}% | zero {100 * (allt == 0).mean():.2f}%')

    # Dataset_original.read_cell_df calls this once per cell (without return_times); keep the times for that cell.
    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity, return_times=False):
        if return_times:
            return super().get_charge_discharge_curves(file_name, df, early_cycle_threshold, nominal_capacity, return_times=True)
        curves, times = super().get_charge_discharge_curves(file_name, df, early_cycle_threshold, nominal_capacity,
                                                            return_times=True)
        self._pending_times = times
        return curves

    def read_samples_from_one_cell(self, file_name):
        self._pending_times = None
        out = super().read_samples_from_one_cell(file_name)
        curves, eol = out[0], out[3]
        if eol is not None and curves is not None:     # read_data keeps exactly these samples, in this order
            if self._pending_times is None:
                raise RuntimeError(f'{file_name}: no times captured')
            tau = times_to_tau(self._pending_times[:self.early_cycle_threshold], self._dt_c, self._dt_clip)
            self._tau_list += [tau] * len(curves)      # same array object for every prefix of the cell
        return out

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['cycle_tau'] = torch.from_numpy(self._tau_list[index])
        return sample


def my_collate_fn_pooled_dt(samples):
    """my_collate_fn_pooled's 8 items + cycle_tau [B, L, P] as a 9th."""
    return (*my_collate_fn_pooled(samples), torch.stack([s['cycle_tau'] for s in samples]))


def data_provider_pooled_dt(args, flag, label_scaler=None, life_class_scaler=None, chemistries=None, split_seed=2021):
    """Same as data_factory.data_provider_pooled, with Dataset_pooled_dt and the 9-item collate."""
    train = (flag == 'train')
    data_set = Dataset_pooled_dt(args, flag=flag, chemistries=chemistries, split_seed=split_seed,
                                 label_scaler=label_scaler, life_class_scaler=life_class_scaler)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers,
                             drop_last=train, collate_fn=my_collate_fn_pooled_dt)
    return data_set, data_loader
