"""
Pooled multi-chemistry dataset.

Question this supports: can cells of one chemistry benefit from training on cells of other chemistries?

Training pools all chemistries (Li-ion = MIX_large, CALB, Zn-ion, Na-ion). Every sample is exactly what
Dataset_original produces for that cell (cycles 1..early_cycle_threshold, same exclusions: no life label,
missing pkl, eol <= early_cycle_threshold) plus a `chemistry_id` (index into CHEMISTRIES).

Validation / test stay split by chemistry: build one Dataset_pooled per chemistry with
`chemistries=['Zn-ion']` etc. Each of those contains exactly the val/test cells the per-chemistry
baseline uses, so reported numbers are apples-to-apples with the paper protocol.

Nothing here changes the behavior of Dataset_original for existing datasets.
"""
import copy
import numpy as np
import torch

from data_provider.data_loader import (Dataset_original, my_collate_fn_baseline, CHEMISTRIES, POOLED_SPLIT_SEEDS,
                                       dataset_id_to_chemistry_id)


class Dataset_pooled(Dataset_original):
    def __init__(self, args, flag='train', chemistries=None, split_seed=2021, **kwargs):
        """
        :param chemistries: subset of CHEMISTRIES to include (default: all four).
                            e.g. ['Zn-ion'] gives that chemistry's own split, unchanged.
        :param split_seed: 2021 / 42 / 2024. Selects the data split for CALB, Zn-ion and Na-ion (as the paper's seed does);
                           Li-ion (MIX_large) has a single split.
        Other kwargs are those of Dataset_original (label_scaler, life_class_scaler, ...).
        """
        chemistries = list(chemistries) if chemistries else list(CHEMISTRIES)
        for c in chemistries:
            assert c in CHEMISTRIES, f'unknown chemistry {c}, expected one of {CHEMISTRIES}'
        assert split_seed in POOLED_SPLIT_SEEDS, f'split_seed must be one of {POOLED_SPLIT_SEEDS}'
        self.pooled_chemistries = chemistries          # read by the 'POOLED' branch in Dataset_original.__init__
        self.pooled_split_seed = split_seed
        args = copy.copy(args)                          # don't mutate the caller's args
        args.dataset = 'POOLED'
        super().__init__(args, flag=flag, **kwargs)
        self.total_chemistry_ids = np.array([dataset_id_to_chemistry_id(i) for i in self.total_dataset_ids], dtype=np.int64)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['chemistry_id'] = int(self.total_chemistry_ids[index])
        return sample

    def chemistry_counts(self):
        """{chemistry name: number of samples} (samples, not cells: one cell yields many samples)."""
        return {c: int((self.total_chemistry_ids == i).sum()) for i, c in enumerate(CHEMISTRIES)}


def my_collate_fn_pooled(samples):
    """Same 7 outputs as my_collate_fn_baseline, plus chemistry_ids (LongTensor [B]) as an 8th."""
    out = my_collate_fn_baseline(samples)
    chemistry_ids = torch.LongTensor([i['chemistry_id'] for i in samples])
    return (*out, chemistry_ids)
