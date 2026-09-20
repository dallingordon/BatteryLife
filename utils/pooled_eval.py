"""
Bookkeeping for pooled multi-chemistry training (run_main.py --pooled). Pure python/numpy so it is easy to test.

vali_baseline() returns, per evaluation set:
  val  : (rmse, mae, mape, acc15, acc10)
  test : (rmse, mae, mape, acc15, acc10, unseen_mape, seen_mape, unseen_acc15, seen_acc15, unseen_acc10, seen_acc10)
where the seen/unseen entries are -10000 when a split has no such cells.

Two views of the same training run are tracked, both evaluated on the per-chemistry val/test sets:
  single-checkpoint : the epoch with the best macro-average val MAPE (mean over chemistries, each chemistry weighted
                      equally). This is what one saved model would give you, and what early stopping follows.
  per-chem best-val : for each chemistry, the epoch where THAT chemistry's val MAPE was lowest. This mirrors how the
                      per-chemistry baselines were early-stopped, so it is the apples-to-apples view vs the paper.
"""
import numpy as np

SENTINEL = -10000


def macro_tuple(tuples):
    """Element-wise mean over chemistries of equal-length metric tuples, ignoring the -10000 'no such cells' sentinel."""
    out = []
    for vals in zip(*tuples):
        ok = [v for v in vals if v > SENTINEL + 1]
        out.append(float(np.mean(ok)) if ok else float(SENTINEL))
    return tuple(out)


class PooledTracker:
    def __init__(self, chemistries):
        self.chemistries = list(chemistries)
        self.best_per_chem = {c: None for c in self.chemistries}
        self.single = None
        self.best_macro_val_mape = float('inf')

    def update(self, epoch, vali_res, test_res):
        """vali_res / test_res: {chemistry: metric tuple} for this epoch."""
        for c in self.chemistries:
            b = self.best_per_chem[c]
            if b is None or vali_res[c][2] < b['val'][2]:
                self.best_per_chem[c] = dict(epoch=epoch, val=vali_res[c], test=test_res[c])
        macro_val_mape = float(np.mean([vali_res[c][2] for c in self.chemistries]))
        if macro_val_mape < self.best_macro_val_mape:
            self.best_macro_val_mape = macro_val_mape
            self.single = {c: dict(epoch=epoch, val=vali_res[c], test=test_res[c]) for c in self.chemistries}

    @staticmethod
    def _line(tag, c, r):
        t, v = r['test'], r['val']
        return (f"Pooled {tag} | chem={c} | epoch={r['epoch']} | Test MAE: {t[1]:.4f} | Test RMSE: {t[0]:.4f} | "
                f"Test MAPE: {t[2]:.4f} | Test 15%-accuracy: {t[3]:.4f} | Test 10%-accuracy: {t[4]:.4f} | "
                f"Test Seen MAPE: {t[6]:.4f} | Test Unseen MAPE: {t[5]:.4f} | Val MAPE: {v[2]:.4f} | Val 15%-accuracy: {v[3]:.4f}")

    def report_lines(self):
        lines = []
        if self.single is not None:
            lines += [self._line('single-checkpoint', c, self.single[c]) for c in self.chemistries]
        lines += [self._line('per-chem-best-val', c, self.best_per_chem[c]) for c in self.chemistries if self.best_per_chem[c] is not None]
        return lines
