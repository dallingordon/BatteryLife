"""
Geometric ("15%-band") bins for turning the battery-life regression target
into a classification problem.

The eval metric everyone already reports is 15%-accuracy: a prediction
counts as correct if it's within +/-15% of the true label. This module
builds bin edges that are exactly that width in relative terms - each bin
spans [center*0.85, center*1.15] - so predicting the right bin is
(approximately) the same thing as being within 15% of the true value.

Because "within 15%" is a *relative* window, the bins have to grow in
absolute size as the value increases (a label of 1000 needs a +/-150
window, a label of 100 needs +/-15). That means uniform spacing in
log-space, i.e. a fixed multiplicative ratio between consecutive bin
edges/centers:

    r = (1 + tol) / (1 - tol)

For tol=0.15, r ~= 1.3529. The number of bins needed to span [range_min,
range_max] is ceil(log(range_max/range_min) / log(r)).

This is deliberately dataset-agnostic for now: one fixed global range
(range_min, range_max) is used for every dataset/model, matching the
observed usable_length range across the full BatteryLife cycle-length
audit (~1 to ~3842 cycles - see notes/notes_9_8.txt).
"""
import json
import numpy as np


class GeoBins:
    def __init__(self, range_min=1.0, range_max=3842.0, tol=0.15):
        if range_min <= 0:
            raise ValueError(f"range_min must be > 0 for geometric bins, got {range_min}")
        if range_max <= range_min:
            raise ValueError(f"range_max ({range_max}) must be > range_min ({range_min})")
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.tol = float(tol)
        self.ratio = (1.0 + tol) / (1.0 - tol)

        n_bins = int(np.ceil(np.log(self.range_max / self.range_min) / np.log(self.ratio)))
        n_bins = max(n_bins, 1)
        self.num_bins = n_bins

        # edges[0] = range_min, edges[num_bins] = range_min * ratio**num_bins (>= range_max)
        exponents = np.arange(n_bins + 1)
        self.edges = self.range_min * (self.ratio ** exponents)  # shape [num_bins+1]
        # geometric center of each bin = sqrt(edge_i * edge_{i+1})
        self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])  # shape [num_bins]

        self._log_edges = np.log(self.edges)

    def value_to_bin(self, values):
        """
        values: array-like of positive floats (raw, unscaled label values).
        returns: int64 array of bin indices in [0, num_bins - 1].
        Values below range_min clip to bin 0; values at/above range_max clip
        to the last bin (num_bins - 1).
        """
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, self.range_min, self.range_max * (1 - 1e-9))
        log_values = np.log(values)
        # searchsorted(edges, v, side='right') - 1 gives i such that edges[i] <= v < edges[i+1]
        bins = np.searchsorted(self._log_edges, log_values, side='right') - 1
        bins = np.clip(bins, 0, self.num_bins - 1)
        return bins.astype(np.int64)

    def bin_to_center(self, bins):
        """bins: array-like of int bin indices. returns: float array of geometric-center values."""
        bins = np.asarray(bins, dtype=np.int64)
        bins = np.clip(bins, 0, self.num_bins - 1)
        return self.centers[bins]

    def to_dict(self):
        return {
            "range_min": self.range_min,
            "range_max": self.range_max,
            "tol": self.tol,
            "ratio": self.ratio,
            "num_bins": self.num_bins,
            "edges": self.edges.tolist(),
            "centers": self.centers.tolist(),
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(range_min=d["range_min"], range_max=d["range_max"], tol=d["tol"])
