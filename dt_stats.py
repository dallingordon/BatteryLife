"""
Δt stats pass (10/13): how much real time each resampled point spans, per dataset, before adding Δt to S4D / Mamba.
Li-ion (MIX_large) by default. Run on the SCC from the repo root:

    python dt_stats.py --root_path ./dataset --workers 8                 # cycles 1..100 (what the <=100 runs see)
    python dt_stats.py --root_path ./dataset --workers 8 --max_cycles 0  # every cycle up to eol-1 (full history)

Uses the exact baseline segmentation/resampling (Dataset_original.get_charge_discharge_curves(return_times=True)):
300 points = 150 charge + 150 discharge, evenly spaced in RECORD INDEX, rests (|I| < 0.01C) dropped. For every cycle:
  - within-segment Δt between consecutive resampled points (charge half, discharge half separately). The FIRST step
    of each half is kept apart: the baseline segmentation starts the chronologically second half with the last record
    of the first half, so that step spans the rest in between (rest rows are dropped). c and the tau percentiles use
    the INTERIOR steps; the first steps are reported separately (that is where a rest token would go).
  - segment durations, the curve-order charge->discharge boundary gap (can be NEGATIVE for discharge-first
    datasets, e.g. RWTH: the curve is always charge then discharge, whatever the chronological order)
  - gap from this cycle's chronological end to the next cycle's start (only meaningful for an absolute clock)
  - clock type: does time_in_s reset every cycle?  non-monotonic / zero steps inside a segment?
Proposes c = median within-segment Δt over the Li-ion TRAIN cells (tau = Δt / c, so tau ~ 1 for a typical step) and
prints tau percentiles (for a clip). Writes <out_dir>/per_cell.csv, per_dataset.csv, summary.json.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data_provider.data_loader import CHEMISTRIES, POOLED_SPLIT_SEEDS, pooled_split_files
from data_provider.data_loader_full import make_cell_reader, _one_cycle_curve

_reader = None
HALF = None
QS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]


def _init(root_path, cdl):
    global _reader, HALF
    _reader = make_cell_reader(root_path, cdl)
    HALF = cdl // 2


def dataset_name(f):
    return f.split('_')[0]


def clock_type(pc):
    """'absolute' if each cycle starts after the previous one, 'reset' if time_in_s restarts most cycles."""
    if len(pc) < 2:
        return 'unknown'
    return 'reset' if (pc['t0'].diff().iloc[1:] <= 0).mean() > 0.5 else 'absolute'


def cell_stats(job):
    """-> (row dict, interior Δt array, first-step Δt array (float32, positive, finite), per-cycle frame) for one cell."""
    f, split, max_cycles = job
    row = dict(file=f, dataset=dataset_name(f), split=split, status='ok', error='')
    try:
        data, eol = _reader.read_cell_data_according_to_prefix(f)
        if data is None:
            row['status'] = 'no_pkl'; return row, np.zeros(0, np.float32), np.zeros(0, np.float32), None
        if eol is None:
            row['status'] = 'no_label'; return row, np.zeros(0, np.float32), np.zeros(0, np.float32), None
        if f.startswith('RWTH'):
            nominal = 1.85
        elif f.startswith('SNL_18650_NCA_25C_20-80'):
            nominal = 3.2
        else:
            nominal = data['nominal_capacity_in_Ah']
        cyc = data['cycle_data']
        n = min(len(cyc), math.ceil(eol) - 1)
        if max_cycles:
            n = min(n, max_cycles)
        row.update(eol=eol, n_cycles_used=n, has_time_key=bool(len(cyc) and 'time_in_s' in cyc[0]))
        dts, firsts, per_cycle = [], [], []
        for i in range(n):
            try:
                out = _one_cycle_curve(_reader, f, cyc[i], i + 1, nominal, return_times=True)
            except Exception as e:
                row.update(status='stopped', error=f'cycle {i + 1}: {type(e).__name__}: {e}'[:200]); break
            if out is None:
                row.update(status='stopped', error=f'cycle {i + 1}: no usable rows'); break
            t = out[1]
            if not np.isfinite(t).all():
                row.update(status='stopped', error=f'cycle {i + 1}: non-finite time'); break
            ch, dis = t[:HALF], t[HALF:]
            dch, ddis = np.diff(ch), np.diff(dis)
            d = np.concatenate([dch[1:], ddis[1:]])
            dts.append(d.astype(np.float32))
            firsts.append(np.array([dch[0], ddis[0]], dtype=np.float32))
            per_cycle.append(dict(cycle=i + 1, t0=ch[0], ch_dur=ch[-1] - ch[0], dis_dur=dis[-1] - dis[0],
                                  boundary=dis[0] - ch[-1], t_end=max(ch[-1], dis[-1]), t_start=min(ch[0], dis[0]),
                                  n_neg=int((d < 0).sum()), n_zero=int((d == 0).sum())))
        if not per_cycle:
            row['status'] = 'no_cycles' if row['status'] == 'ok' else row['status']
            return row, np.zeros(0, np.float32), np.zeros(0, np.float32), None
        pc = pd.DataFrame(per_cycle)
        pc['next_gap'] = pc['t_start'].shift(-1) - pc['t_end']
        d = np.concatenate(dts)
        pos = d[np.isfinite(d) & (d > 0)]
        fs = np.concatenate(firsts)
        fpos = fs[np.isfinite(fs) & (fs > 0)]
        row.update(
            n_cycles_ok=len(pc),
            clock=clock_type(pc),
            frac_neg_steps=float((d < 0).mean()), frac_zero_steps=float((d == 0).mean()),
            dt_median=float(np.median(pos)) if len(pos) else np.nan,
            dt_p01=float(np.quantile(pos, 0.01)) if len(pos) else np.nan,
            dt_p99=float(np.quantile(pos, 0.99)) if len(pos) else np.nan,
            first_step_median_s=float(np.median(fpos)) if len(fpos) else np.nan,
            ch_dur_median_min=float(pc['ch_dur'].median() / 60), dis_dur_median_min=float(pc['dis_dur'].median() / 60),
            boundary_median_min=float(pc['boundary'].median() / 60), frac_boundary_neg=float((pc['boundary'] < 0).mean()),
            next_gap_median_min=float(pc['next_gap'].median() / 60) if len(pc) > 1 else np.nan,
            cycle_dur_median_min=float((pc['t_end'] - pc['t_start']).median() / 60),
        )
        return row, pos, fpos, pc
    except Exception as e:
        row.update(status='error', error=f'{type(e).__name__}: {e}'[:200])
        return row, np.zeros(0, np.float32), np.zeros(0, np.float32), None


def qtable(x, scale=1.0):
    return {f'p{100 * q:g}': float(np.quantile(x, q) / scale) for q in QS} if len(x) else {}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', default='./dataset')
    p.add_argument('--chemistries', nargs='+', default=['Li-ion'], choices=CHEMISTRIES)
    p.add_argument('--split_seed', type=int, default=2021, choices=POOLED_SPLIT_SEEDS, help='Li-ion has one split anyway')
    p.add_argument('--charge_discharge_length', type=int, default=300)
    p.add_argument('--max_cycles', type=int, default=100, help='0 = every cycle up to eol-1')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit', type=int, default=None, help='debug: first N cells')
    p.add_argument('--out_dir', default=None, help='default dt_stats_<chems>_<max_cycles or all>')
    a = p.parse_args()
    out_dir = a.out_dir or f"dt_stats_{'-'.join(a.chemistries)}_{a.max_cycles or 'all'}"
    os.makedirs(out_dir, exist_ok=True)

    jobs, seen = [], set()
    for c in a.chemistries:
        for split in ('train', 'val', 'test'):
            for f in pooled_split_files(c, split, a.split_seed):
                if f not in seen:
                    seen.add(f); jobs.append((f, split, a.max_cycles or None))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{len(jobs)} cells ({", ".join(a.chemistries)}; split seed {a.split_seed}) | max_cycles={a.max_cycles or "all"} | workers={a.workers}')

    reader = make_cell_reader(a.root_path, a.charge_discharge_length)
    merged = f'{a.root_path}/total_MICH/'
    if any(f.startswith('MICH') for f, _, _ in jobs) and not (os.path.isdir(merged) and os.listdir(merged)):
        reader.merge_MICH(merged)

    rows, dt_by, first_by = [], defaultdict(list), defaultdict(list)   # [(dataset, split)] -> positive Δt arrays (s)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.root_path, a.charge_discharge_length)) as pool:
        for k, (row, pos, fpos, pc) in enumerate(pool.imap_unordered(cell_stats, jobs), 1):
            rows.append(row)
            if len(pos):
                dt_by[(row['dataset'], row['split'])].append(pos)
            if len(fpos):
                first_by[(row['dataset'], row['split'])].append(fpos)
            if k % 25 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} cells | {time.time() - t0:.0f}s', flush=True)

    cells = pd.DataFrame(rows).sort_values(['dataset', 'split', 'file'])
    cells.to_csv(f'{out_dir}/per_cell.csv', index=False)

    def pooled(pred, src=None):
        xs = [np.concatenate(v) for (ds, sp), v in (dt_by if src is None else src).items() if pred(ds, sp)]
        return np.concatenate(xs) if xs else np.zeros(0)

    train = pooled(lambda ds, sp: sp == 'train')
    if not len(train):
        raise SystemExit('no usable training cells -- see per_cell.csv')
    c = float(np.median(train))
    ok = cells[cells['status'].isin(['ok', 'stopped'])]

    # per dataset (all splits pooled)
    ds_rows = []
    for ds in sorted(cells['dataset'].unique()):
        x = pooled(lambda d, sp: d == ds)
        xf = pooled(lambda d, sp: d == ds, first_by)
        g = cells[cells['dataset'] == ds]
        go = ok[ok['dataset'] == ds]
        ds_rows.append(dict(
            dataset=ds, cells=len(g), ok=int((g['status'] == 'ok').sum()), stopped=int((g['status'] == 'stopped').sum()),
            failed=int((~g['status'].isin(['ok', 'stopped'])).sum()),
            train=int((g['split'] == 'train').sum()), val=int((g['split'] == 'val').sum()), test=int((g['split'] == 'test').sum()),
            clock=','.join(sorted(go['clock'].dropna().astype(str).unique())) if len(go) else '',
            dt_median_s=float(np.median(x)) if len(x) else np.nan,
            tau_p01=float(np.quantile(x, 0.01) / c) if len(x) else np.nan, tau_p50=float(np.median(x) / c) if len(x) else np.nan,
            tau_p99=float(np.quantile(x, 0.99) / c) if len(x) else np.nan, tau_max=float(x.max() / c) if len(x) else np.nan,
            first_tau_p50=float(np.median(xf) / c) if len(xf) else np.nan,
            first_tau_p99=float(np.quantile(xf, 0.99) / c) if len(xf) else np.nan,
            frac_zero_steps=float(go['frac_zero_steps'].mean()) if len(go) else np.nan,
            frac_neg_steps=float(go['frac_neg_steps'].mean()) if len(go) else np.nan,
            ch_dur_min=float(go['ch_dur_median_min'].median()) if len(go) else np.nan,
            dis_dur_min=float(go['dis_dur_median_min'].median()) if len(go) else np.nan,
            boundary_min=float(go['boundary_median_min'].median()) if len(go) else np.nan,
            frac_boundary_neg=float(go['frac_boundary_neg'].mean()) if len(go) else np.nan,
            next_gap_min=float(go['next_gap_median_min'].median()) if len(go) else np.nan,
            cycle_dur_min=float(go['cycle_dur_median_min'].median()) if len(go) else np.nan,
        ))
    dsf = pd.DataFrame(ds_rows)
    dsf.to_csv(f'{out_dir}/per_dataset.csv', index=False)

    summary = dict(
        chemistries=a.chemistries, max_cycles=a.max_cycles or 'all', n_cells=len(cells),
        status_counts=cells['status'].value_counts().to_dict(),
        c_seconds=c, c_rule='median positive interior within-segment dt over train cells',
        tau_train=qtable(train, c), tau_train_max=float(train.max() / c),
        first_step_tau_train=qtable(pooled(lambda ds, sp: sp == 'train', first_by), c),
        tau_val=qtable(pooled(lambda ds, sp: sp == 'val'), c), tau_test=qtable(pooled(lambda ds, sp: sp == 'test'), c),
        dt_train_seconds=qtable(train),
    )
    json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=2)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
    print(f'\nstatus: {summary["status_counts"]}')
    bad = cells[~cells['status'].isin(['ok'])]
    if len(bad):
        print(f'\ncells not fully ok ({len(bad)}):')
        print(bad[['file', 'split', 'status', 'error']].head(40).to_string(index=False))
    print(f'\nproposed c = {c:.3f} s  (median interior within-segment dt, train)')
    print('tau = dt / c percentiles (train): ' + '  '.join(f'{k} {v:.3g}' for k, v in summary['tau_train'].items())
          + f'  max {summary["tau_train_max"]:.3g}')
    print('first-step tau (spans the dropped rest between halves), train: '
          + '  '.join(f'{k} {v:.3g}' for k, v in summary['first_step_tau_train'].items()))
    print('\nper dataset (durations in minutes, medians; boundary = curve-order charge end -> discharge start):')
    print(dsf.round(3).to_string(index=False))
    print(f'\nwrote {out_dir}/per_cell.csv, per_dataset.csv, summary.json')
