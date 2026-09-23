#!/usr/bin/env python3
"""
Collect pooled / full-timescale run results from results_logs/*.log into CSVs. Standard library only, so it runs on
the login node without any venv:

    python3 collect_results.py                          # reads results_logs/*.log, writes results_logs/results_*.csv
    python3 collect_results.py --logs 'results_logs/CPMamba_*.log' --out_prefix results_logs/mamba

Reads the per-run logs the qsub scripts tee (<MODEL>_Pooled<TAG>_seed<seed>.log, <MODEL>_Full<TAG>_seed<seed>.log),
one file per seed, and parses the lines run_main.py prints at the end (utils/pooled_eval.PooledTracker):
    Pooled single-checkpoint | chem=Li-ion | epoch=12 | Test MAE: ... | Test MAPE: ... | Val MAPE: ...
    Pooled per-chem-best-val | chem=Li-ion | ...

Writes:
  <out_prefix>_runs.csv     one row per (log file, view, chemistry): model, run_type, tag, seed, epoch, all metrics
  <out_prefix>_summary.csv  mean / population std over seeds per (model, run_type, tag, view, chemistry), n_seeds, seeds
  <out_prefix>_missing.csv  logs with no result lines (still running, killed, crashed) + the last log line
Seen/unseen entries of -10000 (no such cells in that split) are written as empty.
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

NAME_RE = re.compile(r'^(?P<model>[A-Za-z0-9]+)_(?P<run_type>Pooled|Full)(?P<tag>.*)_seed(?P<seed>\d+)\.log$')
LINE_RE = re.compile(r'Pooled (?P<view>single-checkpoint|per-chem-best-val) \| chem=(?P<chem>[^|]+?) \| epoch=(?P<epoch>\d+) \| (?P<rest>.*)$')
METRIC_RE = re.compile(r'([A-Za-z0-9 %\-]+?):\s*(-?[\d.]+)')
METRICS = ['Test MAE', 'Test RMSE', 'Test MAPE', 'Test 15%-accuracy', 'Test 10%-accuracy',
           'Test Seen MAPE', 'Test Unseen MAPE', 'Val MAPE', 'Val 15%-accuracy']
COL = {m: m.lower().replace(' ', '_').replace('%-', '_').replace('-', '_') for m in METRICS}  # 'Test 15%-accuracy' -> test_15_accuracy


def parse_log(path):
    rows = {}
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    for line in lines:
        m = LINE_RE.search(line)
        if not m:
            continue
        metrics = {k.strip(): float(v) for k, v in METRIC_RE.findall(m['rest'])}
        row = {'view': m['view'], 'chem': m['chem'].strip(), 'epoch': int(m['epoch'])}
        for name in METRICS:
            v = metrics.get(name)
            row[COL[name]] = '' if v is None or v <= -9999 else v
        rows[(row['view'], row['chem'])] = row          # last occurrence wins (e.g. a log reused across attempts)
    last = next((l for l in reversed(lines) if l.strip()), '')
    return list(rows.values()), last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='results_logs/*.log', help='glob of per-run logs')
    ap.add_argument('--out_prefix', default='results_logs/results')
    args = ap.parse_args()

    run_rows, missing = [], []
    for path in sorted(glob.glob(args.logs)):
        name = os.path.basename(path)
        nm = NAME_RE.match(name)
        if not nm:
            continue                                       # not a pooled/full per-run log (e.g. old per-chemistry logs)
        rows, last = parse_log(path)
        info = dict(model=nm['model'], run_type=nm['run_type'], tag=nm['tag'].lstrip('_') or 'base', seed=int(nm['seed']))
        if not rows:
            missing.append(dict(file=name, **info, last_line=last[:300]))
            continue
        for r in rows:
            run_rows.append(dict(file=name, **info, **r))

    metric_cols = [COL[m] for m in METRICS]
    base_cols = ['model', 'run_type', 'tag', 'seed', 'view', 'chem', 'epoch']
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    with open(f'{args.out_prefix}_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=base_cols + metric_cols + ['file'])
        w.writeheader()
        w.writerows(sorted(run_rows, key=lambda r: (r['model'], r['run_type'], r['tag'], r['view'], r['chem'], r['seed'])))

    groups = defaultdict(list)
    for r in run_rows:
        groups[(r['model'], r['run_type'], r['tag'], r['view'], r['chem'])].append(r)
    summary = []
    for (model, run_type, tag, view, chem), rs in sorted(groups.items()):
        out = dict(model=model, run_type=run_type, tag=tag, view=view, chem=chem, n_seeds=len(rs),
                   seeds=' '.join(str(s) for s in sorted(r['seed'] for r in rs)))
        for c in metric_cols:
            vals = [r[c] for r in rs if r[c] != '']
            out[f'{c}_mean'] = round(statistics.mean(vals), 4) if vals else ''
            out[f'{c}_std'] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else ('' if not vals else 0.0)
        summary.append(out)
    with open(f'{args.out_prefix}_summary.csv', 'w', newline='') as f:
        cols = ['model', 'run_type', 'tag', 'view', 'chem', 'n_seeds', 'seeds'] + \
               [f'{c}_{s}' for c in metric_cols for s in ('mean', 'std')]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)

    with open(f'{args.out_prefix}_missing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'model', 'run_type', 'tag', 'seed', 'last_line'])
        w.writeheader()
        w.writerows(missing)

    print(f'{len(run_rows)} result rows from {len({r["file"] for r in run_rows})} logs -> {args.out_prefix}_runs.csv')
    print(f'{len(summary)} (model, tag, view, chem) groups -> {args.out_prefix}_summary.csv')
    print(f'{len(missing)} logs without results -> {args.out_prefix}_missing.csv')


if __name__ == '__main__':
    main()
