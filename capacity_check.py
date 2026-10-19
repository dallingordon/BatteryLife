"""
Per-cycle capacity and the cycle-life label (10/18).

How BatteryLife makes the label (process_scripts/Extract_life_labels.py, and Extract_life_labels_tools/XJTU_tools.py
for XJTU):
    capacity of cycle k   Qd_k  = max(discharge_capacity_in_Ah) over that cycle's records (the tester's running total,
                                  so the max is the end-of-discharge value)
    state of health       SOH_k = Qd_k / nominal_capacity / SOC_span
                                  nominal: data['nominal_capacity_in_Ah'], except RWTH 1.85 Ah and SNL NCA 20-80 3.2 Ah;
                                  SOC_span = SOC_interval[1] - SOC_interval[0] (0 -> 1, full cycles)
    label (cycle life)    last cycle SOH >= 0.825      -> cell dropped (no label)
                          0.8 < last cycle SOH < 0.825 -> linear fit cycle ~ SOH over the last 20 cycles, read off at 0.80
                          otherwise                    -> first cycle (1-based position in cycle_data) with SOH <= 0.8
    XJTU instead          Qd_k = max discharge capacity over records with current < 0; label = last downward crossing
                          of 0.8, linearly interpolated between cycles, rounded up (extrapolation as above if needed)
The model only sees cycles 1..100, and only cells with label > 100 are used.

    python capacity_check.py                          # one Li-ion train cell (MATR_b1c5): plots + label vs recomputed
    python capacity_check.py --cell HUST_1-6.pkl      # another cell
    python capacity_check.py --compare_all --no_plot  # every Li-ion train cell (slow)

Outputs (figures/capacity_check/): <cell>_all_data.png, <cell>_cycles.png, label_check.csv, label_check.png
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression

from data_provider.data_split_recorder import split_recorder

PREFIX_DIR = {'MATR': 'MATR', 'HUST': 'HUST', 'SNL': 'SNL', 'CALCE': 'CALCE', 'HNEI': 'HNEI', 'MICH': 'total_MICH',
              'RWTH': 'RWTH', 'UL-PUR': 'UL_PUR', 'SMICH': 'MICH_EXP', 'BIT2': 'BIT2', 'Tongji': 'Tongji',
              'Stanford': 'Stanford', 'ISU-ILCC': 'ISU_ILCC', 'XJTU': 'XJTU'}


# ------------------------------------------------------------------ loading (same paths / label files as the loader)
def load_cell(root, file_name):
    prefix = file_name.split('_')[0]
    key = next(k for k in PREFIX_DIR if prefix.startswith(k))
    if key == 'SMICH':
        path = f'{root}/MICH_EXP/{file_name[1:]}'
    elif key == 'MICH' and not os.path.isdir(f'{root}/total_MICH'):
        path = next(p for p in (f'{root}/MICH/{file_name}', f'{root}/MICH_EXP/{file_name}') if os.path.exists(p))
    else:
        path = f'{root}/{PREFIX_DIR[key]}/{file_name}'
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_label(root, file_name):
    prefix = file_name.split('_')[0]
    name = file_name
    if prefix == 'MICH':
        lab = 'total_MICH_labels.json'
    elif prefix.startswith('Tongji'):
        lab, name = 'Tongji_labels.json', file_name.replace('--', '-#')
    else:
        lab = f'{prefix}_labels.json'
    with open(f'{root}/Life labels/{lab}') as f:
        return json.load(f).get(name)


def nominal_capacity(data, file_name):
    if file_name.startswith('RWTH'):
        return 1.85
    if file_name.startswith('SNL_18650_NCA_25C_20-80'):
        return 3.2
    return float(data['nominal_capacity_in_Ah'])


def soc_span(data):
    s = data['SOC_interval']
    span = float(s[1]) - float(s[0])
    return 1.0 if span == 0 or not np.isfinite(span) else span


# ------------------------------------------------------------------ the function: capacity per cycle -> label
def cycle_capacity(data, file_name):
    """-> cycle number (1-based position in cycle_data) [L], Qd in Ah [L], SOH [L], nominal, SOC span."""
    nom, span = nominal_capacity(data, file_name), soc_span(data)
    xjtu = file_name.startswith('XJTU')
    qd = []
    for cyc in data['cycle_data']:
        d = np.asarray(cyc['discharge_capacity_in_Ah'], dtype=float)
        if xjtu:
            i = np.asarray(cyc['current_in_A'], dtype=float)[:len(d)]
            d = d[:len(i)]
            ok = np.isfinite(d) & np.isfinite(i)
            sel = ok & (i < 0)
            d = d[sel] if sel.any() else d[ok]
        qd.append(np.nanmax(d) if np.isfinite(d).any() else np.nan)
    qd = np.array(qd)
    return np.arange(1, len(qd) + 1), qd, qd / nom / span, nom, span


def life_from_capacity(cycles, soh, file_name, threshold=0.8, regress_cycle_num=20):
    """Cycle life from the SOH curve, same rules as BatteryLife. -> (life or None, rule)."""
    if file_name.startswith('XJTU'):
        x, y = cycles[np.isfinite(soh)].astype(float), soh[np.isfinite(soh)]
        cross = [x[k - 1] + (threshold - y[k - 1]) / (y[k] - y[k - 1]) * (x[k] - x[k - 1]) if y[k] != y[k - 1] else x[k]
                 for k in range(1, len(x)) if y[k - 1] > threshold >= y[k]]
        if cross:
            return int(np.ceil(max(cross))), 'xjtu: last 0.8 crossing (interp)'
        if len(y) and y[0] <= threshold:
            return int(np.ceil(x[0])), 'xjtu: starts below 0.8'
        if len(y) and threshold < y[-1] < threshold + 0.025:
            n = min(regress_cycle_num, len(x))
            reg = LinearRegression().fit(y[-n:].reshape(-1, 1), x[-n:])
            return int(np.ceil(int(reg.predict([[threshold]])[0]))), 'xjtu: extrapolated'
        return None, 'xjtu: never reaches 0.8'
    last = soh[-1]
    if last >= threshold + 0.025:
        return None, 'dropped: last SOH >= 0.825'
    if last > threshold:
        n = regress_cycle_num
        reg = LinearRegression().fit(soh[-n:].reshape(-1, 1), cycles[-n:])
        return int(reg.predict([[threshold]])[0]), 'extrapolated (last 20 cycles)'
    below = np.nonzero(soh <= threshold)[0]
    return int(below[0]) + 1, 'first cycle with SOH <= 0.8'


# ------------------------------------------------------------------ plots
def plot_cell(data, file_name, label, out_dir, model_cycles=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cycles, qd, soh, nom, span = cycle_capacity(data, file_name)
    life, rule = life_from_capacity(cycles, soh, file_name)
    cd = data['cycle_data']
    keys = [('voltage_in_V', 'Voltage (V)'), ('current_in_A', 'Current (A)'),
            ('charge_capacity_in_Ah', 'Charge capacity (Ah)'), ('discharge_capacity_in_Ah', 'Discharge capacity (Ah)')]

    # every record of every cycle, on one time axis. Each cycle's own clock is shifted to start where the previous
    # cycle ended (testers differ: some reset time per cycle, some don't), so gaps between cycles are dropped.
    T, cols, cid, off = [], {k: [] for k, _ in keys}, [], 0.0
    for k, c in enumerate(cd):
        t = np.asarray(c['time_in_s'], dtype=float)
        n = min([len(t)] + [len(c[kk]) for kk, _ in keys])
        t = t[:n]
        t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
        T.append((t - t0 + off) / 3600.0)
        off += (np.nanmax(t) - t0 if np.isfinite(t).any() else 0.0) + 1.0
        for kk, _ in keys:
            cols[kk].append(np.asarray(c[kk], dtype=float)[:n])
        cid.append(np.full(n, k + 1))
    T = np.concatenate(T)
    cid = np.concatenate(cid)
    n_rec = len(T)
    t100 = T[cid <= model_cycles].max() if (cid <= model_cycles).any() else None

    fig, ax = plt.subplots(len(keys) + 1, 1, figsize=(14, 13), sharex=False)
    for a, (kk, lab) in zip(ax, keys):
        a.plot(T, np.concatenate(cols[kk]), lw=0.3, color='#1f4e79', rasterized=True)
        a.set_ylabel(lab)
        if t100 is not None:
            a.axvspan(0, t100, color='#f0b429', alpha=0.15, lw=0)
    ax[3].set_xlabel('cumulative test time (h), gaps between cycles removed')
    ax[0].set_title(f'{file_name}: all {n_rec:,} records, {len(cd)} cycles '
                    f'(shaded = cycles 1..{model_cycles}, what the model sees)')
    a = ax[-1]
    a.plot(cycles, qd, '.', ms=2, color='#1f4e79', label='Qd per cycle = max discharge capacity')
    thr = 0.8 * nom * span
    a.axhline(thr, color='#c0392b', lw=1, ls='--', label=f'0.8 x nominal x SOC span = {thr:.3g} Ah')
    if label is not None:
        a.axvline(label, color='#c0392b', lw=1, label=f'label (json) = {label}')
    if life is not None and life != label:
        a.axvline(life, color='#27ae60', lw=1, ls=':', label=f'recomputed = {life}')
    a.axvspan(0.5, model_cycles + 0.5, color='#f0b429', alpha=0.15, lw=0)
    a.set_xlabel('cycle'); a.set_ylabel('Qd (Ah)')
    a.set_title(f'capacity per cycle | nominal {nom:g} Ah, SOC span {span:g} | recomputed life {life} ({rule})')
    a.legend(fontsize=8, loc='best')
    fig.tight_layout()
    p1 = f'{out_dir}/{os.path.splitext(file_name)[0]}_all_data.png'
    fig.savefig(p1, dpi=130); plt.close(fig)

    # per-cycle curves: every cycle, coloured by cycle number (V vs time in cycle, V vs Qd, I vs time in cycle)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.get_cmap('viridis')
    L = len(cd)
    for k, c in enumerate(cd):
        t = np.asarray(c['time_in_s'], dtype=float)
        n = min(len(t), len(c['voltage_in_V']), len(c['current_in_A']), len(c['discharge_capacity_in_Ah']))
        t = (t[:n] - np.nanmin(t[:n])) / 60.0
        v = np.asarray(c['voltage_in_V'], float)[:n]
        i = np.asarray(c['current_in_A'], float)[:n]
        q = np.asarray(c['discharge_capacity_in_Ah'], float)[:n]
        col = cmap(k / max(1, L - 1))
        ax[0].plot(t, v, lw=0.3, color=col, rasterized=True)
        ax[1].plot(q[i < 0], v[i < 0], lw=0.3, color=col, rasterized=True)
        ax[2].plot(t, i, lw=0.3, color=col, rasterized=True)
    ax[0].set(xlabel='time in cycle (min)', ylabel='V', title='voltage, every cycle')
    ax[1].set(xlabel='discharge capacity (Ah)', ylabel='V', title='discharge V vs Qd (records with I < 0)')
    ax[2].set(xlabel='time in cycle (min)', ylabel='A', title='current, every cycle')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, L))
    fig.colorbar(sm, ax=ax, label='cycle', fraction=0.02)
    fig.suptitle(file_name)
    p2 = f'{out_dir}/{os.path.splitext(file_name)[0]}_cycles.png'
    fig.savefig(p2, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'{file_name}: {len(cd)} cycles, {n_rec:,} records | nominal {nom:g} Ah, SOC span {span:g} | '
          f'Qd cycle 1 {qd[0]:.4g} Ah, cycle {min(100, L)} {qd[min(100, L) - 1]:.4g} Ah, last {qd[-1]:.4g} Ah '
          f'(SOH {soh[-1]:.3f}) | label {label} | recomputed {life} ({rule})')
    print(f'  wrote {p1}\n  wrote {p2}')


# ------------------------------------------------------------------ comparison over many cells
def compare(root, files, out_dir):
    import csv
    import time
    rows, t0 = [], time.time()
    for j, fn in enumerate(files):
        print(f'  [{j + 1}/{len(files)}] {fn} ({time.time() - t0:.0f}s)', flush=True)
        label = load_label(root, fn)
        try:
            data = load_cell(root, fn)
        except (FileNotFoundError, StopIteration):
            rows.append(dict(file=fn, dataset=fn.split('_')[0], label=label, recomputed='', diff='', rule='pkl missing',
                             n_cycles='', soh_first='', soh_last=''))
            continue
        cycles, qd, soh, nom, span = cycle_capacity(data, fn)
        life, rule = life_from_capacity(cycles, soh, fn)
        diff = (life - label) if (life is not None and label is not None) else ''
        rows.append(dict(file=fn, dataset=fn.split('_')[0], label=label, recomputed=life, diff=diff, rule=rule,
                         n_cycles=len(cycles), soh_first=round(float(soh[0]), 4), soh_last=round(float(soh[-1]), 4)))
    p = f'{out_dir}/label_check.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f'\n=== recomputed life vs json label: {len(rows)} cells ===')
    print(f'{"dataset":10s} {"cells":>5s} {"exact":>5s} {"|d|<=1":>6s} {"mismatch":>8s} {"no label":>8s} {"missing":>7s}  worst')
    by = {}
    for r in rows:
        by.setdefault(r['dataset'], []).append(r)
    for ds, rs in sorted(by.items()):
        d = [r['diff'] for r in rs if r['diff'] != '']
        miss = sum(r['rule'] == 'pkl missing' for r in rs)
        nolab = sum(r['label'] is None and r['rule'] != 'pkl missing' for r in rs)
        worst = max(rs, key=lambda r: abs(r['diff']) if r['diff'] != '' else -1)
        wtxt = f"{worst['file']} label {worst['label']} vs {worst['recomputed']}" if worst['diff'] not in ('', 0) else ''
        print(f'{ds:10s} {len(rs):5d} {sum(x == 0 for x in d):5d} {sum(abs(x) <= 1 for x in d):6d} '
              f'{sum(x != 0 for x in d):8d} {nolab:8d} {miss:7d}  {wtxt}')
    print(f'  wrote {p}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r['diff'] != '']
    if ok:
        fig, a = plt.subplots(figsize=(6, 6))
        for ds in sorted({r['dataset'] for r in ok}):
            rr = [r for r in ok if r['dataset'] == ds]
            a.scatter([r['label'] for r in rr], [r['recomputed'] for r in rr], s=10, label=ds)
        m = max(max(r['label'] for r in ok), max(r['recomputed'] for r in ok))
        a.plot([1, m], [1, m], 'k--', lw=0.8)
        a.set(xscale='log', yscale='log', xlabel='json label (cycle life)', ylabel='recomputed from capacity',
              title='cycle life: label vs recomputed')
        a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f'{out_dir}/label_check.png', dpi=130); plt.close(fig)
        print(f'  wrote {out_dir}/label_check.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_path', default='./dataset')
    ap.add_argument('--cell', default='MATR_b1c5.pkl', help='a Li-ion (MIX_large) train cell')
    ap.add_argument('--out_dir', default='figures/capacity_check')
    ap.add_argument('--splits', nargs='+', default=['train'], choices=['train', 'val', 'test'])
    ap.add_argument('--compare_all', action='store_true', help='also recompute every Li-ion cell (slow: use the qsub script)')
    ap.add_argument('--datasets', nargs='+', default=None, help='only these dataset prefixes, e.g. MATR HUST')
    ap.add_argument('--no_plot', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.no_plot:
        in_train = a.cell in split_recorder.MIX_large_train_files
        print(f'{a.cell}: {"in" if in_train else "NOT in"} the Li-ion (MIX_large) train split')
        plot_cell(load_cell(a.root_path, a.cell), a.cell, load_label(a.root_path, a.cell), a.out_dir)
    if a.compare_all:
        files = []
        for s in a.splits:
            files += getattr(split_recorder, f'MIX_large_{s}_files')
        files = list(dict.fromkeys(files))
        if a.datasets:
            files = [f for f in files if f.split('_')[0] in a.datasets]
        compare(a.root_path, files, a.out_dir)
