"""HierGen readout from checkpoints/HierGen_<tag>_seed<S>/rollouts_<split>_<mode>.npz.
Per run: MAPE, Spearman rho, median pred/label, pred p10/50/90, never-crossed count, best epoch.
Usage: python scc/hiergen_readout.py TAG [TAG ...] [--split test] [--modes latent full]
"""
import argparse, glob, os, re
import numpy as np
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument('tags', nargs='+')
p.add_argument('--split', default='test')
p.add_argument('--modes', nargs='+', default=['latent', 'full'])
p.add_argument('--max_c', type=int, default=4000)
a = p.parse_args()

first = True
for tag in a.tags:
    for d in sorted(glob.glob(f'checkpoints/HierGen_{tag}_seed*')):
        seed = d.rsplit('seed', 1)[1]
        log = f'results_logs/HierGen_{tag}_seed{seed}.log'
        ep = '?'
        if os.path.exists(log):
            m = re.findall(r'checkpoint from epoch (\d+)', open(log, errors='ignore').read())
            ep = m[-1] if m else 'none'
        for mode in a.modes:
            f = os.path.join(d, f'rollouts_{a.split}_{mode}.npz')
            if not os.path.exists(f):
                print(f'{tag:34s} s{seed:5s} {mode:7s} MISSING (best ep {ep})'); continue
            z = np.load(f)
            life, lab = z['life'].astype(float), z['label'].astype(float)
            if first:
                print(f'n={len(lab)}  label p10/50/90 = ' + ' / '.join(f'{v:.0f}' for v in np.percentile(lab, [10, 50, 90])))
                print(f'constant-101 MAPE {np.mean(np.abs(101 - lab) / lab):.3f} | constant-median MAPE {np.mean(np.abs(np.median(lab) - lab) / lab):.3f}')
                print(f'{"tag":34s} {"seed":6s} {"mode":7s} {"MAPE":>6s} {"rho":>6s} {"medR":>5s} {"p10":>5s} {"p50":>5s} {"p90":>5s} {"never":>5s} ep')
                first = False
            mape = np.mean(np.abs(life - lab) / lab)
            rho = spearmanr(life, lab).correlation
            q = np.percentile(life, [10, 50, 90])
            print(f'{tag:34s} s{seed:5s} {mode:7s} {mape:6.3f} {rho:6.2f} {np.median(life / lab):5.2f} '
                  f'{q[0]:5.0f} {q[1]:5.0f} {q[2]:5.0f} {int((life >= a.max_c).sum()):5d} {ep}')
