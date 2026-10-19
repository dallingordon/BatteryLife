"""
Train / evaluate HierGen (models/HierGen.py) on Li-ion (MIX_large) full histories. Single GPU, plain PyTorch.

    python precompute_gen_cells.py --workers 4                       # once: data_provider/data_loader_gen.py cache
    python train_hiergen.py --seed 2021 --cycle_layer vanilla --read_layer vanilla --tag mamba

Training (teacher forcing), per step a batch of cells, each with its WHOLE history (all cached cycles):
    loss = w_soh * SOH NLL (every cycle, from the state before it)
         + w_latent * ||zhat_c - sg(z_c)||^2 (every cycle)
         + w_read * (V NLL + dQ NLL) on --read_cycles random cycles per cell (300-step reading sequences)
Model selection: val EOL MAPE of the LATENT rollout from cycle --prefix (every --eval_every epochs; cheap).
End of training (best checkpoint): latent and full rollouts on val and test -> EOL MAPE / 15%-acc / MAE, seen / unseen,
per dataset. EOL = first generated cycle with SOH <= 0.8 (same threshold as the label); never reached by
--max_rollout_cycles -> counted as --max_rollout_cycles (and reported).
NOTE: the paper's MAPE averages over prefixes 1..100 per cell; this evaluates one prefix per cell (--prefix, default
100), so compare with the baseline at prefix 100, not its averaged number.
Outputs: results_logs/HierGen_<tag>_seed<S>.log (via the qsub tee), checkpoints/HierGen_<tag>_seed<S>/{model.pt,
rollouts_<split>_<mode>.npz, trajectories_test.png}.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader_gen import GenCells, collate_cells
from models.HierGen import HierGen

p = argparse.ArgumentParser()
p.add_argument('--root_path', default='./dataset')
p.add_argument('--cache_dir', default=None)
p.add_argument('--tag', default='default')
p.add_argument('--seed', type=int, default=2021)
p.add_argument('--checkpoints', default='./checkpoints')
# model
p.add_argument('--n_points', type=int, default=300)
p.add_argument('--d_model', type=int, default=128)
p.add_argument('--d_ff', type=int, default=256)
p.add_argument('--cycle_layer', default='vanilla', choices=['vanilla', 's4d'])
p.add_argument('--read_layer', default='vanilla', choices=['vanilla', 's4d'])
p.add_argument('--n_cycle_layers', type=int, default=4)
p.add_argument('--n_read_layers', type=int, default=4)
p.add_argument('--d_state', type=int, default=16)
p.add_argument('--d_conv', type=int, default=4)
p.add_argument('--expand', type=int, default=2)
p.add_argument('--cyc_dt_min', type=float, default=1e-4, help='S4D timescales of the cycle stack (<= ~4000 steps)')
p.add_argument('--cyc_dt_max', type=float, default=1e-1)
p.add_argument('--read_dt_min', type=float, default=1e-3, help='S4D timescales of the reading stack (300 steps)')
p.add_argument('--read_dt_max', type=float, default=1e-1)
p.add_argument('--s4_backend', default='fft', choices=['fft', 'scan'])
p.add_argument('--mamba_scan', default='cuda', choices=['cuda', 'ref'])
p.add_argument('--grad_ckpt', type=int, default=0)
# training
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--batch_cells', type=int, default=8)
p.add_argument('--read_cycles', type=int, default=32, help='cycles per cell per step that get the reading loss')
p.add_argument('--max_train_cycles', type=int, default=None, help='cap on cached cycles per cell (default: all)')
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--wd', type=float, default=1e-2)
p.add_argument('--warmup_epochs', type=float, default=2)
p.add_argument('--grad_clip', type=float, default=1.0)
p.add_argument('--w_soh', type=float, default=1.0)
p.add_argument('--w_latent', type=float, default=1.0)
p.add_argument('--w_read', type=float, default=1.0)
p.add_argument('--num_workers', type=int, default=2)
# evaluation
p.add_argument('--prefix', type=int, default=100)
p.add_argument('--eval_every', type=int, default=10)
p.add_argument('--eval_batch', type=int, default=64)
p.add_argument('--max_rollout_cycles', type=int, default=4000)
p.add_argument('--final_modes', nargs='+', default=['latent', 'full'], choices=['latent', 'full'])
p.add_argument('--final_splits', nargs='+', default=['val', 'test'])
p.add_argument('--sample', type=int, default=0, help='full rollout: sample from the predicted Gaussians (else means)')
p.add_argument('--eval_only', type=int, default=0, help='skip training, load the checkpoint, run the final evaluation')
p.add_argument('--quick', type=int, default=0, help='debug: N cells per split, 2 epochs, small rollouts')
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run = f'HierGen_{args.tag}_seed{args.seed}'
ckpt_dir = os.path.join(args.checkpoints, run)
os.makedirs(ckpt_dir, exist_ok=True)
print(f'=== {run} | device {dev} ===\n' + json.dumps(vars(args)), flush=True)

train = GenCells(args.root_path, 'train', args.cache_dir, max_cycles=args.max_train_cycles)
val = GenCells(args.root_path, 'val', args.cache_dir)
test = GenCells(args.root_path, 'test', args.cache_dir)
if args.quick:
    for d in (train, val, test):
        d.cells = d.cells[:args.quick]
    args.epochs, args.eval_every, args.max_rollout_cycles = min(args.epochs, 2), 1, min(args.max_rollout_cycles, 300)
mean, std = train.channel_stats()
print(f'channel mean {mean.tolist()} std {std.tolist()} (V volts, I C-rate, Q / nominal)', flush=True)

model = HierGen(args, mean, std).to(dev)
n_par = sum(p.numel() for p in model.parameters())
print(f'parameters: {n_par / 1e6:.2f}M | cycle {args.cycle_layer} x{args.n_cycle_layers} | read {args.read_layer} '
      f'x{args.n_read_layers}', flush=True)

no_wd = [p for n, p in model.named_parameters() if getattr(p, '_ssm_no_wd', False) or n.endswith('A_log')
         or n.endswith('.D') or p.dim() < 2]
rest = [p for n, p in model.named_parameters() if all(p is not q for q in no_wd)]
opt = torch.optim.AdamW([{'params': rest, 'weight_decay': args.wd}, {'params': no_wd, 'weight_decay': 0.0}], lr=args.lr)
loader = DataLoader(train, batch_size=args.batch_cells, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_cells, drop_last=False, persistent_workers=args.num_workers > 0)
steps_per_epoch = len(loader)
total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                          0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))


def pick_read_idx(mask, R):
    """R random real cycles per cell (all of them if fewer), -1 padded. [B, R]"""
    B = mask.shape[0]
    out = torch.full((B, R), -1, dtype=torch.long)
    for b in range(B):
        n = int(mask[b].sum())
        k = min(R, n)
        out[b, :k] = torch.randperm(n)[:k]
    return out


def to_dev(batch):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def life_metrics(pred, label, unseen, datasets, max_c):
    pred = np.where(np.isfinite(pred), pred, max_c)
    err = np.abs(pred - label) / label
    r = dict(MAPE=err.mean(), acc15=100 * (err <= 0.15).mean(), MAE=np.abs(pred - label).mean(),
             seen_MAPE=err[~unseen].mean() if (~unseen).any() else float('nan'),
             unseen_MAPE=err[unseen].mean() if unseen.any() else float('nan'), n=len(err))
    r['per_dataset'] = {d: float(err[datasets == d].mean()) for d in sorted(set(datasets))}
    return r


@torch.no_grad()
def evaluate(ds, mode, max_c, keep=False):
    model.eval()
    lives, heads, sohs, files = [], [], [], []
    dl = DataLoader(ds, batch_size=args.eval_batch, shuffle=False, num_workers=0, collate_fn=collate_cells)
    for batch in dl:
        b = to_dev(batch)
        span = torch.tensor([ds.cells[i]['soc_span'] for i in batch['idx'].tolist()], device=dev)
        out = model.rollout(b['curves'], b['mask'], b['discharge_first'], span, prefix=args.prefix, max_cycles=max_c,
                            mode=mode, sample=bool(args.sample))
        lives.append(out['life'].cpu().numpy())
        if mode == 'full':
            heads.append(out['life_head'].cpu().numpy())
        if keep:
            sohs.append(out['soh'].cpu().numpy().astype(np.float16))
        files += [ds.cells[i]['file'] for i in batch['idx'].tolist()]
    label = np.array([c['eol'] for c in ds.cells], dtype=np.float64)
    unseen = np.array([c['unseen'] for c in ds.cells])
    datasets = np.array([c['dataset'] for c in ds.cells])
    life = np.concatenate(lives)
    res = {mode: life_metrics(life, label, unseen, datasets, max_c)}
    res[mode]['not_crossed'] = int((~np.isfinite(life)).sum())
    extra = dict(life=life, label=label, files=np.array(files))
    if mode == 'full':
        lh = np.concatenate(heads)
        res['full_head'] = life_metrics(lh, label, unseen, datasets, max_c)
        res['full_head']['not_crossed'] = int((~np.isfinite(lh)).sum())
        extra['life_head'] = lh
    if keep:
        extra['soh'] = np.concatenate(sohs)
    model.train()
    return res, extra


@torch.no_grad()
def teacher_forced_val(ds):
    model.eval()
    agg, n = {}, 0
    dl = DataLoader(ds, batch_size=args.batch_cells, shuffle=False, num_workers=0, collate_fn=collate_cells)
    for batch in dl:
        b = to_dev(batch)
        ri = pick_read_idx(batch['mask'], args.read_cycles).to(dev)
        out = model(b['curves'], b['mask'], b['soh'], b['discharge_first'], ri)
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / n for k, v in agg.items()}


def fmt(name, r):
    return (f'{name} | MAPE {r["MAPE"]:.4f} | 15%-acc {r["acc15"]:.1f} | MAE {r["MAE"]:.1f} | seen MAPE '
            f'{r["seen_MAPE"]:.4f} | unseen MAPE {r["unseen_MAPE"]:.4f} | not crossed {r.get("not_crossed", 0)}/{r["n"]}')


best, best_epoch = float('inf'), -1
ckpt_path = os.path.join(ckpt_dir, 'model.pt')
if not args.eval_only:
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0, agg, nb = time.time(), {}, 0
        for batch in loader:
            ri = pick_read_idx(batch['mask'], args.read_cycles)
            b = to_dev(batch)
            out = model(b['curves'], b['mask'], b['soh'], b['discharge_first'], ri.to(dev))
            loss = args.w_soh * out['soh_nll'] + args.w_latent * out['latent_mse']
            if 'read_v_nll' in out:
                loss = loss + args.w_read * (out['read_v_nll'] + out['read_q_nll'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            for k, v in out.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg['loss'] = agg.get('loss', 0.0) + float(loss)
            nb += 1
        msg = ' | '.join(f'{k} {v / nb:.4f}' for k, v in agg.items())
        print(f'Epoch {epoch} | {time.time() - t0:.0f}s | lr {sched.get_last_lr()[0]:.2e} | train {msg}', flush=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            t1 = time.time()
            tf = teacher_forced_val(val)
            res, _ = evaluate(val, 'latent', args.max_rollout_cycles)
            r = res['latent']
            print(f'  val teacher-forced: ' + ' | '.join(f'{k} {v:.4f}' for k, v in tf.items()), flush=True)
            print('  ' + fmt(f'val latent rollout (prefix {args.prefix}) epoch {epoch}', r) + f' | {time.time() - t1:.0f}s',
                  flush=True)
            if r['MAPE'] < best:
                best, best_epoch = r['MAPE'], epoch
                torch.save(dict(model=model.state_dict(), args=vars(args), mean=mean, std=std, epoch=epoch), ckpt_path)
                print(f'  saved best (val latent MAPE {best:.4f}) -> {ckpt_path}', flush=True)
    print(f'best epoch {best_epoch} | val latent MAPE {best:.4f}', flush=True)

ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
model.load_state_dict(ck['model'])
print(f'=== final evaluation, checkpoint from epoch {ck["epoch"]} ===', flush=True)
for split in args.final_splits:
    ds = val if split == 'val' else test
    for mode in args.final_modes:
        t1 = time.time()
        res, extra = evaluate(ds, mode, args.max_rollout_cycles, keep=True)
        for k, r in res.items():
            print(fmt(f'HierGen result | split={split} | mode={k} | prefix={args.prefix}', r) + f' | {time.time() - t1:.0f}s')
            print(f'  per dataset MAPE: ' + ', '.join(f'{d} {v:.3f}' for d, v in r['per_dataset'].items()), flush=True)
        np.savez_compressed(os.path.join(ckpt_dir, f'rollouts_{split}_{mode}.npz'), **extra)

# plot: true SOH (raw, whole history) vs rollouts for a few test cells
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted({c['dataset'] for c in test.cells})
    pick = []
    for d in files:
        pick += [i for i, c in enumerate(test.cells) if c['dataset'] == d][:1]
    pick = pick[:9]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    ro = {m: np.load(os.path.join(ckpt_dir, f'rollouts_test_{m}.npz')) for m in args.final_modes
          if os.path.exists(os.path.join(ckpt_dir, f'rollouts_test_{m}.npz'))}
    for ax, i in zip(axs.ravel(), pick):
        c = test.cells[i]
        ax.plot(np.arange(1, len(c['soh']) + 1), c['soh'], color='k', lw=1, label='true (raw)')
        for m, col in (('latent', '#1f77b4'), ('full', '#d62728')):
            if m in ro and 'soh' in ro[m]:
                s = ro[m]['soh'][i].astype(np.float32)
                k = np.isfinite(s)
                ax.plot(np.nonzero(k)[0] + 1, s[k], color=col, lw=1, label=f'{m} (life {ro[m]["life"][i]:.0f})')
        ax.axhline(0.8, color='gray', ls='--', lw=0.8)
        ax.axvline(c['eol'], color='k', ls=':', lw=0.8, label=f'label {c["eol"]:.0f}')
        ax.axvspan(0, args.prefix, color='#f0b429', alpha=0.15, lw=0)
        ax.set_title(c['file'], fontsize=9)
        ax.set_xlabel('cycle'); ax.set_ylabel('SOH')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, 'trajectories_test.png'), dpi=120)
    print(f'wrote {os.path.join(ckpt_dir, "trajectories_test.png")}')
except Exception as e:
    print(f'plot skipped: {e!r}')
