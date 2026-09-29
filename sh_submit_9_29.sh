#!/bin/bash
# Session submit file for 9/29. Append submit commands during the session; run ONCE at the end, from the SCC repo root:
#   DRY_RUN=1 bash sh_submit_9_29.sh    # print every qsub without submitting
#   bash sh_submit_9_29.sh
# Comment out a numbered block to drop it.
#
# BEFORE SUBMITTING (block 2 uses new code: --prediction_mode dual): smoke-test it in an interactive GPU session (qrsh),
# 2 epochs, no Li-ion so it loads fast. Both should end with 'Pooled single-checkpoint ... | Head: bins|reg' lines
# plus Pooled[bins] / Pooled[reg] lines:
#   EPOCHS=2 SEEDS=2021 POOLED_CHEMS="CALB Zn-ion Na-ion" PRED_MODE=dual WD=1e-3 bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh 2>&1 | tail -30
#   EPOCHS=2 SEEDS=2021 POOLED_CHEMS="CALB Zn-ion Na-ion" PRED_MODE=dual CHEM_FUSION=early_concat CHEM_EMBED_DIM=16 WD=1e-3 bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh 2>&1 | tail -30
cd "$(dirname "$0")" || exit 1
CFG=experiments/sweep_9_29.tsv

# 1) More seeds (1-5) for unconditioned / alpha0.5 / alpha1.0 / early_concat16, + early_concat16+alpha0.5 combo on 8 seeds (11 jobs)
GROUP='^(more_seeds|combo)$' bash train_eval_scripts/submit_from_config.sh "$CFG"

# 2) Dual heads (geo_bins + regression on one backbone, head picked per chemistry on val), 8 seeds each (6 jobs):
#    on the unconditioned pooled model and on early_concat16. Tag: CPMLP_Pooled_dual[_early_concatE16]_wd1e-3_seed*.log
GROUP='^dual_heads$' bash train_eval_scripts/submit_from_config.sh "$CFG"

# 3) Ablation: early_concat chemistry embedding size E=32 and E=64 (E=16 is Me16 above), geo_bins wd1e-3, 8 seeds each (6 jobs).
#    Tags: CPMLP_Pooled_geobins_early_concatE{32,64}_wd1e-3_seed*.log
GROUP='^embed_dim$' bash train_eval_scripts/submit_from_config.sh "$CFG"

# 4) Mamba at max100 (training prefixes 1..100, same data CPMLP sees) x per-chemistry loss weighting alpha {0, 0.5, 1.0}
#    x {vanilla, mamba_init}, 3 seeds (seed 2021 alpha=0 already exists) (16 jobs, 1 seed each, 30h).
#    Needs the 9/29 full-loader change (--chem_loss_alpha now works with --full_timescale). Check an alpha run's first
#    epoch printout: '[full-timescale train] ... chem_loss_alpha=0.5' with a 'loss weight' column per chemistry.
bash train_eval_scripts/submit_from_config.sh experiments/sweep_9_29_mamba.tsv

# 5) Long Mamba v1 (NEW MODEL, models/LongMamba.py): Mamba over the ~30k-step point-level sequence of 100 cycles,
#    output CLS at the end; boundary token none / shared / index x {vanilla, mamba_init} x 3 seeds (18 jobs, 1 seed each).
#    BEFORE SUBMITTING, in a qrsh GPU session (gpu_c=7.0):
#      python test_longmamba.py --gpu        # must print ALL PASSED; read the B=1..32 time / memory lines and, if
#                                            # needed, add BATCH=/ACCUM= columns to experiments/sweep_9_29_longmamba.tsv
#      EPOCHS=1 SEEDS=2021 POOLED_CHEMS="CALB Zn-ion Na-ion" BOUNDARY=index bash train_eval_scripts/LongMamba_pooled_qsub.sh 2>&1 | tail -30
bash train_eval_scripts/submit_from_config.sh experiments/sweep_9_29_longmamba.tsv
