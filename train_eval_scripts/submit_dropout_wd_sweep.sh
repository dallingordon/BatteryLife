#!/bin/bash
# Submit the pooled-training dropout x weight-decay sweep -- STAGE 1 of the staged dropout/wd + fusion/alpha
# search (run on an SCC login node from the repo root):
#   bash train_eval_scripts/submit_dropout_wd_sweep.sh           # submit
#   DRY_RUN=1 bash train_eval_scripts/submit_dropout_wd_sweep.sh # just print the qsub commands
# Fixes CHEM_FUSION=none, CHEM_LOSS_ALPHA=0 (unconditioned pooled model) and sweeps:
#   DROPOUT in {0, 0.05, 0.1}   WD in {0, 1e-4, 1e-3}
# for both models (CPMLP, CPTransformer) x both modes (regression, geo_bins). The dropout=0, wd=0.0 cell is
# skipped -- that's exactly the original 9/19-9/20 pooled baseline (cpmlp/cptransformer_pool_{regression,
# geo_bins}_none), already run, so this submits the other 8 of the 9 grid cells.
# Each job runs seeds 2021/42/2024 sequentially (seed also picks the CALB/Zn-ion/Na-ion split).
#
# Once this stage's results are in, pick the best (dropout, wd) per model (or overall) and feed it into stage 2
# (train_eval_scripts/submit_pooled_sweep.sh) to sweep fusion x alpha at that fixed regularization setting:
#   DROPOUT=<winner> WD=<winner> bash train_eval_scripts/submit_pooled_sweep.sh
cd "$(dirname "$0")/.." || exit 1
DROPOUTS=${DROPOUTS:-"0 0.05 0.1"}
WDS=${WDS:-"0 1e-4 1e-3"}

for model in CPMLP CPTransformer; do
  script=train_eval_scripts/${model}_pooled_3seeds_qsub.sh
  lc=$(echo "$model" | tr 'A-Z' 'a-z')
  for mode in regression geo_bins; do
    for dropout in $DROPOUTS; do
      for wd in $WDS; do
        # skip the already-completed dropout=0, wd=0 baseline
        { [ "$dropout" = "0" ] || [ "$dropout" = "0.0" ]; } && { [ "$wd" = "0" ] || [ "$wd" = "0.0" ]; } && continue
        name="${lc}_pool_${mode}_none_d${dropout}_w${wd}"
        # short job name (qstat shows 10 chars): <M|T><r|g>_d<dropout>w<wd>, e.g. Mg_d05w4 = CPMLP geo_bins d0.05 wd1e-4
        short="${model:2:1}${mode:0:1}_d${dropout#0.}w${wd#1e-}"
        cmd="qsub -N $short -o ${name}.qlog -v PRED_MODE=$mode,CHEM_FUSION=none,CHEM_EMBED_DIM=0,CHEM_LOSS_ALPHA=0,DROPOUT=$dropout,WD=$wd $script"
        echo "$cmd"
        [ -z "$DRY_RUN" ] && $cmd
      done
    done
  done
done
