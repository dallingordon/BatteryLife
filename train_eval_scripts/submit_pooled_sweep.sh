#!/bin/bash
# Submit the pooled-training fusion x alpha sweep (run on an SCC login node from the repo root):
#   bash train_eval_scripts/submit_pooled_sweep.sh                       # submit, dropout/wd at their defaults (0 / 0.0)
#   DRY_RUN=1 bash train_eval_scripts/submit_pooled_sweep.sh             # just print the qsub commands
#   WITH_CONTROL=1 ...                                                   # also submit the capacity controls (embed_dim=0, no chemistry input), at alpha=0 only
#   DROPOUT=0.1 WD=0.0001 bash train_eval_scripts/submit_pooled_sweep.sh # fix dropout/wd to a chosen setting (e.g. the
#                                                                          winner from submit_dropout_wd_sweep.sh) while
#                                                                          sweeping fusion x alpha at that setting -- this
#                                                                          is "stage 2" of the staged dropout/wd + fusion/
#                                                                          alpha search (see submit_dropout_wd_sweep.sh for stage 1)
# Each job runs seeds 2021/42/2024 sequentially (seed also picks the CALB/Zn-ion/Na-ion split).
# Compare, per chemistry: pooled baseline (fusion none, alpha 0) vs late-fusion (late_mlp E=16, chemistry only seen by
# the output head) vs early-fusion (early_concat E=16, chemistry embedding concatenated onto the raw input before the
# model's first layer, so every downstream layer sees it) -- each crossed with per-chemistry loss weighting alpha in
# {0, 0.5, 1.0} -- vs the per-chemistry results.
cd "$(dirname "$0")/.." || exit 1
E=${E:-16}
DROPOUT=${DROPOUT:-0}
WD=${WD:-0.0}
ALPHAS=${ALPHAS:-"0 0.5 1.0"}

extra_tag=""
{ [ "$DROPOUT" != "0" ] && [ "$DROPOUT" != "0.0" ]; } && extra_tag="${extra_tag}_d${DROPOUT}"
{ [ "$WD" != "0" ] && [ "$WD" != "0.0" ]; } && extra_tag="${extra_tag}_w${WD}"

for model in CPMLP CPTransformer; do
  script=train_eval_scripts/${model}_pooled_3seeds_qsub.sh
  lc=$(echo "$model" | tr 'A-Z' 'a-z')
  for mode in regression geo_bins; do
    variants="none:0 late_mlp:$E early_concat:$E"
    for v in $variants; do
      fusion=${v%%:*}; emb=${v##*:}
      for alpha in $ALPHAS; do
        name="${lc}_pool_${mode}_${fusion}"
        [ "$fusion" != "none" ] && name="${name}${emb}"
        { [ "$alpha" != "0" ] && [ "$alpha" != "0.0" ]; } && name="${name}_a${alpha}"
        name="${name}${extra_tag}"
        # short job name (qstat shows 10 chars): <M|T><r|g><n|l|e><emb>a<alpha>, e.g. Mge16a5 = CPMLP geo_bins early_concat16 alpha0.5
        short="${model:2:1}${mode:0:1}${fusion:0:1}$([ "$fusion" != none ] && echo "$emb")a$(echo "$alpha" | sed 's/^0\.//; s/\.0$//')"
        cmd="qsub -N $short -o ${name}.qlog -v PRED_MODE=$mode,CHEM_FUSION=$fusion,CHEM_EMBED_DIM=$emb,CHEM_LOSS_ALPHA=$alpha,DROPOUT=$DROPOUT,WD=$WD $script"
        echo "$cmd"
        [ -z "$DRY_RUN" ] && $cmd
      done
    done
    # capacity controls stay at alpha=0 only -- they're about architecture capacity, not loss weighting
    if [ -n "$WITH_CONTROL" ]; then
      for v in late_mlp:0 early_concat:0; do
        fusion=${v%%:*}; emb=${v##*:}
        name="${lc}_pool_${mode}_${fusion}${emb}${extra_tag}"
        short="${model:2:1}${mode:0:1}${fusion:0:1}${emb}ctl"
        cmd="qsub -N $short -o ${name}.qlog -v PRED_MODE=$mode,CHEM_FUSION=$fusion,CHEM_EMBED_DIM=$emb,CHEM_LOSS_ALPHA=0,DROPOUT=$DROPOUT,WD=$WD $script"
        echo "$cmd"
        [ -z "$DRY_RUN" ] && $cmd
      done
    fi
  done
done
