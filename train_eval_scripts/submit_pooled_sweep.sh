#!/bin/bash
# Submit the pooled-training sweep (run on an SCC login node from the repo root):
#   bash train_eval_scripts/submit_pooled_sweep.sh              # submit
#   DRY_RUN=1 bash train_eval_scripts/submit_pooled_sweep.sh    # just print the qsub commands
#   WITH_CONTROL=1 ...                                          # also submit the capacity control (late_mlp, no chemistry input)
# Each job runs seeds 2021/42/2024 sequentially (seed also picks the CALB/Zn-ion/Na-ion split).
# Compare, per chemistry: pooled baseline (fusion none) vs chemistry-conditioned (late_mlp E=16) vs the per-chemistry results.
cd "$(dirname "$0")/.." || exit 1
E=${E:-16}
for model in CPMLP CPTransformer; do
  script=train_eval_scripts/${model}_pooled_3seeds_qsub.sh
  lc=$(echo "$model" | tr 'A-Z' 'a-z')
  for mode in regression geo_bins; do
    variants="none:0 late_mlp:$E"
    [ -n "$WITH_CONTROL" ] && variants="$variants late_mlp:0"
    for v in $variants; do
      fusion=${v%%:*}; emb=${v##*:}
      name="${lc}_pool_${mode}_${fusion}"
      [ "$fusion" = "late_mlp" ] && name="${name}${emb}"
      cmd="qsub -N $name -o ${name}.qlog -v PRED_MODE=$mode,CHEM_FUSION=$fusion,CHEM_EMBED_DIM=$emb $script"
      echo "$cmd"
      [ -z "$DRY_RUN" ] && $cmd
    done
  done
done
