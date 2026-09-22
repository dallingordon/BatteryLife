#!/bin/bash
# Fast end-to-end check of every pooled variant (2 epochs, seed 2021, Li-ion left out so loading is quick).
# Run inside an interactive GPU session (qrsh) from the repo root:  bash train_eval_scripts/pooled_quick_test.sh
# Stops at the first failing variant. Logs: results_logs/*_quick_seed2021.log
cd "$(dirname "$0")/.." || exit 1
for model in CPMLP CPTransformer; do
  for mode in regression geo_bins; do
    for v in none:0 late_mlp:16 late_mlp:0 early_concat:16 early_concat:0; do
      fusion=${v%%:*}; emb=${v##*:}
      echo "################ $model | $mode | fusion=$fusion E=$emb"
      EPOCHS=2 SEEDS=2021 POOLED_CHEMS="CALB Zn-ion Na-ion" PRED_MODE=$mode CHEM_FUSION=$fusion CHEM_EMBED_DIM=$emb \
        bash train_eval_scripts/${model}_pooled_3seeds_qsub.sh > /tmp/quick_${model}_${mode}_${fusion}${emb}.out 2>&1 \
        || { echo "FAILED - see /tmp/quick_${model}_${mode}_${fusion}${emb}.out"; tail -30 /tmp/quick_${model}_${mode}_${fusion}${emb}.out; exit 1; }
      grep -h "Pooled single-checkpoint" /tmp/quick_${model}_${mode}_${fusion}${emb}.out | cut -c1-140
    done
  done
done
echo "ALL VARIANTS RAN"
