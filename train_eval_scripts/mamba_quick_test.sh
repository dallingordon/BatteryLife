#!/bin/bash
# Fast end-to-end check of CPMamba on the full-timescale loader: 2 epochs, 5 prefixes per cell, prefixes capped at
# 300 cycles, Li-ion left out so loading is quick. Both mixer types x regression/geo_bins, plus one late_mlp run.
# Run inside an interactive GPU session with compute capability >= 7.0, from the repo root:
#   qrsh -P nsf-energize -l gpus=1 -l gpu_c=7.0 -l h_rt=2:00:00 -pe omp 4
#   bash train_eval_scripts/mamba_quick_test.sh
# Stops at the first failing variant. Full output: /tmp/quick_mamba_*.out
cd "$(dirname "$0")/.." || exit 1
for layer in vanilla mamba_init; do
  for v in regression:none geo_bins:none regression:late_mlp; do
    mode=${v%%:*}; fusion=${v##*:}
    [ "$layer" = "mamba_init" ] && [ "$fusion" != "none" ] && continue
    out=/tmp/quick_mamba_${layer}_${mode}_${fusion}.out
    echo "################ CPMamba | $layer | $mode | fusion=$fusion"
    EPOCHS=2 SEEDS=2021 K=5 MAX_CYCLES=300 POOLED_CHEMS="CALB Zn-ion Na-ion" ACCUM=4 PRINT_EVERY=100 \
      MAMBA_LAYER=$layer PRED_MODE=$mode CHEM_FUSION=$fusion \
      bash train_eval_scripts/CPMamba_full_qsub.sh > "$out" 2>&1 \
      || { echo "FAILED - see $out"; tail -30 "$out"; exit 1; }
    grep -q "Pooled per-chemistry results | chemistries" "$out" || { echo "NO RESULTS - see $out"; tail -30 "$out"; exit 1; }
    grep -h "epoch 1:\|Epoch: 2 cost time\|Pooled single-checkpoint" "$out" | cut -c1-160
  done
done
echo "ALL VARIANTS RAN"
