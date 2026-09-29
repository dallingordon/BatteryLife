#!/bin/bash
# Smoke tests for everything new in sh_submit_9_29.sh. Run INSIDE an interactive GPU session, from the repo root:
#   qrsh -P nsf-energize -l gpus=1 -l gpu_c=7.0 -pe omp 4 -l h_rt=04:00:00
#   cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife && bash smoke_9_29.sh 2>&1 | tee smoke_9_29.out
# Quick runs leave out Li-ion (fast loading) and train 1-2 epochs; their logs get a _quick tag, so they never mix
# with real results. Prints PASS / FAIL per check and a summary at the end; full output of each check is in /tmp.
cd "$(dirname "$0")" || exit 1
QUICK="CALB Zn-ion Na-ion"
declare -a RESULTS

run_check () {   # name, required patterns joined by ';;' (ALL must appear in the output), command...
  local name=$1 patterns=$2; shift 2
  local out="/tmp/smoke_9_29_${name}.out"
  echo "################ $name"
  ( "$@" ) > "$out" 2>&1
  local ok=1 pat
  IFS=';' read -r -a _pats <<< "${patterns//;;/;}"
  for pat in "${_pats[@]}"; do grep -q -- "$pat" "$out" || ok=0; done
  grep -q "Traceback" "$out" && ok=0
  if [ "$ok" = 1 ]; then
    RESULTS+=("PASS  $name"); echo "PASS"
  else
    RESULTS+=("FAIL  $name  (see $out)"); echo "FAIL -- last lines of $out:"; tail -25 "$out"
  fi
}

# 1) Block 1: extra seeds reuse the paper's splits (seed 1 -> split 42). Plain geo_bins CPMLP, 1 epoch.
run_check block1_seed1_split "split_seed=42 | seed=1;;Pooled single-checkpoint" \
  env EPOCHS=1 SEEDS=1 POOLED_CHEMS="$QUICK" PRED_MODE=geo_bins WD=1e-3 bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh
grep -h "Pooled single-checkpoint" /tmp/smoke_9_29_block1_seed1_split.out | cut -c1-150

# 2) Block 2: dual heads (unconditioned and early_concat16), 2 epochs. Must print '| Head: ...' + Pooled[bins]/[reg] lines.
run_check block2_dual "Pooled\[reg\] single-checkpoint;;| Head: " \
  env EPOCHS=2 SEEDS=2021 POOLED_CHEMS="$QUICK" PRED_MODE=dual WD=1e-3 bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh
grep -h "single-checkpoint" /tmp/smoke_9_29_block2_dual.out | cut -c1-110
run_check block2_dual_e16 "Pooled\[reg\] single-checkpoint;;| Head: " \
  env EPOCHS=2 SEEDS=2021 POOLED_CHEMS="$QUICK" PRED_MODE=dual CHEM_FUSION=early_concat CHEM_EMBED_DIM=16 WD=1e-3 \
  bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh

# 3) Block 3 needs no new code (early_concat E=32/64 is an existing option) -- one quick build check for E=64.
run_check block3_e64 "Pooled single-checkpoint" \
  env EPOCHS=1 SEEDS=2021 POOLED_CHEMS="$QUICK" PRED_MODE=geo_bins CHEM_FUSION=early_concat CHEM_EMBED_DIM=64 WD=1e-3 \
  bash train_eval_scripts/CPMLP_pooled_3seeds_qsub.sh

# 4) Block 4: Mamba full loader with chem_loss_alpha at max100. Must print the per-chemistry 'loss weight' column.
run_check block4_mamba_alpha "loss weight;;Pooled single-checkpoint" \
  env EPOCHS=1 SEEDS=2021 POOLED_CHEMS="$QUICK" MAMBA_LAYER=vanilla PRED_MODE=geo_bins WD=1e-3 K=200 MAX_CYCLES=100 \
  CHEM_LOSS_ALPHA=0.5 bash train_eval_scripts/CPMamba_full_qsub.sh
grep -h -A5 "full-timescale train\] epoch 0" /tmp/smoke_9_29_block4_mamba_alpha.out | head -6

# 5) Block 5: Long Mamba unit tests (CPU + GPU, incl. the real-size time / memory probe), then a 1-epoch run.
run_check block5_longmamba_tests "ALL PASSED" \
  bash -lc "module load gcc/12.2.0 cuda/12.2 && source venv_mamba/bin/activate && python test_longmamba.py --gpu"
grep -h "100 cycles:" /tmp/smoke_9_29_block5_longmamba_tests.out
run_check block5_longmamba_run "Pooled single-checkpoint" \
  env EPOCHS=1 SEEDS=2021 POOLED_CHEMS="$QUICK" BOUNDARY=index bash train_eval_scripts/LongMamba_pooled_qsub.sh
grep -h "cost time" /tmp/smoke_9_29_block5_longmamba_run.out | head -2

echo; echo "======== SUMMARY"; printf '%s\n' "${RESULTS[@]}"
echo "If everything passed: DRY_RUN=1 bash sh_submit_9_29.sh (expect 11 + 6 + 6 + 16 + 18 = 57 jobs), then bash sh_submit_9_29.sh"
