#!/bin/bash -l
#$ -N longmamba_pooled
#$ -P nsf-energize
#$ -l h_rt=30:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o longmamba_pooled.qlog

# LongMamba (models/LongMamba.py): one Mamba stack over the raw per-point series of the early cycles
# (100 cycles x 300 points = ~30k steps, 3 channels per step), output CLS token at the end.
# Trained on the ORDINARY pooled loader (all four chemistries, prefixes 1..100 cycles -- the same data, splits and
# val/test as pooled CPMLP), so the comparison with CPMLP is architecture-only.
#   gpu_c=7.0: the mamba-init fork's selective_scan kernels are not built for P100 (sm_60).
#   venv_mamba: see scc/setup_venv_mamba.sh.
# NOTE: BATCH / ACCUM / h_rt are first guesses -- run `python test_longmamba.py --gpu` in a qrsh first; it prints
# fwd+bwd time and peak memory at 100 cycles for B = 1..32.
#
# Variants via environment variables (`qsub -v VAR=val,... this_script`, or VAR=val bash this_script in a qrsh):
#   MAMBA_LAYER     vanilla (default) | mamba_init
#   BOUNDARY        none (default) | shared | index   (cycle-boundary token in front of every cycle; see LongMamba.py)
#   PRED_MODE       geo_bins (default) | regression | dual
#   POOLED_CHEMS    "" / all = all four (default); e.g. "CALB Zn-ion Na-ion" for a fast test (logs get a _quick tag)
#   BATCH ACCUM     8 4 (defaults; effective batch 32 like CPMLP)
#   LR WD           1e-4 1e-3 (defaults)
#   D_MODEL N_LAYERS D_STATE     64 4 16 (defaults, same as CPMamba)
#   SEEDS           "2021 42 2024" (default); other seeds reuse the paper's 3 splits (see run_one)
#   EPOCHS PATIENCE 100 5 (defaults)
#   PRINT_EVERY     200 (default)

module load gcc/12.2.0      # selective_scan_cuda needs gcc 12's libstdc++
module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

MAMBA_LAYER=${MAMBA_LAYER:-vanilla}
BOUNDARY=${BOUNDARY:-none}
PRED_MODE=${PRED_MODE:-geo_bins}
POOLED_CHEMS=${POOLED_CHEMS:-}
[ "$POOLED_CHEMS" = all ] && POOLED_CHEMS=""   # "all" = every chemistry (qsub -v cannot reliably pass an empty value)
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
D_MODEL=${D_MODEL:-64}
N_LAYERS=${N_LAYERS:-4}
D_STATE=${D_STATE:-16}
SEEDS=${SEEDS:-"2021 42 2024"}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-5}
PRINT_EVERY=${PRINT_EVERY:-200}

TAG="_${MAMBA_LAYER}_b${BOUNDARY}"
[ "$PRED_MODE" = "geo_bins" ] && TAG="${TAG}_geobins"
[ "$PRED_MODE" = "dual" ] && TAG="${TAG}_dual"
[ "$WD" != "0.0" ] && [ "$WD" != "0" ] && TAG="${TAG}_wd${WD}"
EXTRA_ARGS="--prediction_mode $PRED_MODE --long_boundary $BOUNDARY --wd $WD"
if [ -n "$POOLED_CHEMS" ]; then
  TAG="${TAG}_quick"
  EXTRA_ARGS="$EXTRA_ARGS --pooled_chemistries $POOLED_CHEMS"
fi

run_one () {
  local seed=$1
  # --pooled_split_seed only has the paper's 3 splits (2021/42/2024); other seeds reuse one of them, cycling.
  local split_seed=$seed
  case "$seed" in 2021|42|2024) ;; *) local _splits=(2021 42 2024); split_seed=${_splits[$((seed % 3))]} ;; esac
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/LongMamba_POOLED${TAG}_seed${seed}"
  local log="${RESULTS_DIR}/LongMamba_Pooled${TAG}_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== LongMamba | POOLED${TAG} seed=$seed (split_seed=$split_seed) epochs=$EPOCHS batch=$BATCH x accum $ACCUM ==="
  accelerate launch --num_processes 1 --main_process_port 20445 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id LongMamba --model LongMamba --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model "$D_MODEL" --batch_size "$BATCH" --learning_rate "$LR" \
    --mamba_layer "$MAMBA_LAYER" --mamba_n_layers "$N_LAYERS" --mamba_d_state "$D_STATE" \
    --train_epochs "$EPOCHS" --model_comment "LongMamba_Pooled${TAG}_s${seed}" --accumulation_steps "$ACCUM" \
    --charge_discharge_length 300 --dataset POOLED --num_workers 4 --print_every "$PRINT_EVERY" \
    --patience "$PATIENCE" --early_cycle_threshold 100 --lradj constant --loss MSE \
    --pooled --pooled_split_seed "$split_seed" $EXTRA_ARGS \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
  run_one "$seed"
done

echo "=== Pooled per-chemistry results (variant tag: '${TAG}') ==="
grep -h "Pooled\(\[[a-z]*\]\)\? \(single-checkpoint\|per-chem-best-val\)\|=== Pooled per-chemistry" "$RESULTS_DIR"/LongMamba_Pooled${TAG}_seed*.log
