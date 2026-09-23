#!/bin/bash -l
#$ -N cpmamba_full
#$ -P nsf-energize
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o cpmamba_full.qlog

# CPMamba trained on the FULL-TIMESCALE loader (prefixes of any length, all stored cycles), evaluated on the usual
# pooled val/test (published cells, prefixes 1..100, per chemistry).
#   gpu_c=7.0: the mamba-init fork's selective_scan kernels are not built for P100 (sm_60).
#   venv_mamba: see scc/setup_venv_mamba.sh (layered on /projectnb/textconv/dgordon/mamba/venv).
# NOTE: h_rt and every hyperparameter default below are first guesses -- check epoch time on the first run.
#
# Variants via environment variables (`qsub -v VAR=val,... this_script`, or VAR=val bash this_script in a qrsh):
#   MAMBA_LAYER     vanilla (default) | mamba_init
#   POOLED_CHEMS    "Li-ion" (default: lithium first); "" = all four
#   K               full_prefixes_per_cell, 50 (default); "" = every prefix
#   MAX_CYCLES      full_max_cycles, "" (default, no max)
#   ACCUM           gradient accumulation steps (train batch size is 1), 16 (default)
#   LR              1e-4 (default)
#   D_MODEL D_FF E_LAYERS N_LAYERS D_STATE     64 256 4 4 16 (defaults)
#   PRED_MODE       regression (default) | geo_bins
#   CHEM_FUSION     none (default) | late_mlp | early_concat;  CHEM_EMBED_DIM 16 (default)
#   DROPOUT WD      0 0.0 (defaults)
#   SEEDS           "2021" (default)
#   EPOCHS          50 (default);  PATIENCE 5 (default)
#   PRINT_EVERY     500 (default)

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
POOLED_CHEMS=${POOLED_CHEMS-Li-ion}
K=${K-50}
MAX_CYCLES=${MAX_CYCLES:-}
ACCUM=${ACCUM:-16}
LR=${LR:-1e-4}
D_MODEL=${D_MODEL:-64}
D_FF=${D_FF:-256}
E_LAYERS=${E_LAYERS:-4}
N_LAYERS=${N_LAYERS:-4}
D_STATE=${D_STATE:-16}
PRED_MODE=${PRED_MODE:-regression}
CHEM_FUSION=${CHEM_FUSION:-none}
CHEM_EMBED_DIM=${CHEM_EMBED_DIM:-16}
DROPOUT=${DROPOUT:-0}
WD=${WD:-0.0}
SEEDS=${SEEDS:-"2021"}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-5}
PRINT_EVERY=${PRINT_EVERY:-500}

TAG="_${MAMBA_LAYER}_K${K:-all}"
[ -n "$MAX_CYCLES" ] && TAG="${TAG}_max${MAX_CYCLES}"
[ -n "$POOLED_CHEMS" ] && TAG="${TAG}_$(echo "$POOLED_CHEMS" | tr -d ' -')"
[ "$PRED_MODE" = "geo_bins" ] && TAG="${TAG}_geobins"
[ "$CHEM_FUSION" != "none" ] && TAG="${TAG}_${CHEM_FUSION}E${CHEM_EMBED_DIM}"
[ "$DROPOUT" != "0" ] && [ "$DROPOUT" != "0.0" ] && TAG="${TAG}_drop${DROPOUT}"
[ "$WD" != "0.0" ] && [ "$WD" != "0" ] && TAG="${TAG}_wd${WD}"

EXTRA_ARGS="--prediction_mode $PRED_MODE --chem_fusion $CHEM_FUSION --chem_embed_dim $CHEM_EMBED_DIM --dropout $DROPOUT --wd $WD"
[ -n "$POOLED_CHEMS" ] && EXTRA_ARGS="$EXTRA_ARGS --pooled_chemistries $POOLED_CHEMS"
[ -n "$K" ] && EXTRA_ARGS="$EXTRA_ARGS --full_prefixes_per_cell $K"
[ -n "$MAX_CYCLES" ] && EXTRA_ARGS="$EXTRA_ARGS --full_max_cycles $MAX_CYCLES"

run_one () {
  local seed=$1
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMamba_FULL${TAG}_seed${seed}"
  local log="${RESULTS_DIR}/CPMamba_Full${TAG}_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPMamba | FULL${TAG} seed=$seed (split_seed=$seed) epochs=$EPOCHS ==="
  accelerate launch --num_processes 1 --main_process_port 20443 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPMamba --model CPMamba --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --e_layers "$E_LAYERS" --batch_size 32 --learning_rate "$LR" \
    --mamba_layer "$MAMBA_LAYER" --mamba_n_layers "$N_LAYERS" --mamba_d_state "$D_STATE" \
    --train_epochs "$EPOCHS" --model_comment "CPMamba_Full${TAG}_s${seed}" --accumulation_steps "$ACCUM" \
    --charge_discharge_length 300 --dataset POOLED --num_workers 4 --print_every "$PRINT_EVERY" \
    --patience "$PATIENCE" --early_cycle_threshold 100 --lradj constant --loss MSE \
    --pooled --pooled_split_seed "$seed" --full_timescale $EXTRA_ARGS \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
  run_one "$seed"
done

echo "=== Pooled per-chemistry results (variant tag: '${TAG}') ==="
grep -h "Pooled single-checkpoint\|Pooled per-chem-best-val\|=== Pooled per-chemistry" "$RESULTS_DIR"/CPMamba_Full${TAG}_seed*.log
