#!/bin/bash
set -e

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cpmlp () {
  local seed=$1 dataset=$2 batch_size=$3 d_model=$4 d_ff=$5 e_layers=$6 d_layers=$7 dropout=$8 lr=$9
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMLP_CALB_seed${seed}"
  local log="${RESULTS_DIR}/CPMLP_${dataset}_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPMLP | dataset=$dataset seed=$seed bs=$batch_size d_model=$d_model d_ff=$d_ff e_layers=$e_layers d_layers=$d_layers dropout=$dropout lr=$lr ==="

  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 20435 run_main.py \
    --task_name classification \
    --data Dataset_original \
    --is_training 1 \
    --root_path ./dataset \
    --model_id CPMLP \
    --model CPMLP \
    --features MS \
    --seq_len 1 \
    --label_len 50 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1 \
    --seed "$seed" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$batch_size" \
    --learning_rate "$lr" \
    --train_epochs 100 \
    --model_comment "CPMLP_${dataset}_s${seed}" \
    --accumulation_steps 1 \
    --charge_discharge_length 300 \
    --dataset "$dataset" \
    --num_workers 4 \
    --e_layers "$e_layers" \
    --lstm_layers 2 \
    --d_layers "$d_layers" \
    --patience 5 \
    --n_heads 8 \
    --early_cycle_threshold 100 \
    --dropout "$dropout" \
    --lradj constant \
    --loss MSE \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

# seed  dataset    batch(doubled) d_model d_ff e_layers d_layers dropout  lr
run_cpmlp 2021 CALB      16  32  32  12 6 0.1  5e-05
run_cpmlp 42   CALB42     8 128 128  7 9 0.05  5e-05
run_cpmlp 2024 CALB2024  16 256 128 12 6 0     5e-05

echo "=== Best model performance lines across all 3 seeds ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPMLP_CALB*.log
