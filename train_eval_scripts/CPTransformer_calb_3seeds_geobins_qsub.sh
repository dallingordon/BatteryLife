#!/bin/bash -l
#$ -N cptransformer_calb_3seeds_geobins
#$ -P nsf-energize
#$ -l h_rt=01:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cptransformer_calb_3seeds_geobins.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cptransformer () {
  local seed=$1 dataset=$2 batch_size=$3 d_model=$4 d_ff=$5 e_layers=$6 d_layers=$7 dropout=$8 lr=$9
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPTransformer_CALB_geobins_seed${seed}"
  local log="${RESULTS_DIR}/CPTransformer_geobins_${dataset}_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPTransformer geo_bins | dataset=$dataset seed=$seed bs=$batch_size d_model=$d_model d_ff=$d_ff e_layers=$e_layers d_layers=$d_layers dropout=$dropout lr=$lr ==="

  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 25219 run_main.py \
    --task_name classification \
    --data Dataset_original \
    --is_training 1 \
    --root_path ./dataset \
    --model_id CPTransformer \
    --model CPTransformer \
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
    --model_comment "CPTransformer_geobins_${dataset}_s${seed}" \
    --accumulation_steps 1 \
    --charge_discharge_length 300 \
    --dataset "$dataset" \
    --num_workers 4 \
    --e_layers "$e_layers" \
    --lstm_layers 6 \
    --d_layers "$d_layers" \
    --patience 5 \
    --n_heads 4 \
    --early_cycle_threshold 100 \
    --dropout "$dropout" \
    --lradj constant \
    --loss MSE \
    --prediction_mode geo_bins --geo_bin_tol 0.15 --geo_bin_min 1 --geo_bin_max 3842 \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

# seed  dataset    batch(doubled) d_model d_ff e_layers d_layers dropout  lr
run_cptransformer 2021 CALB      16  64  256  9 9 0.1  5e-05
run_cptransformer 42   CALB42   128 256  256  6 7 0.05 5e-05
run_cptransformer 2024 CALB2024   8 128  256  7 6 0    5e-05

echo "=== Best model performance lines across all 3 seeds (geo_bins) ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPTransformer_geobins_CALB*.log
