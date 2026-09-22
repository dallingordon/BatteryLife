#!/bin/bash -l
#$ -N cpmlp_znion_3seeds_geobins
#$ -P nsf-energize
#$ -l h_rt=01:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cpmlp_znion_3seeds_geobins.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cpmlp () {
  local seed=$1 dataset=$2
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMLP_ZNION_geobins_seed${seed}"
  local log="${RESULTS_DIR}/CPMLP_geobins_Znion_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPMLP geo_bins | dataset=$dataset seed=$seed ==="
  accelerate launch --num_processes 1 --main_process_port 20441 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPMLP --model CPMLP --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 64 --d_ff 64 --batch_size 128 --learning_rate 5e-04 \
    --train_epochs 100 --model_comment "CPMLP_geobins_Znion_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset "$dataset" --num_workers 4 \
    --e_layers 5 --lstm_layers 2 --d_layers 9 --patience 5 --n_heads 8 \
    --early_cycle_threshold 100 --dropout 0.1 --lradj constant --loss MSE \
    --prediction_mode geo_bins --geo_bin_tol 0.15 --geo_bin_min 1 --geo_bin_max 3842 \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

run_cpmlp 2021 ZN-coin
run_cpmlp 42   ZN-coin42
run_cpmlp 2024 ZN-coin2024

echo "=== Best model performance lines across all 3 seeds (geo_bins) ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPMLP_geobins_Znion_seed*.log
