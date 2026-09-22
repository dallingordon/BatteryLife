#!/bin/bash -l
#$ -N cptransformer_liion_3seeds_geobins
#$ -P nsf-energize
#$ -l h_rt=04:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cptransformer_liion_3seeds_geobins.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cptransformer () {
  local seed=$1
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPTransformer_MIXLARGE_geobins_seed${seed}"
  local log="${RESULTS_DIR}/CPTransformer_geobins_Liion_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPTransformer geo_bins | dataset=MIX_large seed=$seed ==="
  accelerate launch --num_processes 1 --main_process_port 25221 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPTransformer --model CPTransformer --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 256 --d_ff 64 --batch_size 256 --learning_rate 5e-05 \
    --train_epochs 100 --model_comment "CPTransformer_geobins_Liion_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset MIX_large --num_workers 4 \
    --e_layers 1 --lstm_layers 6 --d_layers 12 --patience 5 --n_heads 4 \
    --early_cycle_threshold 100 --dropout 0 --lradj constant --loss MSE \
    --prediction_mode geo_bins --geo_bin_tol 0.15 --geo_bin_min 1 --geo_bin_max 3842 \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

run_cptransformer 2021
run_cptransformer 42
run_cptransformer 2024

echo "=== Best model performance lines across all 3 seeds (geo_bins) ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPTransformer_geobins_Liion_seed*.log
