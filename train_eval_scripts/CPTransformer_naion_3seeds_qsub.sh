#!/bin/bash -l
#$ -N cptransformer_naion_3seeds
#$ -P nsf-energize
#$ -l h_rt=01:30:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cptransformer_naion_3seeds.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cptransformer () {
  local seed=$1 dataset=$2
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPTransformer_NAION_seed${seed}"
  local log="${RESULTS_DIR}/CPTransformer_Naion_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPTransformer | dataset=$dataset seed=$seed ==="
  accelerate launch --num_processes 1 --main_process_port 25224 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPTransformer --model CPTransformer --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 128 --d_ff 128 --batch_size 32 --learning_rate 5e-05 \
    --train_epochs 100 --model_comment "CPTransformer_Naion_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset "$dataset" --num_workers 4 \
    --e_layers 4 --lstm_layers 6 --d_layers 9 --patience 5 --n_heads 4 \
    --early_cycle_threshold 100 --dropout 0.1 --lradj constant --loss MSE \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

run_cptransformer 2021 NAion
run_cptransformer 42   NAion42
run_cptransformer 2024 NAion2024

echo "=== Best model performance lines across all 3 seeds ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPTransformer_Naion_seed*.log
