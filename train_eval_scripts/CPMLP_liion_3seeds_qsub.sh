#!/bin/bash -l
#$ -N cpmlp_liion_3seeds
#$ -P nsf-energize
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cpmlp_liion_3seeds.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

run_cpmlp () {
  local seed=$1
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMLP_MIXLARGE_seed${seed}"
  local log="${RESULTS_DIR}/CPMLP_Liion_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPMLP | dataset=MIX_large seed=$seed ==="
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 20438 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPMLP --model CPMLP --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 32 --d_ff 256 --batch_size 32 --learning_rate 5e-05 \
    --train_epochs 100 --model_comment "CPMLP_Liion_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset MIX_large --num_workers 4 \
    --e_layers 12 --lstm_layers 2 --d_layers 7 --patience 5 --n_heads 8 \
    --early_cycle_threshold 100 --dropout 0 --lradj constant --loss MSE \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

run_cpmlp 2021
run_cpmlp 42
run_cpmlp 2024

echo "=== Best model performance lines across all 3 seeds ==="
grep -h "Best model performance" "$RESULTS_DIR"/CPMLP_Liion_seed*.log
