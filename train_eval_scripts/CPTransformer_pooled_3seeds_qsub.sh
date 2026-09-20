#!/bin/bash -l
#$ -N cptransformer_pooled_3seeds
#$ -P nsf-energize
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cptransformer_pooled_3seeds.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

# ONE CPTransformer trained on all four chemistries pooled (Li-ion + CALB + Zn-ion + Na-ion), no chemistry conditioning yet.
# Val/test are evaluated per chemistry on the same cells as the per-chemistry baselines.
# Seed s also selects the data split for CALB/Zn-ion/Na-ion (--pooled_split_seed), like the paper's seeds do.
# Hyperparameters: Li-ion row of assets/Selected_hyperparameters.md, batch size doubled for single GPU (repo convention).
# NOTE: h_rt is a guess - check epoch time on the first run and adjust.
run_pooled () {
  local seed=$1
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPTransformer_POOLED_seed${seed}"
  local log="${RESULTS_DIR}/CPTransformer_Pooled_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPTransformer | POOLED seed=$seed (split_seed=$seed) ==="
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 25218 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPTransformer --model CPTransformer --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 256 --d_ff 64 --batch_size 256 --learning_rate 5e-05 \
    --train_epochs 100 --model_comment "CPTransformer_Pooled_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset POOLED --num_workers 4 \
    --e_layers 1 --lstm_layers 2 --d_layers 12 --patience 5 --n_heads 8 \
    --early_cycle_threshold 100 --dropout 0 --lradj constant --loss MSE \
    --pooled --pooled_split_seed "$seed" \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

run_pooled 2021
run_pooled 42
run_pooled 2024

echo "=== Pooled per-chemistry results, all 3 seeds ==="
grep -h "Pooled single-checkpoint\|Pooled per-chem-best-val\|=== Pooled per-chemistry" "$RESULTS_DIR"/CPTransformer_Pooled_seed*.log
