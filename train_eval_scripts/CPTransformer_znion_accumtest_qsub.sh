#!/bin/bash -l
#$ -N cptransformer_znion_accumtest
#$ -P nsf-energize
#$ -l h_rt=00:30:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cptransformer_znion_accumtest.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

seed=2021
ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPTransformer_ZNION_accumtest_seed${seed}"
log="${RESULTS_DIR}/CPTransformer_znion_accumtest_seed${seed}.log"
mkdir -p "$ckpt"

accelerate launch --num_processes 1 --main_process_port 20437 run_main.py \
  --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
  --model_id CPTransformer --model CPTransformer --features MS --seq_len 1 --label_len 50 --factor 3 \
  --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
  --d_model 64 --d_ff 128 --batch_size 32 --learning_rate 0.001 \
  --train_epochs 100 --model_comment "CPTransformer_znion_accumtest_s${seed}" --accumulation_steps 2 \
  --charge_discharge_length 300 --dataset Znion --num_workers 4 \
  --e_layers 1 --lstm_layers 2 --d_layers 11 --patience 5 --n_heads 8 \
  --early_cycle_threshold 100 --dropout 0 --lradj constant --loss MSE \
  --checkpoints "$ckpt" 2>&1 | tee "$log"

echo "=== Best model performance (accumtest) ==="
grep -h "Best model performance" "$log"
