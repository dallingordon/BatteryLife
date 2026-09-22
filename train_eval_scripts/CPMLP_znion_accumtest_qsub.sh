#!/bin/bash -l
#$ -N cpmlp_znion_accumtest
#$ -P nsf-energize
#$ -l h_rt=00:30:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cpmlp_znion_accumtest.qlog

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

# Batch-size hypothesis test for Zn-ion regression: table batch_size (64) at
# accumulation_steps=2 instead of doubled batch_size (128) at accumulation_steps=1.
# Same effective global batch (128) and same optimizer-step count per epoch as the
# paper's 2-GPU DDP run either way if the math is right - this run is here to check
# whether that's actually true in practice, or whether something else explains why
# CPMLP_Znion_seed*.log (doubled-batch) badly missed published numbers while geo_bins
# on the identical doubled-batch setting did not.
seed=2021
ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMLP_ZNION_accumtest_seed${seed}"
log="${RESULTS_DIR}/CPMLP_znion_accumtest_seed${seed}.log"
mkdir -p "$ckpt"

accelerate launch --num_processes 1 --main_process_port 20436 run_main.py \
  --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
  --model_id CPMLP --model CPMLP --features MS --seq_len 1 --label_len 50 --factor 3 \
  --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
  --d_model 64 --d_ff 64 --batch_size 64 --learning_rate 0.0005 \
  --train_epochs 100 --model_comment "CPMLP_znion_accumtest_s${seed}" --accumulation_steps 2 \
  --charge_discharge_length 300 --dataset Znion --num_workers 4 \
  --e_layers 5 --lstm_layers 2 --d_layers 9 --patience 5 --n_heads 8 \
  --early_cycle_threshold 100 --dropout 0.1 --lradj constant --loss MSE \
  --checkpoints "$ckpt" 2>&1 | tee "$log"

echo "=== Best model performance (accumtest) ==="
grep -h "Best model performance" "$log"
