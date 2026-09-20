#!/bin/bash -l
#$ -N cpmlp_pooled_3seeds
#$ -P nsf-energize
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 4
#$ -j y
#$ -o cpmlp_pooled_3seeds.qlog

# ONE CPMLP trained on all four chemistries pooled (Li-ion + CALB + Zn-ion + Na-ion).
# Val/test are evaluated per chemistry on the same cells as the per-chemistry baselines.
# Seed s also selects the data split for CALB/Zn-ion/Na-ion (--pooled_split_seed), like the paper's seeds do.
# Hyperparameters: Li-ion row of assets/Selected_hyperparameters.md, batch size doubled for single GPU (repo convention).
# NOTE: h_rt is a guess - check epoch time on the first run and adjust.
#
# Variants are chosen with environment variables (use submit_pooled_sweep.sh, or `qsub -v VAR=val,... this_script`):
#   PRED_MODE       regression (default) | geo_bins
#   CHEM_FUSION     none (default) | late_mlp      (late_mlp = output head over [features ; chemistry embedding])
#   CHEM_EMBED_DIM  16 (default); 0 with late_mlp = same head but NO chemistry input (capacity control)
#   SEEDS           "2021 42 2024" (default)
#   EPOCHS          100 (default)
#   POOLED_CHEMS    unset = all four; e.g. "CALB Zn-ion Na-ion" for a fast test (logs get a _quick tag)

module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_train_p100/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

PRED_MODE=${PRED_MODE:-regression}
CHEM_FUSION=${CHEM_FUSION:-none}
CHEM_EMBED_DIM=${CHEM_EMBED_DIM:-16}
SEEDS=${SEEDS:-"2021 42 2024"}
EPOCHS=${EPOCHS:-100}
POOLED_CHEMS=${POOLED_CHEMS:-}

TAG=""
[ "$PRED_MODE" = "geo_bins" ] && TAG="${TAG}_geobins"
[ "$CHEM_FUSION" != "none" ] && TAG="${TAG}_${CHEM_FUSION}E${CHEM_EMBED_DIM}"
EXTRA_ARGS="--prediction_mode $PRED_MODE --chem_fusion $CHEM_FUSION --chem_embed_dim $CHEM_EMBED_DIM"
if [ -n "$POOLED_CHEMS" ]; then
  TAG="${TAG}_quick"
  EXTRA_ARGS="$EXTRA_ARGS --pooled_chemistries $POOLED_CHEMS"
fi

run_pooled () {
  local seed=$1
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/CPMLP_POOLED${TAG}_seed${seed}"
  local log="${RESULTS_DIR}/CPMLP_Pooled${TAG}_seed${seed}.log"
  mkdir -p "$ckpt"
  echo "=== CPMLP | POOLED${TAG} seed=$seed (split_seed=$seed) epochs=$EPOCHS ==="
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 20441 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id CPMLP --model CPMLP --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model 32 --d_ff 256 --batch_size 32 --learning_rate 5e-05 \
    --train_epochs "$EPOCHS" --model_comment "CPMLP_Pooled${TAG}_s${seed}" --accumulation_steps 1 \
    --charge_discharge_length 300 --dataset POOLED --num_workers 4 \
    --e_layers 12 --lstm_layers 2 --d_layers 7 --patience 5 --n_heads 8 \
    --early_cycle_threshold 100 --dropout 0 --lradj constant --loss MSE \
    --pooled --pooled_split_seed "$seed" $EXTRA_ARGS \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
  run_pooled "$seed"
done

echo "=== Pooled per-chemistry results (variant tag: '${TAG}') ==="
grep -h "Pooled single-checkpoint\|Pooled per-chem-best-val\|=== Pooled per-chemistry" "$RESULTS_DIR"/CPMLP_Pooled${TAG}_seed*.log
