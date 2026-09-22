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
#   CHEM_FUSION     none (default) | late_mlp | early_concat
#                   late_mlp = output head over [features ; chemistry embedding]
#                   early_concat = chemistry embedding concatenated onto the raw input before the model's first layer
#   CHEM_EMBED_DIM  16 (default); 0 = same architecture but NO chemistry input (capacity control; a TRUE control for
#                   early_concat -- identical param count to CHEM_FUSION=none -- but not for late_mlp, whose E=0 head
#                   still has the MLP shape)
#   CHEM_LOSS_ALPHA 0.0 (default, off). Per-chemistry loss weighting: sample in chemistry c weighted by
#                   N_c^-alpha, renormalized to mean 1. 1.0 = chemistries contribute equally to the loss;
#                   0.5 = softer sqrt balancing. Orthogonal to CHEM_FUSION; composes with --weighted_loss.
#   DROPOUT         0 (default -- was hardcoded at 0 before 9/22, the Li-ion row's tuned value; CALB/Zn-ion/
#                   Na-ion's own per-chemistry tuned dropout was nonzero, 0.05-0.1, so this had never been
#                   swept for pooled training)
#   WD              0.0 (default -- weight decay was never set for pooled training before 9/22)
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
CHEM_LOSS_ALPHA=${CHEM_LOSS_ALPHA:-0.0}
DROPOUT=${DROPOUT:-0}
WD=${WD:-0.0}
SEEDS=${SEEDS:-"2021 42 2024"}
EPOCHS=${EPOCHS:-100}
POOLED_CHEMS=${POOLED_CHEMS:-}

TAG=""
[ "$PRED_MODE" = "geo_bins" ] && TAG="${TAG}_geobins"
[ "$CHEM_FUSION" != "none" ] && TAG="${TAG}_${CHEM_FUSION}E${CHEM_EMBED_DIM}"
[ "$CHEM_LOSS_ALPHA" != "0.0" ] && [ "$CHEM_LOSS_ALPHA" != "0" ] && TAG="${TAG}_alpha${CHEM_LOSS_ALPHA}"
[ "$DROPOUT" != "0" ] && [ "$DROPOUT" != "0.0" ] && TAG="${TAG}_drop${DROPOUT}"
[ "$WD" != "0.0" ] && [ "$WD" != "0" ] && TAG="${TAG}_wd${WD}"
EXTRA_ARGS="--prediction_mode $PRED_MODE --chem_fusion $CHEM_FUSION --chem_embed_dim $CHEM_EMBED_DIM --chem_loss_alpha $CHEM_LOSS_ALPHA --dropout $DROPOUT --wd $WD"
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
    --early_cycle_threshold 100 --lradj constant --loss MSE \
    --pooled --pooled_split_seed "$seed" $EXTRA_ARGS \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
  run_pooled "$seed"
done

echo "=== Pooled per-chemistry results (variant tag: '${TAG}') ==="
grep -h "Pooled single-checkpoint\|Pooled per-chem-best-val\|=== Pooled per-chemistry" "$RESULTS_DIR"/CPMLP_Pooled${TAG}_seed*.log
