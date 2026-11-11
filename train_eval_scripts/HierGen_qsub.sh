#!/bin/bash -l
# HierGen (models/HierGen.py) training + final latent / full rollouts (train_hiergen.py). Env vars (all optional):
#   CYCLE_LAYER READ_LAYER   vanilla (default) | s4d          N_CYCLE N_READ   4 4       D_MODEL 128
#   EPOCHS 200  LR 5e-4  BATCH_CELLS 8  READ_CYCLES 32  W_SOH 1  W_LATENT 1  W_READ 1  PREFIX 100  MAX_ROLLOUT 4000
#   SEEDS "2021"   TAG (default <cycle>_<read>)   EXTRA (more train_hiergen.py flags, quoted)
#   UNROLL_K 0  UNROLL_W 1  READ_VNOISE 0  READ_QNOISE 0  FINAL_MODES "latent full fullcc"  FINAL_SPLITS "val test"
#   GEN_K 0  GEN_W 1  GEN_READ 4  SELECT_MODE latent (or full)  SELECT_CELLS 32      (11/10 generated-history unroll)
#   READ_SELFGEN 0  SELFGEN_SAMPLE 0  SELFGEN_W 1   INIT_FROM "" (checkpoint, {seed} filled in)   (11/11 self-gen reading)
#   D_FF 256  GRAD_CKPT 0  WARMUP 2  EVAL_EVERY 10  WD 1e-2
#   EVAL_ONLY 0 (1 = load checkpoints/HierGen_<TAG>_seed<S>/model.pt, only run the final eval)
#   LOG_SUFFIX "" (appended to the log name, e.g. _evalcc, so an eval-only rerun does not overwrite the training log)
# Log: results_logs/HierGen_<TAG>_seed<S>.log ; checkpoint + rollouts + trajectories_test.png in checkpoints/HierGen_<TAG>_seed<S>/
#$ -P nsf-energize
#$ -N HierGen
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=40G
# RTXP6000 (scc-703 etc.) is Blackwell, compute capability 12.0: torch 2.2+cu121 and the selective_scan build have no sm_120 kernels ("no kernel image is available", 4/4 jobs on 11/10)
#$ -l gpu_type=!RTXP6000
#$ -pe omp 4
module load gcc/12.2.0
module load cuda/12.2
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
mkdir -p results_logs
CYCLE_LAYER=${CYCLE_LAYER:-vanilla}
READ_LAYER=${READ_LAYER:-vanilla}
TAG=${TAG:-${CYCLE_LAYER}_${READ_LAYER}}
for seed in ${SEEDS:-2021}; do
  python train_hiergen.py --seed "$seed" --tag "$TAG" --cycle_layer "$CYCLE_LAYER" --read_layer "$READ_LAYER" \
    --n_cycle_layers "${N_CYCLE:-4}" --n_read_layers "${N_READ:-4}" --d_model "${D_MODEL:-128}" --d_ff "${D_FF:-256}" \
    --grad_ckpt "${GRAD_CKPT:-0}" --warmup_epochs "${WARMUP:-2}" --eval_every "${EVAL_EVERY:-10}" --wd "${WD:-1e-2}" \
    --read_selfgen "${READ_SELFGEN:-0}" --selfgen_sample "${SELFGEN_SAMPLE:-0}" --selfgen_w "${SELFGEN_W:-1}" \
    --init_from "${INIT_FROM:-}" \
    --epochs "${EPOCHS:-200}" --lr "${LR:-5e-4}" --batch_cells "${BATCH_CELLS:-8}" --read_cycles "${READ_CYCLES:-32}" \
    --w_soh "${W_SOH:-1}" --w_latent "${W_LATENT:-1}" --w_read "${W_READ:-1}" --prefix "${PREFIX:-100}" \
    --max_rollout_cycles "${MAX_ROLLOUT:-4000}" --unroll_k "${UNROLL_K:-0}" --unroll_w "${UNROLL_W:-1}" \
    --read_vnoise "${READ_VNOISE:-0}" --read_qnoise "${READ_QNOISE:-0}" \
    --gen_k "${GEN_K:-0}" --gen_w "${GEN_W:-1}" --gen_read "${GEN_READ:-4}" \
    --select_mode "${SELECT_MODE:-latent}" --select_cells "${SELECT_CELLS:-32}" --final_modes ${FINAL_MODES:-latent full fullcc} \
    --final_splits ${FINAL_SPLITS:-val test} --eval_only "${EVAL_ONLY:-0}" \
    ${EXTRA} 2>&1 | tee "results_logs/HierGen_${TAG}_seed${seed}${LOG_SUFFIX}.log"
done
