#!/bin/bash -l
# HierGen (models/HierGen.py) training + final latent / full rollouts (train_hiergen.py). Env vars (all optional):
#   CYCLE_LAYER READ_LAYER   vanilla (default) | s4d          N_CYCLE N_READ   4 4       D_MODEL 128
#   EPOCHS 200  LR 5e-4  BATCH_CELLS 8  READ_CYCLES 32  W_SOH 1  W_LATENT 1  W_READ 1  PREFIX 100  MAX_ROLLOUT 4000
#   SEEDS "2021"   TAG (default <cycle>_<read>)   EXTRA (more train_hiergen.py flags, quoted)
# Log: results_logs/HierGen_<TAG>_seed<S>.log ; checkpoint + rollouts + trajectories_test.png in checkpoints/HierGen_<TAG>_seed<S>/
#$ -P nsf-energize
#$ -N HierGen
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
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
    --n_cycle_layers "${N_CYCLE:-4}" --n_read_layers "${N_READ:-4}" --d_model "${D_MODEL:-128}" \
    --epochs "${EPOCHS:-200}" --lr "${LR:-5e-4}" --batch_cells "${BATCH_CELLS:-8}" --read_cycles "${READ_CYCLES:-32}" \
    --w_soh "${W_SOH:-1}" --w_latent "${W_LATENT:-1}" --w_read "${W_READ:-1}" --prefix "${PREFIX:-100}" \
    --max_rollout_cycles "${MAX_ROLLOUT:-4000}" ${EXTRA} 2>&1 | tee "results_logs/HierGen_${TAG}_seed${seed}.log"
done
