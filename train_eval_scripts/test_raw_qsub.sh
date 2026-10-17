#!/bin/bash -l
#$ -N test_raw
#$ -P nsf-energize
#$ -l h_rt=03:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o test_raw.qlog

# Checks for the raw-record code as a batch job (test_raw.py): CPU checks (raw path == grid path, variable cycle
# lengths, padding), CUDA timing / peak memory at raw lengths, and Dataset_pooled_raw vs Dataset_pooled on the Li-ion
# val cells (+ raw tau stats per dataset, accelerate-prepared loader).
#   qsub train_eval_scripts/test_raw_qsub.sh                      -> test_raw.qlog (ends with ALL PASSED or the failures)
#   qsub -v DT_C=5,DT_CLIP=100 train_eval_scripts/test_raw_qsub.sh
module load gcc/12.2.0
module load cuda/12.2
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python test_raw.py --gpu --data --dt_c "${DT_C:-5}" --dt_clip "${DT_CLIP:-100}"
