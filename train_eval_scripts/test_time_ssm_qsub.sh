#!/bin/bash -l
#$ -N test_time_ssm
#$ -P nsf-energize
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o test_time_ssm.qlog

# Checks for the recorded-time-step code as a batch job (test_time_ssm.py): CPU checks, CUDA scan == ref for
# TimeS4DMixer / TimeMambaMixer, real-size fwd+bwd time / memory, and Dataset_pooled_dt == Dataset_pooled on the
# Li-ion val cells (+ tau stats with DT_C).
#   qsub train_eval_scripts/test_time_ssm_qsub.sh               -> test_time_ssm.qlog (ends with ALL PASSED or the failures)
#   qsub -v DT_C=<c>,DT_CLIP=<clip> train_eval_scripts/test_time_ssm_qsub.sh   (tau stats with the real constant)
module load gcc/12.2.0
module load cuda/12.2
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python test_time_ssm.py --gpu --data --dt_c "${DT_C:-20}" --dt_clip "${DT_CLIP:-0}"
