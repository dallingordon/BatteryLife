#!/bin/bash -l
#$ -N test_s4d
#$ -P nsf-energize
#$ -l h_rt=01:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o test_s4d.qlog

# GPU checks for models/S4D.py as a batch job (when no qrsh is available):
#   scan backend (CUDA selective scan, complex A) == fft, speed side by side at 100 cycles,
#   and the scan backend at full-history lengths (1000 / 2000 / 3842 cycles).
#   qsub train_eval_scripts/test_s4d_qsub.sh        -> test_s4d.qlog (ends with ALL PASSED or the failures)
module load gcc/12.2.0
module load cuda/12.2
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python test_s4d.py --gpu --long
