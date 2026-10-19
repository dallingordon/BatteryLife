#!/bin/bash -l
# test_hiergen.py on a GPU (CPU S4D checks + vanilla Mamba on CUDA + the real cache).  -> test_hiergen.qlog
#$ -P nsf-energize
#$ -N test_hiergen
#$ -o test_hiergen.qlog
#$ -j y
#$ -l h_rt=01:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
module load gcc/12.2.0
module load cuda/12.2
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python test_hiergen.py --gpu --data
