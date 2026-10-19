#!/bin/bash -l
# Recompute cycle life from per-cycle capacity for every Li-ion cell and compare with the json labels (capacity_check.py).
#   qsub train_eval_scripts/capacity_check_qsub.sh                         -> capacity_check.qlog, figures/capacity_check/
#   qsub -v SPLITS="train val test" train_eval_scripts/capacity_check_qsub.sh
#$ -P nsf-energize
#$ -N capcheck
#$ -o capacity_check.qlog
#$ -j y
#$ -l h_rt=4:00:00
#$ -l mem_per_core=8G
#$ -pe omp 2
module load gcc/12.2.0
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
python capacity_check.py --compare_all --no_plot --splits ${SPLITS:-train}
