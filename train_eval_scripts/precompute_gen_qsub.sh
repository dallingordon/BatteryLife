#!/bin/bash -l
# Build the HierGen cycle cache (precompute_gen_cells.py): Li-ion train/val/test, all cycles, 300-point grid in volts.
#   qsub train_eval_scripts/precompute_gen_qsub.sh        -> precompute_gen.qlog
#$ -P nsf-energize
#$ -N gencache
#$ -o precompute_gen.qlog
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
module load gcc/12.2.0
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
python precompute_gen_cells.py --root_path ./dataset --workers 8 --check
