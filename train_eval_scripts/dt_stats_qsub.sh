#!/bin/bash -l
#$ -N dt_stats
#$ -P nsf-energize
#$ -l h_rt=04:00:00
#$ -pe omp 8
#$ -j y
#$ -o dt_stats.qlog

# CPU-only Δt stats pass over the Li-ion cells (dt_stats.py), two passes:
#   cycles 1..100 (what the <=100 runs see)            -> dt_stats_Li-ion_100/
#   every cycle up to eol-1 (full history, for later)   -> dt_stats_Li-ion_all/
#   qsub train_eval_scripts/dt_stats_qsub.sh   -> dt_stats.qlog (each pass ends with "proposed c" and the per-dataset table)
# Skip the full-history pass with: qsub -v ONLY100=1 train_eval_scripts/dt_stats_qsub.sh
module load gcc/12.2.0
cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate
python dt_stats.py --root_path ./dataset --workers 8 --max_cycles 100 2>&1 | grep -v "WARNING: pkl file not found"
[ "${ONLY100:-0}" = 1 ] || python dt_stats.py --root_path ./dataset --workers 8 --max_cycles 0 2>&1 | grep -v "WARNING: pkl file not found"
