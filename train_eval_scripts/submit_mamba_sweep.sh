#!/bin/bash
# Submit the unified (all four chemistries pooled) CPMamba full-timescale sweep: mixer x chemistry conditioning,
# at the stage-1 winning setting (geo_bins, wd 1e-3, dropout 0). Run on an SCC login node from the repo root:
#   DRY_RUN=1 bash train_eval_scripts/submit_mamba_sweep.sh     # just print the qsub commands
#   H_RT=12:00:00 bash train_eval_scripts/submit_mamba_sweep.sh # override the wall-clock limit (set from the timing probe)
#   WITH_CONTROL=1 ...                                          # also the E=0 capacity controls (same shape, no chemistry input)
#   HORIZON_CONTROL=1 ...                                       # also train-horizon controls: same full-timescale trainer but
#                                                                 MAX_CYCLES=100 (prefixes 1..100, the range val/test and the
#                                                                 CPMLP/CPTransformer baselines use), fusion none, both mixers.
#                                                                 Separates "Mamba architecture" from "training on long histories".
# K defaults to 200 here (CPMamba_full_qsub.sh alone defaults to 50).
# Grid: MAMBA_LAYERS {vanilla, mamba_init} x FUSIONS {none, late_mlp E16, early_concat E16}  (+ E0 controls)
# Any other CPMamba_full_qsub.sh variable (K, ACCUM, LR, EPOCHS, SEEDS, ...) can be set here and is passed through.
cd "$(dirname "$0")/.." || exit 1
MAMBA_LAYERS=${MAMBA_LAYERS:-"vanilla mamba_init"}
FUSIONS=${FUSIONS:-"none late_mlp early_concat"}
E=${E:-16}
PRED_MODE=${PRED_MODE:-geo_bins}
WD=${WD:-1e-3}
DROPOUT=${DROPOUT:-0}
H_RT=${H_RT:-24:00:00}
K=${K-200}   # prefixes per cell per epoch: cells with <=K eligible prefixes contribute all of them ("" = every prefix)
PASS=""
for v in K MAX_CYCLES ACCUM LR D_MODEL D_FF E_LAYERS N_LAYERS D_STATE SEEDS EPOCHS PATIENCE PRINT_EVERY; do
  [ -n "${!v+x}" ] && PASS="$PASS,$v=${!v}"
done

submit () {  # layer fusion emb short
  local layer=$1 fusion=$2 emb=$3 short=$4
  local extra=${5:-}   # e.g. ",MAX_CYCLES=100" for the horizon control
  local mx=$(echo "$extra$PASS" | grep -oP 'MAX_CYCLES=\K[0-9]+' | tail -1)
  local name="cpmamba_full_${layer}_all_${PRED_MODE}_${fusion}$([ "$fusion" != none ] && echo "$emb")_w${WD}$([ -n "$mx" ] && echo "_max$mx")"
  local cmd="qsub -N $short -l h_rt=$H_RT -o ${name}.qlog -v MAMBA_LAYER=$layer,POOLED_CHEMS=,PRED_MODE=$PRED_MODE,CHEM_FUSION=$fusion,CHEM_EMBED_DIM=$emb,DROPOUT=$DROPOUT,WD=$WD$PASS$extra train_eval_scripts/CPMamba_full_qsub.sh"
  echo "$cmd"
  [ -z "$DRY_RUN" ] && $cmd
}

for layer in $MAMBA_LAYERS; do
  L=$([ "$layer" = vanilla ] && echo Zv || echo Zi)      # Zv = vanilla Mamba, Zi = mamba_init
  for fusion in $FUSIONS; do
    if [ "$fusion" = none ]; then submit "$layer" none "$E" "${L}_none"
    else submit "$layer" "$fusion" "$E" "${L}_${fusion:0:1}${E}"; fi
  done
  if [ -n "$WITH_CONTROL" ]; then
    for fusion in $FUSIONS; do
      [ "$fusion" = none ] && continue
      submit "$layer" "$fusion" 0 "${L}_${fusion:0:1}0ctl"
    done
  fi
  [ -n "$HORIZON_CONTROL" ] && submit "$layer" none "$E" "${L}_max100" ",MAX_CYCLES=100"
done
