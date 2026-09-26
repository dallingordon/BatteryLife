#!/bin/bash
# Submit jobs from an experiment config TSV (one row per job; see experiments/sweep_9_26.tsv). Run on an SCC login
# node from the repo root:
#   DRY_RUN=1 bash train_eval_scripts/submit_from_config.sh experiments/sweep_9_26.tsv      # print the qsub commands
#   bash train_eval_scripts/submit_from_config.sh experiments/sweep_9_26.tsv                # submit every row
#   GROUP=mamba_main bash train_eval_scripts/submit_from_config.sh experiments/sweep_9_26.tsv   # one group (regex on `group`)
#   ONLY=Zi_none,Zv_none bash ...                                                                 # specific rows by `short`
# Columns: short, group, script, then env vars for the qsub script ("-" = don't pass), H_RT, purpose.
# Refuses to submit anything while a row still has TBD in it.
cd "$(dirname "$0")/.." || exit 1
CFG=${1:?usage: submit_from_config.sh <config.tsv>}
tag=$(basename "$CFG" .tsv)
header=$(grep -v '^#' "$CFG" | head -1)
IFS=$'\t' read -r -a cols <<< "$header"
n=0
while IFS=$'\t' read -r -a f; do
  declare -A row=()
  for i in "${!cols[@]}"; do row[${cols[$i]}]="${f[$i]}"; done
  short=${row[short]}; group=${row[group]}
  [ -n "$GROUP" ] && ! [[ "$group" =~ $GROUP ]] && { unset row; continue; }
  [ -n "$ONLY" ] && [[ ",$ONLY," != *",$short,"* ]] && { unset row; continue; }
  if printf '%s\n' "${f[@]}" | grep -qx 'TBD'; then echo "!! $short still has TBD fields -- not submitting" >&2; unset row; continue; fi
  vars=""
  for c in "${cols[@]}"; do
    case "$c" in short|group|script|H_RT|purpose) continue ;; esac
    v=${row[$c]}; [ "$v" = "-" ] || [ -z "$v" ] && continue
    vars="${vars:+$vars,}$c=$v"
  done
  args=(qsub -N "$short" -o "${tag}_${short}.qlog")
  [ -n "${row[H_RT]}" ] && [ "${row[H_RT]}" != "-" ] && args+=(-l "h_rt=${row[H_RT]}")
  args+=(-v "$vars" "${row[script]}")
  printf '%q ' "${args[@]}"; echo
  [ -z "$DRY_RUN" ] && "${args[@]}"
  n=$((n+1)); unset row
done < <(grep -v '^#' "$CFG" | tail -n +2 | grep -v '^[[:space:]]*$')
echo "$([ -n "$DRY_RUN" ] && echo 'would submit' || echo 'submitted') $n job(s) from $CFG"
