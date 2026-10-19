#!/bin/bash -l
#$ -N longmamba_pooled
#$ -P nsf-energize
#$ -l h_rt=30:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -j y
#$ -o longmamba_pooled.qlog

# LongMamba (models/LongMamba.py): one Mamba stack over the raw per-point series of the early cycles
# (100 cycles x 300 points = ~30k steps, 3 channels per step), output CLS token at the end.
# Trained on the ORDINARY pooled loader (all four chemistries, prefixes 1..100 cycles -- the same data, splits and
# val/test as pooled CPMLP), so the comparison with CPMLP is architecture-only.
#   gpu_c=7.0: the mamba-init fork's selective_scan kernels are not built for P100 (sm_60).
#   venv_mamba: see scc/setup_venv_mamba.sh.
# NOTE: BATCH / ACCUM / h_rt are first guesses -- run `python test_longmamba.py --gpu` in a qrsh first; it prints
# fwd+bwd time and peak memory at 100 cycles for B = 1..32.
#
# Variants via environment variables (`qsub -v VAR=val,... this_script`, or VAR=val bash this_script in a qrsh):
#   MAMBA_LAYER     vanilla (default) | mamba_init | s4d (time-invariant S4D in the same block; models/S4D.py)
#   S4_DT_MIN S4_DT_MAX  1e-5 1e-1 (defaults; s4d only): initial step-size range, memory ~ 2/dt steps
#                   (tag _dtmin<v> / _dtmax<v> if not default)
#   S4_BACKEND      fft (default) | scan (s4d only): same LTI model run by mamba_ssm's selective-scan kernel with constant
#                   dt/B/C -- linear in length, use it for full-history / long-tail runs           (tag _scan)
#   INIT_CKPT       "" (default) | checkpoint dir or model.safetensors to start from (finetune after pretraining).
#                   "{seed}" in it is replaced by the seed, e.g.
#                   INIT_CKPT=/projectnb/.../checkpoints/LongMamba_POOLED_s4d_breadout_rfirst_..._fulltailK1_Lionly_seed{seed}
#   INIT_TAG        short name for the checkpoint, required with INIT_CKPT                          (tag _ft<INIT_TAG>)
#   BOUNDARY        none (default) | shared | index | readout   (cycle-boundary token in front of every cycle; see LongMamba.py)
#                   readout = R token between cycles + training-only guidance head (eval uses the output CLS F only):
#   R_FIRST         0 (default) | 1: also an R in front of cycle 1 (never supervised)            (tag _rfirst)
#   R_DIM           32 (default): guidance-head bottleneck Linear(d_model -> R_DIM)            (tag _rd<R_DIM> if not 32)
#   R_WEIGHT        1.0 (default): loss = F loss + R_WEIGHT * R loss; 0 = R tokens unsupervised (tag _rw<R_WEIGHT> if not 1.0)
#   CHEM_CLS        0 (default) | 1: per-chemistry output CLS token                               (log tag _ccls)
#   CHEM_BOUNDARY   0 (default) | 1: per-chemistry cycle-start embedding; replaces the token with BOUNDARY=shared,
#                   is added to the cycle-index token with BOUNDARY=index; not allowed with none   (log tag _cbnd)
#   PRED_MODE       geo_bins (default) | regression | dual
#   POOLED_CHEMS    "" / all = all four (default); e.g. "CALB Zn-ion Na-ion" for a fast test (logs get a _quick tag)
#   CHEM_TAG        log tag used instead of _quick when POOLED_CHEMS is set on purpose, e.g. CHEM_TAG=Lionly
#   FULL            0 (default) | 1: full-timescale training loader (prefixes past cycle 100; batch 1, ACCUM default 32;
#                   val/test unchanged). Needs BOUNDARY none, shared or readout. Log tag _full<sampling>K<K>[_max<M>]
#   FULL_SAMPLING   stratified (default) = every prefix 1..100 + FULL_K longer ones per cell per epoch | uniform
#                   | tail = long-tail pretraining: FULL_K lengths per cell per epoch drawn from 100..L only; with
#                   BOUNDARY=readout each sample is one pass (R before every cycle, F after the last drawn cycle)
#   FULL_K          100 (default): random prefixes per cell per epoch (the longer ones, for stratified)
#   FULL_MAX_CYCLES "" (default, no cap) | int: longest training prefix
#   GRAD_CKPT       0 (default) | 1: recompute Mamba blocks in backward (memory for very long prefixes)  (tag _gc not added)
#   TIME_DT         off (default) | const | real: time-aware mixers (models/time_ssm.py; MAMBA_LAYER s4d needs
#                   S4_BACKEND=scan, or vanilla). const = tau 1 (control), real = recorded step / DT_C   (tag _tdt<TIME_DT>)
#   DT_INPUT        0 (default) | 1: log(tau) as a 4th per-point input feature                        (tag _dtin)
#   DT_C            seconds, needed for TIME_DT=real / DT_INPUT=1 (median interior step from dt_stats.py; not in the tag)
#   DT_CLIP         0 (default, none) | upper clip on tau                                              (tag _clip<DT_CLIP>)
#   RAW             0 (default) | 1: raw records instead of the 300-point grid (data_provider/data_loader_raw.py):
#                   every logged record of cycles 1..100, rests kept, 4 channels (V, I, Q_charge, Q_discharge),
#                   tau = real step to the previous record / DT_C                                       (tag _raw)
#   RAW_VNORM       cycle_max (default) | none: voltage / that cycle's max, or volts                    (tag _vvolts)
#   RAW_TAU         real (default) | cycle_norm (10/18, needs RAW=1 TIME_DT=real): tau = RAW_TAU_SCALE * step / that
#                   cycle's duration, rests included; DT_C x DT_CLIP (s) still caps steps if both set   (tag _cnorm)
#   RAW_TAU_SCALE   300 (default): tau per cycle under cycle_norm                                     (tag _cs<S>)
#   RAW_TAU_BND     1 (default) | 0: tau of each cycle's first record under cycle_norm                (tag _cb<B>)
#   LRADJ           constant (default) | cosine (10/18): per-step cosine, peak LR, warmup + floor     (tag _cos<E>w<W>)
#   COS_EPOCHS      epochs the cosine spans (default EPOCHS); WARMUP epochs of linear warmup (0); MIN_LR_RATIO floor (0.01,
#                   tag _min<R> if not 0.01). With cosine, set PATIENCE >= EPOCHS so early stopping doesn't cut the schedule.
#   LR              1e-4 (default)                                                                   (tag _lr<LR> if not 1e-4)
#   BATCH_STEPS     0 (default, none) | RAW=1: samples x longest length per batch <= BATCH_STEPS (BATCH = max samples)
#   BATCH ACCUM     8 4 (defaults; effective batch 32 like CPMLP)
#   LR WD           1e-4 1e-3 (defaults)
#   D_MODEL N_LAYERS D_STATE MAMBA_EXPAND   64 4 16 2 (defaults, same as CPMamba); non-default values add
#                   _dm<D_MODEL> / _L<N_LAYERS> / _ds<D_STATE> / _ex<MAMBA_EXPAND> to the log tag
#   SEEDS           "2021 42 2024" (default); other seeds reuse the paper's 3 splits (see run_one)
#   EPOCHS PATIENCE 100 5 (defaults)
#   PRINT_EVERY     200 (default)

module load gcc/12.2.0      # selective_scan_cuda needs gcc 12's libstdc++
module load cuda/12.2
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife
source venv_mamba/bin/activate

RESULTS_DIR=/projectnb/nsf-energize/dgordon/Projects/BatteryLife/results_logs
mkdir -p "$RESULTS_DIR"

MAMBA_LAYER=${MAMBA_LAYER:-vanilla}
BOUNDARY=${BOUNDARY:-none}
CHEM_CLS=${CHEM_CLS:-0}
CHEM_BOUNDARY=${CHEM_BOUNDARY:-0}
PRED_MODE=${PRED_MODE:-geo_bins}
POOLED_CHEMS=${POOLED_CHEMS:-}
[ "$POOLED_CHEMS" = all ] && POOLED_CHEMS=""   # "all" = every chemistry (qsub -v cannot reliably pass an empty value)
FULL=${FULL:-0}
FULL_SAMPLING=${FULL_SAMPLING:-stratified}
FULL_K=${FULL_K:-100}
FULL_MAX_CYCLES=${FULL_MAX_CYCLES:-}
GRAD_CKPT=${GRAD_CKPT:-0}
R_FIRST=${R_FIRST:-0}
R_DIM=${R_DIM:-32}
R_WEIGHT=${R_WEIGHT:-1.0}
S4_DT_MIN=${S4_DT_MIN:-1e-5}
S4_DT_MAX=${S4_DT_MAX:-1e-1}
S4_BACKEND=${S4_BACKEND:-fft}
INIT_CKPT=${INIT_CKPT:-}
TIME_DT=${TIME_DT:-off}
DT_INPUT=${DT_INPUT:-0}
DT_C=${DT_C:-0}
DT_CLIP=${DT_CLIP:-0}
RAW=${RAW:-0}
RAW_VNORM=${RAW_VNORM:-cycle_max}
BATCH_STEPS=${BATCH_STEPS:-0}
RAW_TAU=${RAW_TAU:-real}
LRADJ=${LRADJ:-constant}
WARMUP=${WARMUP:-0}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.01}
RAW_TAU_SCALE=${RAW_TAU_SCALE:-300}
RAW_TAU_BND=${RAW_TAU_BND:-1}
INIT_TAG=${INIT_TAG:-}
CHEM_TAG=${CHEM_TAG:-}
BATCH=${BATCH:-8}                                         # full-timescale: train batch is 1, BATCH is val/test only
if [ "$FULL" = "1" ]; then ACCUM=${ACCUM:-32}; else ACCUM=${ACCUM:-4}; fi   # effective train batch 32 either way
LR=${LR:-1e-4}
WD=${WD:-1e-3}
D_MODEL=${D_MODEL:-64}
N_LAYERS=${N_LAYERS:-4}
D_STATE=${D_STATE:-16}
MAMBA_EXPAND=${MAMBA_EXPAND:-2}
SEEDS=${SEEDS:-"2021 42 2024"}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-5}
PRINT_EVERY=${PRINT_EVERY:-200}
COS_EPOCHS=${COS_EPOCHS:-$EPOCHS}

TAG="_${MAMBA_LAYER}_b${BOUNDARY}"
[ "$RAW" = "1" ] && TAG="_raw${TAG}"
[ "$RAW" = "1" ] && [ "$RAW_VNORM" = "none" ] && TAG="${TAG}_vvolts"
if [ "$BOUNDARY" = "readout" ]; then
  [ "$R_FIRST" = "1" ] && TAG="${TAG}_rfirst"
  [ "$R_DIM" != "32" ] && TAG="${TAG}_rd${R_DIM}"
  [ "$R_WEIGHT" != "1.0" ] && [ "$R_WEIGHT" != "1" ] && TAG="${TAG}_rw${R_WEIGHT}"
fi
if [ "$MAMBA_LAYER" = "s4d" ]; then
  [ "$S4_DT_MIN" != "1e-5" ] && TAG="${TAG}_dtmin${S4_DT_MIN}"
  [ "$S4_DT_MAX" != "1e-1" ] && TAG="${TAG}_dtmax${S4_DT_MAX}"
  [ "$S4_BACKEND" != "fft" ] && TAG="${TAG}_scan"
fi
[ "$TIME_DT" != "off" ] && TAG="${TAG}_tdt${TIME_DT}"
[ "$DT_INPUT" = "1" ] && TAG="${TAG}_dtin"
if [ "$RAW_TAU" = "cycle_norm" ]; then
  if [ "$RAW" != "1" ] || [ "$TIME_DT" != "real" ]; then echo "RAW_TAU=cycle_norm needs RAW=1 TIME_DT=real"; exit 1; fi
  TAG="${TAG}_cnorm"
  [ "$RAW_TAU_SCALE" != "300" ] && TAG="${TAG}_cs${RAW_TAU_SCALE}"
  TAG="${TAG}_cb${RAW_TAU_BND}"
  [ "$DT_CLIP" != "0" ] && [ "$DT_C" != "0" ] && TAG="${TAG}_clip${DT_CLIP}"
elif [ "$TIME_DT" = "real" ] || [ "$DT_INPUT" = "1" ]; then
  if [ "$DT_C" = "0" ]; then echo "TIME_DT=real / DT_INPUT=1 need DT_C (seconds, from dt_stats.py)"; exit 1; fi
  [ "$DT_CLIP" != "0" ] && TAG="${TAG}_clip${DT_CLIP}"
fi
[ "$CHEM_CLS" = "1" ] && TAG="${TAG}_ccls"
[ "$CHEM_BOUNDARY" = "1" ] && TAG="${TAG}_cbnd"
[ "$D_MODEL" != "64" ] && TAG="${TAG}_dm${D_MODEL}"
[ "$N_LAYERS" != "4" ] && TAG="${TAG}_L${N_LAYERS}"
[ "$D_STATE" != "16" ] && TAG="${TAG}_ds${D_STATE}"
[ "$MAMBA_EXPAND" != "2" ] && TAG="${TAG}_ex${MAMBA_EXPAND}"
[ "$PRED_MODE" = "geo_bins" ] && TAG="${TAG}_geobins"
[ "$PRED_MODE" = "dual" ] && TAG="${TAG}_dual"
[ "$WD" != "0.0" ] && [ "$WD" != "0" ] && TAG="${TAG}_wd${WD}"
[ "$LR" != "1e-4" ] && TAG="${TAG}_lr${LR}"
if [ "$LRADJ" = "cosine" ]; then
  TAG="${TAG}_cos${COS_EPOCHS}w${WARMUP}"
  [ "$MIN_LR_RATIO" != "0.01" ] && TAG="${TAG}_min${MIN_LR_RATIO}"
elif [ "$LRADJ" != "constant" ]; then
  TAG="${TAG}_lradj${LRADJ}"
fi
EXTRA_ARGS="--prediction_mode $PRED_MODE --long_boundary $BOUNDARY --long_chem_cls $CHEM_CLS --long_chem_boundary $CHEM_BOUNDARY --wd $WD --long_grad_ckpt $GRAD_CKPT --long_r_first $R_FIRST --long_r_dim $R_DIM --long_r_weight $R_WEIGHT --s4_dt_min $S4_DT_MIN --s4_dt_max $S4_DT_MAX --s4_backend $S4_BACKEND --long_time_dt $TIME_DT --long_dt_input $DT_INPUT --dt_c $DT_C --dt_clip $DT_CLIP --long_raw $RAW --raw_v_norm $RAW_VNORM --raw_max_batch_steps $BATCH_STEPS --raw_tau_mode $RAW_TAU --raw_tau_scale $RAW_TAU_SCALE --raw_tau_boundary $RAW_TAU_BND --cos_epochs $COS_EPOCHS --warmup_epochs $WARMUP --min_lr_ratio $MIN_LR_RATIO"
if [ "$FULL" = "1" ]; then
  if [ "$BOUNDARY" = "index" ]; then echo "FULL=1 needs BOUNDARY none, shared or readout (index has 100 cycle tokens)"; exit 1; fi
  TAG="${TAG}_full${FULL_SAMPLING}K${FULL_K}"
  EXTRA_ARGS="$EXTRA_ARGS --full_timescale --full_sampling $FULL_SAMPLING --full_prefixes_per_cell $FULL_K"
  if [ -n "$FULL_MAX_CYCLES" ]; then
    TAG="${TAG}_max${FULL_MAX_CYCLES}"
    EXTRA_ARGS="$EXTRA_ARGS --full_max_cycles $FULL_MAX_CYCLES"
  fi
fi
if [ -n "$INIT_CKPT" ]; then
  if [ -z "$INIT_TAG" ]; then echo "INIT_CKPT needs INIT_TAG (short name for the log tag)"; exit 1; fi
  TAG="${TAG}_ft${INIT_TAG}"
fi
if [ -n "$POOLED_CHEMS" ]; then
  if [ -n "$CHEM_TAG" ]; then TAG="${TAG}_${CHEM_TAG}"; else TAG="${TAG}_quick"; fi
  EXTRA_ARGS="$EXTRA_ARGS --pooled_chemistries $POOLED_CHEMS"
fi

run_one () {
  local seed=$1
  # --pooled_split_seed only has the paper's 3 splits (2021/42/2024); other seeds reuse one of them, cycling.
  local split_seed=$seed
  case "$seed" in 2021|42|2024) ;; *) local _splits=(2021 42 2024); split_seed=${_splits[$((seed % 3))]} ;; esac
  local ckpt="/projectnb/nsf-energize/dgordon/Projects/BatteryLife/checkpoints/LongMamba_POOLED${TAG}_seed${seed}"
  local log="${RESULTS_DIR}/LongMamba_Pooled${TAG}_seed${seed}.log"
  local init_args=""
  if [ -n "$INIT_CKPT" ]; then
    local init="${INIT_CKPT//\{seed\}/$seed}"
    [ -e "$init" ] || { echo "INIT_CKPT $init does not exist"; return 1; }
    init_args="--init_checkpoint $init"
  fi
  mkdir -p "$ckpt"
  echo "=== LongMamba | POOLED${TAG} seed=$seed (split_seed=$split_seed) epochs=$EPOCHS batch=$BATCH x accum $ACCUM ==="
  accelerate launch --num_processes 1 --main_process_port 20445 run_main.py \
    --task_name classification --data Dataset_original --is_training 1 --root_path ./dataset \
    --model_id LongMamba --model LongMamba --features MS --seq_len 1 --label_len 50 --factor 3 \
    --enc_in 3 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --seed "$seed" \
    --d_model "$D_MODEL" --batch_size "$BATCH" --learning_rate "$LR" \
    --mamba_layer "$MAMBA_LAYER" --mamba_n_layers "$N_LAYERS" --mamba_d_state "$D_STATE" --mamba_expand "$MAMBA_EXPAND" \
    --train_epochs "$EPOCHS" --model_comment "LongMamba_Pooled${TAG}_s${seed}" --accumulation_steps "$ACCUM" \
    --charge_discharge_length 300 --dataset POOLED --num_workers 4 --print_every "$PRINT_EVERY" \
    --patience "$PATIENCE" --early_cycle_threshold 100 --lradj "$LRADJ" --loss MSE \
    --pooled --pooled_split_seed "$split_seed" $EXTRA_ARGS $init_args \
    --checkpoints "$ckpt" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
  run_one "$seed"
done

echo "=== Pooled per-chemistry results (variant tag: '${TAG}') ==="
grep -h "Pooled\(\[[a-z]*\]\)\? \(single-checkpoint\|per-chem-best-val\)\|=== Pooled per-chemistry" "$RESULTS_DIR"/LongMamba_Pooled${TAG}_seed*.log
