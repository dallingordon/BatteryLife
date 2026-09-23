#!/bin/bash -l
# One-time setup of venv_mamba: the BatteryLife venv for CPMamba jobs.
#
# It is LAYERED on the mamba-init project's venv instead of copying it:
#   * torch 2.2.0+cu121, triton, and the mamba_ssm fork (editable) + its compiled selective_scan_cuda kernel come
#     straight from /projectnb/textconv/dgordon/mamba/venv via a .pth file -- nothing duplicated, no kernel rebuild.
#     (The kernel .so is built against torch 2.2.0, so BatteryLife's own torch 2.4.1 venv can't load it.)
#   * BatteryLife's other requirements are installed into venv_mamba only, with torch/triton pinned to what the
#     mamba venv has, so pip can't pull a second torch. Nothing is installed into the mamba project's venv.
#   * Packages in venv_mamba win over the mamba venv's (its site-packages is searched first), e.g. BatteryLife's
#     pinned accelerate/transformers.
#
# Run once from anywhere (login node is fine, ~5-10 min):  bash scc/setup_venv_mamba.sh
# Rerunning is safe (reuses the venv, pip skips what's installed).
set -euo pipefail

MAMBA_REPO=/projectnb/textconv/dgordon/mamba
MAMBA_VENV=$MAMBA_REPO/venv
BL=/projectnb/nsf-energize/dgordon/Projects/BatteryLife
OLD_VENV=$BL/venv_train_p100          # only used to find how batteryml was installed

module load gcc/12.2.0 cuda/12.2      # selective_scan_cuda needs gcc 12's libstdc++ (GLIBCXX_3.4.29)
cd "$BL"

PY="$MAMBA_VENV/bin/python"
PYVER=$("$PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
[ -d venv_mamba ] || "$PY" -m venv venv_mamba
SP="venv_mamba/lib/$PYVER/site-packages"

# layer the mamba venv (addsitedir also processes its .pth files, incl. the editable mamba_ssm install),
# plus the repo root, where selective_scan_cuda*.so lives
cat > "$SP/zz_mamba_init_venv.pth" <<EOF
import site; site.addsitedir('$MAMBA_VENV/lib/$PYVER/site-packages')
$MAMBA_REPO
EOF

source venv_mamba/bin/activate
python -m pip install -q --upgrade pip

TORCH_V=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
TRITON_V=$(python -c 'import triton; print(triton.__version__)')
echo "layered from mamba venv: torch $TORCH_V, triton $TRITON_V"
printf 'torch==%s\ntriton==%s\n' "$TORCH_V" "$TRITON_V" > venv_mamba/constraints.txt

# BatteryLife requirements minus torch (from the mamba venv) and BatteryML (not on PyPI; handled below)
grep -viE '^(torch|batteryml)==' requirements.txt > venv_mamba/requirements_nontorch.txt
python -m pip install -r venv_mamba/requirements_nontorch.txt -c venv_mamba/constraints.txt

# batteryml: install the same thing the existing BatteryLife venv has
if ! python -c 'import batteryml' 2>/dev/null; then
  BML=$("$OLD_VENV/bin/pip" freeze 2>/dev/null | grep -i '^batteryml' || true)
  echo "batteryml in $OLD_VENV: '${BML}'"
  if [ -n "$BML" ]; then
    python -m pip install --no-deps "$BML" -c venv_mamba/constraints.txt || echo "!! could not install '$BML' -- install batteryml manually"
  else
    echo "!! batteryml not found in $OLD_VENV -- install it manually into venv_mamba"
  fi
fi

echo "=== import check ==="
python - <<'EOF'
import importlib
for m in ['torch', 'mamba_ssm', 'selective_scan_cuda', 'mamba_ssm.modules.mamba_init', 'mamba_ssm.models.mixer_seq_simple',
          'accelerate', 'deepspeed', 'evaluate', 'peft', 'wandb', 'transformers', 'sklearn', 'joblib', 'batteryml',
          'reformer_pytorch', 'denseweight']:
    try:
        mod = importlib.import_module(m)
        print(f'  ok   {m:34s} {getattr(mod, "__version__", "")}  {getattr(mod, "__file__", "")}')
    except Exception as e:
        print(f'  FAIL {m:34s} {type(e).__name__}: {e}')
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
EOF
echo "done. Use:  module load gcc/12.2.0 cuda/12.2 && source $BL/venv_mamba/bin/activate"
