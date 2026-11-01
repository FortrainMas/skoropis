#!/usr/bin/env bash
# All experiments for one GPU session, in priority order (see docs/plan.md).
#
#   tmux new -s htr 'bash scripts/server_run.sh; bash'
#
# tmux keeps the run alive if SSH drops. Each step writes logs/<step>.log; a failed
# step is reported and the session goes on with the steps that do not depend on it.
# Environment checks come first and stop everything, so a broken setup costs minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation when memory is tight

step() {
  local name=$1
  shift
  echo "=== $(date '+%F %T') $name"
  if "$@" 2>&1 | tee "logs/$name.log"; then
    echo "=== $name: ok"
  else
    echo "=== $name: FAILED, see logs/$name.log"
  fi
}

# Image decoding and augmentation run on the CPU; on a fast GPU they are the bottleneck.
WORKERS=$(nproc)
[ "$WORKERS" -gt 16 ] && WORKERS=16

crnn_train() {
  uv run diploma train --workers "$WORKERS" "$@"
}

crnn_eval() {  # crnn_eval RUN DATASET...
  local run=$1
  shift
  local data=()
  for spec in "$@"; do data+=(-d "$spec"); done
  uv run diploma eval diploma.models.crnn:load -a "checkpoint=checkpoints/$run/best.pt" -a device=cuda \
    "${data[@]}" --batch-size 64 --name "$run"
}

# --- 0. environment -----------------------------------------------------------
uv sync --frozen || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || exit 1
uv run python -c "
import torch
assert torch.cuda.is_available(), 'torch does not see the GPU: driver too old for this CUDA build, see docs/plan.md'
x = torch.ones(256, 256, device='cuda', dtype=torch.float16)  # fails if this build has no kernels for the card
assert (x @ x)[0, 0].item() == 256
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0),
      'bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp16')
" || exit 1
uv run diploma data > logs/data.log || { echo "data not found: set DIPLOMA_DATA"; exit 1; }
uv run pytest -q > logs/pytest.log 2>&1 || { echo "tests failed, see logs/pytest.log"; exit 1; }

# --- E004: fine-tune Kansallisarkisto on Makarov ---------------------------------
# First, because it is the most likely best result for Makarov (see docs/plan.md).
# Two recipes: everything at once, and the authors' own (encoder alone for the first epochs).
#
# Weights, gradients and optimizer state of the 610M-parameter model take ~10 GB. On a 16 GB
# card (T4) the batch is halved and accumulated twice as long: the same 16 lines per step.
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 20000 ]; then
  TROCR_BATCH=(--batch-size 4 --accumulate 4)
  EVAL_BATCH=8
else
  TROCR_BATCH=()
  EVAL_BATCH=16
fi
trocr_eval() {  # trocr_eval RUN
  uv run diploma eval diploma.models.trocr:TrOCR -a "checkpoint=checkpoints/$1/best" -a device=cuda \
    -d makarov:test --batch-size "$EVAL_BATCH" --name "$1"
}
step e004-full-train uv run diploma train-trocr --name trocr-makarov --workers "$WORKERS" "${TROCR_BATCH[@]}" \
  --train makarov:train --val makarov:val --epochs 20
step e004-full-eval trocr_eval trocr-makarov
step e004-encfirst-train uv run diploma train-trocr --name trocr-makarov-encfirst --workers "$WORKERS" "${TROCR_BATCH[@]}" \
  --train makarov:train --val makarov:val --epochs 20 --freeze-decoder-epochs 6
step e004-encfirst-eval trocr_eval trocr-makarov-encfirst

# --- E002: CRNN on Digital Peter ------------------------------------------------
# Makarov is the target; Peter is here as extra training data and as the start
# point for E003. Its own test score only checks that the training code is sane.
step e002-train crnn_train --name crnn-peter --train peter:train --val peter:val --epochs 60
step e002-eval crnn_eval crnn-peter peter:test makarov:test

# --- E003: CRNN on Makarov ------------------------------------------------------
step e003-scratch-train crnn_train --name crnn-makarov \
  --train makarov:train --val makarov:val --epochs 100
step e003-scratch-eval crnn_eval crnn-makarov makarov:test

if [ -f checkpoints/crnn-peter/best.pt ]; then
  step e003-finetune-train crnn_train --name crnn-peter-makarov --init checkpoints/crnn-peter/best.pt \
    --train makarov:train --val makarov:val --epochs 60 --lr 3e-4
  step e003-finetune-eval crnn_eval crnn-peter-makarov makarov:test peter:test
fi

step e003-joint-train crnn_train --name crnn-joint \
  --train peter:train --train 'makarov:train*4' --val makarov:val --val peter:val --epochs 60
step e003-joint-eval crnn_eval crnn-joint makarov:test peter:test

uv run diploma compare | tee logs/compare.md
echo "=== $(date '+%F %T') done. Bring back runs/, logs/ and checkpoints/*/history.jsonl"
echo "(checkpoints/trocr-*/best are 2.4 GB each: copy them only if you want to keep the weights)"
