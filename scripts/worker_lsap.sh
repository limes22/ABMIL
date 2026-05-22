#!/bin/bash
# worker_lsap.sh GPU CELLS_FILE
# Reads cells_lsap.txt: CELL|TASK|SPLIT|FE|MIL|NCLS|ALPHA|EXPSUF|SEED
# Skip-if-done: any *_${EXPSUF}_s${SEED}_*_s${SEED}/split_9_results.pkl

GPU=${1:-0}
CELLS=${2:-/workspace/scripts/cells_lsap.txt}
FREE_MB=${FREE_MB:-80000}

cd /workspace
PY=/usr/bin/python3
[ ! -x "$PY" ] && PY=/workspace/venv/bin/python

mkdir -p /workspace/exp_logs /workspace/.locks

wait_for_gpu_free() {
  local gpu=$1
  while :; do
    local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$mem" ]; then sleep 60; continue; fi
    if [ "$mem" -lt "$FREE_MB" ]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU $gpu busy (${mem}MB used) — wait 120s"
    sleep 120
  done
}

while IFS='|' read -r CELL TASK SPLIT FE MIL NCLS ALPHA EXPSUF SEED; do
  [ -z "$CELL" ] && continue
  [[ "$CELL" =~ ^# ]] && continue

  FULL_EXP_PREFIX="${CELL}_${EXPSUF}_s${SEED}"
  LOCK="/workspace/.locks/${FULL_EXP_PREFIX}.lock"

  # Check if any existing run with this prefix has split_9 done → skip
  if compgen -G "/workspace/results/${FULL_EXP_PREFIX}_*/split_9_results.pkl" > /dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $FULL_EXP_PREFIX (done)"
    continue
  fi

  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $FULL_EXP_PREFIX (locked by other worker)"
    continue
  fi

  # Check if any existing run with this prefix has any split done (running)
  if compgen -G "/workspace/results/${FULL_EXP_PREFIX}_*/split_0_results.pkl" > /dev/null \
     && ! compgen -G "/workspace/results/${FULL_EXP_PREFIX}_*/split_9_results.pkl" > /dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $FULL_EXP_PREFIX (running)"
    rm -rf "$LOCK"
    continue
  fi

  wait_for_gpu_free $GPU

  TS=$(date +%Y%m%d_%H%M%S)
  EXP=${FULL_EXP_PREFIX}_${TS}
  LOG=/workspace/exp_logs/${EXP}.log
  echo "[$(date +%H:%M:%S)] LAUNCH ${EXP} on GPU ${GPU} (α=${ALPHA})"

  CUDA_VISIBLE_DEVICES=$GPU $PY main.py \
    --data_root_dir /workspace/features --model_size small --k 10 \
    --max_epochs 200 --lr 1e-4 --reg 1e-5 --drop_out 0.25 --label_frac 1.0 \
    --bag_loss ce --bag_weight 0.7 --B 8 --early_stopping --log_data \
    --seed $SEED --no_inst_cluster \
    --feature_extractor $FE --task $TASK --split_dir $SPLIT --model_type $MIL \
    --attn_norm entmax_alpha --lsap_no_tau --lsap_alpha $ALPHA \
    --results_dir /workspace/results --exp_code ${EXP} > $LOG 2>&1
  EXIT=$?
  rm -rf "$LOCK"
  echo "[$(date +%H:%M:%S)] DONE ${EXP} exit=$EXIT"
done < $CELLS
