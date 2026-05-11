#!/bin/bash
# NanoRollout demo — Qwen3-4B SWE-RL agentic GRPO, v5p-8.
#
# Wrapper banks 1 ckpt per attempt; 3 attempts × 25min cap absorbs the
# known intermittent SIGSEGV from tpu_inference + JAX shard_map (bug #3,
# see tasks/debug_tokio/summary.md). Phase 9 empirical: ~17 banked
# steps / 42 min typical. 25 min cap covers ≥1 banked step per attempt
# at p>0.9.

set -uo pipefail

PROJECT=hao-ai-lab-trc
ZONE=us-west1-a
MIG=tunix-worker-mig
SANDBOX=/home/yuxuan/tunix/submodules/test_nanorollout_tunix
CKPT_DIR=/mnt/disks/tunix-data/checkpoints/nro_demo
LOG_DIR=/home/yuxuan/tunix/tasks/organize_running_scripts/_logs
mkdir -p "$CKPT_DIR" "$LOG_DIR"

MAX_ATTEMPTS=3
PER_ATTEMPT_TIMEOUT=1500  # 25 min

count_banked() {
  ls -1d "$CKPT_DIR"/actor/[0-9]* 2>/dev/null | grep -oE '[0-9]+$' \
    | sort -n | tail -1
}

resize_mig() {
  gcloud compute instance-groups managed resize "$MIG" \
    --size="$1" --zone="$ZONE" --project="$PROJECT" --quiet 2>&1 | head -1 \
    || true
}

cleanup() {
  echo "[$(date '+%H:%M:%S')] cleanup: MIG to 0"
  resize_mig 0
  for pid in $(ps -ef \
       | grep -E "tunix.cli.grpo_main|VLLM::|vllm.*tpu_inference|vllm.*spmd|wandb-xpu" \
       | grep -v grep | grep -v claude | awk '{print $2}'); do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

############################## Outer retry loop ##############################
for attempt in $(seq 1 $MAX_ATTEMPTS); do
  banked=$(count_banked); banked=${banked:-0}
  echo "[$(date '+%H:%M:%S')] === attempt $attempt/$MAX_ATTEMPTS; banked=$banked ==="
  [ "$banked" -ge 1 ] && { echo "TARGET met (banked=$banked). Done."; break; }

  echo "[$(date '+%H:%M:%S')] resizing MIG to 4"
  resize_mig 4

  echo "[$(date '+%H:%M:%S')] waiting for 4 RUNNING workers..."
  until [ "$(gcloud compute instances list \
              --project="$PROJECT" \
              --filter='name~^tunix-worker AND zone:'"$ZONE" \
              --format='value(status)' 2>/dev/null \
              | grep -c RUNNING)" -ge 4 ]; do
    sleep 10
  done
  sleep 30  # daemon HealthCheck startup

  # Use existing train venv (has JAX/vLLM/etc); inject nanorollout + tunix
  # via PYTHONPATH instead of pip-installing — keeps train venv uncontaminated.
  # shellcheck disable=SC1091
  source /mnt/disks/tunix-data/venvs/train/bin/activate

  export HF_TOKEN="${HF_TOKEN:-placeholder}"
  export HF_HOME=/mnt/disks/tunix-data/hf
  export HF_DATASETS_CACHE=/mnt/disks/tunix-data/dataset_cache
  export SKIP_JAX_PRECOMPILE=true
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=disabled
  export WANDB__DISABLE_SERVICE=true
  export WANDB_DISABLE_SERVICE=true
  export PYTHONPATH="$SANDBOX/trainers/tunix:$SANDBOX:$SANDBOX/trainers/tunix/examples/nanorollout:${PYTHONPATH:-}"

  RUN_NAME="nro_demo_attempt$(printf '%02d' "$attempt")_$(date +%Y%m%d_%H%M%S)"
  ATTEMPT_LOG="$LOG_DIR/nro_demo_attempt_${attempt}.log"
  echo "[$(date '+%H:%M:%S')] launching grpo_main run_name=$RUN_NAME"

  # No CLI override — passing a nested key like
  # rl_training_config.metrics_logging_options.run_name=...
  # makes omegaconf replace the whole `rl_training_config` dict, dropping the
  # YAML-anchor-merged siblings (log_dir, project_name, ...) and erroring on
  # MetricsLoggerOptions.__init__. RUN_NAME goes into the log filename only.
  timeout --kill-after=30 "$PER_ATTEMPT_TIMEOUT" \
    python -m tunix.cli.grpo_main \
      "$SANDBOX/trainers/tunix/examples/nanorollout/config.yaml" \
      &>"$ATTEMPT_LOG"
  rc=$?

  banked_after=$(count_banked); banked_after=${banked_after:-0}
  echo "[$(date '+%H:%M:%S')] attempt $attempt exit=$rc banked: $banked → $banked_after"
  tail -30 "$ATTEMPT_LOG"
done

final=$(count_banked); final=${final:-0}
echo "[$(date '+%H:%M:%S')] === run.sh done; final banked = $final ==="
[ "$final" -ge 1 ]
