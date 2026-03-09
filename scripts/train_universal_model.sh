#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
OUTPUT="${OUTPUT:-runs/universal}"
UPDATES="${UPDATES:-100}"
EPISODES_PER_UPDATE="${EPISODES_PER_UPDATE:-24}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
EVALUATION_MATCHES="${EVALUATION_MATCHES:-4}"
EVALUATION_EVERY="${EVALUATION_EVERY:-1}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
SEED="${SEED:-0}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$OUTPUT/tensorboard}"
RESUME_FROM="${RESUME_FROM:-}"

CMD=(
  uv run romanian-whist train
  --output "$OUTPUT"
  --updates "$UPDATES"
  --episodes-per-update "$EPISODES_PER_UPDATE"
  --learning-rate "$LEARNING_RATE"
  --evaluation-matches "$EVALUATION_MATCHES"
  --evaluation-every "$EVALUATION_EVERY"
  --rollout-workers "$ROLLOUT_WORKERS"
  --eval-workers "$EVAL_WORKERS"
  --device "$DEVICE"
  --tensorboard-logdir "$TENSORBOARD_LOGDIR"
  --universal
  --seed "$SEED"
)

if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume-from "$RESUME_FROM")
fi

"${CMD[@]}"
