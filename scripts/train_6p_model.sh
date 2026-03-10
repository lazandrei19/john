#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
OUTPUT="${OUTPUT:-runs/six-player}"
UPDATES="${UPDATES:-300}"
EPISODES_PER_UPDATE="${EPISODES_PER_UPDATE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
EMBED_DIM="${EMBED_DIM:-256}"
EVALUATION_MATCHES="${EVALUATION_MATCHES:-8}"
EVALUATION_EVERY="${EVALUATION_EVERY:-5}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
REWARD_SHAPING="${REWARD_SHAPING:-0.5}"
IMITATION_EPISODES="${IMITATION_EPISODES:-256}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SEED="${SEED:-0}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$OUTPUT/tensorboard}"
RESUME_FROM="${RESUME_FROM:-}"

CMD=(
  uv run romanian-whist train
  --output "$OUTPUT"
  --updates "$UPDATES"
  --episodes-per-update "$EPISODES_PER_UPDATE"
  --learning-rate "$LEARNING_RATE"
  --embed-dim "$EMBED_DIM"
  --players 6
  --fixed-player-count
  --evaluation-matches "$EVALUATION_MATCHES"
  --evaluation-every "$EVALUATION_EVERY"
  --entropy-coef "$ENTROPY_COEF"
  --gae-lambda "$GAE_LAMBDA"
  --reward-shaping "$REWARD_SHAPING"
  --imitation-episodes "$IMITATION_EPISODES"
  --rollout-workers "$ROLLOUT_WORKERS"
  --eval-workers "$EVAL_WORKERS"
  --device "$DEVICE"
  --tensorboard-logdir "$TENSORBOARD_LOGDIR"
  --seed "$SEED"
)

if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume-from "$RESUME_FROM")
fi

"${CMD[@]}"
