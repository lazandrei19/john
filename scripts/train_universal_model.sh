#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
OUTPUT="${OUTPUT:-runs/universal}"
UPDATES="${UPDATES:-100}"
EPISODES_PER_UPDATE="${EPISODES_PER_UPDATE:-24}"
EVALUATION_MATCHES="${EVALUATION_MATCHES:-4}"
SEED="${SEED:-0}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$OUTPUT/tensorboard}"

uv run romanian-whist train \
  --output "$OUTPUT" \
  --updates "$UPDATES" \
  --episodes-per-update "$EPISODES_PER_UPDATE" \
  --evaluation-matches "$EVALUATION_MATCHES" \
  --device "$DEVICE" \
  --tensorboard-logdir "$TENSORBOARD_LOGDIR" \
  --universal \
  --seed "$SEED"
