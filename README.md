# Romanian Whist RL

Python package for training and evaluating deep reinforcement learning agents for Romanian whist.

## Highlights

- Supports Romanian whist for `3-6` players with the custom `8-1-8` schedule.
- Includes a deterministic rules engine, an AEC-style multi-agent environment, scripted baselines, PPO training, league self-play, tournament evaluation, and CLI play/spectate flows.
- Trains in PyTorch and exports checkpoints to an MLX-friendly `.npz` format for Apple Silicon inference/evaluation.

## Quick start

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
romanian-whist train --output runs/dev --updates 2 --episodes-per-update 4
romanian-whist spectate --players 4 --bot heuristic
```

## Universal training

```bash
uv run romanian-whist train \
  --output runs/universal \
  --updates 100 \
  --episodes-per-update 24 \
  --learning-rate 0.0003 \
  --embed-dim 128 \
  --evaluation-matches 4 \
  --evaluation-every 1 \
  --reward-shaping 0.5 \
  --rollout-workers 1 \
  --eval-workers 1 \
  --device cpu \
  --tensorboard-logdir runs/universal/tensorboard \
  --universal \
  --seed 0
```

This trains one shared model across `3-6` players, writes rolling checkpoints under `runs/universal`, and keeps `best.pt` plus per-update evaluation reports aggregated and broken out by player count.

TensorBoard logs are written to `runs/universal/tensorboard` by default. Start TensorBoard with:

```bash
uv run tensorboard --logdir runs
```

Resume a previous run from its latest checkpoint:

```bash
uv run romanian-whist train \
  --output runs/universal \
  --updates 500 \
  --episodes-per-update 24 \
  --learning-rate 0.0003 \
  --embed-dim 128 \
  --evaluation-matches 4 \
  --evaluation-every 1 \
  --reward-shaping 0.5 \
  --rollout-workers 1 \
  --eval-workers 1 \
  --device cuda \
  --tensorboard-logdir runs/universal/tensorboard \
  --universal \
  --seed 0 \
  --resume-from runs/universal/update-0500.pt
```

For long CUDA runs, reduce evaluation overhead by evaluating every few updates instead of every update:

```bash
uv run romanian-whist train \
  --output runs/universal-fast \
  --updates 500 \
  --episodes-per-update 64 \
  --learning-rate 0.0001 \
  --embed-dim 128 \
  --evaluation-matches 16 \
  --evaluation-every 10 \
  --reward-shaping 0.5 \
  --rollout-workers 8 \
  --eval-workers 4 \
  --device cuda \
  --tensorboard-logdir runs/universal-fast/tensorboard \
  --universal \
  --seed 0
```

Useful training knobs:
- `REWARD_SHAPING`: scales dense contract-progress shaping during rollouts. Default `0.5`.

## Six-player training

If real-world play is usually six-handed, train a dedicated 6-player model first:

```bash
uv run romanian-whist train \
  --output runs/six-player \
  --updates 300 \
  --episodes-per-update 32 \
  --learning-rate 0.0001 \
  --embed-dim 256 \
  --players 6 \
  --fixed-player-count \
  --evaluation-matches 8 \
  --evaluation-every 5 \
  --entropy-coef 0.01 \
  --final-entropy-coef 0.001 \
  --gae-lambda 0.95 \
  --reward-shaping 0.5 \
  --final-reward-shaping 0.1 \
  --latest-weight 0.5 \
  --snapshot-weight 0.35 \
  --scripted-weight 0.15 \
  --rollout-workers 8 \
  --eval-workers 4 \
  --device cuda \
  --tensorboard-logdir runs/six-player/tensorboard \
  --seed 0
```

Example six-player run with the larger fixed-player setup:

```bash
uv run romanian-whist train \
  --output runs/six-player-strong \
  --updates 300 \
  --episodes-per-update 64 \
  --learning-rate 0.0001 \
  --embed-dim 256 \
  --players 6 \
  --fixed-player-count \
  --evaluation-matches 16 \
  --evaluation-every 5 \
  --entropy-coef 0.01 \
  --final-entropy-coef 0.001 \
  --gae-lambda 0.95 \
  --reward-shaping 0.5 \
  --final-reward-shaping 0.1 \
  --latest-weight 0.5 \
  --snapshot-weight 0.35 \
  --scripted-weight 0.15 \
  --rollout-workers 8 \
  --eval-workers 4 \
  --device cuda \
  --tensorboard-logdir runs/six-player-strong/tensorboard \
  --seed 0
```

## Evaluate and play a trained model

Evaluate the best universal checkpoint across all supported player counts:

```bash
uv run romanian-whist eval runs/universal/best.pt --universal --matches 8
```

Play against a trained checkpoint:

```bash
uv run romanian-whist play --checkpoint runs/universal/best.pt --players 4 --seat 0
```

Use `--seat-config` for arbitrary human/model/bot combinations. Example: one human, two trained seats, one random bot:

```bash
uv run romanian-whist play \
  --checkpoint runs/universal/best.pt \
  --players 4 \
  --seat-config human,model,model,random
```

Example with two humans and two trained seats:

```bash
uv run romanian-whist play \
  --checkpoint runs/universal/best.pt \
  --players 4 \
  --seat-config human,human,model,model
```

Watch a trained checkpoint play using the built-in evaluation command and the saved JSON reports, or spectate heuristic bots with:

```bash
uv run romanian-whist spectate --players 4 --bot heuristic
```

You can also spectate mixed bot/model tables:

```bash
uv run romanian-whist spectate \
  --checkpoint runs/universal/best.pt \
  --players 5 \
  --seat-config model,heuristic,random,safe,model
```

## Web UI

Launch the local browser UI:

```bash
uv run romanian-whist ui --checkpoint runs/universal/best.pt --host 127.0.0.1 --port 8000
```

The app exposes three modes:

- `Play`: a real game table for mixed human/model/bot seats.
- `Inspect`: replay and step through bot or mixed games with hidden-card reveal and model output.
- `Advisor`: enter a real-world game state for one seat and get legal bid/play recommendations from a checkpoint.

Open [http://127.0.0.1:8000/play](http://127.0.0.1:8000/play) after launching the server.

Examples:

```bash
uv run romanian-whist ui --checkpoint runs/universal/best.pt --mode play
uv run romanian-whist ui --checkpoint runs/universal/best.pt --mode inspect
uv run romanian-whist ui --checkpoint runs/universal/best.pt --mode advisor
```

## Commands

```bash
romanian-whist train --help
romanian-whist eval --help
romanian-whist play --help
romanian-whist spectate --help
romanian-whist export-mlx --help
```
