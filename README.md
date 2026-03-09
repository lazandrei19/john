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
./scripts/train_universal_model.sh
```

This trains one shared model across `3-6` players, writes rolling checkpoints under `runs/universal`, and keeps `best.pt` plus per-update evaluation reports aggregated and broken out by player count.

TensorBoard logs are written to `runs/universal/tensorboard` by default. Start TensorBoard with:

```bash
uv run tensorboard --logdir runs
```

Resume a previous run from its latest checkpoint:

```bash
RESUME_FROM=runs/universal/update-0500.pt \
DEVICE=cuda \
UPDATES=500 \
OUTPUT=runs/universal \
./scripts/train_universal_model.sh
```

For long CUDA runs, reduce evaluation overhead by evaluating every few updates instead of every update:

```bash
DEVICE=cuda \
UPDATES=500 \
EPISODES_PER_UPDATE=64 \
LEARNING_RATE=0.0001 \
EMBED_DIM=128 \
EVALUATION_MATCHES=16 \
EVALUATION_EVERY=10 \
ROLLOUT_WORKERS=8 \
EVAL_WORKERS=4 \
OUTPUT=runs/universal-fast \
./scripts/train_universal_model.sh
```

Useful training knobs:
- `REWARD_SHAPING`: scales dense contract-progress shaping during rollouts. Default `0.5`.
- `IMITATION_EPISODES`: runs behavior-cloning warmup from scripted experts before PPO. Default `128` in the universal script.

## Six-player training

If real-world play is usually six-handed, train a dedicated 6-player model first:

```bash
./scripts/train_6p_model.sh
```

The 6-player script defaults to a larger model and fixed 6-player training:

```bash
DEVICE=cuda \
UPDATES=300 \
EPISODES_PER_UPDATE=32 \
LEARNING_RATE=0.0001 \
EMBED_DIM=256 \
EVALUATION_MATCHES=8 \
EVALUATION_EVERY=5 \
ROLLOUT_WORKERS=8 \
EVAL_WORKERS=4 \
OUTPUT=runs/six-player \
./scripts/train_6p_model.sh
```

The 6-player script also enables stronger warm-start settings by default:
- `REWARD_SHAPING=0.5`
- `IMITATION_EPISODES=256`

Example long 6-player run with the new shaping and imitation warmup enabled explicitly:

```bash
DEVICE=cuda \
UPDATES=300 \
EPISODES_PER_UPDATE=64 \
LEARNING_RATE=0.0001 \
EMBED_DIM=256 \
EVALUATION_MATCHES=16 \
EVALUATION_EVERY=5 \
REWARD_SHAPING=0.5 \
IMITATION_EPISODES=256 \
ROLLOUT_WORKERS=8 \
EVAL_WORKERS=4 \
OUTPUT=runs/six-player-strong \
./scripts/train_6p_model.sh
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
