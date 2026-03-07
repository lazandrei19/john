from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.checkpoint import load_policy_checkpoint
from romanian_whist.agents.model import PolicyAgent
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.mlx_support.converter import CheckpointConverter
from romanian_whist.rules.cards import card_label
from romanian_whist.rules.config import WhistVariantConfig, normalize_one_card_modes
from romanian_whist.train.league import LeagueConfig, LeagueTrainer
from romanian_whist.train.ppo import PPOConfig
from romanian_whist.web.app import run_ui

app = typer.Typer(add_completion=False)
VALID_SEAT_ROLES = {"human", "model", "heuristic", "safe", "random", "bot"}


def _config(players: int, seed: int, one_card_modes: str) -> WhistVariantConfig:
    modes = normalize_one_card_modes([mode.strip() for mode in one_card_modes.split(",") if mode.strip()])
    return WhistVariantConfig(players=players, seed=seed, one_card_modes=modes)


def _normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized == "bot":
        return "heuristic"
    return normalized


def _parse_seat_config(players: int, seat_config: str) -> list[str]:
    roles = [_normalize_role(role) for role in seat_config.split(",") if role.strip()]
    if len(roles) != players:
        raise typer.BadParameter(
            "Seat config must specify exactly {players} roles.".format(players=players)
        )
    invalid = [role for role in roles if role not in VALID_SEAT_ROLES]
    if invalid:
        raise typer.BadParameter(
            "Invalid seat roles: {roles}. Use human, model, heuristic, safe, random, or bot.".format(
                roles=", ".join(sorted(set(invalid)))
            )
        )
    return roles


def _default_play_roles(players: int, seat: int, checkpoint: Optional[Path], bot: str) -> list[str]:
    if seat < 0 or seat >= players:
        raise typer.BadParameter("Seat must be between 0 and {max_seat}.".format(max_seat=players - 1))
    roles = [_normalize_role(bot) for _ in range(players)]
    roles[seat] = "human"
    if checkpoint is not None:
        roles[(seat + 1) % players] = "model"
    return roles


def _seat_roles(players: int, seat_config: Optional[str], *, default_roles: list[str], require_model_checkpoint: bool) -> list[str]:
    roles = _parse_seat_config(players, seat_config) if seat_config else list(default_roles)
    if require_model_checkpoint and "model" in roles:
        raise typer.BadParameter("A checkpoint is required when any seat uses the model role.")
    return roles


def _build_bot_agent(role: str, seed: int, checkpoint_agent: Optional[PolicyAgent]) -> object:
    if role == "model":
        if checkpoint_agent is None:
            raise typer.BadParameter("A checkpoint is required when any seat uses the model role.")
        return checkpoint_agent
    if role == "random":
        return RandomLegalAgent(seed=seed)
    if role == "safe":
        return SafeHeuristicAgent(seed=seed)
    return BidPlayHeuristicAgent(seed=seed)


def _bot_agents_for_roles(roles: list[str], seed: int, checkpoint_agent: Optional[PolicyAgent]) -> list[object | None]:
    agents = []
    for index, role in enumerate(roles):
        if role == "human":
            agents.append(None)
            continue
        agents.append(_build_bot_agent(role, seed + index, checkpoint_agent))
    return agents


@app.command()
def train(
    output: Path = typer.Option(Path("runs/universal"), "--output"),
    updates: int = typer.Option(100, "--updates"),
    episodes_per_update: int = typer.Option(24, "--episodes-per-update"),
    players: int = typer.Option(4, "--players"),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    one_card_modes: str = typer.Option("regular,forehead,blind", "--one-card-modes"),
    universal: bool = typer.Option(True, "--universal/--fixed-player-count"),
    evaluation_matches: int = typer.Option(4, "--evaluation-matches"),
    tensorboard_logdir: Optional[Path] = typer.Option(None, "--tensorboard-logdir"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    evaluation_player_counts = (3, 4, 5, 6) if universal else (players,)
    resolved_tensorboard_logdir = tensorboard_logdir or (output / "tensorboard")
    trainer = LeagueTrainer(
        variant_config=_config(players, seed, one_card_modes),
        ppo_config=PPOConfig(),
        league_config=LeagueConfig(
            total_updates=updates,
            episodes_per_update=episodes_per_update,
            checkpoint_dir=output,
            device=device,
            seed=seed,
            evaluation_matches=evaluation_matches,
            evaluation_player_counts=evaluation_player_counts,
            tensorboard_log_dir=resolved_tensorboard_logdir,
        ),
    )
    history = trainer.train(updates=updates)
    typer.echo(json.dumps(history[-1], indent=2))
    typer.echo(json.dumps(trainer.evaluate(matches=evaluation_matches), indent=2))


@app.command()
def eval(
    checkpoint: Path = typer.Argument(...),
    players: int = typer.Option(4, "--players"),
    seed: int = typer.Option(0, "--seed"),
    matches: int = typer.Option(8, "--matches"),
    device: str = typer.Option("cpu", "--device"),
    universal: bool = typer.Option(True, "--universal/--single-player-count"),
) -> None:
    policy, _ = load_policy_checkpoint(checkpoint, device=device)
    evaluation_player_counts = (3, 4, 5, 6) if universal else (players,)
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=players, seed=seed),
        league_config=LeagueConfig(device=device, seed=seed, evaluation_player_counts=evaluation_player_counts),
    )
    trainer.policy.load_state_dict(policy.state_dict())
    typer.echo(json.dumps(trainer.evaluate(matches=matches), indent=2))


@app.command()
def export_mlx(checkpoint: Path = typer.Argument(...), output: Path = typer.Option(Path("mlx-export"), "--output")) -> None:
    converter = CheckpointConverter()
    result = converter.export(checkpoint, output)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def ui(
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    mode: str = typer.Option("play", "--mode"),
    players: int = typer.Option(4, "--players"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    run_ui(
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        host=host,
        port=port,
        mode=mode,
        players=players,
        device=device,
    )


@app.command()
def spectate(
    players: int = typer.Option(4, "--players"),
    seed: int = typer.Option(0, "--seed"),
    bot: str = typer.Option("heuristic", "--bot"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    device: str = typer.Option("cpu", "--device"),
    seat_config: Optional[str] = typer.Option(None, "--seat-config"),
) -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=players, seed=seed))
    env.reset(seed=seed)
    checkpoint_agent = None
    if checkpoint is not None:
        policy, _ = load_policy_checkpoint(checkpoint, device=device)
        checkpoint_agent = PolicyAgent(policy, device=device, greedy=True)
    roles = _seat_roles(
        players,
        seat_config,
        default_roles=[_normalize_role(bot) for _ in range(players)],
        require_model_checkpoint=checkpoint is None,
    )
    if "human" in roles:
        raise typer.BadParameter("Spectate mode does not support human seats. Use the play command instead.")
    agents = _bot_agents_for_roles(roles, seed, checkpoint_agent)

    while not all(env.terminations.values()):
        typer.echo(env.render())
        actor = env.agent_selection
        seat = env.agent_index(actor)
        action = agents[seat].select_action(env.observe(actor))
        if action < 9:
            typer.echo("P{seat} ({role}) bids {bid}".format(seat=seat, role=roles[seat], bid=action))
        else:
            typer.echo(
                "P{seat} ({role}) plays {card}".format(
                    seat=seat, role=roles[seat], card=card_label(action - 9)
                )
            )
        env.step(action)
    typer.echo("Final scores: {scores}".format(scores=env.serialize_replay()["scores"]))


@app.command()
def play(
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    players: int = typer.Option(4, "--players"),
    seat: int = typer.Option(0, "--seat"),
    seed: int = typer.Option(0, "--seed"),
    bot: str = typer.Option("heuristic", "--bot"),
    device: str = typer.Option("cpu", "--device"),
    seat_config: Optional[str] = typer.Option(None, "--seat-config"),
) -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=players, seed=seed))
    env.reset(seed=seed)
    model_agent = None
    if checkpoint is not None:
        policy, _ = load_policy_checkpoint(checkpoint, device=device)
        model_agent = PolicyAgent(policy, device=device, greedy=True)
    roles = _seat_roles(
        players,
        seat_config,
        default_roles=_default_play_roles(players, seat, checkpoint, bot),
        require_model_checkpoint=checkpoint is None,
    )
    agents = _bot_agents_for_roles(roles, seed, model_agent)

    while not all(env.terminations.values()):
        typer.echo(env.render())
        actor = env.agent_selection
        actor_seat = env.agent_index(actor)
        observation = env.observe(actor)
        if roles[actor_seat] == "human":
            legal_actions = [index for index, enabled in enumerate(observation["legal_action_mask"]) if enabled]
            typer.echo(
                "P{seat} is human. Legal actions: {actions}".format(
                    seat=actor_seat, actions=_format_actions(legal_actions)
                )
            )
            chosen = int(typer.prompt("Action"))
            env.step(chosen)
            continue
        action = agents[actor_seat].select_action(observation)
        typer.echo(
            "P{seat} ({role}) -> {action}".format(
                seat=actor_seat, role=roles[actor_seat], action=_format_actions([action])[0]
            )
        )
        env.step(action)
    typer.echo("Final scores: {scores}".format(scores=env.serialize_replay()["scores"]))


def _format_actions(actions: list[int]) -> list[str]:
    formatted = []
    for action in actions:
        if action < 9:
            formatted.append("bid:{value}".format(value=action))
        else:
            formatted.append("card:{value}".format(value=card_label(action - 9)))
    return formatted


if __name__ == "__main__":
    app()
