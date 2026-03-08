from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from torch.utils.tensorboard import SummaryWriter

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.checkpoint import save_checkpoint
from romanian_whist.agents.model import PolicyAgent, WhistPolicyNetwork
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.config import WhistVariantConfig
from romanian_whist.train.curriculum import CurriculumScheduler
from romanian_whist.train.eval import TournamentRunner, average_stat_dicts, stats_to_dict
from romanian_whist.train.ppo import PPOConfig, PPOTrainer, RolloutBuffer


@dataclass
class LeagueConfig:
    total_updates: int = 100
    episodes_per_update: int = 24
    snapshot_interval: int = 5
    max_snapshots: int = 8
    latest_weight: float = 0.50
    snapshot_weight: float = 0.35
    scripted_weight: float = 0.15
    checkpoint_dir: Optional[Path] = None
    device: str = "cpu"
    seed: int = 0
    balanced_player_count_sampling: bool = True
    evaluation_matches: int = 4
    evaluation_interval: int = 1
    evaluation_player_counts: tuple[int, ...] = (3, 4, 5, 6)
    save_best_checkpoint: bool = True
    tensorboard_log_dir: Optional[Path] = None


@dataclass
class LeagueTrainer:
    variant_config: WhistVariantConfig
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    league_config: LeagueConfig = field(default_factory=LeagueConfig)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.league_config.seed)
        self.policy = WhistPolicyNetwork()
        self.ppo = PPOTrainer(self.policy, self.ppo_config, device=self.league_config.device)
        self.curriculum = CurriculumScheduler(self.league_config.total_updates)
        self.snapshots = []  # type: List[WhistPolicyNetwork]
        self.best_selection_score = float("-inf")
        self.writer = (
            SummaryWriter(log_dir=str(self.league_config.tensorboard_log_dir))
            if self.league_config.tensorboard_log_dir is not None
            else None
        )
        self.scripted_pool = [
            RandomLegalAgent(seed=self.league_config.seed),
            SafeHeuristicAgent(seed=self.league_config.seed + 1),
            BidPlayHeuristicAgent(seed=self.league_config.seed + 2),
        ]

    def train(self, updates: Optional[int] = None, start_update: int = 0) -> List[Dict[str, float]]:
        total_updates = updates or self.league_config.total_updates
        history = []
        try:
            for local_update_index in range(total_updates):
                update_index = start_update + local_update_index + 1
                stage = self.curriculum.stage_for_update(update_index - 1)
                buffer = self.collect_rollouts(stage.player_counts, stage.one_card_modes, self.league_config.episodes_per_update)
                metrics = self.ppo.update(buffer)
                metrics["stage"] = stage.name
                metrics["transitions"] = float(len(buffer))
                evaluation = None
                should_evaluate = (
                    update_index % self.league_config.evaluation_interval == 0
                    or update_index == start_update + total_updates
                )
                if should_evaluate:
                    evaluation = self.evaluate(matches=self.league_config.evaluation_matches)
                    metrics["selection_score"] = self.selection_score(evaluation)
                history.append(metrics)
                self._log_update(update_index, metrics, evaluation)
                if update_index % self.league_config.snapshot_interval == 0:
                    self._promote_snapshot()
                self._save_checkpoint(update_index, metrics, evaluation)
                self._save_best_checkpoint(update_index, metrics, evaluation)
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        return history

    def collect_rollouts(
        self,
        player_counts: tuple[int, ...],
        one_card_modes: tuple[object, ...],
        episodes: int,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer()
        player_count_schedule = self._player_count_schedule(player_counts, episodes)
        for episode_index, players in enumerate(player_count_schedule):
            config = self.variant_config.replace(players=players, one_card_modes=tuple(one_card_modes))
            env = RomanianWhistEnv(config)
            env.reset(seed=self.rng.randint(0, 1_000_000))
            focal_seat = episode_index % players
            focal_agent_name = env.possible_agents[focal_seat]
            opponent_agents = self._sample_opponents(players, focal_seat)
            last_transition_index = None
            while not all(env.terminations.values()):
                acting_agent = env.agent_selection
                seat = env.agent_index(acting_agent)
                observation = env.observe(acting_agent)
                if seat == focal_seat:
                    action, log_prob, value = self.ppo.select_action(observation)
                    transition = env.step(action)
                    reward = transition.rewards[focal_agent_name]
                    done = transition.terminations[focal_agent_name]
                    last_transition_index = buffer.add(
                        observation=observation,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        done=done,
                    )
                else:
                    action = opponent_agents[seat].select_action(observation)
                    transition = env.step(action)
                    if last_transition_index is not None:
                        buffer.rewards[last_transition_index] += transition.rewards[focal_agent_name]
                        buffer.dones[last_transition_index] = transition.terminations[focal_agent_name]
        return buffer

    def evaluate(self, matches: int = 8) -> Dict[str, object]:
        per_player_count = {}  # type: Dict[str, Dict[str, Dict[str, float]]]
        stat_dicts = []
        for players in self.league_config.evaluation_player_counts:
            runner = TournamentRunner(self.variant_config.replace(players=players))
            participants, label_groups = self._evaluation_participants(players)
            raw_stats = stats_to_dict(runner.run(participants, matches=matches, seed=self.league_config.seed + players))
            stats = self._aggregate_label_groups(raw_stats, label_groups)
            per_player_count[str(players)] = stats
            stat_dicts.append(stats)
        overall = average_stat_dicts(stat_dicts)
        return {
            "overall": overall,
            "by_player_count": per_player_count,
        }

    @staticmethod
    def selection_score(evaluation: Dict[str, object]) -> float:
        overall = evaluation["overall"]
        return float(overall["average_scores"]["policy"])

    def _player_count_schedule(self, player_counts: Sequence[int], episodes: int) -> List[int]:
        counts = list(dict.fromkeys(player_counts))
        if not counts:
            raise ValueError("At least one player count must be provided.")
        if not self.league_config.balanced_player_count_sampling or len(counts) == 1:
            return [self.rng.choice(counts) for _ in range(episodes)]
        self.rng.shuffle(counts)
        return [counts[index % len(counts)] for index in range(episodes)]

    def _evaluation_participants(self, players: int) -> tuple[List[tuple[str, object]], Dict[str, str]]:
        participants = [
            ("policy_0", PolicyAgent(self.policy, device=self.league_config.device, greedy=True)),
        ]
        label_groups = {"policy_0": "policy"}
        family_factories = [
            ("random", lambda offset: RandomLegalAgent(seed=self.league_config.seed + 100 + offset)),
            ("safe", lambda offset: SafeHeuristicAgent(seed=self.league_config.seed + 200 + offset)),
            ("heuristic", lambda offset: BidPlayHeuristicAgent(seed=self.league_config.seed + 300 + offset)),
        ]
        for seat_index in range(1, players):
            family_name, factory = family_factories[(seat_index - 1) % len(family_factories)]
            seat_label = "{family}_{index}".format(family=family_name, index=seat_index - 1)
            participants.append((seat_label, factory(seat_index)))
            label_groups[seat_label] = family_name
        return participants, label_groups

    @staticmethod
    def _aggregate_label_groups(
        raw_stats: Dict[str, Dict[str, float]],
        label_groups: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        families = ("policy", "random", "safe", "heuristic")
        aggregated = {}  # type: Dict[str, Dict[str, float]]
        for metric_name, values in raw_stats.items():
            aggregated[metric_name] = {}
            for family in families:
                family_values = [metric_value for label, metric_value in values.items() if label_groups[label] == family]
                aggregated[metric_name][family] = (
                    sum(family_values) / float(len(family_values)) if family_values else 0.0
                )
        return aggregated

    def _sample_opponents(self, players: int, focal_seat: int) -> List[object]:
        pool = []
        for seat in range(players):
            if seat == focal_seat:
                pool.append(PolicyAgent(self.policy, device=self.league_config.device, greedy=False))
                continue
            roll = self.rng.random()
            if self.snapshots and roll < self.league_config.snapshot_weight:
                snapshot = self.rng.choice(self.snapshots)
                pool.append(PolicyAgent(snapshot, device=self.league_config.device, greedy=True))
            elif roll < self.league_config.snapshot_weight + self.league_config.scripted_weight:
                pool.append(copy.deepcopy(self.rng.choice(self.scripted_pool)))
            else:
                pool.append(PolicyAgent(self.policy, device=self.league_config.device, greedy=True))
        return pool

    def _promote_snapshot(self) -> None:
        snapshot = copy.deepcopy(self.policy).cpu()
        snapshot.eval()
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.league_config.max_snapshots:
            self.snapshots = self.snapshots[-self.league_config.max_snapshots :]

    def _save_checkpoint(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.league_config.checkpoint_dir is None:
            return
        path = self.league_config.checkpoint_dir / "update-{index:04d}.pt".format(index=update_index)
        save_checkpoint(
            path,
            self.policy,
            optimizer=self.ppo.optimizer,
            metadata={"metrics": metrics, "evaluation": evaluation, "update": update_index},
        )
        if evaluation is not None:
            report_path = self.league_config.checkpoint_dir / "update-{index:04d}.eval.json".format(index=update_index)
            report_path.write_text(json.dumps(evaluation, indent=2))

    def _save_best_checkpoint(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.league_config.checkpoint_dir is None or not self.league_config.save_best_checkpoint or evaluation is None:
            return
        selection_score = self.selection_score(evaluation)
        if selection_score <= self.best_selection_score:
            return
        self.best_selection_score = selection_score
        path = self.league_config.checkpoint_dir / "best.pt"
        save_checkpoint(
            path,
            self.policy,
            optimizer=self.ppo.optimizer,
            metadata={"metrics": metrics, "evaluation": evaluation, "update": update_index, "best": True},
        )
        report_path = self.league_config.checkpoint_dir / "best.eval.json"
        report_path.write_text(json.dumps(evaluation, indent=2))

    def _log_update(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.writer is None:
            return
        stage_map = {"stage_1": 1.0, "stage_2": 2.0, "stage_3": 3.0}
        for key, value in metrics.items():
            if key == "stage":
                self.writer.add_scalar("train/stage_index", stage_map.get(str(value), 0.0), update_index)
                self.writer.add_text("train/stage_name", str(value), global_step=update_index)
                continue
            self.writer.add_scalar("train/{key}".format(key=key), float(value), update_index)
        if evaluation is not None:
            for tag, value in self._flatten_scalars(evaluation):
                self.writer.add_scalar(tag, value, update_index)
        self.writer.flush()

    def _flatten_scalars(self, value: object, prefix: str = "eval") -> List[tuple[str, float]]:
        scalars = []  # type: List[tuple[str, float]]
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_prefix = "{prefix}/{key}".format(prefix=prefix, key=key)
                scalars.extend(self._flatten_scalars(nested_value, nested_prefix))
        elif isinstance(value, (int, float)):
            scalars.append((prefix, float(value)))
        return scalars
