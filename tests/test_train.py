import pathlib
import platform
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from romanian_whist.agents.checkpoint import load_policy_checkpoint
from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.model import PolicyAgent, PolicyNetworkConfig, WhistPolicyNetwork
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.mlx_support.converter import CheckpointConverter
from romanian_whist.rules.config import WhistVariantConfig
from romanian_whist.train.league import LeagueConfig, LeagueTrainer
from romanian_whist.train.ppo import PPOConfig, PPOTrainer, RolloutBuffer


def test_ppo_smoke_and_checkpoint_roundtrip(tmp_path: pathlib.Path) -> None:
    tensorboard_dir = tmp_path / "tensorboard"
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=21),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=2,
            checkpoint_dir=tmp_path,
            seed=21,
            evaluation_matches=1,
            evaluation_player_counts=(3, 4),
            tensorboard_log_dir=tensorboard_dir,
        ),
    )
    history = trainer.train(updates=1)
    assert history
    checkpoint = tmp_path / "update-0001.pt"
    assert checkpoint.exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "best.eval.json").exists()
    policy, payload = load_policy_checkpoint(checkpoint)
    assert payload["metadata"]["update"] == 1
    assert "overall" in payload["metadata"]["evaluation"]
    assert "timing/rollout_sec" in payload["metadata"]["metrics"]
    assert "timing/eval_sec" in payload["metadata"]["metrics"]
    assert "timing/update_sec" in history[-1]
    assert list(tensorboard_dir.glob("events.out.tfevents.*"))
    assert policy is not None


def test_checkpoint_roundtrip_preserves_model_config(tmp_path: pathlib.Path) -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=6, seed=31),
        policy_config=PolicyNetworkConfig(embed_dim=256),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=1,
            checkpoint_dir=tmp_path,
            seed=31,
            evaluation_matches=1,
            evaluation_player_counts=(6,),
        ),
    )
    trainer.train(updates=1)
    policy, payload = load_policy_checkpoint(tmp_path / "update-0001.pt")
    assert policy.config.embed_dim == 256
    assert payload["model_config"]["embed_dim"] == 256


def test_resume_training_continues_checkpoint_numbering(tmp_path: pathlib.Path) -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=17),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=1,
            checkpoint_dir=tmp_path,
            seed=17,
            evaluation_matches=1,
            evaluation_player_counts=(3, 4),
        ),
    )
    trainer.train(updates=1)
    policy, payload = load_policy_checkpoint(tmp_path / "update-0001.pt")

    resumed = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=17),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=1,
            checkpoint_dir=tmp_path,
            seed=17,
            evaluation_matches=1,
            evaluation_player_counts=(3, 4),
        ),
    )
    resumed.policy.load_state_dict(policy.state_dict())
    resumed.ppo.optimizer.load_state_dict(payload["optimizer_state"])
    resumed.best_selection_score = resumed.selection_score(payload["metadata"]["evaluation"])
    resumed.train(updates=1, start_update=int(payload["metadata"]["update"]))

    checkpoint = tmp_path / "update-0002.pt"
    assert checkpoint.exists()
    _, resumed_payload = load_policy_checkpoint(checkpoint)
    assert resumed_payload["metadata"]["update"] == 2


def test_sparse_evaluation_only_writes_reports_on_interval(tmp_path: pathlib.Path) -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=9),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=2,
            episodes_per_update=1,
            checkpoint_dir=tmp_path,
            seed=9,
            evaluation_matches=1,
            evaluation_interval=2,
            evaluation_player_counts=(3, 4),
        ),
    )
    history = trainer.train(updates=2)
    assert "selection_score" not in history[0]
    assert "selection_score" in history[1]
    assert not (tmp_path / "update-0001.eval.json").exists()
    assert (tmp_path / "update-0002.eval.json").exists()


def test_policy_agent_uses_model_device_for_inference() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=4, seed=3))
    env.reset(seed=3)
    agent = PolicyAgent(policy=WhistPolicyNetwork(), device="cuda", greedy=True)
    action = agent.select_action(env.observe(env.agent_selection))
    assert isinstance(action, int)


def test_scripted_agents_fast_path_matches_observation_path() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=4, seed=16))
    env.reset(seed=16)
    actor = env.agent_selection
    seat = env.agent_index(actor)
    observation = env.observe(actor)

    for agent_cls in (RandomLegalAgent, SafeHeuristicAgent, BidPlayHeuristicAgent):
        from_observation = agent_cls(seed=16)
        from_game = agent_cls(seed=16)
        assert from_observation.select_action(observation) == from_game.select_action_from_game(env.game, seat)


@pytest.mark.skipif(platform.system() == "Windows", reason="Process pool smoke test is not used on Windows here.")
def test_parallel_rollout_and_evaluation_smoke(tmp_path: pathlib.Path) -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=11),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=2,
            checkpoint_dir=tmp_path,
            seed=11,
            evaluation_matches=2,
            evaluation_interval=1,
            evaluation_player_counts=(3, 4),
            rollout_workers=2,
            eval_workers=2,
        ),
    )
    history = trainer.train(updates=1)
    assert history
    assert (tmp_path / "update-0001.pt").exists()
    assert (tmp_path / "update-0001.eval.json").exists()


@pytest.mark.skipif(platform.system() == "Windows", reason="Process pool smoke test is not used on Windows here.")
def test_persistent_rollout_pool_is_reused() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=13),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=2,
            seed=13,
            rollout_workers=2,
        ),
    )
    try:
        buffer_one = trainer.collect_rollouts((3, 4), (trainer.variant_config.one_card_modes[0],), 2)
        pool = trainer.rollout_pool
        buffer_two = trainer.collect_rollouts((3, 4), (trainer.variant_config.one_card_modes[0],), 2)
        assert pool is trainer.rollout_pool
        assert len(buffer_one) > 0
        assert len(buffer_two) > 0
    finally:
        if trainer.rollout_pool is not None:
            trainer.rollout_pool.close()
            trainer.rollout_pool = None


def test_cuda_parallel_rollouts_route_to_gpu_inference_path() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=14),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=2, seed=14, rollout_workers=2),
    )
    trainer.league_config.device = "cuda"
    captured = []

    def fake_gpu_collect(self: LeagueTrainer, player_counts: tuple[int, ...], one_card_modes: tuple[object, ...], episodes: int) -> RolloutBuffer:
        captured.append((player_counts, one_card_modes, episodes))
        return RolloutBuffer()

    trainer._collect_rollouts_parallel_gpu_inference = MethodType(fake_gpu_collect, trainer)

    trainer.collect_rollouts((3, 4), (trainer.variant_config.one_card_modes[0],), 2)

    assert captured == [((3, 4), (trainer.variant_config.one_card_modes[0],), 2)]


def test_checkpoint_converter_exports_npz(tmp_path: pathlib.Path) -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=4),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=1,
            checkpoint_dir=tmp_path,
            seed=4,
            evaluation_matches=1,
            evaluation_player_counts=(3,),
        ),
    )
    trainer.train(updates=1)
    converter = CheckpointConverter()
    result = converter.export(tmp_path / "update-0001.pt", tmp_path / "mlx")
    assert pathlib.Path(result["weights"]).exists()
    assert pathlib.Path(result["metadata"]).exists()


def test_league_evaluation_contains_expected_metrics() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=2),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=1, seed=2, evaluation_matches=1),
    )
    stats = trainer.evaluate(matches=1)
    assert set(stats.keys()) == {"overall", "by_player_count"}
    assert set(stats["overall"].keys()) == {"average_scores", "contract_hit_rate", "trick_differential", "elo_like"}
    assert set(stats["by_player_count"].keys()) == {"3", "4", "5", "6"}


def test_balanced_player_count_schedule_covers_all_counts() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=5),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=8, seed=5),
    )
    schedule = trainer._player_count_schedule((3, 4, 5, 6), 8)
    assert sorted(schedule) == [3, 3, 4, 4, 5, 5, 6, 6]


def test_fixed_player_count_training_constrains_rollout_counts() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=6, seed=6),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(
            total_updates=1,
            episodes_per_update=1,
            seed=6,
            evaluation_interval=999,
            rollout_player_counts=(6,),
        ),
    )
    captured = []

    def fake_collect_rollouts(self: LeagueTrainer, player_counts: tuple[int, ...], one_card_modes: tuple[object, ...], episodes: int) -> RolloutBuffer:
        captured.append((player_counts, one_card_modes, episodes))
        return RolloutBuffer()

    trainer.collect_rollouts = MethodType(fake_collect_rollouts, trainer)
    trainer.ppo.update = lambda buffer: {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}  # type: ignore[method-assign]
    trainer._save_checkpoint = lambda *args, **kwargs: None  # type: ignore[method-assign]
    trainer._save_best_checkpoint = lambda *args, **kwargs: None  # type: ignore[method-assign]

    history = trainer.train(updates=1)

    assert history
    assert captured == [((6,), trainer.curriculum.stage_for_update(0).one_card_modes, 1)]


def test_imitation_pretraining_smoke() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=8),
        ppo_config=PPOConfig(epochs=1, batch_size=8, imitation_epochs=1, imitation_batch_size=16),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=1, seed=8, imitation_episodes=2),
    )
    metrics = trainer.pretrain_imitation(episodes=2)
    assert set(metrics.keys()) == {"imitation/loss", "imitation/action_loss", "imitation/belief_loss"}


def test_ppo_update_restores_train_mode_after_rollout_inference() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=4, seed=15))
    observation = env.reset(seed=15)
    trainer = PPOTrainer(WhistPolicyNetwork(), PPOConfig(epochs=1, batch_size=1, mixed_precision=False), device="cpu")
    action, log_prob, value = trainer.select_action(observation)

    buffer = RolloutBuffer(
        observations=[observation],
        actions=[action],
        log_probs=[log_prob],
        values=[value],
        next_values=[0.0],
        rewards=[0.0],
        dones=[True],
        trajectory_ids=[0],
    )

    trainer.update(buffer)

    assert trainer.policy.training


def test_returns_and_advantages_use_explicit_next_values_for_interleaved_trajectories() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=12),
        ppo_config=PPOConfig(gamma=1.0, gae_lambda=1.0, epochs=1, batch_size=8),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=1, seed=12),
    )
    buffer = RolloutBuffer(
        observations=[{}, {}, {}],
        actions=[0, 0, 0],
        log_probs=[0.0, 0.0, 0.0],
        values=[1.0, 10.0, 2.0],
        next_values=[2.0, 0.0, 0.0],
        rewards=[5.0, 7.0, 11.0],
        dones=[False, True, True],
        trajectory_ids=[0, 1, 0],
    )

    returns, advantages = trainer.ppo._returns_and_advantages(buffer)

    assert returns.tolist() == pytest.approx([16.0, 7.0, 11.0])
    assert advantages.tolist() == pytest.approx([15.0, -3.0, 9.0])
