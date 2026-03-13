import json
import pathlib
import platform
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from romanian_whist.agents.checkpoint import load_policy_checkpoint
from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.model import PolicyAgent, PolicyForwardOutputs, PolicyNetworkConfig, WhistPolicyNetwork
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.mlx_support.converter import CheckpointConverter
from romanian_whist.rules.config import WhistVariantConfig
from romanian_whist.train.league import (
    LeagueConfig,
    LeagueTrainer,
    _apply_focal_bid_feedback,
    _bid_alignment_reward,
    _round_potential,
)
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
    assert "policy_bid/min" in history[-1]
    assert "policy_bid/max" in history[-1]
    assert "policy_bid/mae_vs_actual" in history[-1]
    assert "policy_bid/strong_hand_underbid_rate" in history[-1]
    assert list(tensorboard_dir.glob("events.out.tfevents.*"))
    diagnostics = json.loads((tmp_path / "training_diagnostics.json").read_text())
    assert diagnostics["summary"]["latest_metrics"]["timing/update_sec"] == history[-1]["timing/update_sec"]
    assert diagnostics["summary"]["best_update"] == 1
    assert diagnostics["history"][-1]["update"] == 1
    assert isinstance(diagnostics["pain_points"], list)
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
    diagnostics = json.loads((tmp_path / "training_diagnostics.json").read_text())
    assert [entry["update"] for entry in diagnostics["history"]] == [1, 2]
    assert diagnostics["run"]["resumed"] is True
    assert diagnostics["run"]["start_update"] == 1


def test_old_checkpoint_format_is_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "old.pt"
    torch.save(
        {
            "model_state": WhistPolicyNetwork().state_dict(),
            "model_config": WhistPolicyNetwork().config_dict(),
            "metadata": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="checkpoint_version"):
        load_policy_checkpoint(path)


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
    diagnostics = json.loads((tmp_path / "training_diagnostics.json").read_text())
    assert diagnostics["summary"]["evaluated_updates"] == [2]
    assert diagnostics["history"][0]["evaluation"] is None
    assert diagnostics["history"][1]["evaluation"] is not None


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
    assert set(stats["overall"].keys()) == {
        "average_scores",
        "contract_hit_rate",
        "trick_differential",
        "elo_like",
        "average_bid",
        "min_bid",
        "max_bid",
        "bid_mae",
        "underbid_rate",
        "overbid_rate",
        "strong_hand_underbid_rate",
    }
    assert set(stats["by_player_count"].keys()) == {"3", "4", "5", "6"}


def test_rollout_bid_metrics_include_hand_size_breakdown() -> None:
    trainer = LeagueTrainer(
        variant_config=WhistVariantConfig(players=4, seed=24),
        ppo_config=PPOConfig(epochs=1, batch_size=8),
        league_config=LeagueConfig(total_updates=1, episodes_per_update=1, seed=24),
    )

    buffer = trainer.collect_rollouts((4,), trainer.variant_config.one_card_modes, 1)

    assert len(buffer) > 0
    assert "policy_bid/count" in trainer.last_rollout_stats
    assert "policy_bid/by_hand_size/1/mean" in trainer.last_rollout_stats
    assert "policy_bid/by_hand_size/8/rate_1" in trainer.last_rollout_stats


def test_apply_focal_bid_feedback_retargets_bid_transition_to_actual_tricks() -> None:
    rewards = [0.0]
    observations = [{"expected_trick_target": 3}]
    events = [
        {"type": "round_start", "round": 0, "hand_size": 4},
        {"type": "bid", "player": 1, "bid": 2},
        {"type": "trick_win", "player": 1, "round": 0},
        {"type": "trick_win", "player": 1, "round": 0},
        {"type": "round_score", "round": 0},
    ]

    _apply_focal_bid_feedback(
        events,
        focal_seat=1,
        bid_transition_indices={0: 0},
        rewards=rewards,
        observations=observations,
        shaping_coef=0.5,
        strong_hand_underbid_penalty=1.0,
    )

    assert rewards[0] == pytest.approx(0.5)
    assert observations[0]["expected_trick_target"] == 2


def test_bid_alignment_reward_penalizes_strong_hand_underbids_more_than_overbids() -> None:
    underbid_reward = _bid_alignment_reward(2, 4, 5, strong_hand_underbid_penalty=1.0)
    overbid_reward = _bid_alignment_reward(6, 4, 5, strong_hand_underbid_penalty=1.0)

    assert underbid_reward < overbid_reward


def test_round_potential_penalizes_winning_risk_after_contract_is_met() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=4, seed=9))
    env.reset(seed=9)
    state = env.game.round_state
    assert state is not None
    state.phase = "play"
    state.trump_suit = 0
    state.bids[0] = 1
    state.tricks_won[0] = 1
    state.hand_size = 3

    state.hands[0] = [12, 11]
    risky_potential = _round_potential(env, 0)

    state.hands[0] = [1, 14]
    safe_potential = _round_potential(env, 0)

    assert risky_potential < safe_potential


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


def test_expected_trick_loss_only_uses_bidding_samples_and_masks_large_classes() -> None:
    trainer = PPOTrainer(
        WhistPolicyNetwork(),
        PPOConfig(epochs=1, batch_size=1, mixed_precision=False),
        device="cpu",
    )
    outputs = PolicyForwardOutputs(
        logits=torch.zeros((2, 61), dtype=torch.float32),
        values=torch.zeros(2, dtype=torch.float32),
        belief_logits=torch.zeros((2, 52, trainer.policy.max_players + 2), dtype=torch.float32),
        expected_trick_logits=torch.tensor(
            [
                [0.0, 0.0, 10.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    batch_obs = {
        "expected_trick_target": torch.tensor([2, 8], dtype=torch.int64),
        "phase": torch.tensor([0, 1], dtype=torch.int64),
        "hand_size": torch.tensor([2, 8], dtype=torch.int64),
    }

    loss = trainer._expected_trick_loss(outputs, batch_obs)

    assert float(loss.item()) < 1e-3


def test_entropy_bonus_scales_bidding_and_play_differently() -> None:
    trainer = PPOTrainer(
        WhistPolicyNetwork(),
        PPOConfig(epochs=1, batch_size=1, mixed_precision=False, entropy_coef=0.5, bid_entropy_scale=2.0, play_entropy_scale=1.0),
        device="cpu",
    )

    bonus = trainer._entropy_bonus(
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        {"phase": torch.tensor([0, 1], dtype=torch.int64)},
    )

    assert float(bonus.item()) == pytest.approx(0.75)


def test_selection_score_penalizes_strong_hand_underbidding() -> None:
    evaluation = {
        "overall": {
            "average_scores": {"policy": 1.5},
            "contract_hit_rate": {"policy": 0.4},
            "trick_differential": {"policy": -0.5},
            "strong_hand_underbid_rate": {"policy": 0.2},
        }
    }

    score = LeagueTrainer.selection_score(evaluation)

    assert score == pytest.approx(1.5 + (20.0 * 0.4) + (2.0 * -0.5) - (5.0 * 0.2))


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
