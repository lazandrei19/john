from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.config import OneCardMode, WhistVariantConfig


def test_env_reset_is_deterministic() -> None:
    config = WhistVariantConfig(players=4, seed=11, one_card_modes=(OneCardMode.REGULAR,))
    env_a = RomanianWhistEnv(config)
    env_b = RomanianWhistEnv(config)
    obs_a = env_a.reset(seed=11)
    obs_b = env_b.reset(seed=11)
    assert obs_a["hand_mask"].tolist() == obs_b["hand_mask"].tolist()
    assert obs_a["history_tokens"].tolist() == obs_b["history_tokens"].tolist()


def test_env_has_legal_action_mask() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=5, seed=5))
    observation = env.reset(seed=5)
    assert sum(observation["legal_action_mask"]) > 0
    assert len(observation["legal_action_mask"]) == 61


def test_replay_round_trip_contains_scores() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=3, seed=9))
    env.reset(seed=9)
    for _ in range(12):
        if all(env.terminations.values()):
            break
        actor = env.agent_selection
        observation = env.observe(actor)
        legal = [index for index, enabled in enumerate(observation["legal_action_mask"]) if enabled]
        env.step(legal[0])
    replay = env.serialize_replay()
    assert "events" in replay
    assert "scores" in replay
