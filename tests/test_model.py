import pytest
import torch

from romanian_whist.agents import model as model_module
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.config import WhistVariantConfig


def test_batch_observations_tensorizes_each_observation_once(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = [
        {"hand_mask": [0, 1], "phase": 0, "legal_action_mask": [1, 0]},
        {"hand_mask": [1, 0], "phase": 1, "legal_action_mask": [0, 1]},
    ]
    original = model_module.tensorize_observation
    calls = {"count": 0}

    def wrapped(observation, device=None):
        calls["count"] += 1
        return original(observation, device=device)

    monkeypatch.setattr(model_module, "tensorize_observation", wrapped)
    batch = model_module.batch_observations(observations)
    assert calls["count"] == len(observations)
    assert tuple(batch["hand_mask"].shape) == (2, 2)


def test_policy_forward_with_aux_outputs_belief_logits() -> None:
    env = RomanianWhistEnv(WhistVariantConfig(players=4, seed=7))
    observation = env.reset(seed=7)
    batch = model_module.batch_observations([observation], device=torch.device("cpu"))
    policy = model_module.WhistPolicyNetwork()
    outputs = policy.forward_with_aux(batch)
    assert tuple(outputs.logits.shape) == (1, 61)
    assert tuple(outputs.values.shape) == (1,)
    assert tuple(outputs.belief_logits.shape) == (1, 52, policy.max_players + 2)
