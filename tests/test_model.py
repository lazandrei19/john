import pytest

from romanian_whist.agents import model as model_module


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
