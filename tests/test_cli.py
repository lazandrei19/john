import pathlib

import pytest

from romanian_whist.cli.main import (
    _default_play_roles,
    _parse_seat_config,
    _seat_roles,
    _train_command,
    _write_resume_scripts,
)


def test_parse_seat_config_accepts_supported_roles() -> None:
    roles = _parse_seat_config(4, "human,model,bot,random")
    assert roles == ["human", "model", "heuristic", "random"]


def test_parse_seat_config_rejects_wrong_length() -> None:
    with pytest.raises(Exception):
        _parse_seat_config(4, "human,model")


def test_default_play_roles_keep_backward_compatibility() -> None:
    assert _default_play_roles(4, seat=0, checkpoint=pathlib.Path("best.pt"), bot="safe") == [
        "human",
        "model",
        "safe",
        "safe",
    ]


def test_model_role_requires_checkpoint() -> None:
    with pytest.raises(Exception):
        _seat_roles(
            4,
            "human,model,heuristic,random",
            default_roles=["human", "heuristic", "heuristic", "heuristic"],
            require_model_checkpoint=True,
        )


def test_write_resume_scripts_creates_latest_and_best_helpers(tmp_path: pathlib.Path) -> None:
    command = _train_command(
        output=tmp_path,
        updates=10,
        episodes_per_update=4,
        learning_rate=3e-4,
        embed_dim=128,
        players=4,
        seed=5,
        device="cpu",
        one_card_modes="regular,forehead,blind",
        universal=True,
        evaluation_matches=2,
        evaluation_every=1,
        entropy_coef=0.01,
        final_entropy_coef=None,
        gae_lambda=0.95,
        reward_shaping=0.5,
        final_reward_shaping=None,
        latest_weight=0.5,
        snapshot_weight=0.35,
        scripted_weight=0.15,
        rollout_workers=1,
        eval_workers=1,
        batch_size=64,
        tensorboard_logdir=tmp_path / "tensorboard",
    )

    _write_resume_scripts(tmp_path, command)

    latest = tmp_path / "resume_latest.sh"
    best = tmp_path / "resume_best.sh"
    assert latest.exists()
    assert best.exists()
    assert '--resume-from "$CHECKPOINT"' in latest.read_text()
    assert 'update-*.pt' in latest.read_text()
    assert 'CHECKPOINT="$OUTPUT/best.pt"' in best.read_text()
    assert "imitation" not in latest.read_text()
    assert "imitation" not in best.read_text()
