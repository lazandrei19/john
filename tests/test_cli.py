import pathlib

import pytest

from romanian_whist.cli.main import _default_play_roles, _parse_seat_config, _seat_roles


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
