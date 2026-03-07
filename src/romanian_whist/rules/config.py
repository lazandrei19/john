from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Tuple


class OneCardMode(str, Enum):
    REGULAR = "regular"
    FOREHEAD = "forehead"
    BLIND = "blind"


@dataclass(frozen=True)
class WhistVariantConfig:
    players: int = 4
    seed: int = 0
    one_card_modes: Tuple[OneCardMode, ...] = field(default_factory=lambda: (OneCardMode.REGULAR,))
    max_history_tokens: int = 64
    max_players: int = 6
    max_hand_size: int = 8

    def __post_init__(self) -> None:
        if self.players < 3 or self.players > 6:
            raise ValueError("Romanian whist supports between 3 and 6 players.")
        if self.max_hand_size != 8:
            raise ValueError("This implementation assumes an 8-card maximum hand size.")
        if not self.one_card_modes:
            raise ValueError("At least one one-card mode must be configured.")

    def schedule(self) -> Tuple[int, ...]:
        plateau = tuple(self.max_hand_size for _ in range(self.players))
        descending = tuple(range(self.max_hand_size - 1, 0, -1))
        one_plateau = tuple(1 for _ in range(self.players))
        ascending = tuple(range(2, self.max_hand_size + 1))
        return plateau + descending + one_plateau + ascending + plateau

    def one_card_mode_for_index(self, one_card_round_index: int) -> OneCardMode:
        return self.one_card_modes[one_card_round_index % len(self.one_card_modes)]

    def replace(self, **changes: object) -> "WhistVariantConfig":
        values = {
            "players": self.players,
            "seed": self.seed,
            "one_card_modes": self.one_card_modes,
            "max_history_tokens": self.max_history_tokens,
            "max_players": self.max_players,
            "max_hand_size": self.max_hand_size,
        }
        values.update(changes)
        return WhistVariantConfig(**values)


def normalize_one_card_modes(modes: Sequence[str]) -> Tuple[OneCardMode, ...]:
    return tuple(OneCardMode(mode) for mode in modes)
