from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

SUITS = ("clubs", "diamonds", "hearts", "spades")
SUIT_SYMBOLS = ("C", "D", "H", "S")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
_CARD_LOOKUP = {
    "{rank}{symbol}".format(rank=rank_value, symbol=symbol): (suit_index * len(RANKS)) + rank_index
    for suit_index, symbol in enumerate(SUIT_SYMBOLS)
    for rank_index, rank_value in enumerate(RANKS)
}  # type: Dict[str, int]


def suit(card_id: int) -> int:
    return card_id // len(RANKS)


def rank(card_id: int) -> int:
    return card_id % len(RANKS)


def card_label(card_id: int) -> str:
    return "{rank}{suit}".format(rank=RANKS[rank(card_id)], suit=SUIT_SYMBOLS[suit(card_id)])


def parse_card(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().upper()
    if text.isdigit():
        return int(text)
    if text not in _CARD_LOOKUP:
        raise ValueError("Unknown card label: {value}".format(value=value))
    return _CARD_LOOKUP[text]


def shuffled_deck(rng: random.Random) -> List[int]:
    deck = list(range(len(SUITS) * len(RANKS)))
    rng.shuffle(deck)
    return deck


def sorted_hand(cards: Iterable[int]) -> List[int]:
    return sorted(cards, key=lambda value: (suit(value), rank(value)))


def active_deck(full_deck: Sequence[int], players: int, max_hand_size: int = 8) -> Tuple[List[int], List[int]]:
    active_count = players * max_hand_size
    return list(full_deck[:active_count]), list(full_deck[active_count:])
