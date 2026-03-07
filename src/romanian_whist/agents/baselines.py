from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from romanian_whist.rules.cards import rank, suit
from romanian_whist.rules.game import CARD_ACTION_OFFSET


def legal_actions_from_mask(mask: Sequence[int]) -> List[int]:
    return [index for index, enabled in enumerate(mask) if enabled]


def card_ids_from_mask(mask: Sequence[int]) -> List[int]:
    return [index for index, enabled in enumerate(mask) if enabled]


@dataclass
class RandomLegalAgent:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        legal_actions = legal_actions_from_mask(observation["legal_action_mask"])
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return self.rng.choice(legal_actions)


@dataclass
class SafeHeuristicAgent:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        legal_actions = legal_actions_from_mask(observation["legal_action_mask"])
        if int(observation["phase"]) == 0:
            return self._choose_bid(observation, legal_actions)
        return self._choose_card(observation, legal_actions, aggressive=False)

    def _choose_bid(self, observation: Dict[str, np.ndarray], legal_actions: List[int]) -> int:
        visible_cards = card_ids_from_mask(observation["hand_mask"])
        trump_suit = int(observation["trump_suit"])
        strength = 0.0
        for card_id in visible_cards:
            if rank(card_id) >= 10:
                strength += 0.7
            elif rank(card_id) >= 8:
                strength += 0.4
            if trump_suit >= 0 and suit(card_id) == trump_suit:
                strength += 0.35
        target = int(round(min(int(observation["hand_size"]), max(0.0, strength))))
        if target in legal_actions:
            return target
        return min(legal_actions, key=lambda action: abs(action - target))

    def _choose_card(
        self,
        observation: Dict[str, np.ndarray],
        legal_actions: List[int],
        aggressive: bool,
    ) -> int:
        legal_cards = [action - CARD_ACTION_OFFSET for action in legal_actions if action >= CARD_ACTION_OFFSET]
        if not legal_cards:
            return self.rng.choice(legal_actions)
        lead_suit = int(observation["lead_suit"])
        trump_suit = int(observation["trump_suit"])
        current_trick_cards = [int(card) for card in observation["current_trick_cards"] if int(card) >= 0]
        if not current_trick_cards:
            key = max if aggressive else min
            chosen = key(legal_cards, key=lambda card_id: (rank(card_id), suit(card_id)))
            return CARD_ACTION_OFFSET + chosen

        winning_card = current_trick_cards[0]
        for card_id in current_trick_cards[1:]:
            if self._card_beats(card_id, winning_card, lead_suit, trump_suit):
                winning_card = card_id

        winning_options = [
            card_id for card_id in legal_cards if self._card_beats(card_id, winning_card, lead_suit, trump_suit)
        ]
        if winning_options:
            chosen = min(winning_options, key=lambda card_id: (rank(card_id), suit(card_id)))
        else:
            chosen = min(legal_cards, key=lambda card_id: (rank(card_id), suit(card_id)))
        return CARD_ACTION_OFFSET + chosen

    @staticmethod
    def _card_beats(candidate: int, current: int, lead_suit: int, trump_suit: int) -> bool:
        candidate_is_trump = trump_suit >= 0 and suit(candidate) == trump_suit
        current_is_trump = trump_suit >= 0 and suit(current) == trump_suit
        if candidate_is_trump and not current_is_trump:
            return True
        if candidate_is_trump and current_is_trump:
            return rank(candidate) > rank(current)
        if current_is_trump:
            return False
        if suit(candidate) == lead_suit and suit(current) != lead_suit:
            return True
        if suit(candidate) == suit(current) == lead_suit:
            return rank(candidate) > rank(current)
        return False


@dataclass
class BidPlayHeuristicAgent(SafeHeuristicAgent):
    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        legal_actions = legal_actions_from_mask(observation["legal_action_mask"])
        if int(observation["phase"]) == 0:
            return self._aggressive_bid(observation, legal_actions)
        return self._choose_card(observation, legal_actions, aggressive=True)

    def _aggressive_bid(self, observation: Dict[str, np.ndarray], legal_actions: List[int]) -> int:
        visible_cards = card_ids_from_mask(observation["hand_mask"])
        trump_suit = int(observation["trump_suit"])
        target = 0
        for card_id in visible_cards:
            if rank(card_id) >= 10:
                target += 1
            elif rank(card_id) >= 8:
                target += 0.5
            if trump_suit >= 0 and suit(card_id) == trump_suit:
                target += 0.5
        target = int(round(min(int(observation["hand_size"]), target)))
        if target in legal_actions:
            return target
        return max([action for action in legal_actions if action <= target] or [min(legal_actions)])
