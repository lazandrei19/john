from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from romanian_whist.rules.cards import active_deck, card_label, rank, shuffled_deck, sorted_hand, suit
from romanian_whist.rules.config import OneCardMode, WhistVariantConfig

ACTION_COUNT = 61
BID_ACTIONS = 9
CARD_ACTION_OFFSET = 9
PAD_CARD_ID = -1
PAD_VALUE = -1
TOKEN_PAD = 0
TOKEN_ROUND_START = 1
TOKEN_BID_BASE = 16
TOKEN_PLAY_BASE = 64
TOKEN_TRICK_WIN_BASE = 128
TOKEN_ROUND_SCORE = 160
TOKEN_MODE_BASE = 176


def action_from_bid(bid: int) -> int:
    return bid


def action_from_card(card_id: int) -> int:
    return CARD_ACTION_OFFSET + card_id


def decode_action(action: int) -> Tuple[str, int]:
    if action < BID_ACTIONS:
        return "bid", action
    return "play", action - CARD_ACTION_OFFSET


@dataclass
class StepOutcome:
    rewards: List[float]
    round_finished: bool
    match_finished: bool
    next_player: Optional[int]


@dataclass
class RoundState:
    round_index: int
    hand_size: int
    one_card_mode: OneCardMode
    dealer: int
    leader: int
    trump_card: Optional[int]
    trump_suit: Optional[int]
    bids: List[Optional[int]]
    tricks_won: List[int]
    hands: List[List[int]]
    current_trick: List[Tuple[int, int]] = field(default_factory=list)
    played_cards: Set[int] = field(default_factory=set)
    phase: str = "bidding"
    turn_index: int = 0

    @property
    def bidding_order(self) -> List[int]:
        players = len(self.hands)
        start = (self.dealer + 1) % players
        return [(start + offset) % players for offset in range(players)]

    @property
    def current_player(self) -> int:
        if self.phase == "bidding":
            return self.bidding_order[self.turn_index]
        if self.current_trick:
            return (self.current_trick[-1][0] + 1) % len(self.hands)
        return self.leader

    @property
    def lead_suit(self) -> Optional[int]:
        if not self.current_trick:
            return None
        return suit(self.current_trick[0][1])


class RomanianWhistGame:
    def __init__(self, config: WhistVariantConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.scores = [0 for _ in range(config.players)]
        self.round_index = -1
        self.one_card_round_index = 0
        self.dealer = config.players - 1
        self.round_state = None  # type: Optional[RoundState]
        self.match_finished = False
        self.history_tokens = deque(maxlen=config.max_history_tokens)  # type: Deque[int]
        self.replay = []  # type: List[Dict[str, object]]

    def reset(self, seed: Optional[int] = None) -> "RomanianWhistGame":
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random(self.config.seed)
        self.scores = [0 for _ in range(self.config.players)]
        self.round_index = -1
        self.one_card_round_index = 0
        self.dealer = self.config.players - 1
        self.match_finished = False
        self.history_tokens.clear()
        self.replay = []
        self._start_next_round()
        return self

    @property
    def schedule(self) -> Tuple[int, ...]:
        return self.config.schedule()

    @property
    def current_player(self) -> int:
        if self.round_state is None:
            raise RuntimeError("Game has not been reset.")
        return self.round_state.current_player

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        if self.round_state is None:
            raise RuntimeError("Game has not been reset.")
        actor = self.current_player if player is None else player
        if actor != self.current_player:
            return []
        if self.round_state.phase == "bidding":
            return [action_from_bid(bid) for bid in self._legal_bids(actor)]
        return [action_from_card(card_id) for card_id in self._legal_cards(actor)]

    def observe(self, player: int) -> Dict[str, np.ndarray]:
        if self.round_state is None:
            raise RuntimeError("Game has not been reset.")
        state = self.round_state
        players = self.config.max_players
        hand_mask = np.zeros(52, dtype=np.int8)
        visible_hand = self._visible_hand_for_player(player)
        for card_id in visible_hand:
            hand_mask[card_id] = 1

        played_mask = np.zeros(52, dtype=np.int8)
        for card_id in state.played_cards:
            played_mask[card_id] = 1

        current_trick_cards = np.full(players, PAD_CARD_ID, dtype=np.int16)
        current_trick_players = np.full(players, PAD_VALUE, dtype=np.int8)
        for index, (seat, card_id) in enumerate(state.current_trick):
            current_trick_cards[index] = card_id
            current_trick_players[index] = seat

        bids = np.full(players, PAD_VALUE, dtype=np.int8)
        tricks_won = np.full(players, 0, dtype=np.int8)
        cumulative_scores = np.full(players, 0, dtype=np.int16)
        public_cards = np.full(players, PAD_CARD_ID, dtype=np.int16)
        for seat in range(self.config.players):
            bids[seat] = state.bids[seat] if state.bids[seat] is not None else PAD_VALUE
            tricks_won[seat] = state.tricks_won[seat]
            cumulative_scores[seat] = self.scores[seat]
            public_cards[seat] = self._public_card_for_player(player, seat)

        history_tokens = np.zeros(self.config.max_history_tokens, dtype=np.int16)
        token_list = list(self.history_tokens)
        history_tokens[: len(token_list)] = token_list

        legal_mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if player == self.current_player and not self.match_finished:
            for action in self.legal_actions(player):
                legal_mask[action] = 1

        observation = {
            "hand_mask": hand_mask,
            "current_trick_cards": current_trick_cards,
            "current_trick_players": current_trick_players,
            "played_card_mask": played_mask,
            "public_card_by_player": public_cards,
            "bids": bids,
            "tricks_won": tricks_won,
            "cumulative_scores": cumulative_scores,
            "history_tokens": history_tokens,
            "legal_action_mask": legal_mask,
            "trump_suit": np.array(state.trump_suit if state.trump_suit is not None else PAD_VALUE, dtype=np.int8),
            "lead_suit": np.array(state.lead_suit if state.lead_suit is not None else PAD_VALUE, dtype=np.int8),
            "seat_index": np.array(player, dtype=np.int8),
            "dealer_index": np.array(state.dealer, dtype=np.int8),
            "leader_index": np.array(state.leader, dtype=np.int8),
            "player_count": np.array(self.config.players, dtype=np.int8),
            "hand_size": np.array(state.hand_size, dtype=np.int8),
            "one_card_mode": np.array(self._mode_index(state.one_card_mode), dtype=np.int8),
            "phase": np.array(0 if state.phase == "bidding" else 1, dtype=np.int8),
            "round_index": np.array(state.round_index, dtype=np.int16),
        }
        return observation

    def observe_for_baseline(self, player: int) -> Dict[str, np.ndarray]:
        if self.round_state is None:
            raise RuntimeError("Game has not been reset.")
        state = self.round_state
        hand_mask = np.zeros(52, dtype=np.int8)
        for card_id in self._visible_hand_for_player(player):
            hand_mask[card_id] = 1

        current_trick_cards = np.full(self.config.max_players, PAD_CARD_ID, dtype=np.int16)
        for index, (_, card_id) in enumerate(state.current_trick):
            current_trick_cards[index] = card_id

        legal_mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if player == self.current_player and not self.match_finished:
            for action in self.legal_actions(player):
                legal_mask[action] = 1

        return {
            "hand_mask": hand_mask,
            "current_trick_cards": current_trick_cards,
            "legal_action_mask": legal_mask,
            "trump_suit": np.array(state.trump_suit if state.trump_suit is not None else PAD_VALUE, dtype=np.int8),
            "lead_suit": np.array(state.lead_suit if state.lead_suit is not None else PAD_VALUE, dtype=np.int8),
            "hand_size": np.array(state.hand_size, dtype=np.int8),
            "phase": np.array(0 if state.phase == "bidding" else 1, dtype=np.int8),
        }

    def step(self, action: int) -> StepOutcome:
        if self.round_state is None:
            raise RuntimeError("Game has not been reset.")
        if self.match_finished:
            raise RuntimeError("Match is already finished.")
        actor = self.current_player
        legal = self.legal_actions(actor)
        if action not in legal:
            raise ValueError("Illegal action {action} for player {player}".format(action=action, player=actor))

        rewards_before = list(self.scores)
        state = self.round_state
        action_type, value = decode_action(action)
        if state.phase == "bidding" and action_type != "bid":
            raise ValueError("Expected a bid action.")
        if state.phase == "play" and action_type != "play":
            raise ValueError("Expected a play action.")

        if action_type == "bid":
            state.bids[actor] = value
            self._append_tokens(TOKEN_BID_BASE + actor, TOKEN_BID_BASE + 8 + value)
            self.replay.append({"type": "bid", "player": actor, "bid": value, "round": state.round_index})
            if state.turn_index == self.config.players - 1:
                state.phase = "play"
                state.turn_index = 0
            else:
                state.turn_index += 1
            return StepOutcome(
                rewards=[0.0 for _ in range(self.config.players)],
                round_finished=False,
                match_finished=False,
                next_player=self.current_player,
            )

        card_id = value
        state.hands[actor].remove(card_id)
        state.current_trick.append((actor, card_id))
        state.played_cards.add(card_id)
        self._append_tokens(TOKEN_PLAY_BASE + actor, TOKEN_PLAY_BASE + 32 + card_id)
        self.replay.append({"type": "play", "player": actor, "card": card_id, "round": state.round_index})

        if len(state.current_trick) == self.config.players:
            winner = self._determine_trick_winner(state.current_trick, state.trump_suit)
            state.tricks_won[winner] += 1
            state.leader = winner
            state.current_trick = []
            self._append_tokens(TOKEN_TRICK_WIN_BASE + winner)
            self.replay.append({"type": "trick_win", "player": winner, "round": state.round_index})

            if all(not hand for hand in state.hands):
                self._score_round()
                round_finished = True
                if self.round_index == len(self.schedule) - 1:
                    self.match_finished = True
                    next_player = None
                else:
                    self.dealer = (self.dealer + 1) % self.config.players
                    self._start_next_round()
                    next_player = self.current_player
            else:
                round_finished = False
                next_player = self.current_player
        else:
            round_finished = False
            next_player = self.current_player

        rewards = [float(after - before) for before, after in zip(rewards_before, self.scores)]
        return StepOutcome(
            rewards=rewards,
            round_finished=round_finished,
            match_finished=self.match_finished,
            next_player=next_player,
        )

    def serialize_replay(self) -> Dict[str, object]:
        state = self.round_state
        summary = {
            "config": {
                "players": self.config.players,
                "one_card_modes": [mode.value for mode in self.config.one_card_modes],
                "schedule": list(self.schedule),
            },
            "scores": list(self.scores),
            "round_index": self.round_index,
            "match_finished": self.match_finished,
            "events": list(self.replay),
        }
        if state is not None:
            summary["current_round"] = {
                "hand_size": state.hand_size,
                "dealer": state.dealer,
                "leader": state.leader,
                "trump_card": state.trump_card,
                "trump_suit": state.trump_suit,
                "bids": state.bids,
                "tricks_won": state.tricks_won,
            }
        return summary

    def _start_next_round(self) -> None:
        self.round_index += 1
        hand_size = self.schedule[self.round_index]
        if hand_size == 1:
            mode = self.config.one_card_mode_for_index(self.one_card_round_index)
            self.one_card_round_index += 1
        else:
            mode = OneCardMode.REGULAR

        deck = shuffled_deck(self.rng)
        active, _ = active_deck(deck, self.config.players, self.config.max_hand_size)
        hands = [[] for _ in range(self.config.players)]  # type: List[List[int]]
        deal_order = [((self.dealer + 1) + offset) % self.config.players for offset in range(self.config.players)]
        for card_index in range(hand_size):
            for order_index, player in enumerate(deal_order):
                deck_offset = card_index * self.config.players + order_index
                hands[player].append(active[deck_offset])
        hands = [sorted_hand(hand) for hand in hands]

        dealt_count = hand_size * self.config.players
        trump_card = active[dealt_count] if dealt_count < len(active) else None
        trump_suit = suit(trump_card) if trump_card is not None else None
        leader = (self.dealer + 1) % self.config.players
        self.round_state = RoundState(
            round_index=self.round_index,
            hand_size=hand_size,
            one_card_mode=mode,
            dealer=self.dealer,
            leader=leader,
            trump_card=trump_card,
            trump_suit=trump_suit,
            bids=[None for _ in range(self.config.players)],
            tricks_won=[0 for _ in range(self.config.players)],
            hands=hands,
        )
        self._append_tokens(
            TOKEN_ROUND_START,
            TOKEN_MODE_BASE + self._mode_index(mode),
            TOKEN_MODE_BASE + 8 + hand_size,
        )
        self.replay.append(
            {
                "type": "round_start",
                "round": self.round_index,
                "hand_size": hand_size,
                "dealer": self.dealer,
                "mode": mode.value,
                "trump_card": trump_card,
            }
        )

    def _score_round(self) -> None:
        if self.round_state is None:
            return
        for seat in range(self.config.players):
            bid = self.round_state.bids[seat]
            tricks = self.round_state.tricks_won[seat]
            if bid is None:
                continue
            if tricks == bid:
                delta = 5 + bid
            else:
                delta = -abs(tricks - bid)
            self.scores[seat] += delta
            self._append_tokens(TOKEN_ROUND_SCORE + seat, TOKEN_ROUND_SCORE + 8 + (delta + 8))
        self.replay.append({"type": "round_score", "scores": list(self.scores), "round": self.round_index})

    def _visible_hand_for_player(self, player: int) -> Sequence[int]:
        if self.round_state is None:
            return []
        state = self.round_state
        if not state.hands[player]:
            return []
        if state.hand_size != 1:
            return state.hands[player]
        if state.one_card_mode in (OneCardMode.FOREHEAD, OneCardMode.BLIND):
            return []
        return state.hands[player]

    def visible_hand_for_player(self, player: int) -> Sequence[int]:
        return self._visible_hand_for_player(player)

    def _public_card_for_player(self, observer: int, target: int) -> int:
        if self.round_state is None or self.round_state.hand_size != 1:
            return PAD_CARD_ID
        mode = self.round_state.one_card_mode
        if mode == OneCardMode.REGULAR:
            return PAD_CARD_ID
        if mode == OneCardMode.FOREHEAD and observer != target and self.round_state.hands[target]:
            return self.round_state.hands[target][0]
        return PAD_CARD_ID

    def _legal_bids(self, player: int) -> List[int]:
        if self.round_state is None:
            return []
        bids = list(range(self.round_state.hand_size + 1))
        order = self.round_state.bidding_order
        if player == order[-1]:
            running_sum = sum(bid for bid in self.round_state.bids if bid is not None)
            forbidden = self.round_state.hand_size - running_sum
            bids = [bid for bid in bids if bid != forbidden]
        return bids

    def _legal_cards(self, player: int) -> List[int]:
        if self.round_state is None:
            return []
        hand = list(self.round_state.hands[player])
        if not self.round_state.current_trick:
            return hand
        lead_suit = self.round_state.lead_suit
        assert lead_suit is not None
        follow_cards = [card_id for card_id in hand if suit(card_id) == lead_suit]
        if follow_cards:
            return sorted_hand(follow_cards)
        trump_suit = self.round_state.trump_suit
        if trump_suit is not None:
            trump_cards = [card_id for card_id in hand if suit(card_id) == trump_suit]
            if trump_cards:
                return sorted_hand(trump_cards)
        return sorted_hand(hand)

    @staticmethod
    def _determine_trick_winner(current_trick: Iterable[Tuple[int, int]], trump_suit: Optional[int]) -> int:
        trick = list(current_trick)
        winning_player, winning_card = trick[0]
        lead = suit(winning_card)
        for player, card_id in trick[1:]:
            winning_is_trump = trump_suit is not None and suit(winning_card) == trump_suit
            card_is_trump = trump_suit is not None and suit(card_id) == trump_suit
            if card_is_trump and not winning_is_trump:
                winning_player, winning_card = player, card_id
                continue
            if card_is_trump and winning_is_trump and rank(card_id) > rank(winning_card):
                winning_player, winning_card = player, card_id
                continue
            if not winning_is_trump and suit(card_id) == lead and rank(card_id) > rank(winning_card):
                winning_player, winning_card = player, card_id
        return winning_player

    def _append_tokens(self, *tokens: int) -> None:
        for token in tokens:
            self.history_tokens.append(max(TOKEN_PAD, min(token, 255)))

    @staticmethod
    def _mode_index(mode: OneCardMode) -> int:
        mapping = {
            OneCardMode.REGULAR: 0,
            OneCardMode.FOREHEAD: 1,
            OneCardMode.BLIND: 2,
        }
        return mapping[mode]

    def summary_lines(self) -> List[str]:
        if self.round_state is None:
            return ["Match not started."]
        state = self.round_state
        lines = [
            "Round {round} | hand size {hand_size} | mode {mode} | dealer P{dealer} | scores {scores}".format(
                round=state.round_index,
                hand_size=state.hand_size,
                mode=state.one_card_mode.value,
                dealer=state.dealer,
                scores=self.scores,
            )
        ]
        lines.append("Phase: {phase} | current player: P{player}".format(phase=state.phase, player=self.current_player))
        if state.trump_card is not None:
            lines.append("Trump: {label}".format(label=card_label(state.trump_card)))
        if state.phase == "bidding":
            lines.append("Bids: {bids}".format(bids=state.bids))
        else:
            lines.append("Tricks won: {tricks}".format(tricks=state.tricks_won))
            lines.append(
                "Current trick: {cards}".format(
                    cards=[(seat, card_label(card_id)) for seat, card_id in state.current_trick]
                )
            )
        return lines
