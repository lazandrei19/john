from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.model import ActionRecommendation, PolicyAgent
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.cards import SUITS, card_label, parse_card, suit
from romanian_whist.rules.config import OneCardMode, WhistVariantConfig
from romanian_whist.rules.game import (
    ACTION_COUNT,
    PAD_CARD_ID,
    PAD_VALUE,
    TOKEN_BID_BASE,
    TOKEN_MODE_BASE,
    TOKEN_PLAY_BASE,
    TOKEN_ROUND_SCORE,
    TOKEN_ROUND_START,
    TOKEN_TRICK_WIN_BASE,
    RomanianWhistGame,
    action_from_bid,
    action_from_card,
)
from romanian_whist.agents.checkpoint import load_policy_checkpoint


def _plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _action_label(action: int) -> str:
    if action < 9:
        return "Bid {value}".format(value=action)
    return card_label(action - 9)


def _card_entries(cards: Sequence[int]) -> List[Dict[str, object]]:
    return [{"id": card_id, "label": card_label(card_id)} for card_id in cards]


def _roles(players: int, roles: Sequence[str]) -> List[str]:
    if len(roles) != players:
        raise ValueError("Seat roles must match player count.")
    return [role.strip().lower() for role in roles]


@dataclass
class SeatRoleConfig:
    roles: List[str]


@dataclass
class CheckpointSelection:
    path: Optional[str]
    resolved_path: Optional[str]
    exists: bool
    device: str
    source: str


@dataclass
class Recommendation:
    chosen_action: int
    chosen_label: str
    legal_actions: List[int]
    legal_labels: List[str]
    top_actions: List[Dict[str, object]]
    value: float


@dataclass
class PublicSeatTrackerState:
    players: int
    advised_seat: int
    dealer: int
    leader: int
    current_player: int
    hand_size: int
    round_index: int
    phase: str
    trump_card: Optional[int]
    trump_suit: Optional[int]
    hand: List[int]
    bids: List[Optional[int]]
    tricks_won: List[int]
    scores: List[int]
    current_trick: List[tuple[int, int]]
    played_cards: List[int]
    one_card_mode: str = OneCardMode.REGULAR.value
    event_log: List[Dict[str, object]] = field(default_factory=list)
    history_tokens: List[int] = field(default_factory=list)
    plays_in_round: int = 0
    round_finished: bool = False


class RecommendationService:
    def __init__(self, default_checkpoint: Optional[Path] = None):
        self.default_checkpoint = default_checkpoint or Path("runs/universal/best.pt")
        self._cache = {}  # type: Dict[tuple[str, str], PolicyAgent]

    def resolve_checkpoint(self, checkpoint_path: Optional[str], device: str = "cpu") -> CheckpointSelection:
        if checkpoint_path:
            path = Path(checkpoint_path).expanduser()
            return CheckpointSelection(
                path=str(path),
                resolved_path=str(path.resolve()) if path.exists() else str(path),
                exists=path.exists(),
                device=device,
                source="explicit",
            )
        path = self.default_checkpoint
        return CheckpointSelection(
            path=str(path),
            resolved_path=str(path.resolve()) if path.exists() else str(path),
            exists=path.exists(),
            device=device,
            source="default",
        )

    def policy_agent(self, checkpoint_path: Optional[str], device: str = "cpu") -> PolicyAgent:
        selection = self.resolve_checkpoint(checkpoint_path, device=device)
        if not selection.exists or selection.resolved_path is None:
            raise FileNotFoundError("Checkpoint not found: {path}".format(path=selection.path))
        cache_key = (selection.resolved_path, device)
        if cache_key not in self._cache:
            policy, _ = load_policy_checkpoint(Path(selection.resolved_path), device=device)
            self._cache[cache_key] = PolicyAgent(policy, device=device, greedy=True)
        return self._cache[cache_key]

    def recommend(self, observation: Dict[str, object], checkpoint_path: Optional[str], device: str = "cpu", top_k: int = 5) -> Recommendation:
        agent = self.policy_agent(checkpoint_path, device=device)
        result = agent.recommend(observation, top_k=top_k)
        return Recommendation(
            chosen_action=result.chosen_action,
            chosen_label=_action_label(result.chosen_action),
            legal_actions=result.legal_actions,
            legal_labels=[_action_label(action) for action in result.legal_actions],
            top_actions=[
                {
                    "action": item.action,
                    "label": _action_label(item.action),
                    "probability": item.probability,
                    "logit": item.logit,
                }
                for item in result.top_actions
            ],
            value=result.value,
        )


class FullGameSession:
    def __init__(
        self,
        *,
        session_id: str,
        mode: str,
        players: int,
        seed: int,
        roles: Sequence[str],
        checkpoint_path: Optional[str],
        device: str,
        recommender: RecommendationService,
    ):
        self.session_id = session_id
        self.mode = mode
        self.device = device
        self.roles = _roles(players, roles)
        self.recommender = recommender
        self.checkpoint_path = checkpoint_path
        self.config = WhistVariantConfig(players=players, seed=seed, one_card_modes=(OneCardMode.REGULAR,))
        self.env = RomanianWhistEnv(self.config)
        self.env.reset(seed=seed)
        self.model_agent = recommender.policy_agent(checkpoint_path, device=device) if "model" in self.roles else None
        self.agents = self._build_agents()
        self.snapshots = []  # type: List[Dict[str, object]]
        self.view_index = 0
        self._record_snapshot()

    def _build_agents(self) -> List[object | None]:
        agents = []
        for seat, role in enumerate(self.roles):
            if role == "human":
                agents.append(None)
            elif role == "model":
                if self.model_agent is None:
                    raise FileNotFoundError("Model role requested without a checkpoint.")
                agents.append(self.model_agent)
            elif role == "random":
                agents.append(RandomLegalAgent(seed=self.config.seed + seat))
            elif role == "safe":
                agents.append(SafeHeuristicAgent(seed=self.config.seed + seat))
            else:
                agents.append(BidPlayHeuristicAgent(seed=self.config.seed + seat))
        return agents

    @property
    def live_index(self) -> int:
        return len(self.snapshots) - 1

    def _serialize_game(self) -> Dict[str, object]:
        game = self.env.game
        state = game.round_state
        if state is None:
            raise RuntimeError("Game not initialized.")
        observations = {
            str(seat): _plain(self.env.observe(agent_name))
            for seat, agent_name in enumerate(self.env.possible_agents)
        }
        legal_actions = []
        if not game.match_finished:
            legal_actions = game.legal_actions()
        return {
            "match_finished": game.match_finished,
            "current_player": game.current_player if not game.match_finished else None,
            "phase": state.phase,
            "round_index": state.round_index,
            "hand_size": state.hand_size,
            "dealer": state.dealer,
            "leader": state.leader,
            "one_card_mode": state.one_card_mode.value,
            "trump_card": state.trump_card,
            "trump_suit": state.trump_suit,
            "bids": list(state.bids),
            "tricks_won": list(state.tricks_won),
            "scores": list(game.scores),
            "current_trick": [{"seat": seat, "card": card} for seat, card in state.current_trick],
            "hands": [list(hand) for hand in state.hands],
            "events": copy.deepcopy(game.serialize_replay()["events"]),
            "observations": observations,
            "legal_actions": list(legal_actions),
        }

    def _record_snapshot(self) -> None:
        snapshot = self._serialize_game()
        snapshot["step_index"] = len(self.snapshots)
        self.snapshots.append(snapshot)
        self.view_index = self.live_index

    def current_state(self, *, reveal_all: bool = False, include_recommendation: bool = False) -> Dict[str, object]:
        snapshot = self.snapshots[self.view_index]
        return self._render_snapshot(snapshot, reveal_all=reveal_all, include_recommendation=include_recommendation)

    def _render_snapshot(self, snapshot: Dict[str, object], *, reveal_all: bool, include_recommendation: bool) -> Dict[str, object]:
        current_player = snapshot["current_player"]
        human_seats = [seat for seat, role in enumerate(self.roles) if role == "human"]
        primary_human = human_seats[0] if human_seats else None
        hands = []
        for seat, cards in enumerate(snapshot["hands"]):
            visible = reveal_all or self.mode == "inspect" or seat == primary_human or self.roles[seat] == "human"
            hands.append(
                {
                    "seat": seat,
                    "role": self.roles[seat],
                    "hidden": not visible,
                    "cards": _card_entries(cards) if visible else [{"label": "Hidden"} for _ in cards],
                    "count": len(cards),
                }
            )
        current_role = self.roles[current_player] if current_player is not None else None
        state = {
            "session_id": self.session_id,
            "mode": self.mode,
            "players": self.config.players,
            "roles": list(self.roles),
            "step_index": snapshot["step_index"],
            "total_steps": len(self.snapshots),
            "is_live_view": snapshot["step_index"] == self.live_index,
            "match_finished": snapshot["match_finished"],
            "current_player": current_player,
            "current_role": current_role,
            "phase": snapshot["phase"],
            "round_index": snapshot["round_index"],
            "hand_size": snapshot["hand_size"],
            "dealer": snapshot["dealer"],
            "leader": snapshot["leader"],
            "trump_card": None if snapshot["trump_card"] is None else {"id": snapshot["trump_card"], "label": card_label(snapshot["trump_card"])},
            "trump_suit": None if snapshot["trump_suit"] is None else SUITS[snapshot["trump_suit"]],
            "bids": snapshot["bids"],
            "tricks_won": snapshot["tricks_won"],
            "scores": snapshot["scores"],
            "current_trick": [
                {"seat": item["seat"], "role": self.roles[item["seat"]], "card": {"id": item["card"], "label": card_label(item["card"])}}
                for item in snapshot["current_trick"]
            ],
            "hands": hands,
            "event_log": self._format_events(snapshot["events"]),
            "legal_actions": [
                {"action": action, "label": _action_label(action)}
                for action in snapshot["legal_actions"]
            ],
            "human_turn": current_player is not None and current_role == "human" and snapshot["step_index"] == self.live_index,
            "can_step": current_player is not None and current_role != "human" and snapshot["step_index"] == self.live_index and not snapshot["match_finished"],
            "checkpoint_path": self.checkpoint_path,
            "observations": snapshot["observations"] if self.mode == "inspect" or reveal_all else None,
        }
        if include_recommendation and current_player is not None:
            try:
                recommendation = self.recommender.recommend(
                    snapshot["observations"][str(current_player)],
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                )
                state["current_recommendation"] = asdict(recommendation)
            except FileNotFoundError:
                state["current_recommendation"] = None
        return state

    def _format_events(self, events: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        formatted = []
        for event in list(events)[-40:]:
            if event["type"] == "bid":
                message = "P{player} bid {bid}".format(player=event["player"], bid=event["bid"])
            elif event["type"] == "play":
                message = "P{player} played {card}".format(player=event["player"], card=card_label(event["card"]))
            elif event["type"] == "trick_win":
                message = "P{player} won the trick".format(player=event["player"])
            elif event["type"] == "round_start":
                message = "Round {round} started with {hand_size} cards".format(
                    round=event["round"], hand_size=event["hand_size"]
                )
            elif event["type"] == "round_score":
                message = "Scores updated to {scores}".format(scores=event["scores"])
            else:
                message = json.dumps(event)
            formatted.append({"type": event["type"], "message": message})
        return formatted

    def submit_human_action(self, action: int) -> Dict[str, object]:
        if self.view_index != self.live_index:
            self.view_index = self.live_index
        current_player = self.env.game.current_player
        if self.roles[current_player] != "human":
            raise ValueError("It is not a human turn.")
        self.env.step(action)
        self._record_snapshot()
        return self.current_state(include_recommendation=self.mode == "inspect")

    def step_once(self) -> Dict[str, object]:
        if self.view_index != self.live_index:
            self.view_index = self.live_index
        game = self.env.game
        if game.match_finished:
            return self.current_state(include_recommendation=self.mode == "inspect")
        current_player = game.current_player
        role = self.roles[current_player]
        if role == "human":
            return self.current_state(include_recommendation=self.mode == "inspect")
        observation = self.env.observe(self.env.agent_selection)
        action = self.agents[current_player].select_action(observation)
        self.env.step(action)
        self._record_snapshot()
        return self.current_state(include_recommendation=self.mode == "inspect")

    def autoplay(self, max_steps: int = 64) -> Dict[str, object]:
        steps = 0
        while steps < max_steps:
            game = self.env.game
            if game.match_finished:
                break
            current_player = game.current_player
            if self.roles[current_player] == "human":
                break
            self.step_once()
            steps += 1
        return self.current_state(include_recommendation=self.mode == "inspect")

    def recommend(self, seat: Optional[int] = None) -> Dict[str, object]:
        snapshot = self.snapshots[self.view_index]
        target_seat = snapshot["current_player"] if seat is None else seat
        if target_seat is None:
            raise ValueError("No active player.")
        recommendation = self.recommender.recommend(
            snapshot["observations"][str(target_seat)],
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        return asdict(recommendation)

    def jump(self, step_index: int) -> Dict[str, object]:
        if step_index < 0 or step_index >= len(self.snapshots):
            raise ValueError("Step out of range.")
        self.view_index = step_index
        return self.current_state(reveal_all=True, include_recommendation=True)

    def export_state(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "roles": list(self.roles),
            "checkpoint_path": self.checkpoint_path,
            "players": self.config.players,
            "device": self.device,
            "snapshots": copy.deepcopy(self.snapshots),
        }


class ReplaySession:
    def __init__(self, payload: Dict[str, object]):
        self.session_id = str(payload.get("session_id") or uuid.uuid4())
        self.mode = "inspect"
        self.roles = list(payload["roles"])
        self.checkpoint_path = payload.get("checkpoint_path")
        self.device = str(payload.get("device") or "cpu")
        self.recommender = RecommendationService()
        self.snapshots = list(payload["snapshots"])
        self.view_index = 0
        self.config = WhistVariantConfig(players=int(payload["players"]), seed=0, one_card_modes=(OneCardMode.REGULAR,))

    @property
    def live_index(self) -> int:
        return len(self.snapshots) - 1

    def current_state(self, *, reveal_all: bool = True, include_recommendation: bool = True) -> Dict[str, object]:
        snapshot = self.snapshots[self.view_index]
        renderer = FullGameSession.__dict__["_render_snapshot"]
        return renderer(self, snapshot, reveal_all=reveal_all, include_recommendation=include_recommendation)

    def _format_events(self, events: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        return FullGameSession._format_events(self, events)

    def jump(self, step_index: int) -> Dict[str, object]:
        if step_index < 0 or step_index >= len(self.snapshots):
            raise ValueError("Step out of range.")
        self.view_index = step_index
        return self.current_state()

    def export_state(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "roles": list(self.roles),
            "checkpoint_path": self.checkpoint_path,
            "players": self.config.players,
            "device": self.device,
            "snapshots": copy.deepcopy(self.snapshots),
        }


class PublicSeatTracker:
    def __init__(self, state: PublicSeatTrackerState):
        self.state = state

    @classmethod
    def create(
        cls,
        *,
        players: int,
        advised_seat: int,
        dealer: int,
        hand_size: int,
        hand: Sequence[int],
        trump_card: Optional[int],
        scores: Optional[Sequence[int]] = None,
        round_index: int = 0,
    ) -> "PublicSeatTracker":
        if players < 3 or players > 6:
            raise ValueError("Advisor mode supports 3-6 players.")
        if advised_seat < 0 or advised_seat >= players:
            raise ValueError("Advised seat out of range.")
        if dealer < 0 or dealer >= players:
            raise ValueError("Dealer out of range.")
        if hand_size < 1 or hand_size > 8:
            raise ValueError("Hand size must be between 1 and 8.")
        if len(hand) != hand_size:
            raise ValueError("Hand size does not match provided cards.")
        if len(set(hand)) != len(hand):
            raise ValueError("Duplicate cards in advised hand.")
        if scores is not None and len(scores) != players:
            raise ValueError("Scores must match player count.")
        leader = (dealer + 1) % players
        current_player = leader
        state = PublicSeatTrackerState(
            players=players,
            advised_seat=advised_seat,
            dealer=dealer,
            leader=leader,
            current_player=current_player,
            hand_size=hand_size,
            round_index=round_index,
            phase="bidding",
            trump_card=trump_card,
            trump_suit=suit(trump_card) if trump_card is not None else None,
            hand=sorted(hand),
            bids=[None for _ in range(players)],
            tricks_won=[0 for _ in range(players)],
            scores=list(scores) if scores is not None else [0 for _ in range(players)],
            current_trick=[],
            played_cards=[],
            event_log=[],
            history_tokens=[TOKEN_ROUND_START, TOKEN_MODE_BASE, TOKEN_MODE_BASE + 8 + hand_size],
        )
        return cls(state)

    def _validate_player(self, player: int, *, label: str) -> None:
        if player < 0 or player >= self.state.players:
            raise ValueError("{label} out of range.".format(label=label))

    def _validate_exact_length(self, values: Sequence[object], *, label: str) -> None:
        if len(values) != self.state.players:
            raise ValueError("{label} must contain exactly {players} values.".format(label=label, players=self.state.players))

    def _validate_current_trick(self, trick: Sequence[tuple[int, int]]) -> None:
        if len(trick) > self.state.players:
            raise ValueError("Current trick cannot contain more cards than players.")
        seen_seats = set()
        seen_cards = set()
        for seat, card_id in trick:
            self._validate_player(seat, label="Trick seat")
            if seat in seen_seats:
                raise ValueError("Current trick contains duplicate seats.")
            if card_id in seen_cards:
                raise ValueError("Current trick contains duplicate cards.")
            seen_seats.add(seat)
            seen_cards.add(card_id)

    @property
    def bidding_order(self) -> List[int]:
        start = (self.state.dealer + 1) % self.state.players
        return [(start + offset) % self.state.players for offset in range(self.state.players)]

    def legal_actions(self) -> List[int]:
        if self.state.current_player != self.state.advised_seat or self.state.round_finished:
            return []
        if self.state.phase == "bidding":
            bids = list(range(self.state.hand_size + 1))
            if self.state.current_player == self.bidding_order[-1]:
                running_sum = sum(bid for bid in self.state.bids if bid is not None)
                forbidden = self.state.hand_size - running_sum
                bids = [bid for bid in bids if bid != forbidden]
            return [action_from_bid(bid) for bid in bids]
        return [action_from_card(card_id) for card_id in self._legal_cards_for_hand(self.state.hand)]

    def _legal_cards_for_hand(self, hand: Sequence[int]) -> List[int]:
        if not self.state.current_trick:
            return sorted(hand)
        lead_suit = suit(self.state.current_trick[0][1])
        follow = [card_id for card_id in hand if suit(card_id) == lead_suit]
        if follow:
            return sorted(follow)
        if self.state.trump_suit is not None:
            trumps = [card_id for card_id in hand if suit(card_id) == self.state.trump_suit]
            if trumps:
                return sorted(trumps)
        return sorted(hand)

    def observe(self) -> Dict[str, object]:
        hand_mask = [0 for _ in range(52)]
        for card_id in self.state.hand:
            hand_mask[card_id] = 1
        played_mask = [0 for _ in range(52)]
        for card_id in self.state.played_cards:
            played_mask[card_id] = 1
        current_trick_cards = [PAD_CARD_ID for _ in range(6)]
        current_trick_players = [PAD_VALUE for _ in range(6)]
        for index, (seat, card_id) in enumerate(self.state.current_trick):
            current_trick_cards[index] = card_id
            current_trick_players[index] = seat
        bids = [PAD_VALUE for _ in range(6)]
        tricks = [0 for _ in range(6)]
        scores = [0 for _ in range(6)]
        public_cards = [PAD_CARD_ID for _ in range(6)]
        for seat in range(self.state.players):
            bids[seat] = self.state.bids[seat] if self.state.bids[seat] is not None else PAD_VALUE
            tricks[seat] = self.state.tricks_won[seat]
            scores[seat] = self.state.scores[seat]
        legal_mask = [0 for _ in range(ACTION_COUNT)]
        for action in self.legal_actions():
            legal_mask[action] = 1
        history_tokens = [0 for _ in range(64)]
        for index, token in enumerate(self.state.history_tokens[-64:]):
            history_tokens[index] = token
        return {
            "hand_mask": hand_mask,
            "current_trick_cards": current_trick_cards,
            "current_trick_players": current_trick_players,
            "played_card_mask": played_mask,
            "public_card_by_player": public_cards,
            "bids": bids,
            "tricks_won": tricks,
            "cumulative_scores": scores,
            "history_tokens": history_tokens,
            "legal_action_mask": legal_mask,
            "trump_suit": self.state.trump_suit if self.state.trump_suit is not None else PAD_VALUE,
            "lead_suit": suit(self.state.current_trick[0][1]) if self.state.current_trick else PAD_VALUE,
            "seat_index": self.state.advised_seat,
            "dealer_index": self.state.dealer,
            "leader_index": self.state.leader,
            "player_count": self.state.players,
            "hand_size": self.state.hand_size,
            "one_card_mode": 0,
            "phase": 0 if self.state.phase == "bidding" else 1,
            "round_index": self.state.round_index,
        }

    def apply_bid(self, player: int, bid: int) -> None:
        self._validate_player(player, label="Player")
        if self.state.phase != "bidding":
            raise ValueError("Round is not in bidding phase.")
        if player != self.state.current_player:
            raise ValueError("It is not player {player}'s turn.".format(player=player))
        legal = self.legal_actions() if player == self.state.advised_seat else [action_from_bid(item) for item in self._legal_bids_for_player(player)]
        action = action_from_bid(bid)
        if action not in legal:
            raise ValueError("Illegal bid {bid}.".format(bid=bid))
        self.state.bids[player] = bid
        self.state.history_tokens.extend([TOKEN_BID_BASE + player, TOKEN_BID_BASE + 8 + bid])
        self.state.event_log.append({"type": "bid", "player": player, "bid": bid})
        order = self.bidding_order
        turn_index = order.index(player)
        if turn_index == len(order) - 1:
            self.state.phase = "play"
            self.state.current_player = self.state.leader
        else:
            self.state.current_player = order[turn_index + 1]

    def _legal_bids_for_player(self, player: int) -> List[int]:
        bids = list(range(self.state.hand_size + 1))
        if player == self.bidding_order[-1]:
            running_sum = sum(bid for bid in self.state.bids if bid is not None)
            forbidden = self.state.hand_size - running_sum
            bids = [bid for bid in bids if bid != forbidden]
        return bids

    def apply_card(self, player: int, card_id: int) -> None:
        self._validate_player(player, label="Player")
        if self.state.phase != "play":
            raise ValueError("Round is not in play phase.")
        if player != self.state.current_player:
            raise ValueError("It is not player {player}'s turn.".format(player=player))
        if card_id in self.state.played_cards:
            raise ValueError("Card already played.")
        if player == self.state.advised_seat:
            legal = self.legal_actions()
            action = action_from_card(card_id)
            if action not in legal:
                raise ValueError("Illegal advised card.")
            self.state.hand.remove(card_id)
        self.state.current_trick.append((player, card_id))
        self.state.played_cards.append(card_id)
        self.state.plays_in_round += 1
        self.state.history_tokens.extend([TOKEN_PLAY_BASE + player, TOKEN_PLAY_BASE + 32 + card_id])
        self.state.event_log.append({"type": "play", "player": player, "card": card_id})
        if len(self.state.current_trick) == self.state.players:
            winner = RomanianWhistGame._determine_trick_winner(self.state.current_trick, self.state.trump_suit)
            self.state.tricks_won[winner] += 1
            self.state.event_log.append({"type": "trick_win", "player": winner})
            self.state.history_tokens.append(TOKEN_TRICK_WIN_BASE + winner)
            self.state.current_trick = []
            self.state.leader = winner
            self.state.current_player = winner
        else:
            self.state.current_player = (player + 1) % self.state.players
        if self.state.plays_in_round == self.state.players * self.state.hand_size:
            self._score_round()

    def _score_round(self) -> None:
        for seat in range(self.state.players):
            bid = self.state.bids[seat]
            if bid is None:
                continue
            tricks = self.state.tricks_won[seat]
            delta = 5 + bid if tricks == bid else -abs(tricks - bid)
            self.state.scores[seat] += delta
            self.state.history_tokens.extend([TOKEN_ROUND_SCORE + seat, TOKEN_ROUND_SCORE + 8 + (delta + 8)])
        self.state.round_finished = True
        self.state.phase = "finished"
        self.state.event_log.append({"type": "round_score", "scores": list(self.state.scores)})

    def manual_update(self, *, hand: Optional[Sequence[int]] = None, bids: Optional[Sequence[Optional[int]]] = None, tricks_won: Optional[Sequence[int]] = None, scores: Optional[Sequence[int]] = None, current_trick: Optional[Sequence[tuple[int, int]]] = None, current_player: Optional[int] = None, phase: Optional[str] = None, leader: Optional[int] = None, trump_card: object = "__unset__") -> None:
        next_hand = list(self.state.hand)
        next_bids = list(self.state.bids)
        next_tricks_won = list(self.state.tricks_won)
        next_scores = list(self.state.scores)
        next_current_trick = list(self.state.current_trick)
        next_played_cards = list(self.state.played_cards)
        next_current_player = self.state.current_player
        next_phase = self.state.phase
        next_round_finished = self.state.round_finished
        next_leader = self.state.leader
        next_trump_card = self.state.trump_card
        next_trump_suit = self.state.trump_suit
        if hand is not None:
            if len(set(hand)) != len(hand):
                raise ValueError("Duplicate cards in advised hand.")
            next_hand = sorted(hand)
        if bids is not None:
            self._validate_exact_length(bids, label="Bids")
            for bid in bids:
                if bid is not None and (bid < 0 or bid > self.state.hand_size):
                    raise ValueError("Bid out of range.")
            next_bids = list(bids)
        if tricks_won is not None:
            self._validate_exact_length(tricks_won, label="Tricks won")
            if any(value < 0 for value in tricks_won):
                raise ValueError("Tricks won cannot be negative.")
            next_tricks_won = list(tricks_won)
        if scores is not None:
            self._validate_exact_length(scores, label="Scores")
            next_scores = list(scores)
        if current_trick is not None:
            self._validate_current_trick(current_trick)
            prior_current_cards = {card_id for _, card_id in self.state.current_trick}
            remaining_played = [card_id for card_id in self.state.played_cards if card_id not in prior_current_cards]
            next_current_trick = list(current_trick)
            for _, card_id in current_trick:
                if card_id not in remaining_played:
                    remaining_played.append(card_id)
            next_played_cards = remaining_played
        if current_player is not None:
            self._validate_player(current_player, label="Current player")
            next_current_player = current_player
        if phase is not None:
            if phase not in {"bidding", "play", "finished"}:
                raise ValueError("Invalid phase.")
            next_phase = phase
            next_round_finished = phase == "finished"
        if leader is not None:
            self._validate_player(leader, label="Leader")
            next_leader = leader
        if trump_card != "__unset__":
            next_trump_card = trump_card if trump_card is not None else None
            next_trump_suit = suit(trump_card) if trump_card is not None else None
        if any(card_id in next_hand for _, card_id in next_current_trick):
            raise ValueError("A card cannot be both in hand and on the table.")
        self.state.hand = next_hand
        self.state.bids = next_bids
        self.state.tricks_won = next_tricks_won
        self.state.scores = next_scores
        self.state.current_trick = next_current_trick
        self.state.played_cards = next_played_cards
        self.state.current_player = next_current_player
        self.state.phase = next_phase
        self.state.round_finished = next_round_finished
        self.state.leader = next_leader
        self.state.trump_card = next_trump_card
        self.state.trump_suit = next_trump_suit

    def start_next_round(self, *, dealer: int, hand_size: int, hand: Sequence[int], trump_card: Optional[int], round_index: Optional[int] = None) -> None:
        self._validate_player(dealer, label="Dealer")
        if hand_size < 1 or hand_size > 8:
            raise ValueError("Hand size must be between 1 and 8.")
        if len(hand) != hand_size:
            raise ValueError("Hand size does not match provided cards.")
        if len(set(hand)) != len(hand):
            raise ValueError("Duplicate cards in advised hand.")
        self.state.dealer = dealer
        self.state.leader = (dealer + 1) % self.state.players
        self.state.current_player = self.state.leader
        self.state.hand_size = hand_size
        self.state.trump_card = trump_card
        self.state.trump_suit = suit(trump_card) if trump_card is not None else None
        self.state.hand = sorted(hand)
        self.state.bids = [None for _ in range(self.state.players)]
        self.state.tricks_won = [0 for _ in range(self.state.players)]
        self.state.current_trick = []
        self.state.played_cards = []
        self.state.history_tokens = [TOKEN_ROUND_START, TOKEN_MODE_BASE, TOKEN_MODE_BASE + 8 + hand_size]
        self.state.event_log = [{"type": "round_start", "round": self.state.round_index if round_index is None else round_index}]
        self.state.plays_in_round = 0
        self.state.phase = "bidding"
        self.state.round_finished = False
        if round_index is not None:
            self.state.round_index = round_index
        else:
            self.state.round_index += 1

    def snapshot(self) -> Dict[str, object]:
        return {
            "players": self.state.players,
            "advised_seat": self.state.advised_seat,
            "dealer": self.state.dealer,
            "leader": self.state.leader,
            "current_player": self.state.current_player,
            "phase": self.state.phase,
            "hand_size": self.state.hand_size,
            "round_index": self.state.round_index,
            "trump_card": None if self.state.trump_card is None else {"id": self.state.trump_card, "label": card_label(self.state.trump_card)},
            "trump_suit": None if self.state.trump_suit is None else SUITS[self.state.trump_suit],
            "hand": _card_entries(self.state.hand),
            "bids": self.state.bids,
            "tricks_won": self.state.tricks_won,
            "scores": self.state.scores,
            "current_trick": [
                {"seat": seat, "card": {"id": card_id, "label": card_label(card_id)}}
                for seat, card_id in self.state.current_trick
            ],
            "played_cards": _card_entries(self.state.played_cards),
            "legal_actions": [{"action": action, "label": _action_label(action)} for action in self.legal_actions()],
            "round_finished": self.state.round_finished,
            "event_log": copy.deepcopy(self.state.event_log[-40:]),
            "manual_state": {
                "hand": ",".join(card_label(card_id) for card_id in self.state.hand),
                "bids": ",".join("" if bid is None else str(bid) for bid in self.state.bids),
                "tricks_won": ",".join(str(value) for value in self.state.tricks_won),
                "scores": ",".join(str(value) for value in self.state.scores),
                "current_trick": ",".join("{seat}:{card}".format(seat=seat, card=card_label(card_id)) for seat, card_id in self.state.current_trick),
            },
        }

class SessionManager:
    def __init__(self, recommender: RecommendationService):
        self.recommender = recommender
        self.sessions = {}  # type: Dict[str, object]

    def create_full_session(self, *, mode: str, players: int, seed: int, roles: Sequence[str], checkpoint_path: Optional[str], device: str) -> FullGameSession:
        session_id = str(uuid.uuid4())
        session = FullGameSession(
            session_id=session_id,
            mode=mode,
            players=players,
            seed=seed,
            roles=roles,
            checkpoint_path=checkpoint_path,
            device=device,
            recommender=self.recommender,
        )
        self.sessions[session_id] = session
        return session

    def create_advisor_session(self, *, players: int, advised_seat: int, dealer: int, hand_size: int, hand: Sequence[int], trump_card: Optional[int], scores: Optional[Sequence[int]], round_index: int) -> str:
        session_id = str(uuid.uuid4())
        tracker = PublicSeatTracker.create(
            players=players,
            advised_seat=advised_seat,
            dealer=dealer,
            hand_size=hand_size,
            hand=hand,
            trump_card=trump_card,
            scores=scores,
            round_index=round_index,
        )
        self.sessions[session_id] = tracker
        return session_id

    def load_replay_session(self, payload: Dict[str, object]) -> ReplaySession:
        session = ReplaySession(payload)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> object:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.sessions[session_id]


def parse_card_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [parse_card(item) for item in text.split(",") if item.strip()]


def parse_optional_int_list(text: str) -> List[Optional[int]]:
    if not text.strip():
        return []
    if all(not item.strip() for item in text.split(",")):
        return []
    values = []
    for item in text.split(","):
        cleaned = item.strip()
        values.append(None if cleaned == "" else int(cleaned))
    return values


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_trick(text: str) -> List[tuple[int, int]]:
    if not text.strip():
        return []
    parsed = []
    for item in text.split(","):
        seat_text, card_text = item.split(":", 1)
        parsed.append((int(seat_text.strip()), parse_card(card_text.strip())))
    return parsed
