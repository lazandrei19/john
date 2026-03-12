from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.config import WhistVariantConfig

SCRIPTED_AGENT_TYPES = (RandomLegalAgent, SafeHeuristicAgent, BidPlayHeuristicAgent)


@dataclass
class TournamentStats:
    average_scores: Dict[str, float]
    contract_hit_rate: Dict[str, float]
    trick_differential: Dict[str, float]
    elo_like: Dict[str, float]
    average_bid: Dict[str, float]
    min_bid: Dict[str, float]
    max_bid: Dict[str, float]
    bid_mae: Dict[str, float]
    underbid_rate: Dict[str, float]
    overbid_rate: Dict[str, float]
    strong_hand_underbid_rate: Dict[str, float]


def bid_records_from_events(
    events: Sequence[Mapping[str, object]],
    seat_labels: Sequence[str],
    target_bids_by_round: Mapping[int, Mapping[int, int]],
) -> list[Dict[str, float | str]]:
    bid_records = []
    current_bids = None  # type: Optional[list[Optional[int]]]
    current_tricks = [0 for _ in seat_labels]
    current_round = -1
    current_hand_size = 0
    for event in events:
        event_type = str(event["type"])
        if event_type == "round_start":
            current_round = int(event["round"])
            current_hand_size = int(event["hand_size"])
            current_bids = [None for _ in seat_labels]
            current_tricks = [0 for _ in seat_labels]
        elif event_type == "bid":
            if current_bids is None:
                current_bids = [None for _ in seat_labels]
                current_tricks = [0 for _ in seat_labels]
            current_bids[int(event["player"])] = int(event["bid"])
        elif event_type == "trick_win" and current_bids is not None:
            current_tricks[int(event["player"])] += 1
        elif event_type == "round_score" and current_bids is not None:
            round_targets = target_bids_by_round.get(current_round, {})
            for seat, label in enumerate(seat_labels):
                bid = current_bids[seat]
                if bid is None or seat not in round_targets:
                    continue
                target_bid = int(round_targets[seat])
                actual_tricks = int(current_tricks[seat])
                bid_records.append(
                    {
                        "label": label,
                        "hand_size": float(current_hand_size),
                        "bid": float(bid),
                        "target_bid": float(target_bid),
                        "actual_tricks": float(actual_tricks),
                        "strong_hand": 1.0 if target_bid >= 3 else 0.0,
                    }
                )
            current_bids = None
    return bid_records


def bid_stats_from_records(
    bid_records: Sequence[Mapping[str, float | str]],
    seat_labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    metrics = {
        "average_bid": {},
        "min_bid": {},
        "max_bid": {},
        "bid_mae": {},
        "underbid_rate": {},
        "overbid_rate": {},
        "strong_hand_underbid_rate": {},
    }  # type: Dict[str, Dict[str, float]]
    for label in seat_labels:
        records = [record for record in bid_records if str(record["label"]) == label]
        if not records:
            for metric in metrics.values():
                metric[label] = 0.0
            continue
        bids = [float(record["bid"]) for record in records]
        actuals = [float(record["actual_tricks"]) for record in records]
        strong_records = [record for record in records if float(record["strong_hand"]) > 0.0]
        metrics["average_bid"][label] = sum(bids) / float(len(bids))
        metrics["min_bid"][label] = min(bids)
        metrics["max_bid"][label] = max(bids)
        metrics["bid_mae"][label] = sum(abs(bid - actual) for bid, actual in zip(bids, actuals)) / float(len(records))
        metrics["underbid_rate"][label] = sum(1.0 for bid, actual in zip(bids, actuals) if bid < actual) / float(len(records))
        metrics["overbid_rate"][label] = sum(1.0 for bid, actual in zip(bids, actuals) if bid > actual) / float(len(records))
        metrics["strong_hand_underbid_rate"][label] = (
            sum(1.0 for record in strong_records if float(record["bid"]) < float(record["target_bid"])) / float(len(strong_records))
            if strong_records
            else 0.0
        )
    return metrics


class TournamentRunner:
    def __init__(
        self,
        base_config: WhistVariantConfig,
        bid_target_resolver: Optional[Callable[[RomanianWhistEnv, int], int]] = None,
    ):
        self.base_config = base_config
        self.bid_target_resolver = bid_target_resolver

    def run(
        self,
        agents: Mapping[str, object] | Sequence[tuple[str, object]],
        matches: int = 8,
        seed: int = 0,
    ) -> TournamentStats:
        participants = list(agents.items()) if isinstance(agents, Mapping) else list(agents)
        if len(participants) != self.base_config.players:
            raise ValueError(
                "Tournament participants ({actual}) must match configured players ({expected}).".format(
                    actual=len(participants), expected=self.base_config.players
                )
            )
        score_totals = dict((name, 0.0) for name, _ in participants)
        contract_hits = dict((name, 0.0) for name, _ in participants)
        trick_diffs = dict((name, 0.0) for name, _ in participants)
        elo = dict((name, 1000.0) for name, _ in participants)
        bid_totals = dict((name, 0.0) for name, _ in participants)
        bid_counts = dict((name, 0.0) for name, _ in participants)
        bid_mins = dict((name, float("inf")) for name, _ in participants)
        bid_maxs = dict((name, float("-inf")) for name, _ in participants)
        bid_mae = dict((name, 0.0) for name, _ in participants)
        underbid_rate = dict((name, 0.0) for name, _ in participants)
        overbid_rate = dict((name, 0.0) for name, _ in participants)
        strong_hand_underbid_rate = dict((name, 0.0) for name, _ in participants)
        player_names = [name for name, _ in participants]
        rounds_per_match = len(self.base_config.schedule())

        for match_index in range(matches):
            env = RomanianWhistEnv(self.base_config)
            env.reset(seed=seed + match_index)
            rotation = match_index % len(participants)
            seat_order = participants[rotation:] + participants[:rotation]
            target_bids_by_round = {}  # type: Dict[int, Dict[int, int]]
            while not all(env.terminations.values()):
                acting_agent = env.agent_selection
                seat = env.agent_index(acting_agent)
                _, agent = seat_order[seat]
                state = env.game.round_state
                if (
                    state is not None
                    and state.phase == "bidding"
                    and self.bid_target_resolver is not None
                ):
                    target_bids_by_round.setdefault(state.round_index, {})[seat] = self.bid_target_resolver(env, seat)
                if isinstance(agent, SCRIPTED_AGENT_TYPES):
                    action = agent.select_action_from_game(env.game, seat)
                else:
                    action = agent.select_action(env.observe(acting_agent))
                env.step(action, include_observation=False)

            replay = env.serialize_replay()
            final_scores = replay["scores"]
            seat_labels = [name for name, _ in seat_order]
            for seat, name in enumerate(seat_labels):
                score_totals[name] += final_scores[seat]
            contract, trick, bid_summary = self._extract_round_metrics(replay, seat_labels)
            bid_records = bid_records_from_events(replay["events"], seat_labels, target_bids_by_round)
            bid_metrics = bid_stats_from_records(bid_records, seat_labels)
            for name in contract:
                contract_hits[name] += contract[name]
                trick_diffs[name] += trick[name]
                bid_totals[name] += bid_summary["total"][name]
                bid_counts[name] += bid_summary["count"][name]
                bid_mins[name] = min(bid_mins[name], bid_summary["min"][name])
                bid_maxs[name] = max(bid_maxs[name], bid_summary["max"][name])
                bid_mae[name] += bid_metrics["bid_mae"][name]
                underbid_rate[name] += bid_metrics["underbid_rate"][name]
                overbid_rate[name] += bid_metrics["overbid_rate"][name]
                strong_hand_underbid_rate[name] += bid_metrics["strong_hand_underbid_rate"][name]
            self._update_elo(elo, seat_labels, final_scores)

        divisor = float(matches)
        round_divisor = float(matches * rounds_per_match)
        return TournamentStats(
            average_scores=dict((name, score / divisor) for name, score in score_totals.items()),
            contract_hit_rate=dict((name, value / round_divisor) for name, value in contract_hits.items()),
            trick_differential=dict((name, value / round_divisor) for name, value in trick_diffs.items()),
            elo_like=elo,
            average_bid={name: (bid_totals[name] / bid_counts[name]) if bid_counts[name] else 0.0 for name in player_names},
            min_bid={name: (bid_mins[name] if bid_counts[name] else 0.0) for name in player_names},
            max_bid={name: (bid_maxs[name] if bid_counts[name] else 0.0) for name in player_names},
            bid_mae={name: (bid_mae[name] / divisor) for name in player_names},
            underbid_rate={name: (underbid_rate[name] / divisor) for name in player_names},
            overbid_rate={name: (overbid_rate[name] / divisor) for name in player_names},
            strong_hand_underbid_rate={name: (strong_hand_underbid_rate[name] / divisor) for name in player_names},
        )

    def _extract_round_metrics(
        self, replay: Mapping[str, object], seat_order: Sequence[str]
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        contract_hits = dict((name, 0.0) for name in seat_order)
        trick_diff = dict((name, 0.0) for name in seat_order)
        bid_total = dict((name, 0.0) for name in seat_order)
        bid_count = dict((name, 0.0) for name in seat_order)
        bid_min = dict((name, float("inf")) for name in seat_order)
        bid_max = dict((name, float("-inf")) for name in seat_order)
        current_tricks = [0 for _ in seat_order]
        current_bids = None
        for event in replay["events"]:
            if event["type"] == "round_start":
                current_bids = None
                current_tricks = [0 for _ in seat_order]
            elif event["type"] == "round_score":
                if current_bids is not None:
                    for seat, agent_name in enumerate(seat_order):
                        diff = abs(current_bids[seat] - current_tricks[seat])
                        contract_hits[agent_name] += 1.0 if diff == 0 else 0.0
                        trick_diff[agent_name] += -float(diff)
                        bid_total[agent_name] += float(current_bids[seat])
                        bid_count[agent_name] += 1.0
                        bid_min[agent_name] = min(bid_min[agent_name], float(current_bids[seat]))
                        bid_max[agent_name] = max(bid_max[agent_name], float(current_bids[seat]))
            elif event["type"] == "bid":
                if current_bids is None:
                    current_bids = [0 for _ in seat_order]
                    current_tricks = [0 for _ in seat_order]
                current_bids[event["player"]] = event["bid"]
            elif event["type"] == "trick_win" and current_bids is not None:
                current_tricks[event["player"]] += 1
        return contract_hits, trick_diff, {"total": bid_total, "count": bid_count, "min": bid_min, "max": bid_max}

    def _update_elo(self, elo: Dict[str, float], seat_order: Sequence[str], final_scores: Sequence[float]) -> None:
        for left in range(len(seat_order)):
            for right in range(left + 1, len(seat_order)):
                player_left = seat_order[left]
                player_right = seat_order[right]
                expected_left = 1.0 / (1.0 + 10 ** ((elo[player_right] - elo[player_left]) / 400.0))
                actual_left = 1.0 if final_scores[left] > final_scores[right] else 0.5 if final_scores[left] == final_scores[right] else 0.0
                delta = 16.0 * (actual_left - expected_left)
                elo[player_left] += delta
                elo[player_right] -= delta

    def summarize_match_results(
        self,
        match_results: Sequence[Mapping[str, object]],
        participants: Sequence[tuple[str, object]],
    ) -> TournamentStats:
        score_totals = dict((name, 0.0) for name, _ in participants)
        contract_hits = dict((name, 0.0) for name, _ in participants)
        trick_diffs = dict((name, 0.0) for name, _ in participants)
        elo = dict((name, 1000.0) for name, _ in participants)
        bid_totals = dict((name, 0.0) for name, _ in participants)
        bid_counts = dict((name, 0.0) for name, _ in participants)
        bid_mins = dict((name, float("inf")) for name, _ in participants)
        bid_maxs = dict((name, float("-inf")) for name, _ in participants)
        bid_mae = dict((name, 0.0) for name, _ in participants)
        underbid_rate = dict((name, 0.0) for name, _ in participants)
        overbid_rate = dict((name, 0.0) for name, _ in participants)
        strong_hand_underbid_rate = dict((name, 0.0) for name, _ in participants)
        rounds_per_match = len(self.base_config.schedule())

        for result in sorted(match_results, key=lambda item: int(item["match_index"])):
            seat_labels = list(result["seat_labels"])
            final_scores = list(result["final_scores"])
            for seat, name in enumerate(seat_labels):
                score_totals[name] += final_scores[seat]
            contract, trick, bid_summary = self._extract_round_metrics({"events": result["events"]}, seat_labels)
            bid_metrics = bid_stats_from_records(
                list(result.get("bid_records", [])),
                seat_labels,
            )
            for name in contract:
                contract_hits[name] += contract[name]
                trick_diffs[name] += trick[name]
                bid_totals[name] += bid_summary["total"][name]
                bid_counts[name] += bid_summary["count"][name]
                bid_mins[name] = min(bid_mins[name], bid_summary["min"][name])
                bid_maxs[name] = max(bid_maxs[name], bid_summary["max"][name])
                bid_mae[name] += bid_metrics["bid_mae"][name]
                underbid_rate[name] += bid_metrics["underbid_rate"][name]
                overbid_rate[name] += bid_metrics["overbid_rate"][name]
                strong_hand_underbid_rate[name] += bid_metrics["strong_hand_underbid_rate"][name]
            self._update_elo(elo, seat_labels, final_scores)

        divisor = float(len(match_results))
        round_divisor = float(len(match_results) * rounds_per_match)
        return TournamentStats(
            average_scores=dict((name, score / divisor) for name, score in score_totals.items()),
            contract_hit_rate=dict((name, value / round_divisor) for name, value in contract_hits.items()),
            trick_differential=dict((name, value / round_divisor) for name, value in trick_diffs.items()),
            elo_like=elo,
            average_bid={name: (bid_totals[name] / bid_counts[name]) if bid_counts[name] else 0.0 for name, _ in participants},
            min_bid={name: (bid_mins[name] if bid_counts[name] else 0.0) for name, _ in participants},
            max_bid={name: (bid_maxs[name] if bid_counts[name] else 0.0) for name, _ in participants},
            bid_mae={name: (bid_mae[name] / divisor) for name, _ in participants},
            underbid_rate={name: (underbid_rate[name] / divisor) for name, _ in participants},
            overbid_rate={name: (overbid_rate[name] / divisor) for name, _ in participants},
            strong_hand_underbid_rate={name: (strong_hand_underbid_rate[name] / divisor) for name, _ in participants},
        )


def stats_to_dict(stats: TournamentStats) -> Dict[str, Dict[str, float]]:
    return {
        "average_scores": stats.average_scores,
        "contract_hit_rate": stats.contract_hit_rate,
        "trick_differential": stats.trick_differential,
        "elo_like": stats.elo_like,
        "average_bid": stats.average_bid,
        "min_bid": stats.min_bid,
        "max_bid": stats.max_bid,
        "bid_mae": stats.bid_mae,
        "underbid_rate": stats.underbid_rate,
        "overbid_rate": stats.overbid_rate,
        "strong_hand_underbid_rate": stats.strong_hand_underbid_rate,
    }


def average_stat_dicts(stat_dicts: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    if not stat_dicts:
        return {}
    metric_names = stat_dicts[0].keys()
    averaged = {}  # type: Dict[str, Dict[str, float]]
    for metric_name in metric_names:
        agent_names = stat_dicts[0][metric_name].keys()
        averaged[metric_name] = {
            agent_name: sum(stat_dict[metric_name][agent_name] for stat_dict in stat_dicts) / float(len(stat_dicts))
            for agent_name in agent_names
        }
    return averaged
