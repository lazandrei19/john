from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

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


class TournamentRunner:
    def __init__(self, base_config: WhistVariantConfig):
        self.base_config = base_config

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
        player_names = [name for name, _ in participants]
        rounds_per_match = len(self.base_config.schedule())

        for match_index in range(matches):
            env = RomanianWhistEnv(self.base_config)
            env.reset(seed=seed + match_index)
            rotation = match_index % len(participants)
            seat_order = participants[rotation:] + participants[:rotation]
            while not all(env.terminations.values()):
                acting_agent = env.agent_selection
                seat = env.agent_index(acting_agent)
                agent_name, agent = seat_order[seat]
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
            contract, trick = self._extract_round_metrics(replay, seat_labels)
            for name in contract:
                contract_hits[name] += contract[name]
                trick_diffs[name] += trick[name]
            self._update_elo(elo, seat_labels, final_scores)

        divisor = float(matches)
        round_divisor = float(matches * rounds_per_match)
        return TournamentStats(
            average_scores=dict((name, score / divisor) for name, score in score_totals.items()),
            contract_hit_rate=dict((name, value / round_divisor) for name, value in contract_hits.items()),
            trick_differential=dict((name, value / round_divisor) for name, value in trick_diffs.items()),
            elo_like=elo,
        )

    def _extract_round_metrics(self, replay: Mapping[str, object], seat_order: Sequence[str]) -> tuple[Dict[str, float], Dict[str, float]]:
        contract_hits = dict((name, 0.0) for name in seat_order)
        trick_diff = dict((name, 0.0) for name in seat_order)
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
            elif event["type"] == "bid":
                if current_bids is None:
                    current_bids = [0 for _ in seat_order]
                    current_tricks = [0 for _ in seat_order]
                current_bids[event["player"]] = event["bid"]
            elif event["type"] == "trick_win" and current_bids is not None:
                current_tricks[event["player"]] += 1
        return contract_hits, trick_diff

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
        rounds_per_match = len(self.base_config.schedule())

        for result in sorted(match_results, key=lambda item: int(item["match_index"])):
            seat_labels = list(result["seat_labels"])
            final_scores = list(result["final_scores"])
            for seat, name in enumerate(seat_labels):
                score_totals[name] += final_scores[seat]
            contract, trick = self._extract_round_metrics({"events": result["events"]}, seat_labels)
            for name in contract:
                contract_hits[name] += contract[name]
                trick_diffs[name] += trick[name]
            self._update_elo(elo, seat_labels, final_scores)

        divisor = float(len(match_results))
        round_divisor = float(len(match_results) * rounds_per_match)
        return TournamentStats(
            average_scores=dict((name, score / divisor) for name, score in score_totals.items()),
            contract_hit_rate=dict((name, value / round_divisor) for name, value in contract_hits.items()),
            trick_differential=dict((name, value / round_divisor) for name, value in trick_diffs.items()),
            elo_like=elo,
        )


def stats_to_dict(stats: TournamentStats) -> Dict[str, Dict[str, float]]:
    return {
        "average_scores": stats.average_scores,
        "contract_hit_rate": stats.contract_hit_rate,
        "trick_differential": stats.trick_differential,
        "elo_like": stats.elo_like,
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
