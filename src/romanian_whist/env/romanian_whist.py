from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

from romanian_whist.rules.config import WhistVariantConfig
from romanian_whist.rules.game import RomanianWhistGame, StepOutcome


@dataclass
class EnvTransition:
    observation: Dict[str, object]
    rewards: Dict[str, float]
    terminations: Dict[str, bool]
    truncations: Dict[str, bool]
    infos: Dict[str, Dict[str, object]]


class RomanianWhistEnv:
    metadata = {"name": "romanian_whist_v0", "is_parallelizable": False}

    def __init__(self, config: Optional[WhistVariantConfig] = None):
        self.config = config or WhistVariantConfig()
        self.possible_agents = ["player_{index}".format(index=index) for index in range(self.config.players)]
        self.agent_to_index = {agent: index for index, agent in enumerate(self.possible_agents)}
        self.agents = list(self.possible_agents)
        self.game = RomanianWhistGame(self.config)
        self.rewards = dict((agent, 0.0) for agent in self.possible_agents)
        self.terminations = dict((agent, False) for agent in self.possible_agents)
        self.truncations = dict((agent, False) for agent in self.possible_agents)
        self.infos = dict((agent, {}) for agent in self.possible_agents)

    @property
    def agent_selection(self) -> str:
        return self.possible_agents[self.game.current_player]

    def reset(self, seed: Optional[int] = None) -> Dict[str, object]:
        self.game.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self.rewards = dict((agent, 0.0) for agent in self.possible_agents)
        self.terminations = dict((agent, False) for agent in self.possible_agents)
        self.truncations = dict((agent, False) for agent in self.possible_agents)
        self.infos = dict((agent, {}) for agent in self.possible_agents)
        return self.observe(self.agent_selection)

    def observe(self, agent: str) -> Dict[str, object]:
        return self.game.observe(self.agent_index(agent))

    def observe_for_baseline(self, agent: str) -> Dict[str, object]:
        return self.game.observe_for_baseline(self.agent_index(agent))

    def agent_index(self, agent: str) -> int:
        return self.agent_to_index[agent]

    def step_outcome(self, action: int) -> StepOutcome:
        outcome = self.game.step(action)
        self._update_from_outcome(outcome)
        return outcome

    def step(self, action: int, *, include_observation: bool = True) -> EnvTransition:
        outcome = self.step_outcome(action)
        observation = self.observe(self.agent_selection) if include_observation and not outcome.match_finished else {}
        return EnvTransition(
            observation=observation,
            rewards=dict(self.rewards),
            terminations=dict(self.terminations),
            truncations=dict(self.truncations),
            infos=dict(self.infos),
        )

    def agent_iter(self) -> Generator[str, None, None]:
        while not all(self.terminations.values()):
            yield self.agent_selection

    def render(self) -> str:
        return "\n".join(self.game.summary_lines())

    def serialize_replay(self) -> Dict[str, object]:
        return self.game.serialize_replay()

    def _update_from_outcome(self, outcome: StepOutcome) -> None:
        for index, agent in enumerate(self.possible_agents):
            self.rewards[agent] = outcome.rewards[index]
            self.terminations[agent] = outcome.match_finished
            self.infos[agent] = {
                "round_finished": outcome.round_finished,
                "match_finished": outcome.match_finished,
            }

    def action_space(self, agent: Optional[str] = None) -> int:
        return 61

    def observation_space(self, agent: Optional[str] = None) -> Dict[str, tuple]:
        max_players = self.config.max_players
        return {
            "hand_mask": (52,),
            "current_trick_cards": (max_players,),
            "current_trick_players": (max_players,),
            "played_card_mask": (52,),
            "public_card_by_player": (max_players,),
            "bids": (max_players,),
            "tricks_won": (max_players,),
            "cumulative_scores": (max_players,),
            "history_tokens": (self.config.max_history_tokens,),
            "legal_action_mask": (61,),
        }
