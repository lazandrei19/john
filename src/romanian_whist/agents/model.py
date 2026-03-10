from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def tensorize_observation(observation: Mapping[str, object], device: Optional[torch.device] = None) -> Dict[str, Tensor]:
    tensor_obs = {}  # type: Dict[str, Tensor]
    for key, value in observation.items():
        tensor = torch.as_tensor(value)
        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            tensor_obs[key] = tensor.to(device=device)
        else:
            tensor_obs[key] = tensor.to(dtype=torch.float32, device=device)
    return tensor_obs


def batch_observations(observations: list[Mapping[str, object]], device: Optional[torch.device] = None) -> Dict[str, Tensor]:
    tensorized = [tensorize_observation(obs) for obs in observations]
    keys = set(tensorized[0].keys())
    for obs in tensorized[1:]:
        keys.intersection_update(obs.keys())
    batches = {}  # type: Dict[str, Tensor]
    for key in tensorized[0].keys():
        if key not in keys:
            continue
        stacked = torch.stack([obs[key] for obs in tensorized], dim=0)
        batches[key] = stacked.to(device=device) if device is not None else stacked
    return batches


def masked_logits(logits: Tensor, legal_action_mask: Tensor) -> Tensor:
    mask = legal_action_mask.to(dtype=torch.bool)
    invalid_fill = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, invalid_fill)


@dataclass
class RankedAction:
    action: int
    probability: float
    logit: float


@dataclass
class ActionRecommendation:
    chosen_action: int
    top_actions: list[RankedAction]
    legal_actions: list[int]
    value: float


@dataclass
class PolicyForwardOutputs:
    logits: Tensor
    values: Tensor
    belief_logits: Tensor


@dataclass(frozen=True)
class PolicyNetworkConfig:
    embed_dim: int = 128
    history_vocab_size: int = 256
    max_players: int = 6
    max_history_tokens: int = 64
    card_encoder_layers: int = 2
    history_encoder_layers: int = 2
    seat_encoder_layers: int = 2
    branch_hidden_multiplier: int = 2
    belief_hidden_multiplier: int = 2


class WhistPolicyNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        history_vocab_size: int = 256,
        max_players: int = 6,
        max_history_tokens: int = 64,
        card_encoder_layers: int = 2,
        history_encoder_layers: int = 2,
        seat_encoder_layers: int = 2,
        branch_hidden_multiplier: int = 2,
        belief_hidden_multiplier: int = 2,
    ):
        super().__init__()
        self.config = PolicyNetworkConfig(
            embed_dim=embed_dim,
            history_vocab_size=history_vocab_size,
            max_players=max_players,
            max_history_tokens=max_history_tokens,
            card_encoder_layers=card_encoder_layers,
            history_encoder_layers=history_encoder_layers,
            seat_encoder_layers=seat_encoder_layers,
            branch_hidden_multiplier=branch_hidden_multiplier,
            belief_hidden_multiplier=belief_hidden_multiplier,
        )
        self.embed_dim = embed_dim
        self.max_players = max_players
        self.max_history_tokens = max_history_tokens
        self.belief_classes = max_players + 2  # player seats, played pile, inactive cards

        self.register_buffer("card_rank_ids", torch.tensor([index % 13 for index in range(52)], dtype=torch.long))
        self.register_buffer("card_suit_ids", torch.tensor([index // 13 for index in range(52)], dtype=torch.long))

        self.card_id_embedding = nn.Embedding(52, embed_dim)
        self.card_rank_embedding = nn.Embedding(13, embed_dim)
        self.card_suit_embedding = nn.Embedding(4, embed_dim)
        self.card_status_embedding = nn.Embedding(6, embed_dim)
        self.owner_hint_embedding = nn.Embedding(self.belief_classes, embed_dim)
        self.belief_class_embedding = nn.Embedding(self.belief_classes, embed_dim)

        self.history_embedding = nn.Embedding(history_vocab_size, embed_dim, padding_idx=0)
        self.history_position_embedding = nn.Embedding(max_history_tokens + 1, embed_dim)
        self.history_validity_embedding = nn.Embedding(2, embed_dim)

        self.seat_embedding = nn.Embedding(max_players, embed_dim)
        self.relative_seat_embedding = nn.Embedding(max_players + 1, embed_dim)
        self.trick_position_embedding = nn.Embedding(max_players + 1, embed_dim)
        self.seat_validity_embedding = nn.Embedding(2, embed_dim)

        scalar_dim = (max_players * 3) + 11
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scalar_to_summary = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.seat_numeric_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        card_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        history_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        seat_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.card_encoder = nn.TransformerEncoder(card_encoder_layer, num_layers=card_encoder_layers)
        self.history_encoder = nn.TransformerEncoder(history_encoder_layer, num_layers=history_encoder_layers)
        self.seat_encoder = nn.TransformerEncoder(seat_encoder_layer, num_layers=seat_encoder_layers)
        self.history_memory = nn.GRU(embed_dim, embed_dim, batch_first=True)

        self.card_summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.history_summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seat_summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        shared_hidden = embed_dim * branch_hidden_multiplier
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim * 5, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.bid_tower = nn.Sequential(
            nn.Linear(embed_dim * 4, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.play_tower = nn.Sequential(
            nn.Linear(embed_dim * 4, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.value_tower = nn.Sequential(
            nn.Linear(embed_dim * 4, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        belief_hidden = embed_dim * belief_hidden_multiplier
        self.belief_head = nn.Sequential(
            nn.Linear(embed_dim * 2, belief_hidden),
            nn.GELU(),
            nn.LayerNorm(belief_hidden),
            nn.Linear(belief_hidden, self.belief_classes),
        )
        self.bid_head = nn.Linear(embed_dim, 9)
        self.play_query = nn.Linear(embed_dim, embed_dim)
        self.play_candidate = nn.Linear(embed_dim * 2, embed_dim)
        self.play_score = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(embed_dim, 1)

        nn.init.normal_(self.card_summary_token, std=0.02)
        nn.init.normal_(self.history_summary_token, std=0.02)
        nn.init.normal_(self.seat_summary_token, std=0.02)

    @classmethod
    def from_config(cls, config: PolicyNetworkConfig) -> "WhistPolicyNetwork":
        return cls(
            embed_dim=config.embed_dim,
            history_vocab_size=config.history_vocab_size,
            max_players=config.max_players,
            max_history_tokens=config.max_history_tokens,
            card_encoder_layers=config.card_encoder_layers,
            history_encoder_layers=config.history_encoder_layers,
            seat_encoder_layers=config.seat_encoder_layers,
            branch_hidden_multiplier=config.branch_hidden_multiplier,
            belief_hidden_multiplier=config.belief_hidden_multiplier,
        )

    def config_dict(self) -> Dict[str, int]:
        return asdict(self.config)

    def forward(self, observation: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        outputs = self.forward_with_aux(observation)
        return outputs.logits, outputs.values

    def forward_with_aux(self, observation: Dict[str, Tensor]) -> PolicyForwardOutputs:
        scalar_features = self._scalar_features(observation)
        scalar_embedding = self.scalar_encoder(scalar_features)
        seat_summary, seat_tokens = self._seat_context(observation, scalar_embedding)
        history_summary, history_memory = self._history_context(observation, scalar_embedding)
        card_summary, card_tokens = self._card_context(observation, scalar_embedding, seat_summary, history_summary)

        shared_input = torch.cat([card_summary, seat_summary, history_summary, history_memory, scalar_embedding], dim=-1)
        shared_latent = self.shared_trunk(shared_input)

        belief_context = torch.cat(
            [card_tokens, shared_latent.unsqueeze(1).expand(-1, card_tokens.shape[1], -1)],
            dim=-1,
        )
        belief_logits = self.belief_head(belief_context)
        belief_probs = torch.softmax(belief_logits, dim=-1)
        belief_context_embed = torch.einsum("bck,kd->bcd", belief_probs, self.belief_class_embedding.weight)
        refined_card_tokens = card_tokens + belief_context_embed
        refined_card_summary = refined_card_tokens.mean(dim=1)

        bid_latent = self.bid_tower(torch.cat([shared_latent, seat_summary, history_summary, history_memory], dim=-1))
        play_latent = self.play_tower(torch.cat([shared_latent, refined_card_summary, seat_summary, history_memory], dim=-1))
        value_latent = self.value_tower(torch.cat([shared_latent, refined_card_summary, seat_summary, scalar_embedding], dim=-1))

        bid_logits = self.bid_head(bid_latent)
        play_logits = self._score_card_actions(refined_card_tokens, play_latent, seat_tokens)
        logits = torch.cat([bid_logits, play_logits], dim=-1)
        values = self.value_head(value_latent).squeeze(-1)
        return PolicyForwardOutputs(logits=logits, values=values, belief_logits=belief_logits)

    def _card_context(
        self,
        observation: Dict[str, Tensor],
        scalar_embedding: Tensor,
        seat_summary: Tensor,
        history_summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = observation["hand_mask"].shape[0]
        device = observation["hand_mask"].device
        base_cards = (
            self.card_id_embedding.weight.unsqueeze(0)
            + self.card_rank_embedding(self.card_rank_ids).unsqueeze(0)
            + self.card_suit_embedding(self.card_suit_ids).unsqueeze(0)
        )

        hand_mask = observation["hand_mask"].bool()
        played_mask = observation["played_card_mask"].bool()
        public_cards = observation["public_card_by_player"].long()
        current_trick_cards = observation["current_trick_cards"].long()
        current_trick_players = observation["current_trick_players"].long()
        seat_index = observation["seat_index"].long()

        status_ids = torch.zeros((batch_size, 52), dtype=torch.long, device=device)
        owner_hints = torch.full((batch_size, 52), self.max_players + 1, dtype=torch.long, device=device)

        status_ids = torch.where(hand_mask, torch.ones_like(status_ids), status_ids)
        owner_hints = torch.where(
            hand_mask,
            seat_index.unsqueeze(1).expand(-1, 52).clamp(max=self.max_players - 1),
            owner_hints,
        )
        status_ids = torch.where(played_mask, torch.full_like(status_ids, 2), status_ids)
        owner_hints = torch.where(played_mask, torch.full_like(owner_hints, self.max_players), owner_hints)

        for trick_slot in range(self.max_players):
            trick_card = current_trick_cards[:, trick_slot]
            valid = trick_card.ge(0)
            if not valid.any():
                continue
            card_index = trick_card.clamp_min(0)
            status_ids.scatter_(1, card_index.unsqueeze(1), torch.where(valid, torch.full_like(card_index, 3), torch.zeros_like(card_index)).unsqueeze(1))
            hinted_owner = torch.where(
                valid,
                current_trick_players[:, trick_slot].clamp(min=0, max=self.max_players - 1),
                torch.zeros_like(card_index),
            )
            owner_hints.scatter_(1, card_index.unsqueeze(1), torch.where(valid, hinted_owner, torch.full_like(hinted_owner, self.max_players + 1)).unsqueeze(1))

        for seat in range(self.max_players):
            public_card = public_cards[:, seat]
            valid = public_card.ge(0)
            if not valid.any():
                continue
            card_index = public_card.clamp_min(0)
            status_ids.scatter_(1, card_index.unsqueeze(1), torch.where(valid, torch.full_like(card_index, 4), torch.zeros_like(card_index)).unsqueeze(1))
            owner_hints.scatter_(1, card_index.unsqueeze(1), torch.where(valid, torch.full_like(card_index, seat), torch.full_like(card_index, self.max_players + 1)).unsqueeze(1))

        status_embed = self.card_status_embedding(status_ids)
        owner_embed = self.owner_hint_embedding(owner_hints.clamp_max(self.belief_classes - 1))
        card_tokens = base_cards + status_embed + owner_embed

        card_summary = self.card_summary_token.expand(batch_size, -1, -1) + self.scalar_to_summary(
            scalar_embedding + seat_summary + history_summary
        ).unsqueeze(1)
        encoded = self.card_encoder(torch.cat([card_summary, card_tokens], dim=1))
        return encoded[:, 0, :], encoded[:, 1:, :]

    def _history_context(self, observation: Dict[str, Tensor], scalar_embedding: Tensor) -> tuple[Tensor, Tensor]:
        history_tokens = observation["history_tokens"].long()
        batch_size = history_tokens.shape[0]
        device = history_tokens.device
        history_valid = history_tokens.ne(0).long()

        positions = torch.arange(1, history_tokens.shape[1] + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        history_embeddings = (
            self.history_embedding(history_tokens)
            + self.history_position_embedding(positions)
            + self.history_validity_embedding(history_valid)
        )
        memory_outputs, _ = self.history_memory(history_embeddings)
        lengths = history_valid.sum(dim=1).clamp_min(1) - 1
        memory_state = memory_outputs[torch.arange(batch_size, device=device), lengths]

        summary = self.history_summary_token.expand(batch_size, -1, -1) + self.scalar_to_summary(
            scalar_embedding + memory_state
        ).unsqueeze(1)
        encoded = self.history_encoder(torch.cat([summary, history_embeddings], dim=1))
        return encoded[:, 0, :], memory_state

    def _seat_context(self, observation: Dict[str, Tensor], scalar_embedding: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = observation["bids"].shape[0]
        device = observation["bids"].device
        seat_ids = torch.arange(self.max_players, device=device).unsqueeze(0).expand(batch_size, -1)
        player_count = observation["player_count"].long().unsqueeze(1)
        valid_mask = seat_ids.lt(player_count).long()
        seat_index = observation["seat_index"].long().unsqueeze(1)
        dealer_index = observation["dealer_index"].long().unsqueeze(1)
        leader_index = observation["leader_index"].long().unsqueeze(1)

        trick_positions = torch.full((batch_size, self.max_players), self.max_players, dtype=torch.long, device=device)
        trick_players = observation["current_trick_players"].long()
        for trick_slot in range(self.max_players):
            slot_players = trick_players[:, trick_slot]
            valid = slot_players.ge(0)
            if not valid.any():
                continue
            trick_positions.scatter_(
                1,
                slot_players.clamp_min(0).unsqueeze(1),
                torch.where(valid, torch.full_like(slot_players, trick_slot), torch.full_like(slot_players, self.max_players)).unsqueeze(1),
            )

        public_cards = observation["public_card_by_player"].long()
        public_known = public_cards.ge(0).float().unsqueeze(-1)
        public_card_embed = torch.zeros((batch_size, self.max_players, self.embed_dim), device=device)
        valid_public = public_cards.ge(0)
        if valid_public.any():
            public_card_embed = torch.where(
                valid_public.unsqueeze(-1),
                self.card_id_embedding(public_cards.clamp_min(0)),
                public_card_embed,
            )

        seat_numeric = torch.stack(
            [
                observation["bids"].float(),
                observation["tricks_won"].float(),
                observation["cumulative_scores"].float(),
                public_known.squeeze(-1),
            ],
            dim=-1,
        )
        seat_tokens = (
            self.seat_embedding(seat_ids)
            + self.relative_seat_embedding(((seat_ids - seat_index) % self.max_players).clamp(max=self.max_players))
            + self.relative_seat_embedding(((seat_ids - dealer_index) % self.max_players).clamp(max=self.max_players))
            + self.relative_seat_embedding(((seat_ids - leader_index) % self.max_players).clamp(max=self.max_players))
            + self.trick_position_embedding(trick_positions)
            + self.seat_validity_embedding(valid_mask)
            + self.seat_numeric_encoder(seat_numeric)
            + public_card_embed
        )
        summary = self.seat_summary_token.expand(batch_size, -1, -1) + self.scalar_to_summary(scalar_embedding).unsqueeze(1)
        encoded = self.seat_encoder(torch.cat([summary, seat_tokens], dim=1))
        return encoded[:, 0, :], encoded[:, 1:, :]

    def _score_card_actions(self, card_tokens: Tensor, play_latent: Tensor, seat_tokens: Tensor) -> Tensor:
        seat_summary = seat_tokens.mean(dim=1)
        query = self.play_query(play_latent + seat_summary).unsqueeze(1)
        candidates = self.play_candidate(torch.cat([card_tokens, seat_summary.unsqueeze(1).expand_as(card_tokens)], dim=-1))
        scores = self.play_score(torch.tanh(candidates + query)).squeeze(-1)
        return scores

    def _scalar_features(self, observation: Dict[str, Tensor]) -> Tensor:
        keys = ["bids", "tricks_won", "cumulative_scores"]
        scalars = [observation[key].float() for key in keys]
        extras = [
            observation["trump_suit"].float().unsqueeze(-1),
            observation["lead_suit"].float().unsqueeze(-1),
            observation["seat_index"].float().unsqueeze(-1),
            observation["dealer_index"].float().unsqueeze(-1),
            observation["leader_index"].float().unsqueeze(-1),
            observation["player_count"].float().unsqueeze(-1),
            observation["hand_size"].float().unsqueeze(-1),
            observation["one_card_mode"].float().unsqueeze(-1),
            observation["phase"].float().unsqueeze(-1),
            observation["round_index"].float().unsqueeze(-1),
            observation["legal_action_mask"].float().sum(dim=-1, keepdim=True),
        ]
        return torch.cat(scalars + extras, dim=-1)


@dataclass
class PolicyAgent:
    policy: WhistPolicyNetwork
    device: str = "cpu"
    greedy: bool = False

    def _policy_device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def select_action(self, observation: Mapping[str, object]) -> int:
        return self.recommend(observation).chosen_action

    def recommend(self, observation: Mapping[str, object], top_k: int = 5) -> ActionRecommendation:
        if self.policy.training:
            self.policy.eval()
        with torch.no_grad():
            batch = batch_observations([observation], device=self._policy_device())
            outputs = self.policy.forward_with_aux(batch)
            legal_mask = batch["legal_action_mask"]
            filtered_logits = masked_logits(outputs.logits, legal_mask)
            distribution = torch.distributions.Categorical(logits=filtered_logits)
            action = torch.argmax(filtered_logits, dim=-1) if self.greedy else distribution.sample()
            probabilities = F.softmax(filtered_logits, dim=-1)
            legal_actions = torch.nonzero(legal_mask[0], as_tuple=False).flatten().tolist()
            top_count = min(top_k, len(legal_actions))
            top_probs, top_indices = torch.topk(probabilities[0], k=top_count)
            return ActionRecommendation(
                chosen_action=int(action.item()),
                top_actions=[
                    RankedAction(
                        action=int(index.item()),
                        probability=float(probability.item()),
                        logit=float(filtered_logits[0, index].item()),
                    )
                    for probability, index in zip(top_probs, top_indices)
                ],
                legal_actions=[int(item) for item in legal_actions],
                value=float(outputs.values.item()),
            )
