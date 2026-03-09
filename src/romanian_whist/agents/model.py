from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
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
    keys = tensorized[0].keys()
    batches = {}  # type: Dict[str, Tensor]
    for key in keys:
        stacked = torch.stack([obs[key] for obs in tensorized], dim=0)
        if device is not None:
            batches[key] = stacked.to(device=device)
        else:
            batches[key] = stacked
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


@dataclass(frozen=True)
class PolicyNetworkConfig:
    embed_dim: int = 128
    history_vocab_size: int = 256
    max_players: int = 6
    max_history_tokens: int = 64
    card_encoder_layers: int = 2
    history_encoder_layers: int = 2
    branch_hidden_multiplier: int = 2


class WhistPolicyNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        history_vocab_size: int = 256,
        max_players: int = 6,
        max_history_tokens: int = 64,
        card_encoder_layers: int = 2,
        history_encoder_layers: int = 2,
        branch_hidden_multiplier: int = 2,
    ):
        super().__init__()
        self.config = PolicyNetworkConfig(
            embed_dim=embed_dim,
            history_vocab_size=history_vocab_size,
            max_players=max_players,
            max_history_tokens=max_history_tokens,
            card_encoder_layers=card_encoder_layers,
            history_encoder_layers=history_encoder_layers,
            branch_hidden_multiplier=branch_hidden_multiplier,
        )
        self.embed_dim = embed_dim
        self.max_players = max_players
        self.max_history_tokens = max_history_tokens
        self.card_token_count = 1 + 52 + 52 + max_players + max_players
        self.card_embedding = nn.Embedding(53, embed_dim)
        self.history_embedding = nn.Embedding(history_vocab_size, embed_dim, padding_idx=0)
        self.card_type_embedding = nn.Embedding(5, embed_dim)
        self.card_seat_embedding = nn.Embedding(max_players + 1, embed_dim)
        self.card_position_embedding = nn.Embedding(self.card_token_count, embed_dim)
        self.history_position_embedding = nn.Embedding(max_history_tokens + 1, embed_dim)
        self.scalar_to_card_context = nn.Sequential(
            nn.Linear((max_players * 3) + 11, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear((max_players * 3) + 11, embed_dim),
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
        self.card_encoder = nn.TransformerEncoder(card_encoder_layer, num_layers=card_encoder_layers)
        self.history_encoder = nn.TransformerEncoder(history_encoder_layer, num_layers=history_encoder_layers)
        self.card_summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.history_summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        shared_hidden = embed_dim * branch_hidden_multiplier
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim * 3, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.bid_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.play_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.value_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Linear(shared_hidden, embed_dim),
            nn.GELU(),
        )
        self.bid_head = nn.Linear(embed_dim, 9)
        self.play_head = nn.Linear(embed_dim, 52)
        self.value_head = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.card_summary_token, std=0.02)
        nn.init.normal_(self.history_summary_token, std=0.02)

    @classmethod
    def from_config(cls, config: PolicyNetworkConfig) -> "WhistPolicyNetwork":
        return cls(
            embed_dim=config.embed_dim,
            history_vocab_size=config.history_vocab_size,
            max_players=config.max_players,
            max_history_tokens=config.max_history_tokens,
            card_encoder_layers=config.card_encoder_layers,
            history_encoder_layers=config.history_encoder_layers,
            branch_hidden_multiplier=config.branch_hidden_multiplier,
        )

    def config_dict(self) -> Dict[str, int]:
        return asdict(self.config)

    def forward(self, observation: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        scalar_features = self._scalar_features(observation)
        scalar_embedding = self.scalar_encoder(scalar_features)
        card_context = self._card_context(observation, scalar_features)
        history_context = self._history_context(observation, scalar_features)
        shared_input = torch.cat([card_context, history_context, scalar_embedding], dim=-1)
        shared_latent = self.shared_trunk(shared_input)
        bid_latent = self.bid_tower(torch.cat([shared_latent, history_context, scalar_embedding], dim=-1))
        play_latent = self.play_tower(torch.cat([shared_latent, card_context, history_context], dim=-1))
        value_latent = self.value_tower(torch.cat([shared_latent, card_context, scalar_embedding], dim=-1))
        logits = torch.cat([self.bid_head(bid_latent), self.play_head(play_latent)], dim=-1)
        values = self.value_head(value_latent).squeeze(-1)
        return logits, values

    def _card_context(self, observation: Dict[str, Tensor], scalar_features: Tensor) -> Tensor:
        batch_size = observation["hand_mask"].shape[0]
        device = observation["hand_mask"].device
        card_ids = torch.arange(1, 53, device=device).unsqueeze(0).expand(batch_size, -1)
        hand_valid = observation["hand_mask"].bool()
        played_valid = observation["played_card_mask"].bool()
        trick_cards = observation["current_trick_cards"].long() + 1
        trick_valid = observation["current_trick_cards"].ge(0)
        public_cards = observation["public_card_by_player"].long() + 1
        public_valid = observation["public_card_by_player"].ge(0)

        pad_seat = self.max_players
        hand_seats = torch.full((batch_size, 52), pad_seat, dtype=torch.long, device=device)
        played_seats = torch.full((batch_size, 52), pad_seat, dtype=torch.long, device=device)
        trick_seats = observation["current_trick_players"].long().clamp_min(0)
        trick_seats = torch.where(trick_valid, trick_seats, torch.full_like(trick_seats, pad_seat))
        public_seats = torch.arange(self.max_players, device=device).unsqueeze(0).expand(batch_size, -1)

        type_ids = [
            torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            torch.full((batch_size, 52), 1, dtype=torch.long, device=device),
            torch.full((batch_size, 52), 2, dtype=torch.long, device=device),
            torch.full((batch_size, self.max_players), 3, dtype=torch.long, device=device),
            torch.full((batch_size, self.max_players), 4, dtype=torch.long, device=device),
        ]
        card_token_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                card_ids,
                card_ids,
                trick_cards.clamp_min(0),
                public_cards.clamp_min(0),
            ],
            dim=1,
        )
        seat_ids = torch.cat(
            [
                torch.full((batch_size, 1), pad_seat, dtype=torch.long, device=device),
                hand_seats,
                played_seats,
                trick_seats,
                public_seats,
            ],
            dim=1,
        )
        valid_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                hand_valid,
                played_valid,
                trick_valid,
                public_valid,
            ],
            dim=1,
        )
        token_embeddings = (
            self.card_embedding(card_token_ids.clamp_min(0))
            + self.card_type_embedding(torch.cat(type_ids, dim=1))
            + self.card_seat_embedding(seat_ids)
            + self.card_position_embedding(
                torch.arange(self.card_token_count, device=device).unsqueeze(0).expand(batch_size, -1)
            )
        )
        token_embeddings[:, :1, :] = token_embeddings[:, :1, :] + self.card_summary_token + self.scalar_to_card_context(
            scalar_features
        ).unsqueeze(1)
        encoded = self.card_encoder(token_embeddings, src_key_padding_mask=~valid_mask)
        return encoded[:, 0, :]

    def _history_context(self, observation: Dict[str, Tensor], scalar_features: Tensor) -> Tensor:
        history_tokens = observation["history_tokens"].long()
        batch_size = history_tokens.shape[0]
        device = history_tokens.device
        history_valid = history_tokens.ne(0)
        summary_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        history_embeddings = self.history_embedding(history_tokens)
        position_ids = torch.arange(1, history_tokens.shape[1] + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        history_embeddings = history_embeddings + self.history_position_embedding(position_ids)
        summary_token = self.history_summary_token + self.scalar_encoder(scalar_features).unsqueeze(1)
        history_input = torch.cat([summary_token, history_embeddings], dim=1)
        history_mask = torch.cat([summary_valid, history_valid], dim=1)
        encoded = self.history_encoder(history_input, src_key_padding_mask=~history_mask)
        return encoded[:, 0, :]

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
        self.policy.eval()
        with torch.no_grad():
            batch = batch_observations([observation], device=self._policy_device())
            logits, values = self.policy(batch)
            legal_mask = batch["legal_action_mask"]
            filtered_logits = masked_logits(logits, legal_mask)
            distribution = torch.distributions.Categorical(logits=filtered_logits)
            if self.greedy:
                action = torch.argmax(filtered_logits, dim=-1)
            else:
                action = distribution.sample()
            probabilities = torch.softmax(filtered_logits, dim=-1)
            legal_actions = torch.nonzero(legal_mask[0], as_tuple=False).flatten().tolist()
            top_count = min(top_k, len(legal_actions))
            top_probs, top_indices = torch.topk(probabilities[0], k=top_count)
            recommendation = ActionRecommendation(
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
                value=float(values.item()),
            )
        return recommendation
