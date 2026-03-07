from __future__ import annotations

from dataclasses import dataclass
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
    keys = observations[0].keys()
    return {
        key: torch.stack([tensorize_observation(obs, device=device)[key] for obs in observations], dim=0)
        for key in keys
    }


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


class WhistPolicyNetwork(nn.Module):
    def __init__(self, embed_dim: int = 128, history_vocab_size: int = 256, max_players: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_players = max_players
        self.card_embedding = nn.Embedding(53, embed_dim)
        self.history_embedding = nn.Embedding(history_vocab_size, embed_dim, padding_idx=0)
        self.scalar_encoder = nn.Sequential(
            nn.Linear((max_players * 3) + 11, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.torso = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.bid_head = nn.Linear(embed_dim, 9)
        self.play_head = nn.Linear(embed_dim, 52)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, observation: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        hand_mask = observation["hand_mask"].float()
        played_mask = observation["played_card_mask"].float()
        current_trick_cards = observation["current_trick_cards"].long() + 1
        public_cards = observation["public_card_by_player"].long() + 1
        history_tokens = observation["history_tokens"].long()

        card_table = self.card_embedding.weight[1:]
        hand_embedding = self._mask_pool(hand_mask, card_table)
        played_embedding = self._mask_pool(played_mask, card_table)
        trick_embedding = self._lookup_pool(current_trick_cards)
        public_embedding = self._lookup_pool(public_cards)

        scalar_features = self._scalar_features(observation)
        scalar_embedding = self.scalar_encoder(scalar_features)

        history_embedding = self.history_embedding(history_tokens)
        history_encoded = self.history_encoder(history_embedding)
        history_mask = history_tokens.ne(0).float().unsqueeze(-1)
        history_sum = (history_encoded * history_mask).sum(dim=1)
        history_count = history_mask.sum(dim=1).clamp_min(1.0)
        history_pooled = history_sum / history_count

        torso_input = torch.cat(
            [hand_embedding, played_embedding, trick_embedding, public_embedding, history_pooled + scalar_embedding], dim=-1
        )
        latent = self.torso(torso_input)
        logits = torch.cat([self.bid_head(latent), self.play_head(latent)], dim=-1)
        values = self.value_head(latent).squeeze(-1)
        return logits, values

    def _mask_pool(self, mask: Tensor, table: Tensor) -> Tensor:
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(mask, table) / counts

    def _lookup_pool(self, card_ids: Tensor) -> Tensor:
        embedded = self.card_embedding(card_ids.clamp_min(0))
        valid = card_ids.gt(0).float().unsqueeze(-1)
        pooled = (embedded * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp_min(1.0)
        return pooled / counts

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

    def select_action(self, observation: Mapping[str, object]) -> int:
        return self.recommend(observation).chosen_action

    def recommend(self, observation: Mapping[str, object], top_k: int = 5) -> ActionRecommendation:
        self.policy.eval()
        with torch.no_grad():
            batch = batch_observations([observation], device=torch.device(self.device))
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
